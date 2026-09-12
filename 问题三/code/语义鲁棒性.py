# -*- coding: utf-8 -*-
from pathlib import Path
from datetime import datetime,timedelta
import sys,time,json,csv
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
from q3_data import *
from q3_opt import build_scenarios
from q3_sim import simulate_strategy
ROOT=HERE.parent.parent; data=load_q3_inputs(ROOT); E=EVAL_START

def save_run(name,s):
    np.savez_compressed(HERE/f'run_{name}.npz',B=s.B,A=s.A,A_stage=s.A_stage,charge=s.charge,discharge=s.discharge,emergency=s.emergency,curtail=s.curtail,grid_import=s.grid_import,unused_contract=s.unused_contract,supply_surplus=s.supply_surplus,battery_dump=s.battery_dump,execution_mode=np.asarray(s.execution_mode),physical_schema_version=np.asarray(2),soc00=s.soc00,soc24=s.soc24,soc_path=s.soc_path,plan_fee=s.plan_fee,adjusted_fee=s.adjusted_fee,natural_regular_fee=s.natural_regular_fee,emergency_fee=s.emergency_fee)
    return {'total_cost_yuan':float(np.sum(s.natural_regular_fee[E:]+s.emergency_fee[E:])), 'regular_cost_yuan':float(np.sum(s.natural_regular_fee[E:])), 'emergency_cost_yuan':float(np.sum(s.emergency_fee[E:])), 'emergency_kwh':float(np.sum(s.emergency[E:])), 'soc_min_kwh':float(np.min(s.soc_path[E:])), 'soc_max_kwh':float(np.max(s.soc_path[E:])), 'solver_max_eq_residual':float(max(r['max_eq_residual'] for r in s.event_rows)), 'solver_max_ub_violation':float(max(r['max_ub_violation'] for r in s.event_rows)), 'revision_count':len(s.revision_rows)}

def cached_run_metrics(name):
    z=np.load(HERE/f'run_{name}.npz')
    return {'total_cost_yuan':float(np.sum(z['natural_regular_fee'][E:]+z['emergency_fee'][E:])), 'regular_cost_yuan':float(np.sum(z['natural_regular_fee'][E:])), 'emergency_cost_yuan':float(np.sum(z['emergency_fee'][E:])), 'emergency_kwh':float(np.sum(z['emergency'][E:])), 'soc_min_kwh':float(np.min(z['soc_path'][E:])), 'soc_max_kwh':float(np.max(z['soc_path'][E:])), 'revision_count':int(np.sum(np.abs(np.diff(z['A_stage'],axis=1))>1e-7)), 'source':'cached_completed_full_reoptimization'}

def revision_clock_price(hour):
    t=data.dates[0]+timedelta(hours=hour); return price_for_time(data,t)

def revision_time_fee_slots(B,Astage):
    """Original-anchor endpoint accounting, only adjustment-price basis switched.
    Adjustment component is priced at the latest revision event that changed the
    accepted quantity for that slot; retained quantity keeps delivery price.
    """
    p=data.price_plan; out=np.zeros(144); last=np.zeros(144,int); hours=(0,6,12,18)
    for r in (1,2,3):
        changed=np.abs(Astage[r]-Astage[r-1])>1e-8; last[changed]=r
    Afinal=Astage[3]
    for j in range(144):
        b=float(B[j]); a=float(Afinal[j]); pd=float(p[j])
        if abs(a-b)<=1e-8: out[j]=pd*b; continue
        r=int(last[j]); pa=revision_clock_price(hours[r]) if r else pd
        if a<b: out[j]=pd*a+0.5*pa*(b-a)
        else: out[j]=pd*b+1.5*pa*(a-b)
    return out

def exact_revision_time_milp_12(day_idx,zD):
    """Exact nonconvex local diagnostic for 12:00-15:00 (18 ten-minute slots).
    B is the original anchor, current q is common across scenarios, and the
    adjustment component uses the 12:00 revision price. No convex relaxation.
    This short point-forecast diagnostic is not the full Q2-inherited objective.
    """
    H=18; bundle=build_scenarios(data,day_idx,12,H,True,1); K=bundle.scenarios.shape[0]; w=bundle.weights
    B=zD['B'][day_idx]; soc0=float(zD['soc_path'][day_idx,72]); js=np.arange(71,71+H); QMAX=5000.0; BIG=1e6
    names=[]; lo=[]; hi=[]; integ=[]; c=[]
    def add(name,l=0,u=np.inf,integer=0,cost=0):
        i=len(names);names.append(name);lo.append(l);hi.append(u);integ.append(integer);c.append(cost);return i
    q={j:add(('q',j),0,QMAX) for j in js}; phi={j:add(('phi',j),0,BIG,0,1.0) for j in js}; zz={j:add(('z',j),0,1,1,0) for j in js}
    x={};y={};e={};s={};rt={}
    for k in range(K):
        for t in range(H):
            x[k,t]=add(('x',k,t),0,XMAX);y[k,t]=add(('y',k,t),0,XMAX);e[k,t]=add(('e',k,t),0,BIG,0,float(w[k]*5*bundle.slot_prices[t]))
        for t in range(H+1):s[k,t]=add(('s',k,t),SOC_MIN,SOC_MAX)
        rt[k]=add(('rt',k),0,BIG,0,float(w[k]*.8))
    rows=[];lb=[];ub=[]
    def con(row,l=-np.inf,u=np.inf):rows.append(row);lb.append(l);ub.append(u)
    for k in range(K):
        con({s[k,0]:1},soc0,soc0)
        for t in range(H):
            j=int(js[t]); con({s[k,t+1]:1,s[k,t]:-1,x[k,t]:-ETA_C,y[k,t]:1/ETA_D},0,0)
            con({q[j]:-1,x[k,t]:1,y[k,t]:-1,e[k,t]:-1},-np.inf,-float(bundle.scenarios[k,t]))
        con({s[k,H]:-1,rt[k]:-1},-np.inf,-6000.0)
    pa=revision_clock_price(12); nonconvex=0
    for t,j0 in enumerate(js):
        j=int(j0); b=float(B[j]); pd=float(data.price_plan[j]); z=zz[j]
        # z=0 down branch q<=B; z=1 up branch q>=B.
        con({q[j]:1,z:-QMAX},-np.inf,b)
        con({q[j]:-1,z:QMAX},-np.inf,QMAX-b)
        ad=pd-.5*pa; cd=.5*pa*b; au=1.5*pa; cu=(pd-1.5*pa)*b
        if pd>2*pa+1e-12: nonconvex+=1
        # Exact branch equalities with one-sided big-M relaxation.
        # z=0 => phi=ad*q+cd; z=1 relaxes this pair.
        con({phi[j]:1,q[j]:-ad,z:-BIG},-np.inf,cd)
        con({phi[j]:1,q[j]:-ad,z:BIG},cd,np.inf)
        # z=1 => phi=au*q+cu; z=0 relaxes this pair.
        con({phi[j]:1,q[j]:-au,z:BIG},-np.inf,cu+BIG)
        con({phi[j]:1,q[j]:-au,z:-BIG},cu-BIG,np.inf)
    rr=[];cc=[];vv=[]
    for r,row in enumerate(rows):
        for j,v in row.items():rr.append(r);cc.append(j);vv.append(v)
    A=coo_matrix((vv,(rr,cc)),shape=(len(rows),len(names))).tocsr(); t0=time.time()
    res=milp(np.asarray(c),integrality=np.asarray(integ),bounds=Bounds(np.asarray(lo),np.asarray(hi)),constraints=LinearConstraint(A,np.asarray(lb),np.asarray(ub)),options={'time_limit':30.0,'mip_rel_gap':1e-8})
    secs=time.time()-t0
    if not res.success: raise RuntimeError(f'MILP failed {data.dates[day_idx].date()}: {res.message}')
    qv=np.asarray([res.x[q[int(j)]] for j in js]); zv=np.asarray([round(res.x[zz[int(j)]]) for j in js],int)
    return {'date':data.dates[day_idx].date().isoformat(),'event_hour':12,'objective':float(res.fun),'solve_seconds':secs,'nonconvex_delivery_slots':int(nonconvex),'up_branch_slots':int(zv.sum()),'down_branch_slots':int(len(zv)-zv.sum()),'q_min':float(qv.min()),'q_max':float(qv.max()),'status':str(res.message)}

sunk_m=json.loads((HERE/'metric_D_sunk.json').read_text(encoding='utf-8'))
step_m=json.loads((HERE/'metric_D_stepwise.json').read_text(encoding='utf-8'))
# revision_time: one-switch full-year endpoint repricing on the main D trajectory.
zD=np.load(HERE/'run_D.npz'); slotfees=np.zeros((365,144))
for d in range(365):slotfees[d]=revision_time_fee_slots(zD['B'][d],zD['A_stage'][d])
rt_regular=0.0
for d in range(E,365):rt_regular+=slotfees[d-1,143]+slotfees[d,:143].sum()
rt_emergency=float(np.sum(zD['emergency'][E:]*5*data.price_calendar[None,:])); rt={'method':'full_year_fixed_policy_repricing_plus_exact_local_MILP','regular_cost_yuan':float(rt_regular),'emergency_cost_yuan':rt_emergency,'total_cost_yuan':float(rt_regular+rt_emergency)}
# Exact nonconvex MILP re-optimization audit at representative 12:00 events.
idx={d.date():i for i,d in enumerate(data.dates)}; audits=[]
for ds in ('2025-03-20','2025-06-21','2025-09-23','2025-12-21'):audits.append(exact_revision_time_milp_12(idx[datetime.fromisoformat(ds).date()],zD))
with open(HERE/'revision_time_milp_audit.csv','w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(audits[0].keys()));w.writeheader();w.writerows(audits)
main=json.loads((HERE/'metrics.json').read_text(encoding='utf-8'))['D']['total_cost_yuan']
out={'main_D_total_cost_yuan':main,'sunk_plan_plus_penalty':sunk_m,'stepwise_revision':step_m,'revision_time':rt,'revision_time_exact_milp_audit':audits,'deltas_vs_main':{'sunk':sunk_m['total_cost_yuan']-main,'stepwise':step_m['total_cost_yuan']-main,'revision_time_fixed_policy':rt['total_cost_yuan']-main}}
(HERE/'semantic_robustness.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
with open(HERE/'semantic_robustness.csv','w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f);w.writerow(['semantics','total_cost_yuan','delta_vs_main_yuan','method']);w.writerow(['main',main,0,'full_reoptimization']);w.writerow(['sunk_plan_plus_penalty',sunk_m['total_cost_yuan'],sunk_m['total_cost_yuan']-main,'full_reoptimization']);w.writerow(['stepwise_revision',step_m['total_cost_yuan'],step_m['total_cost_yuan']-main,'full_reoptimization']);w.writerow(['revision_time',rt['total_cost_yuan'],rt['total_cost_yuan']-main,rt['method']])
print(json.dumps(out,ensure_ascii=False,indent=2),flush=True)
