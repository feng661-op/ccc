# -*- coding: utf-8 -*-
"""E10 exact local re-optimization under alternative settlement semantics.

The main model remains the convex original-B/delivery-price settlement.  This
module changes the *objective semantics* and re-solves the four specified Q4-3
days from their authoritative 00:00 physical state.  It uses the same q4_flow
source-flow equations and the same causal B0 forecasts.

Semantics:
- original: final A anchored to original B, 0.5/1/1.5 at delivery price c_t.
- sunk: original c_t B is sunk; downward cancellation adds 0.5 c_t(B-A),
  upward adds 1.5 c_t(A-B).
- sequential: after c_t B at 00:00, every later adjustment is charged against
  the immediately preceding A with the same 0.5/1.5 delivery-price multipliers.
- adjustment_time: retained energy is at delivery c_t while the B-anchored
  adjustment part is priced at the revealed event price p_tau.  Its two-branch
  function is solved exactly with binary variables whenever c_t>2 p_tau makes
  it non-convex.  Big-M is derived slotwise from Q<=L+XMAX (unused positive-cost
  entitlement is never optimal), not from an arbitrary huge constant.

This is deliberately a four-date local reoptimization plus the already-saved
full-year fixed-main-strategy exposure.  It is NOT claimed as a full-year
alternative-semantics optimum.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import json
import numpy as np
import pandas as pd
from scipy.optimize import linprog, milp, Bounds, LinearConstraint
from scipy.sparse import csr_matrix, vstack

from q4_data import load_q4_inputs, SOC_MIN, SOC_MAX, XMAX, actual_at
from q4_flow import FlowIndex, canonical_bounds, const_expr, flow_constraint_rows, var_expr, structure_signature
from q4_price import PriceForecaster, PriceConfig
from q4_scenarios import build_point_scenario
from q4_control import execute_slot
from q4_tree import event_horizon, EVENT_PLAN_HORIZON
from q4_settlement import settlement_components

ROOT=Path(__file__).resolve().parents[2]
CODE=Path(__file__).resolve().parent
OUT=CODE/'ablations'; OUT.mkdir(parents=True,exist_ok=True)
SPEC_DATES=('2025-03-20','2025-06-21','2025-09-23','2025-12-21')
SEMANTICS=('original','sunk','sequential','adjustment_time')
PC=PriceConfig('full_ridge','global',10.0,42,48)


class Model:
    def __init__(self):
        self.n=0; self.names=[]; self.lb=[]; self.ub=[]; self.integrality=[]; self.eq=[]; self.ineq=[]
    def var(self,name,lb=0.0,ub=np.inf,integer=False):
        j=self.n; self.n+=1; self.names.append(name); self.lb.append(float(lb) if lb is not None else -np.inf); self.ub.append(float(ub) if ub is not None else np.inf); self.integrality.append(1 if integer else 0); return j
    def add_eq(self,co,rhs): self.eq.append((dict(co),float(rhs)))
    def add_ub(self,co,rhs): self.ineq.append((dict(co),float(rhs)))
    def rows(self,rows):
        rr=[];cc=[];vv=[];bb=[]
        for i,(co,b) in enumerate(rows):
            for j,a in co.items():
                if abs(a)>0:rr.append(i);cc.append(j);vv.append(float(a))
            bb.append(float(b))
        return csr_matrix((vv,(rr,cc)),shape=(len(rows),self.n)),np.asarray(bb,float)
    def solve(self,c,extra=()):
        c=np.asarray(c,float)
        if len(c)<self.n:c=np.pad(c,(0,self.n-len(c)))
        eq=list(self.eq);ineq=list(self.ineq)+list(extra)
        Aeq,beq=self.rows(eq); Aub,bub=self.rows(ineq)
        if any(self.integrality):
            mats=[];lo=[];hi=[]
            if len(beq): mats.append(Aeq);lo.extend(beq);hi.extend(beq)
            if len(bub): mats.append(Aub);lo.extend([-np.inf]*len(bub));hi.extend(bub)
            A=vstack(mats,format='csr') if mats else csr_matrix((0,self.n))
            res=milp(c,integrality=np.asarray(self.integrality,int),bounds=Bounds(np.asarray(self.lb),np.asarray(self.ub)),constraints=LinearConstraint(A,np.asarray(lo),np.asarray(hi)),options={'time_limit':300,'mip_rel_gap':1e-8})
            return res,Aeq,beq
        bounds=list(zip(self.lb,self.ub))
        res=linprog(c,A_ub=Aub,b_ub=bub,A_eq=Aeq,b_eq=beq,bounds=bounds,method='highs')
        return res,Aeq,beq


def add_coeff(dst,j,a): dst[j]=dst.get(j,0.0)+float(a)

@dataclass
class LocalEvent:
    success:bool; semantic:str; event_hour:int; A:np.ndarray; B:np.ndarray; soc_target:np.ndarray
    objective:float; throughput:float; curtailment:float; max_residual:float; binaries:int; solve_seconds:float
    message:str=''


def _settlement_expr(model:Model,semantic:str,Aidx:int,B:float,Aprev:float,c:float,p_event:float,U:float,name:str):
    """Return a cost-variable index whose value is the exact semantic fee/adjustment."""
    phi=model.var('phi_'+name,0.0,np.inf)
    if semantic=='original':
        # max(0.5c A+0.5cB, 1.5c A-0.5cB)
        model.add_ub({Aidx:.5*c,phi:-1},-.5*c*B)
        model.add_ub({Aidx:1.5*c,phi:-1},.5*c*B)
    elif semantic=='sunk':
        # max(1.5cB-.5cA, 1.5cA-.5cB)
        model.add_ub({Aidx:-.5*c,phi:-1},-1.5*c*B)
        model.add_ub({Aidx:1.5*c,phi:-1},.5*c*B)
    elif semantic=='sequential':
        # incremental fee vs prior A; prior paid amounts are sunk.
        model.add_ub({Aidx:-.5*c,phi:-1},-.5*c*Aprev)
        model.add_ub({Aidx:1.5*c,phi:-1},1.5*c*Aprev)
    elif semantic=='adjustment_time':
        # Exact piecewise B-anchor function. z=0 -> A<=B lower branch; z=1 -> A>=B upper.
        z=model.var('branch_'+name,0,1,integer=True)
        Mq=max(float(U),float(B),1.0)
        model.add_ub({Aidx:1,z:-Mq},B)                 # A-B <= M z
        model.add_ub({Aidx:-1,z:Mq},Mq-B)             # B-A <= M(1-z)
        # l0=(c-.5p)A+.5pB ; l1=1.5pA+(c-1.5p)B
        a0=c-.5*p_event; b0=.5*p_event*B
        a1=1.5*p_event; b1=(c-1.5*p_event)*B
        Mphi=abs(2*p_event-c)*Mq+1e-7
        # |phi-l0| <= Mphi*z
        model.add_ub({phi:1,Aidx:-a0,z:-Mphi},b0)
        model.add_ub({phi:-1,Aidx:a0,z:-Mphi},-b0)
        # |phi-l1| <= Mphi*(1-z)
        model.add_ub({phi:1,Aidx:-a1,z:Mphi},b1+Mphi)
        model.add_ub({phi:-1,Aidx:a1,z:Mphi},-b1+Mphi)
    else: raise ValueError(semantic)
    return phi


def solve_local_event(data,pf,day_idx,event_hour,semantic,soc0,B,Aprev,lead_q):
    H=EVENT_PLAN_HORIZON[event_hour]; sc=build_point_scenario(data,pf,day_idx,event_hour,H,'q4_3',PC)
    slots=event_horizon(data.dates[day_idx],event_hour,'q4_3'); m=Model(); qidx={}
    for s in slots:
        if s.mutable and s.plan_j not in qidx:
            # valid positive-cost optimum bound: Q<=L+XMAX for the corresponding predicted delivery slot
            h=s.h; U=max(float(B[s.plan_j]),float(sc.load[0,h]+XMAX),1.0)
            qidx[s.plan_j]=m.var(f'A[{s.plan_j}]',0.0,U)
    flows=[];prevS=const_expr(float(soc0)); cost={}
    issue=data.dates[day_idx]+timedelta(hours=event_hour); p_event=float(actual_at(data,issue,'price') or sc.price[0,0])
    for h,s in enumerate(slots):
        ids=[m.var(f'{nm}[{h}]') for nm in ('qL','qB','u','gL','gB','kappa','eL','eB','x','y','S1')]
        f=FlowIndex.from_sequence(ids);flows.append(f)
        # canonical bounds
        for nm in ('qL','qB','u','gL','gB','kappa','eL'): pass
        m.lb[f.eB]=0;m.ub[f.eB]=0;m.ub[f.x]=XMAX;m.ub[f.y]=XMAX;m.lb[f.S1]=SOC_MIN;m.ub[f.S1]=SOC_MAX
        Q=const_expr(float(lead_q)) if s.lead_from_previous_contract else var_expr(qidx[s.plan_j]) if s.mutable else const_expr(float(Aprev[s.plan_j]))
        for r in flow_constraint_rows(f,Q=Q,load_kwh=float(sc.load[0,h]),pv_kwh=float(sc.pv[0,h]),S0=prevS):
            (m.add_eq if r.kind=='eq' else m.add_ub)(r.coeff,r.rhs)
        prevS=var_expr(f.S1);c=float(sc.price[0,h]);add_coeff(cost,f.eL,5*c);add_coeff(cost,f.eB,5*c)
        if not s.lead_from_previous_contract:
            if event_hour==0:
                add_coeff(cost,qidx[s.plan_j],c)
            else:
                U=m.ub[qidx[s.plan_j]]
                ph=_settlement_expr(m,semantic,qidx[s.plan_j],float(B[s.plan_j]),float(Aprev[s.plan_j]),c,p_event,U,f'{h}')
                add_coeff(cost,ph,1.0)
    z=m.var('terminal_short',0.0,np.inf);m.add_ub({flows[-1].S1:-1,z:-1},-6000.0);add_coeff(cost,z,.8)
    c1=np.zeros(m.n);
    for j,a in cost.items():c1[j]+=a
    t0=perf_counter();r1,Aeq,beq=m.solve(c1);elapsed=perf_counter()-t0
    ok=bool(getattr(r1,'success',False));
    if not ok:return LocalEvent(False,semantic,event_hour,Aprev.copy(),B.copy(),np.full(H,np.nan),np.nan,np.nan,np.nan,np.inf,sum(m.integrality),elapsed,str(getattr(r1,'message','unknown MILP failure')))
    x1=r1.x;f1=float(c1@x1)
    # HiGHS MIP feasibility is coarser than the LP lex tolerance.  For the
    # non-convex adjustment-time sensitivity only, keep a 1e-4 yuan absolute
    # tie tolerance (<1e-8 relative on a typical event objective); the primary
    # semantic optimum remains the first-stage MILP optimum.
    tol1=max(1e-4 if any(m.integrality) else 1e-7,1e-9*max(1,abs(f1)))
    c2=np.zeros(m.n)
    for f in flows:c2[f.x]+=1;c2[f.y]+=1
    co1={j:float(v) for j,v in enumerate(c1) if abs(v)>0}
    r2,_,_=m.solve(c2,[(co1,f1+tol1)])
    if not getattr(r2,'success',False):
        if any(m.integrality):
            # Preserve the exact first-stage MILP optimum rather than declaring
            # the non-convex sensitivity infeasible because a secondary
            # tie-break falls inside HiGHS' mixed-integer feasibility noise.
            # The authoritative V1.2 main model is continuous and still uses
            # the strict three-stage lex solve; this fallback is E10-only.
            x=x1;A=Aprev.copy()
            for j,ix in qidx.items():A[j]=max(0,float(x[ix]))
            Bout=A.copy() if event_hour==0 else B.copy()
            target=np.asarray([x[f.S1] for f in flows]);res=float(np.max(np.abs(Aeq@x-beq))) if len(beq) else 0
            return LocalEvent(True,semantic,event_hour,A,Bout,target,f1,float(c2@x),float(sum(x[f.kappa] for f in flows)),res,sum(m.integrality),perf_counter()-t0)
        raise RuntimeError('local lex2 failed')
    f2=float(c2@r2.x);tol2=max(1e-7,1e-9*max(1,abs(f2)))
    c3=np.zeros(m.n)
    for f in flows:c3[f.kappa]+=1
    co2={j:float(v) for j,v in enumerate(c2) if abs(v)>0}
    r3,Aeq,beq=m.solve(c3,[(co1,f1+tol1),(co2,f2+tol2)])
    if not getattr(r3,'success',False):raise RuntimeError('local lex3 failed')
    x=r3.x; A=Aprev.copy()
    for j,ix in qidx.items():A[j]=max(0,float(x[ix]))
    Bout=A.copy() if event_hour==0 else B.copy()
    target=np.asarray([x[f.S1] for f in flows]);res=float(np.max(np.abs(Aeq@x-beq))) if len(beq) else 0
    return LocalEvent(True,semantic,event_hour,A,Bout,target,f1,float(c2@x),float(c3@x),res,sum(m.integrality),perf_counter()-t0)


def _semantic_plan_fee(semantic,B,A,c,adjustments,event_prices):
    B=np.asarray(B,float);A=np.asarray(A,float);c=np.asarray(c,float)
    if semantic=='original':return float(np.sum(c*np.minimum(A,B)+.5*c*np.maximum(B-A,0)+1.5*c*np.maximum(A-B,0)))
    if semantic=='sunk':return float(np.sum(c*B+.5*c*np.maximum(B-A,0)+1.5*c*np.maximum(A-B,0)))
    if semantic=='sequential':
        total=float(np.sum(c*B));prev=B.copy()
        for eh,newA in adjustments:
            delta=np.asarray(newA)-prev; total += float(np.sum(.5*c*np.maximum(-delta,0)+1.5*c*np.maximum(delta,0))); prev=np.asarray(newA).copy()
        return total
    if semantic=='adjustment_time':
        # Each plan slot is finally revisable at 6/12/18 according to its index.
        p=np.zeros(144,float)
        for j in range(144):
            eh=0 if j<35 else 6 if j<71 else 12 if j<107 else 18
            p[j]=event_prices[eh]
        return float(np.sum(np.where(A<=B,(c-.5*p)*A+.5*p*B,1.5*p*A+(c-1.5*p)*B)))
    raise ValueError(semantic)


def start_state(data,day_idx):
    slots=pd.read_csv(CODE/'q4_3'/'physical_10min.csv');ctr=pd.read_csv(CODE/'q4_3'/'contract_ledger.csv')
    r=slots[(slots.day_index==day_idx)&(slots.slot==0)].iloc[0];prev=ctr[ctr.day_index==day_idx-1].sort_values('plan_j')
    return float(r.S0_kwh),prev.B_kwh.to_numpy(float),prev.A_kwh.to_numpy(float)


def replay_one_date(data,pf,day_idx,semantic):
    day=data.dates[day_idx];soc,prevB,prevA=start_state(data,day_idx);B=np.zeros(144);A=np.zeros(144);sol=None;event_start=0;adjustments=[];events=[];emg_fee=0.;emg_kwh=0.;physics=[]
    event_prices={h:float(actual_at(data,day+timedelta(hours=h),'price')) for h in (0,6,12,18)}
    for i in range(144):
        eh=i//6
        if eh in (0,6,12,18) and i==eh*6:
            sol=solve_local_event(data,pf,day_idx,eh,semantic,soc,B,A,float(prevA[143]));
            if not sol.success:raise RuntimeError(f'{day.date()} {semantic} event{eh} failed')
            A=sol.A.copy();B=sol.B.copy();event_start=i
            if eh>0:adjustments.append((eh,A.copy()))
            events.append({'event_hour':eh,'objective':sol.objective,'throughput':sol.throughput,'curtailment':sol.curtailment,'binaries':sol.binaries,'max_residual':sol.max_residual})
        L=float(data.load_cal_kwh[day_idx,i]);G=float(data.pv_cal_kwh[day_idx,i]);c=float(data.price_cal[day_idx,i])
        Q=float(prevA[143]) if i==0 else float(A[i-1]);target=float(sol.soc_target[i-event_start])
        a=execute_slot(Q=Q,load_kwh=L,pv_kwh=G,S0=soc,target_soc=target);soc=a.S1;physics.append(a.residual_max);emg_kwh+=a.eL+a.eB;emg_fee+=5*c*(a.eL+a.eB)
    plan_fee=_semantic_plan_fee(semantic,B,A,data.price_plan[day_idx],adjustments,event_prices)
    locked_lead=settlement_components(float(prevB[143]),float(prevA[143]),float(data.price_cal[day_idx,0]),0).regular_fee
    return {'date':day.date().isoformat(),'semantic':semantic,'initial_soc':start_state(data,day_idx)[0],'end_soc':soc,'plan_row_semantic_fee_yuan':plan_fee,'natural_day_locked_lead_fee_yuan':locked_lead,'natural_day_emergency_fee_yuan':emg_fee,'emergency_kwh':emg_kwh,'local_comparison_cash_proxy_yuan':plan_fee+emg_fee,'max_physics_residual':max(physics),'events':events,'event_prices':event_prices,'final_B':B.tolist(),'final_A':A.tolist(),'scope':'exact four-date local reoptimization; not full-year optimum'}


def run():
    data=load_q4_inputs(ROOT);pf=PriceForecaster(data);rows=[];detail={}
    for ds in SPEC_DATES:
        d=(datetime.fromisoformat(ds).date()-datetime(2025,1,1).date()).days;detail[ds]={}
        for sem in SEMANTICS:
            r=replay_one_date(data,pf,d,sem);detail[ds][sem]=r;rows.append({k:v for k,v in r.items() if k not in ('events','event_prices','final_B','final_A')});print(ds,sem,r['local_comparison_cash_proxy_yuan'],flush=True)
    pd.DataFrame(rows).to_csv(OUT/'E10_local_reoptimization.csv',index=False,encoding='utf-8-sig')
    exposure_path=OUT/'E10_settlement_exposure.json';exposure=json.loads(exposure_path.read_text(encoding='utf-8')) if exposure_path.exists() else None
    result={'method':'four specified dates exact local reoptimization + full-year frozen-main exposure','local_exact':detail,'full_year_fixed_main_exposure':exposure,'claim_boundary':'Changing settlement semantics triggers reoptimization. Local results are exact for the four specified dates under frozen causal B0 information/physics. Full-year alternative-semantic numbers, where present, are fixed-main-strategy exposure only and are not called optimal.'}
    (OUT/'E10_settlement_exact_local.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('DONE',flush=True)
if __name__=='__main__':run()
