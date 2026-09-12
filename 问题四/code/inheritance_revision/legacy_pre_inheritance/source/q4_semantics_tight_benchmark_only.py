"""Exact event-time semantic adapter; production delivery solver is unchanged.

Expected regular fee has the SAME breakpoint B in all scenarios:
 sum_k w_k Phi(B,A;p_k,pe) = Phi(B,A;sum_k w_k p_k,pe).
CVaR penalizes emergency cost only, so this cash aggregation is exact. Net-load
and emergency-price recourse remain K-scenario. It removes duplicate fee
variables and binaries caused only by individual (not mean) delivery prices.
At B=0 only the A>=B linear branch exists; no binary is needed.
The runner records this adapter's hash separately from the unchanged base core.
"""
from q4_inheritance import *
import json
import pandas as pd
from q4_data import natural_interval_label
from q4_sim_v2 import regular_fee
from q4_flow import weighted_cvar

# Original core SHA256: 4ce3c15c319d34e7fe524ab79be1f7d938e9dd0acd0be29870f9c7697ec1b31b

def solve_intraday(bundle,prices,soc0,B,A0,lead,eh,cfg,event_price=None,fixed_contract=None):
    K,H=bundle.scenarios.shape; start=eh*6; startj=EVENT_PLAN_START[eh]
    cost=[]; bounds=[]; eqs=[]; rhs=[]; ubs=[]; urhs=[]; integers=[]
    def add(c=0.0,bound=(0,None),integer=False):
        i=len(cost);cost.append(float(c));bounds.append(bound);integers.append(int(integer));return i
    q={j:add() for j in range(startj,144)}
    if not cfg.allow_contract_revision or fixed_contract is not None:
        fixed=A0 if fixed_contract is None else np.asarray(fixed_contract,float)
        for j,ix in q.items(): bounds[ix]=(float(fixed[j]),float(fixed[j]))
    future={};x={};y={};e={};s={};r={}
    for k in range(K):
        w=bundle.weights[k]
        for t in range(H):
            if start+t>=145: future[k,t]=add(w*prices[k,t])
            x[k,t]=add(bound=(0,XMAX));y[k,t]=add(bound=(0,XMAX));e[k,t]=add(w*5*prices[k,t])
        for t in range(H+1): s[k,t]=add(bound=(SOC_MIN,SOC_MAX))
        r[k]=add(w*cfg.terminal_value)
    zeta=add(cfg.risk_lambda)
    excess={k:add(cfg.risk_lambda*bundle.weights[k]/(1-cfg.alpha)) for k in range(K)}
    for k in range(K):
        eqs.append({s[k,0]:1});rhs.append(float(soc0));risk={zeta:-1,excess[k]:-1}
        for t in range(H):
            eqs.append({s[k,t+1]:1,s[k,t]:-1,x[k,t]:-ETA_C,y[k,t]:1/ETA_D});rhs.append(0)
            row={x[k,t]:1,y[k,t]:-1,e[k,t]:-1};g=start+t;v=-bundle.scenarios[k,t]
            if g==0:v+=lead
            elif g<=144:row[q[g-1]]=-1
            else:row[future[k,t]]=-1
            ubs.append(row);urhs.append(v)
            if g<=144:risk[e[k,t]]=5*prices[k,t]
        ubs.append({s[k,H]:-1,r[k]:-1});urhs.append(-cfg.terminal_reserve)
        ubs.append(risk);urhs.append(0)
    cash_offset=0.0
    for j,ix in q.items():
        t=j+1-start;b=float(B[j]);p=float(np.dot(bundle.weights,prices[:,t]))
        event_clock=getattr(cfg,'settlement_clock','delivery')=='adjustment_time'
        pe=float(event_price) if event_clock else p
        U=max(b,float(A0[j]),float(np.max(bundle.scenarios[:,t])+XMAX),1.0)
        lines=((p-.5*pe,.5*pe*b),(1.5*pe,(p-1.5*pe)*b)) if event_clock else ((.5*p,.5*p*b),(1.5*p,-.5*p*b))
        if event_clock and b>0 and p>2*pe+1e-12:
            # Exact multiple-choice convex-hull formulation. It is materially
            # tighter than big-M epigraph switching, without changing the two
            # feasible segments or relaxing binary integrality.
            m0,b0=lines[0];m1,b1=lines[1]
            z=add(b1-b0,bound=(0,1),integer=True)
            qlow=add(m0,bound=(0,b));qhigh=add(m1,bound=(0,U));cash_offset+=b0
            eqs.append({ix:1,qlow:-1,qhigh:-1});rhs.append(0.)
            ubs.extend(({qlow:1,z:b},{qhigh:-1,z:b},{qhigh:1,z:-U}))
            urhs.extend((b,0.,0.))
            if bounds[ix][0]!=bounds[ix][1]:bounds[ix]=(0,U)
        else:
            phi=add(1.0)
            if b==0:lines=(lines[1],)
            for slope,intercept in lines:ubs.append({ix:slope,phi:-1});urhs.append(-intercept)
    N=len(cost)
    def sparse(rows):
        rr=[];cc=[];vv=[]
        for i,row in enumerate(rows):
            for j,v in row.items():rr.append(i);cc.append(j);vv.append(v)
        return coo_matrix((vv,(rr,cc)),shape=(len(rows),N)).tocsr()
    eq=sparse(eqs);ub=sparse(ubs);t0=perf_counter()
    if any(integers):
        mat=vstack([eq,ub],format='csr');lo=np.r_[rhs,np.full(len(urhs),-np.inf)];hi=np.r_[rhs,urhs]
        lb=[-np.inf if v[0] is None else v[0] for v in bounds];upper=[np.inf if v[1] is None else v[1] for v in bounds]
        sol=milp(cost,integrality=np.asarray(integers),bounds=Bounds(lb,upper),constraints=LinearConstraint(mat,lo,hi),options={'mip_rel_gap':1e-7,'time_limit':60})
    else:
        sol=linprog(cost,A_eq=eq,b_eq=rhs,A_ub=ub,b_ub=urhs,bounds=bounds,method='highs',options={'presolve':True})
    elapsed=perf_counter()-t0
    if not sol.success:raise RuntimeError(f'inherited event LP/MILP failed: {sol.message}')
    v=sol.x;out=A0.copy()
    for j,ix in q.items():out[j]=max(0,float(v[ix]))
    ss=np.array([[v[s[k,t]] for t in range(H+1)] for k in range(K)])
    ee=np.array([[v[e[k,t]] for t in range(H)] for k in range(K)])
    resid=max(float(np.max(np.abs(eq@v-rhs))),float(max(0,np.max(ub@v-urhs))))
    return InheritedEvent(out,B.copy(),ss,ee,np.quantile(ss,.25,axis=0),float(sol.fun)+cash_offset,elapsed,resid,K,
        list(bundle.source_days),prices,int(sum(integers)),float(getattr(sol,'mip_gap',0) or 0))

def solve_event(data,day_idx,eh,soc,B,A,lead,cfg,fixed_contract=None):
    bundle,prices=make_bundle(data,day_idx,eh,cfg)
    if eh==0:
        t0=perf_counter()
        sol=solve_midnight(bundle.scenarios,prices,soc,lead,cfg.risk_lambda,cfg.alpha,cfg.terminal_value)
        out=np.maximum(sol.q_current,0);out[np.abs(out)<1e-9]=0
        return InheritedEvent(out,out.copy(),sol.scenario_soc,sol.scenario_emergency,
            np.quantile(sol.scenario_soc,.25,axis=0),sol.objective,perf_counter()-t0,sol.max_residual,
            sol.scenario_count,list(bundle.source_days),prices)
    p=actual_at(data,data.dates[day_idx]+timedelta(hours=eh),'price')
    if cfg.fixed_price:p=float(data.fixed_price_plan[(eh*6-1)%144])
    return solve_intraday(bundle,prices,soc,B,A,lead,eh,cfg,p,fixed_contract)

def simulate_range(data,pf,cfg,start_day,end_day,state=None,*,progress=None):
    from q4_sim import SimState, SimulationResult, initial_state
    if getattr(cfg,'backend',None)!=VERSION:raise ValueError('unfrozen/legacy backend: refusing to mix old and inherited results')
    if cfg.scenario_k!=9 or cfg.alpha!=.8 or cfg.risk_lambda!=.02:
        raise ValueError('this revision inherits Q2/Q3 K=9, alpha=.8, lambda=.02; no parameter/robustness search')
    if cfg.allow_emergency_charging or cfg.mismatch:raise ValueError('main inherited execution forbids alternate physical semantics')
    state=initial_state() if state is None else state
    soc=float(state.soc);prevB=np.array(state.prev_B,float);prevA=np.array(state.prev_A,float)
    prevPA=getattr(state,'prev_adjustment_prices',None)
    prevPA=np.zeros(144) if prevPA is None else np.asarray(prevPA,float).copy()
    rows=[];events=[];contracts=[];days=[];schema=structure_signature()
    event_clock=cfg.settlement_clock=='adjustment_time'
    if cfg.settlement_clock not in ('delivery','adjustment_time'):raise ValueError('unknown settlement clock')
    for d in range(start_day,end_day):
        ds=data.dates[d].date().isoformat();S00=soc;B=np.zeros(144);A=B.copy();PA=np.zeros(144)
        planp=data.fixed_price_plan if cfg.fixed_price else data.price_plan[d]
        calp=np.r_[data.fixed_price_plan[-1],data.fixed_price_plan[:143]] if cfg.fixed_price else data.price_cal[d]
        event_start=0;sol=None;dayrows=[]
        for i in range(144):
            eh=i//6
            event=(i==0 or (cfg.branch=='q4_3' and i in (36,72,108)))
            # Turning off Q3's added information and trading rights means
            # executing the SAME midnight plan AND reserve, not re-solving it.
            if i>0 and not cfg.use_pv_updates and not cfg.allow_contract_revision:event=False
            if event:
                before=A.copy();sol=solve_event(data,d,eh,soc,B,A,float(prevA[143]),cfg);event_start=i
                A=sol.contract_A.copy()
                if i==0:
                    B=sol.original_B.copy();PA=np.asarray(planp,float).copy()
                else:
                    j0=EVENT_PLAN_START[eh]
                    if not np.array_equal(A[:j0],before[:j0]):raise AssertionError('delivered contract prefix mutated')
                    if not cfg.allow_contract_revision and not np.allclose(A,before,atol=1e-8):raise AssertionError('revision rights violated')
                    if event_clock:PA[j0:]=float(calp[i])
                committed=min(145-i,len(sol.scenario_emergency[0]))
                ecost=np.sum(5*sol.prices[:,:committed]*sol.scenario_emergency[:,:committed],axis=1)
                weights=np.full(sol.scenario_count,1/sol.scenario_count)
                events.append(dict(date=ds,day_index=d,event_hour=eh,branch=cfg.branch,level=cfg.level,label=cfg.label,
                    backend=VERSION,settlement_clock=cfg.settlement_clock,soc_event_kwh=soc,scenario_k=sol.scenario_count,
                    source_days=json.dumps(sol.source_days),scenario_weights=json.dumps(weights.tolist()),
                    objective_with_surrogates_yuan=sol.objective,level1=sol.objective,
                    expected_emergency_cost_yuan=float(np.mean(ecost)),scenario_cvar_emergency_cost_yuan=weighted_cvar(ecost,weights,.8),
                    solve_seconds=sol.solve_seconds,max_eq_residual=sol.max_eq_residual,horizon_intervals=len(sol.reserve_floor)-1,
                    epsilon=0.,risk_lambda=cfg.risk_lambda,alpha=cfg.alpha,contract_changed_kwh=float(np.abs(A-before).sum()) if i else 0.,
                    binaries=sol.binaries,mip_gap=sol.mip_gap,contracts_locked=bool(i>0 and not cfg.allow_contract_revision),schema_hash=schema))
            L=float(data.load_cal_kwh[d,i]);G=float(data.pv_cal_kwh[d,i]);p=float(calp[i])
            if not np.isfinite(L+G+p):
                if d==0 and i==0:continue
                raise AssertionError('unexpected missing actual interval')
            j=143 if i==0 else i-1;owner=d-1 if i==0 else d
            b=float(prevB[j] if i==0 else B[j]);a=float(prevA[j] if i==0 else A[j]);pa=float(prevPA[j] if i==0 else PA[j])
            floor=float(sol.reserve_floor[i-event_start+1])
            f=execute_inherited(a,L,G,soc,floor)
            fee=float(regular_fee(b,a,p,pa if event_clock else None));ef=5*p*f.eL
            row=dict(timestamp=(data.dates[d]+timedelta(minutes=10*i)).isoformat(),date=ds,day_index=d,slot=i,
                interval=natural_interval_label(i),branch=cfg.branch,level=cfg.level,label=cfg.label,backend=VERSION,
                settlement_clock=cfg.settlement_clock,plan_owner_day_index=owner,plan_j=j,B_kwh=b,A_kwh=a,Q_kwh=a,
                price_yuan_per_kwh=p,adjustment_price_yuan_per_kwh=pa if event_clock else p,load_kwh=L,pv_kwh=G,
                qL=f.qL,qB=f.qB,u=f.u,gL=f.gL,gB=f.gB,kappa=f.kappa,eL=f.eL,eB=0.,e_kwh=f.eL,x=f.x,y=f.y,I_kwh=f.I,
                S0_kwh=soc,S1_kwh=f.S1,target_soc_kwh=floor,regular_fee_yuan=fee,emergency_fee_yuan=ef,
                cash_fee_yuan=fee+ef,physics_residual=f.residual_max,schema_hash=schema)
            soc=f.S1;dayrows.append(row)
        rows.extend(dayrows)
        fees=regular_fee(B,A,planp,PA if event_clock else None)
        for j in range(144):
            contracts.append(dict(date=ds,day_index=d,plan_j=j,branch=cfg.branch,level=cfg.level,backend=VERSION,
                settlement_clock=cfg.settlement_clock,B_kwh=float(B[j]),A_kwh=float(A[j]),delta_kwh=float(A[j]-B[j]),
                price_yuan_per_kwh=float(planp[j]),adjustment_price_yuan_per_kwh=float(PA[j]) if event_clock else float(planp[j]),
                plan_reference_fee_yuan=float(B[j]*planp[j]),final_regular_fee_yuan=float(fees[j])))
        def total(k):return float(sum(x[k] for x in dayrows))
        days.append(dict(date=ds,day_index=d,branch=cfg.branch,level=cfg.level,label=cfg.label,backend=VERSION,
            settlement_clock=cfg.settlement_clock,cash_fee_yuan=total('cash_fee_yuan'),regular_fee_yuan=total('regular_fee_yuan'),
            emergency_fee_yuan=total('emergency_fee_yuan'),emergency_kwh=total('e_kwh'),charge_kwh=total('x'),discharge_kwh=total('y'),
            unused_contract_kwh=total('u'),curtailment_kwh=total('kappa'),pv_curtailment_kwh=total('kappa'),S00_kwh=S00,S24_kwh=soc,
            next_tail_target_soc_kwh=float(sol.reserve_floor[145-event_start]),
            max_physics_residual=max((z['physics_residual'] for z in dayrows),default=0.),
            template_plan_reference_fee_yuan=float(np.dot(B,planp)),template_final_regular_fee_yuan=float(np.sum(fees))))
        prevB=B.copy();prevA=A.copy();prevPA=PA.copy()
        if progress is not None:progress(d,days[-1])
    st=SimState(soc,prevB,prevA,prevPA)
    return SimulationResult(cfg,st,pd.DataFrame(rows),pd.DataFrame(events),pd.DataFrame(contracts),pd.DataFrame(days))
