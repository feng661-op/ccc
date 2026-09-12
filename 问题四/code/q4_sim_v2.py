"""Continuous execution of inherited Q4 policies; the four authoritative ledgers."""
from __future__ import annotations
from datetime import timedelta
import json
import numpy as np
import pandas as pd
from q4_data import EVENT_PLAN_START, natural_interval_label
from q4_flow import structure_signature, weighted_cvar
from q4_inheritance import VERSION, solve_event, execute_inherited


def regular_fee(B,A,p,pa=None):
    p=np.asarray(p,float);pa=p if pa is None else np.asarray(pa,float)
    B=np.asarray(B,float);A=np.asarray(A,float)
    return p*np.minimum(B,A)+.5*pa*np.maximum(B-A,0)+1.5*pa*np.maximum(A-B,0)


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
