# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import List,Dict
import numpy as np
from q3_data import *
from q3_opt import *
from q3_physical import physical_flows, project_current_action
from q3_inheritance import q2, RISK_LAMBDA, SCENARIO_COUNT, TERMINAL_VALUE
from dataclasses import replace

@dataclass
class SimulationResult:
    name:str; use_new_vintage:bool; allow_revision:bool; scenario_count:int; down_settlement:str
    B:np.ndarray; A:np.ndarray; A_stage:np.ndarray; charge:np.ndarray; discharge:np.ndarray; emergency:np.ndarray; curtail:np.ndarray
    soc00:np.ndarray; soc24:np.ndarray; soc_path:np.ndarray
    plan_fee:np.ndarray; adjusted_fee:np.ndarray; natural_regular_fee:np.ndarray; emergency_fee:np.ndarray
    event_rows:List[Dict]; revision_rows:List[Dict]; dispatch_rows:List[Dict]; revision_anchor:str='original_anchor'; disabled_vintage_hours:tuple=()
    execution_mode:str='q2_reserve_feedback'
    grid_import:np.ndarray=None; unused_contract:np.ndarray=None
    supply_surplus:np.ndarray=None; battery_dump:np.ndarray=None

def stepwise_fee_slots(Bday,stages,price):
    """Strict stepwise fee by plan slot: p*A_final + 0.5*p*sum revision magnitudes."""
    stages=np.asarray(stages,float); p=np.asarray(price,float)
    diffs=np.abs(stages[1]-stages[0])+np.abs(stages[2]-stages[1])+np.abs(stages[3]-stages[2])
    return p*stages[3]+0.5*p*diffs

def simulate_strategy(data,name='D',use_new_vintage=True,allow_revision=True,scenario_count=SCENARIO_COUNT,down_settlement='cancel_settlement',revision_anchor='original_anchor',end_day=365,verbose=False,disabled_vintage_hours=(),terminal_value=TERMINAL_VALUE,execution_mode='q2_reserve_feedback'):
    if execution_mode != 'q2_reserve_feedback':
        raise ValueError(execution_mode)
    n=end_day; disabled_vintage_hours=tuple(sorted(int(x) for x in disabled_vintage_hours))
    B=np.zeros((n,144)); A=np.zeros((n,144)); astage=np.zeros((n,4,144)); ch=np.zeros((n,144)); dis=np.zeros((n,144)); em=np.zeros((n,144)); cur=np.zeros((n,144)); s00=np.zeros(n); s24=np.zeros(n); sp=np.zeros((n,145)); pf=np.zeros(n); af=np.zeros(n); rf=np.zeros(n); ef=np.zeros(n)
    gi=np.zeros((n,144)); unused=np.zeros((n,144)); surplus=np.zeros((n,144)); dump=np.zeros((n,144))
    events=[]; revisions=[]; dispatches=[]; soc=SOC0
    def event_record(day,eh,sol,soc_now):
        return dict(strategy=name,date=day.date().isoformat(),event_hour=eh,soc_kwh=float(soc_now),scenario_count=sol.scenario_count,node_count=sol.node_count,objective=sol.objective,solve_seconds=sol.solve_seconds,max_eq_residual=sol.max_eq_residual,max_ub_violation=sol.max_ub_violation,latest_history_day=(day-data.dates[0]).days-2,continuation_first_time=str(sol.continuation_first_time or ''),terminal_time=str(sol.terminal_time),scenario_weight_sum=float(np.sum(sol.scenario_weights)),clip_rate=float(sol.clip_rate),terminal_shortfall_expected_kwh=float(sol.terminal_shortfall_expected),terminal_shortfall_binding_scenarios=int(sol.terminal_shortfall_binding))
    for d in range(n):
        day=data.dates[d]; s00[d]=soc; sp[d,0]=soc; lead=0.0 if d==0 else float(A[d-1,143])
        # Midnight is Q2's own planner. Possible later revisions are NOT used to
        # under-purchase now; the next day is valuation-only, as in Q2.
        sol=solve_event_lp(data,d,0,soc,None,None,lead_contract=lead,use_new_vintage=use_new_vintage,allow_revision=allow_revision,scenario_count=scenario_count,down_settlement=down_settlement,revision_anchor=revision_anchor,terminal_value=terminal_value,disabled_vintage_hours=disabled_vintage_hours)
        B[d]=sol.current_contract; A[d]=B[d].copy(); astage[d,0]=A[d]; events.append(event_record(day,0,sol,soc))
        # Contracts remain causal and frozen between events. Current-measurement
        # dispatch is a separate, explicitly declared within-slot approximation.
        def run_block(lo_i,hi_i,event_hour,sol):
            nonlocal soc
            for i in range(lo_i,hi_i):
                if d==0 and i==0 and not np.isfinite(data.net_cal_kwh[d,i]):
                    sp[d,i+1]=soc; continue
                step=i-lo_i; H=len(sol.point_net)-step
                actual=float(data.net_cal_kwh[d,i]); q=float(natural_contract(A,d,i))
                forecast0=float(sol.point_net[step]); floor=float(sol.reserve_floor[step+1])
                # Exactly the same one-way, present-measurement feedback policy
                # as Q2. The forecast SOC reserve does not reset physical SOC.
                soc1,x,y,e_pred,w0=q2.execute_interval(soc,q,actual,floor)
                r=max(0.0,q2.TERMINAL_RESERVE-soc1)
                flow=physical_flows(q,float(data.load_cal_kwh[d,i]),float(data.pv_cal_kwh[d,i]),x,y)
                if abs(flow.emergency-e_pred)>1e-7 or abs(flow.supply_surplus-w0)>1e-7:
                    raise AssertionError('Q2 execution and split physical ledger disagree')
                e=flow.emergency; w=flow.pv_curtail
                gi[d,i]=flow.grid_import; unused[d,i]=flow.unused_contract
                surplus[d,i]=flow.supply_surplus; dump[d,i]=flow.battery_dump
                resid=float(actual-forecast0)
                dispatches.append(dict(strategy=name,date=day.date().isoformat(),interval_i=i,event_hour=event_hour,horizon_slots=H,soc_before_kwh=float(soc),soc_after_kwh=float(soc1),contract_kwh=q,forecast_net_kwh=forecast0,dispatch_net_kwh=actual,execution_mode=execution_mode,measurement_model='ideal_piecewise_constant_current_slot',actual_net_kwh=actual,forecast_residual_kwh=resid,charge_kwh=x,discharge_kwh=y,predicted_emergency_kwh=e_pred,emergency_kwh=e,curtail_kwh=w,pv_curtail_kwh=w,grid_import_kwh=flow.grid_import,unused_contract_kwh=flow.unused_contract,supply_surplus_kwh=flow.supply_surplus,battery_dump_kwh=flow.battery_dump,terminal_shortfall_kwh=r,reserve_floor_kwh=floor))
                soc=soc1; ch[d,i]=x;dis[d,i]=y;em[d,i]=e;cur[d,i]=w;sp[d,i+1]=soc
        run_block(0,36,0,sol)
        for stage_idx,(eh,lo_i,hi_i) in enumerate(((6,36,72),(12,72,108),(18,108,144)),start=1):
            old=A[d].copy()
            if not allow_revision and (not use_new_vintage or eh in disabled_vintage_hours):
                # No additional information or permission: preserve all future
                # midnight actions, rather than silently changing the executor.
                old_floor=sol.reserve_floor
                sol=replace(sol,event_hour=eh,point_net=sol.point_net[36:],
                            point_prices=sol.point_prices[36:],scenario_soc=sol.scenario_soc[:,36:],
                            scenario_emergency=sol.scenario_emergency[:,36:],expected_soc=sol.expected_soc[36:],
                            solve_seconds=0.0)
                sol.reserve_floor=old_floor[36:]
            else:
                sol=solve_event_lp(data,d,eh,soc,B[d],A[d],lead_contract=lead,use_new_vintage=use_new_vintage,allow_revision=allow_revision,scenario_count=scenario_count,down_settlement=down_settlement,revision_anchor=revision_anchor,terminal_value=terminal_value,disabled_vintage_hours=disabled_vintage_hours)
            A[d]=sol.current_contract; astage[d,stage_idx]=A[d]
            startj=EVENT_PLAN_START[eh]
            if np.max(np.abs(A[d,:startj]-old[:startj]))>1e-8: raise AssertionError(f'{day.date()} {eh}点修改了已交付合同')
            diff=A[d]-old
            for j in np.where(np.abs(diff)>1e-7)[0]:
                revisions.append(dict(strategy=name,date=day.date().isoformat(),event_hour=eh,slot_j=int(j),slot_label=data.plan_headers[j],old_kwh=float(old[j]),new_kwh=float(A[d,j]),delta_kwh=float(diff[j]),legal=bool(j>=startj)))
            events.append(event_record(day,eh,sol,soc)); run_block(lo_i,hi_i,eh,sol)
        if soc<SOC_MIN-1e-6 or soc>SOC_MAX+1e-6: raise AssertionError(f'{day.date()} SOC越界 {soc}')
        soc=float(np.clip(soc,SOC_MIN,SOC_MAX)); s24[d]=soc; sp[d,-1]=soc; pf[d]=float(np.dot(data.price_plan,B[d]))
        if revision_anchor=='stepwise_revision':
            fee_slots=stepwise_fee_slots(B[d],astage[d],data.price_plan); af[d]=float(fee_slots.sum())
            if d==0: rf[d]=float(fee_slots[:143].sum())
            else:
                prev_slots=stepwise_fee_slots(B[d-1],astage[d-1],data.price_plan); rf[d]=float(prev_slots[143]+fee_slots[:143].sum())
        else:
            af[d]=float(settlement_components(B[d],A[d],data.price_plan,down_settlement)['F_regular'].sum()); rf[d]=natural_regular_fee(B,A,data,d,down_settlement)
        ef[d]=float(np.dot(5*data.price_calendar,em[d]))
        if verbose and (d%31==0 or d==n-1): print(f'[{name}] {d+1}/{n} {day.date()} SOC={s00[d]:.1f}->{s24[d]:.1f} B={B[d].sum():.1f} A={A[d].sum():.1f} emg={em[d].sum():.1f}',flush=True)
    return SimulationResult(name,use_new_vintage,allow_revision,scenario_count,down_settlement,B,A,astage,ch,dis,em,cur,s00,s24,sp,pf,af,rf,ef,events,revisions,dispatches,revision_anchor,disabled_vintage_hours,execution_mode,gi,unused,surplus,dump)
