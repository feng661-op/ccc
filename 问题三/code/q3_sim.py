# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import List,Dict
import numpy as np
from q3_data import *
from q3_opt import *

@dataclass
class SimulationResult:
    name:str; use_new_vintage:bool; allow_revision:bool; scenario_count:int; down_settlement:str
    B:np.ndarray; A:np.ndarray; A_stage:np.ndarray; charge:np.ndarray; discharge:np.ndarray; emergency:np.ndarray; curtail:np.ndarray
    soc00:np.ndarray; soc24:np.ndarray; soc_path:np.ndarray
    plan_fee:np.ndarray; adjusted_fee:np.ndarray; natural_regular_fee:np.ndarray; emergency_fee:np.ndarray
    event_rows:List[Dict]; revision_rows:List[Dict]; revision_anchor:str='original_anchor'

def stepwise_fee_slots(Bday,stages,price):
    """Strict stepwise fee by plan slot: p*A_final + 0.5*p*sum revision magnitudes."""
    stages=np.asarray(stages,float); p=np.asarray(price,float)
    diffs=np.abs(stages[1]-stages[0])+np.abs(stages[2]-stages[1])+np.abs(stages[3]-stages[2])
    return p*stages[3]+0.5*p*diffs

def simulate_strategy(data,name='D',use_new_vintage=True,allow_revision=True,scenario_count=3,down_settlement='cancel_settlement',revision_anchor='original_anchor',end_day=365,verbose=False):
    n=end_day; B=np.zeros((n,144)); A=np.zeros((n,144)); astage=np.zeros((n,4,144)); ch=np.zeros((n,144)); dis=np.zeros((n,144)); em=np.zeros((n,144)); cur=np.zeros((n,144)); s00=np.zeros(n); s24=np.zeros(n); sp=np.zeros((n,145)); pf=np.zeros(n); af=np.zeros(n); rf=np.zeros(n); ef=np.zeros(n)
    events=[]; revisions=[]; soc=SOC0
    for d in range(n):
        day=data.dates[d]; s00[d]=soc; sp[d,0]=soc; lead=0.0 if d==0 else float(A[d-1,143])
        # event0: create immutable original contract B; A_0=B.
        sol=solve_event_lp(data,d,0,soc,None,None,lead_contract=lead,use_new_vintage=use_new_vintage,allow_revision=allow_revision,scenario_count=scenario_count,down_settlement=down_settlement,revision_anchor=revision_anchor)
        B[d]=sol.current_contract; A[d]=B[d].copy(); astage[d,0]=A[d]
        events.append(dict(strategy=name,date=day.date().isoformat(),event_hour=0,soc_kwh=soc,scenario_count=sol.scenario_count,node_count=sol.node_count,objective=sol.objective,solve_seconds=sol.solve_seconds,max_eq_residual=sol.max_eq_residual,max_ub_violation=sol.max_ub_violation,latest_history_day=d-2,continuation_first_time=str(sol.continuation_first_time or '')))
        # execute natural 00:00-06:00; i0 uses previous plan-day final slot.
        for i in range(0,36):
            if d==0 and i==0 and not np.isfinite(data.net_cal_kwh[d,i]):
                sp[d,i+1]=soc; continue
            q=natural_contract(A,d,i); target=float(sol.expected_soc[min(i+1,len(sol.expected_soc)-1)])
            soc,x,y,e,w=execute_actual_interval(soc,q,float(data.net_cal_kwh[d,i]),target); ch[d,i]=x;dis[d,i]=y;em[d,i]=e;cur[d,i]=w;sp[d,i+1]=soc
        for stage_idx,(eh,lo_i,hi_i) in enumerate(((6,36,72),(12,72,108),(18,108,144)),start=1):
            old=A[d].copy()
            sol=solve_event_lp(data,d,eh,soc,B[d],A[d],lead_contract=lead,use_new_vintage=use_new_vintage,allow_revision=allow_revision,scenario_count=scenario_count,down_settlement=down_settlement,revision_anchor=revision_anchor)
            A[d]=sol.current_contract; astage[d,stage_idx]=A[d]
            startj=EVENT_PLAN_START[eh]
            if np.max(np.abs(A[d,:startj]-old[:startj]))>1e-8: raise AssertionError(f'{day.date()} {eh}点修改了已交付合同')
            diff=A[d]-old
            for j in np.where(np.abs(diff)>1e-7)[0]:
                revisions.append(dict(strategy=name,date=day.date().isoformat(),event_hour=eh,slot_j=int(j),slot_label=data.plan_headers[j],old_kwh=float(old[j]),new_kwh=float(A[d,j]),delta_kwh=float(diff[j]),legal=bool(j>=startj)))
            events.append(dict(strategy=name,date=day.date().isoformat(),event_hour=eh,soc_kwh=soc,scenario_count=sol.scenario_count,node_count=sol.node_count,objective=sol.objective,solve_seconds=sol.solve_seconds,max_eq_residual=sol.max_eq_residual,max_ub_violation=sol.max_ub_violation,latest_history_day=d-2,continuation_first_time=str(sol.continuation_first_time or '')))
            for i in range(lo_i,hi_i):
                q=natural_contract(A,d,i); step=i-lo_i; target=float(sol.expected_soc[min(step+1,len(sol.expected_soc)-1)])
                soc,x,y,e,w=execute_actual_interval(soc,q,float(data.net_cal_kwh[d,i]),target); ch[d,i]=x;dis[d,i]=y;em[d,i]=e;cur[d,i]=w;sp[d,i+1]=soc
        if soc<SOC_MIN-1e-6 or soc>SOC_MAX+1e-6: raise AssertionError(f'{day.date()} SOC越界 {soc}')
        s24[d]=soc
        pf[d]=float(np.dot(data.price_plan,B[d]))
        if revision_anchor=='stepwise_revision':
            fee_slots=stepwise_fee_slots(B[d],astage[d],data.price_plan); af[d]=float(fee_slots.sum())
            if d==0: rf[d]=float(fee_slots[:143].sum())
            else:
                prev_slots=stepwise_fee_slots(B[d-1],astage[d-1],data.price_plan)
                rf[d]=float(prev_slots[143]+fee_slots[:143].sum())
        else:
            af[d]=float(settlement_components(B[d],A[d],data.price_plan,down_settlement)['F_regular'].sum())
            rf[d]=natural_regular_fee(B,A,data,d,down_settlement)
        ef[d]=float(np.dot(5*data.price_calendar,em[d]))
        if verbose and (d%31==0 or d==n-1): print(f'[{name}] {d+1}/{n} {day.date()} SOC={s00[d]:.1f}->{s24[d]:.1f} B={B[d].sum():.1f} A={A[d].sum():.1f} emg={em[d].sum():.1f}')
    return SimulationResult(name,use_new_vintage,allow_revision,scenario_count,down_settlement,B,A,astage,ch,dis,em,cur,s00,s24,sp,pf,af,rf,ef,events,revisions,revision_anchor)
