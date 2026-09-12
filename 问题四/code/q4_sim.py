# -*- coding: utf-8 -*-
"""Continuous causal replay for Q4-2/Q4-3.

Natural-day evaluation uses [00:00,24:00); contracts keep the official plan-row
coordinate [00:10,...,+1 00:10].  Slot 00:00-00:10 therefore uses yesterday's
plan-row j=143 and yesterday's final B/A.  This is explicit in every ledger row.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import timedelta
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from q4_data import Q4Data, SOC0, EVENT_HOURS, actual_at, natural_interval_label
from q4_price import PriceConfig, PriceForecaster
from q4_scenarios import build_joint_scenarios, build_point_scenario, ScenarioSet
from q4_opt import solve_event
from q4_control import execute_slot
from q4_settlement import settlement_components, settlement_vector
from q4_tree import EVENT_PLAN_HORIZON


@dataclass(frozen=True)
class SimConfig:
    branch:str
    level:str='B0'
    price_config:PriceConfig=PriceConfig()
    scenario_k:int=3
    alpha:float=0.80
    risk_lambda:float=0.0
    epsilon:float=0.0
    terminal_reserve:float=6000.0
    terminal_value:float=0.80
    history_days:int=42
    mismatch:bool=False
    allow_emergency_charging:bool=False
    use_pv_updates:bool=True
    allow_contract_revision:bool=True
    price_oracle:bool=False
    # Matched causal no-price-adaptation benchmark: contract optimization uses
    # the known Attachment-1 reference tariff, while realized execution and
    # settlement still use Attachment-4 variable prices.  This is deliberately
    # different from fixed_price, which changes the settlement counterfactual.
    decision_fixed_price:bool=False
    fixed_price:bool=False
    disabled_vintage_hours:Tuple[int,...]=()
    label:str='formal'


@dataclass
class SimState:
    soc:float
    prev_B:np.ndarray
    prev_A:np.ndarray


@dataclass
class SimulationResult:
    config:SimConfig
    state_end:SimState
    slots:pd.DataFrame
    events:pd.DataFrame
    contracts:pd.DataFrame
    days:pd.DataFrame


def initial_state()->SimState:
    return SimState(float(SOC0),np.zeros(144,float),np.zeros(144,float))


def _override_fixed_price(data:Q4Data,sc:ScenarioSet,times):
    vals=[]
    for t in times:
        v=actual_at(data,t,'fixed_price')
        # 附件1固定分时电价本身是已知日循环。Jan-1自然日首10分钟缺少的
        # 是上一计划行实际记录，不是电价规则；决策层可用已知日循环末段价格。
        # 执行账本仍保留Jan-1首段实际量缺失，不伪造实际0。
        vals.append(float(data.fixed_price_plan[143]) if v is None else float(v))
    vals=np.asarray(vals,float)
    sc.price=np.tile(vals[None,:],(len(sc.weights),1)); sc.point_price=vals.copy(); return sc


def _make_scenarios(data:Q4Data,pf:PriceForecaster,day_idx:int,eh:int,cfg:SimConfig)->ScenarioSet:
    H=EVENT_PLAN_HORIZON[eh]
    if cfg.level=='B0':
        sc=build_point_scenario(data,pf,day_idx,eh,H,cfg.branch,cfg.price_config,
                                price_oracle=cfg.price_oracle,use_new_vintage=cfg.use_pv_updates,
                                disabled_vintage_hours=cfg.disabled_vintage_hours)
    else:
        sc=build_joint_scenarios(data,pf,day_idx,eh,H,cfg.branch,cfg.price_config,k=cfg.scenario_k,
                                 history_days=cfg.history_days,mismatch=cfg.mismatch,
                                 use_new_vintage=cfg.use_pv_updates,disabled_vintage_hours=cfg.disabled_vintage_hours)
        if cfg.price_oracle:
            # Price Oracle removes only price uncertainty.  L/PV scenario paths
            # remain causal; every support point gets the same realized price path.
            # The Dec-31 event-0 horizon contains the contractual +1 tail slot
            # outside Attachment 4.  That slot is never executed/scored in the
            # 334-day formal period, so there is no legitimate realized oracle
            # price for it; retain the causal point forecast there rather than
            # injecting NaN or fabricating 2026 information.
            times=[data.dates[day_idx]+timedelta(hours=eh,minutes=10*h) for h in range(H)]
            vals=[]
            for h,t in enumerate(times):
                v=actual_at(data,t,'price')
                vals.append(float(sc.point_price[h]) if v is None or not np.isfinite(v) else float(v))
            vals=np.asarray(vals,float)
            sc.price=np.tile(vals[None,:],(len(sc.weights),1)); sc.point_price=vals.copy()
    if cfg.fixed_price or cfg.decision_fixed_price:
        times=[data.dates[day_idx]+timedelta(hours=eh,minutes=10*h) for h in range(H)]
        sc=_override_fixed_price(data,sc,times)
    return sc


def simulate_range(data:Q4Data,pf:PriceForecaster,cfg:SimConfig,start_day:int,end_day:int,
                   state:Optional[SimState]=None)->SimulationResult:
    """Replay day indices [start_day,end_day), returning authoritative ledgers."""
    if cfg.branch not in ('q4_2','q4_3'): raise ValueError(cfg.branch)
    if cfg.level not in ('B0','B1','B2'): raise ValueError(cfg.level)
    st=initial_state() if state is None else SimState(float(state.soc),state.prev_B.copy(),state.prev_A.copy())
    slot_rows=[]; event_rows=[]; contract_rows=[]; day_rows=[]
    for d in range(int(start_day),int(end_day)):
        day=data.dates[d]
        day_soc0=st.soc
        B=np.zeros(144,float); A=np.zeros(144,float)
        latest_sol=None; latest_event_slot=0
        daily_cash=daily_reg=daily_emg_fee=daily_emg=0.0
        daily_x=daily_y=daily_u=daily_k=0.0
        max_res=0.0
        event_hours=(0,) if cfg.branch=='q4_2' else EVENT_HOURS
        for i in range(144):
            eh=i//6
            if eh in event_hours and i==eh*6:
                # Q4-3 may re-solve state targets at 6/12/18 while contracts are
                # locked (A/B experiment); this is not a hidden contract revision.
                sc=_make_scenarios(data,pf,d,eh,cfg)
                sol=solve_event(scenarios=sc,soc0=st.soc,day=day,event_hour=eh,branch=cfg.branch,
                                original_B=B,current_A=A,lead_q=float(st.prev_A[143]),
                                model_level=cfg.level,risk_lambda=cfg.risk_lambda,alpha=cfg.alpha,
                                epsilon=cfg.epsilon,terminal_reserve=cfg.terminal_reserve,
                                terminal_value=cfg.terminal_value,allow_emergency_charging=cfg.allow_emergency_charging,
                                lock_contracts=(eh>0 and not cfg.allow_contract_revision))
                if not sol.success: raise RuntimeError(f'{day.date()} {eh:02d}:00 {cfg.label} failed: {sol.message}')
                A_before=A.copy(); A=sol.contract_A.copy(); B=sol.original_B.copy()
                latest_sol=sol; latest_event_slot=i
                event_rows.append({
                    'date':day.date().isoformat(),'day_index':d,'event_hour':eh,'branch':cfg.branch,'level':cfg.level,
                    'label':cfg.label,'soc_event':st.soc,'scenario_k':len(sc.weights),'scenario_weights':';'.join(f'{x:.12g}' for x in sc.weights),
                    'scenario_source_days':';'.join(map(str,sc.medoid_source_days)),'history_source_days':';'.join(map(str,sc.source_days)),
                    'expected_cash':sol.expected_cash,'scenario_cvar_cash':sol.weighted_cvar_cash,'level1':sol.level1_objective,
                    'level2_throughput':sol.level2_throughput,'level3_curtailment':sol.level3_curtailment,
                    'pred_emergency_kwh':sol.predicted_emergency_kwh,'pred_unused_kwh':sol.predicted_unused_contract_kwh,
                    'pred_curtailment_kwh':sol.predicted_curtailment_kwh,'solve_seconds':sol.solve_seconds,
                    'max_eq_residual':sol.max_eq_residual,'epsilon':cfg.epsilon,'risk_lambda':cfg.risk_lambda,'alpha':cfg.alpha,
                    'contract_changed_kwh':float(np.abs(A-A_before).sum()),'contracts_locked':bool(eh>0 and not cfg.allow_contract_revision),
                    'schema_hash':sol.physical_schema_hash,
                })
            if latest_sol is None: raise AssertionError('missing event0 solution')
            L=float(data.load_cal_kwh[d,i]) if np.isfinite(data.load_cal_kwh[d,i]) else np.nan
            G=float(data.pv_cal_kwh[d,i]) if np.isfinite(data.pv_cal_kwh[d,i]) else np.nan
            price_arr=data.fixed_price_cal if cfg.fixed_price else data.price_cal
            c=float(price_arr[d,i]) if np.isfinite(price_arr[d,i]) else np.nan
            # Jan-1 00:00 is genuinely unavailable.  It is excluded from any
            # formal evaluation and is not replaced by zero during warm-up.
            if not (np.isfinite(L) and np.isfinite(G) and np.isfinite(c)):
                continue
            if i==0:
                Bb=float(st.prev_B[143]); Aa=float(st.prev_A[143]); plan_owner=d-1; plan_j=143
            else:
                plan_j=i-1; Bb=float(B[plan_j]); Aa=float(A[plan_j]); plan_owner=d
            h=i-latest_event_slot
            target=float(latest_sol.expected_soc_end[min(max(h,0),len(latest_sol.expected_soc_end)-1)])
            act=execute_slot(Q=Aa,load_kwh=L,pv_kwh=G,S0=st.soc,target_soc=target,
                             allow_emergency_charging=cfg.allow_emergency_charging)
            st.soc=act.S1; max_res=max(max_res,act.residual_max)
            ss=settlement_components(Bb,Aa,c,act.eL+act.eB)
            daily_reg+=ss.regular_fee; daily_emg_fee+=ss.emergency_fee; daily_cash+=ss.total_cash
            daily_emg+=act.eL+act.eB; daily_x+=act.x; daily_y+=act.y; daily_u+=act.u; daily_k+=act.kappa
            t=day+timedelta(minutes=10*i)
            slot_rows.append({
                'timestamp':t.isoformat(),'date':day.date().isoformat(),'day_index':d,'slot':i,'interval':natural_interval_label(i),
                'branch':cfg.branch,'level':cfg.level,'label':cfg.label,'plan_owner_day_index':plan_owner,'plan_j':plan_j,
                'B_kwh':Bb,'A_kwh':Aa,'Q_kwh':Aa,'price_yuan_per_kwh':c,'load_kwh':L,'pv_kwh':G,
                'qL':act.qL,'qB':act.qB,'u':act.u,'gL':act.gL,'gB':act.gB,'kappa':act.kappa,
                'eL':act.eL,'eB':act.eB,'e_kwh':act.eL+act.eB,'x':act.x,'y':act.y,'I_kwh':act.I,
                'S0_kwh':act.S0,'S1_kwh':act.S1,'target_soc_kwh':target,'regular_fee_yuan':ss.regular_fee,
                'emergency_fee_yuan':ss.emergency_fee,'cash_fee_yuan':ss.total_cash,'physics_residual':act.residual_max,
                'schema_hash':act.schema_hash,
            })
        # End-of-day final plan contracts; the tail j=143 is executed at next day 00:00.
        sv=settlement_vector(B,A,(data.fixed_price_plan if cfg.fixed_price else data.price_plan[d]))
        for j in range(144):
            contract_rows.append({'date':day.date().isoformat(),'day_index':d,'plan_j':j,'B_kwh':float(B[j]),'A_kwh':float(A[j]),
                                  'delta_A_minus_B_kwh':float(A[j]-B[j]),'price_yuan_per_kwh':float(data.fixed_price_plan[j] if cfg.fixed_price else data.price_plan[d,j]),
                                  'plan_reference_fee_yuan':float(B[j]*(data.fixed_price_plan[j] if cfg.fixed_price else data.price_plan[d,j])),
                                  'final_regular_fee_yuan':float(sv['regular_fee'][j])})
        day_rows.append({'date':day.date().isoformat(),'day_index':d,'branch':cfg.branch,'level':cfg.level,'label':cfg.label,
                         'cash_fee_yuan':daily_cash,'regular_fee_yuan':daily_reg,'emergency_fee_yuan':daily_emg_fee,
                         'emergency_kwh':daily_emg,'charge_kwh':daily_x,'discharge_kwh':daily_y,'unused_contract_kwh':daily_u,
                         'curtailment_kwh':daily_k,'S00_kwh':day_soc0,'S24_kwh':st.soc,'next_tail_target_soc_kwh':float(latest_sol.expected_soc_end[-1]),'max_physics_residual':max_res,
                         'template_plan_reference_fee_yuan':float(np.sum([r['plan_reference_fee_yuan'] for r in contract_rows[-144:]])),
                         'template_final_regular_fee_yuan':float(np.sum(sv['regular_fee']))})
        st.prev_B=B.copy(); st.prev_A=A.copy()
    return SimulationResult(cfg,st,pd.DataFrame(slot_rows),pd.DataFrame(event_rows),pd.DataFrame(contract_rows),pd.DataFrame(day_rows))


def replay_summary(res:SimulationResult)->dict:
    d=res.days
    return {'branch':res.config.branch,'level':res.config.level,'label':res.config.label,'days':int(len(d)),
            'cash_yuan':float(d.cash_fee_yuan.sum()),'regular_yuan':float(d.regular_fee_yuan.sum()),
            'emergency_yuan':float(d.emergency_fee_yuan.sum()),'emergency_kwh':float(d.emergency_kwh.sum()),
            'curtailment_kwh':float(d.curtailment_kwh.sum()),'unused_contract_kwh':float(d.unused_contract_kwh.sum()),
            'soc_min_kwh':float(res.slots.S1_kwh.min()) if len(res.slots) else float('nan'),
            'soc_max_kwh':float(res.slots.S1_kwh.max()) if len(res.slots) else float('nan'),
            'soc_end_kwh':float(res.state_end.soc),'event_solve_p95_sec':float(res.events.solve_seconds.quantile(.95)) if len(res.events) else 0.0,
            'max_physics_residual':float(res.slots.physics_residual.max()) if len(res.slots) else 0.0}
