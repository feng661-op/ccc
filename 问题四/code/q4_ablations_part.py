# -*- coding: utf-8 -*-
"""Parallelizable frozen-model Q4 ablation parts.

Each part writes only its own artifact(s).  q4_ablations_merge.py assembles the
final E0-E15 matrix after all parts finish.  No formal-period result is fed back
into January model selection.
"""
from __future__ import annotations
import argparse, json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

import q4_ablations as qa
import q4_sim, q4_flow, q4_control
from q4_price import PriceForecaster
from q4_sim import SimState, simulate_range
from q4_metrics import day_metrics
from q4_forecast import historical_forecast_kwh


def save(name, obj):
    qa.jsave(name, obj)
    print(json.dumps({name: 'done'}, ensure_ascii=False), flush=True)


def e7():
    out={'main_all_versions':qa.compact(qa.main_result('q4_3'))}
    for h in (6,12,18):
        m,r=qa.runformal(qa.frozen_cfg('q4_3',disabled_vintage_hours=(h,),label=f'E7_disable_{h:02d}'))
        out[f'disable_{h:02d}']=qa.compact(m)
        r.days.to_csv(qa.ADIR/f'E7_disable_{h:02d}_daily.csv',index=False,encoding='utf-8-sig')
    save('E7_event_contribution.json',out)


def e8():
    out={}; orig=q4_sim.solve_event
    def dyn_solve(**kw):
        sc=kw['scenarios']; tail=sc.price[:,-min(12,sc.price.shape[1]):]
        kw['terminal_value']=float(np.median(tail)); return orig(**kw)
    for br in ('q4_2','q4_3'):
        m0,_=qa.runformal(qa.frozen_cfg(br,terminal_value=0.0,label='E8_no_terminal_value'))
        q4_sim.solve_event=dyn_solve
        try: md,_=qa.runformal(qa.frozen_cfg(br,terminal_value=.8,label='E8_dynamic_nu'))
        finally: q4_sim.solve_event=orig
        out[br]={'simple_frozen':qa.compact(qa.main_result(br)),'zero_terminal_value':qa.compact(m0),'dynamic_nu_causal_tail_median':qa.compact(md),
                 'execution_horizon_note':'execution remains 10-min one-step; event horizons are inherited Q2/Q3 information horizons and therefore are not tuned on formal data'}
    save('E8_terminal_horizon.json',out)


def e9():
    out={}; original_exec=q4_sim.execute_slot
    for br in ('q4_2','q4_3'):
        lag={'prevL':float(qa.DATA.load_cal_kwh[30,143]),'prevG':float(qa.DATA.pv_cal_kwh[30,143])}
        def full_lag_wrapper(*,Q,load_kwh,pv_kwh,S0,target_soc,allow_emergency_charging=False,_c=lag):
            tentative=original_exec(Q=Q,load_kwh=_c['prevL'],pv_kwh=_c['prevG'],S0=S0,target_soc=target_soc,allow_emergency_charging=allow_emergency_charging)
            physical=original_exec(Q=Q,load_kwh=load_kwh,pv_kwh=pv_kwh,S0=S0,target_soc=tentative.S1,allow_emergency_charging=allow_emergency_charging)
            _c['prevL']=float(load_kwh); _c['prevG']=float(pv_kwh); return physical
        q4_sim.execute_slot=full_lag_wrapper
        try: lm,lr=qa.runformal(qa.frozen_cfg(br,label='E9_formal_lag1_measurement'))
        finally: q4_sim.execute_slot=original_exec
        current=qa.compact(qa.main_result(br)); lagged=qa.compact(lm)
        out[br]={'formal_current':current,'formal_lag1_measurement':lagged,
                 'cash_delta_pct':100.0*(lagged['cash_total_yuan']-current['cash_total_yuan'])/current['cash_total_yuan'],
                 'interpretation':'one-slot delayed controller measurement; current realized L/PV is used only for automatic physical balancing/emergency recourse, not for the delayed control target',
                 'specified_dates':{}}
        lr.days.to_csv(qa.ADIR/f'E9_{br}_formal_lag1_daily.csv',index=False,encoding='utf-8-sig')
        for ds in qa.SPEC_DATES:
            day_idx=(datetime.fromisoformat(ds).date()-datetime(2025,1,1).date()).days; st=qa.start_state_for_day(br,day_idx)
            times=[qa.DATA.dates[day_idx]+timedelta(minutes=10*i) for i in range(144)]
            Lnom=historical_forecast_kwh(qa.DATA,qa.DATA.dates[day_idx],times,'load'); Gnom=historical_forecast_kwh(qa.DATA,qa.DATA.dates[day_idx],times,'pv')
            vals={}
            for mode in ('current','lagged','nominal'):
                counter={'i':0,'prevL':float(Lnom[0]),'prevG':float(Gnom[0])}
                def wrapper(*,Q,load_kwh,pv_kwh,S0,target_soc,allow_emergency_charging=False,_mode=mode,_c=counter):
                    i=_c['i']; _c['i']+=1
                    if _mode=='current': return original_exec(Q=Q,load_kwh=load_kwh,pv_kwh=pv_kwh,S0=S0,target_soc=target_soc,allow_emergency_charging=allow_emergency_charging)
                    ml,mg=(_c['prevL'],_c['prevG']) if _mode=='lagged' else (float(Lnom[min(i,143)]),float(Gnom[min(i,143)]))
                    tentative=original_exec(Q=Q,load_kwh=ml,pv_kwh=mg,S0=S0,target_soc=target_soc,allow_emergency_charging=allow_emergency_charging)
                    physical=original_exec(Q=Q,load_kwh=load_kwh,pv_kwh=pv_kwh,S0=S0,target_soc=tentative.S1,allow_emergency_charging=allow_emergency_charging)
                    _c['prevL']=float(load_kwh); _c['prevG']=float(pv_kwh); return physical
                q4_sim.execute_slot=wrapper
                try:r=simulate_range(qa.DATA,PriceForecaster(qa.DATA),qa.frozen_cfg(br,label='E9_'+mode),day_idx,day_idx+1,qa.clone(st))
                finally:q4_sim.execute_slot=original_exec
                vals[mode]=qa.compact(day_metrics(r.days))
            out[br]['specified_dates'][ds]=vals
    save('E9_measurement_timing.json',out)


def e11():
    out={}; old=(q4_flow.ETA_C,q4_flow.ETA_D,q4_control.ETA_C,q4_control.ETA_D)
    try:
        q4_flow.ETA_C=q4_flow.ETA_D=q4_control.ETA_C=q4_control.ETA_D=.9
        for br in ('q4_2','q4_3'):
            m,_=qa.runformal(qa.frozen_cfg(br,label='E11_two_oneway_90pct'))
            out[br]={'RTE90_symmetric_main':qa.compact(qa.main_result(br)),'two_oneway90_sensitivity':qa.compact(m)}
    finally:q4_flow.ETA_C,q4_flow.ETA_D,q4_control.ETA_C,q4_control.ETA_D=old
    save('E11_efficiency.json',out)


def e12():
    out={}
    for br in ('q4_2','q4_3'):
        m,r=qa.runformal(qa.frozen_cfg(br,allow_emergency_charging=True,label='E12_emergency_charge'))
        out[br]={'main_eB0':qa.compact(qa.main_result(br)),'sensitivity_eB_allowed':qa.compact(m),'sensitivity_eB_total_kwh':float(r.slots.eB.sum())}
    save('E12_emergency_charging.json',out)


def e13():
    out={}
    for br in ('q4_2','q4_3'):
        out[br]={'42_frozen_main':qa.compact(qa.main_result(br))}
        for H in (28,56):
            m,_=qa.runformal(qa.frozen_cfg(br,price_config=qa.pcfg(br,H),history_days=H,label=f'E13_history_{H}'))
            out[br][str(H)]=qa.compact(m)
    save('E13_history_window.json',out)


PARTS={'E7':e7,'E8':e8,'E9':e9,'E11':e11,'E12':e12,'E13':e13}
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('part',choices=PARTS); a=ap.parse_args(); PARTS[a.part]()
