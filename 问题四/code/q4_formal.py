# -*- coding: utf-8 -*-
"""334-day continuous formal replay with monthly checkpoint/resume and tail bridge."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from datetime import datetime
import argparse, json, shutil
import numpy as np
import pandas as pd

from q4_data import load_q4_inputs, FORMAL_START_DAY, FORMAL_END_DAY
from q4_price import PriceForecaster, PriceConfig
from q4_sim import SimConfig, SimState, simulate_range, replay_summary
from q4_control import execute_slot
from q4_settlement import settlement_components
from q4_metrics import day_metrics

ROOT=Path(__file__).resolve().parents[2]
CODE=Path(__file__).resolve().parent
FREEZE=CODE/'january_frozen_selection.json'


def _cfg(d):
    x=dict(d); pc=PriceConfig(**x.pop('price_config'))
    if 'disabled_vintage_hours' in x: x['disabled_vintage_hours']=tuple(x['disabled_vintage_hours'])
    return SimConfig(price_config=pc,**x)

def _state(d): return SimState(float(d['soc']),np.asarray(d['prev_B'],float),np.asarray(d['prev_A'],float))
def _state_dict(s): return {'soc':float(s.soc),'prev_B':s.prev_B.tolist(),'prev_A':s.prev_A.tolist()}

def month_edges():
    out=[FORMAL_START_DAY]
    cur=datetime(2025,3,1)
    while cur.year==2025:
        out.append((cur.date()-datetime(2025,1,1).date()).days)
        if cur.month==12: break
        cur=datetime(2025,cur.month+1,1)
    out.append(FORMAL_END_DAY)
    return sorted(set(out))

def run_branch(branch:str,resume:bool=True):
    freeze=json.loads(FREEZE.read_text(encoding='utf-8'))[branch]
    cfg=_cfg(freeze['config']); cfg=SimConfig(**{**asdict(cfg),'price_config':cfg.price_config,'label':'FORMAL_FROZEN'})
    state=_state(freeze['state_end'])
    bdir=CODE/branch; cpdir=bdir/'checkpoints'; bdir.mkdir(parents=True,exist_ok=True); cpdir.mkdir(parents=True,exist_ok=True)
    edges=month_edges(); completed=[]
    if resume:
        for a,b in zip(edges[:-1],edges[1:]):
            cp=cpdir/f'checkpoint_{a:03d}_{b:03d}.json'
            if not cp.exists(): break
            meta=json.loads(cp.read_text(encoding='utf-8'))
            if not meta.get('complete',False): break
            state=_state(meta['state_end']); completed.append((a,b))
    start_idx=len(completed)
    data=load_q4_inputs(ROOT); pf=PriceForecaster(data)
    for a,b in zip(edges[:-1][start_idx:],edges[1:][start_idx:]):
        r=simulate_range(data,pf,cfg,a,b,state)
        stem=f'{a:03d}_{b:03d}'
        r.slots.to_csv(cpdir/f'slots_{stem}.csv',index=False,encoding='utf-8-sig')
        r.events.to_csv(cpdir/f'events_{stem}.csv',index=False,encoding='utf-8-sig')
        r.contracts.to_csv(cpdir/f'contracts_{stem}.csv',index=False,encoding='utf-8-sig')
        r.days.to_csv(cpdir/f'days_{stem}.csv',index=False,encoding='utf-8-sig')
        state=r.state_end
        (cpdir/f'checkpoint_{stem}.json').write_text(json.dumps({'branch':branch,'range':[a,b],'complete':True,'state_end':_state_dict(state),'summary':replay_summary(r)},ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'branch':branch,'checkpoint':[a,b],'cash':float(r.days.cash_fee_yuan.sum()),'soc_end':state.soc}),flush=True)
    # Merge only after all 334 days exist.
    slots=[]; events=[]; contracts=[]; days=[]
    for a,b in zip(edges[:-1],edges[1:]):
        stem=f'{a:03d}_{b:03d}'
        if not (cpdir/f'checkpoint_{stem}.json').exists(): raise RuntimeError(f'missing checkpoint {stem}')
        slots.append(pd.read_csv(cpdir/f'slots_{stem}.csv'))
        events.append(pd.read_csv(cpdir/f'events_{stem}.csv'))
        contracts.append(pd.read_csv(cpdir/f'contracts_{stem}.csv'))
        days.append(pd.read_csv(cpdir/f'days_{stem}.csv'))
    slots=pd.concat(slots,ignore_index=True); events=pd.concat(events,ignore_index=True); contracts=pd.concat(contracts,ignore_index=True); days=pd.concat(days,ignore_index=True)
    if len(days)!=334 or len(slots)!=334*144 or len(contracts)!=334*144: raise AssertionError((len(days),len(slots),len(contracts)))
    slots.to_csv(bdir/'physical_10min.csv',index=False,encoding='utf-8-sig')
    events.to_csv(bdir/'event_ledger.csv',index=False,encoding='utf-8-sig')
    contracts.to_csv(bdir/'contract_ledger.csv',index=False,encoding='utf-8-sig')
    days.to_csv(bdir/'daily_ledger.csv',index=False,encoding='utf-8-sig')
    state=SimState(float(days.S24_kwh.iloc[-1]),contracts[contracts.day_index==364].B_kwh.to_numpy(float),contracts[contracts.day_index==364].A_kwh.to_numpy(float))
    # Replay the explicitly committed +1 tail using the actual Dec31 plan-row tail in Attachments 2/4.
    Btail=float(state.prev_B[143]); Atail=float(state.prev_A[143]); ctail=float(data.price_plan[364,143])
    Ltail=float(data.load_plan_kwh[364,143]); Gtail=float(data.pv_plan_kwh[364,143]); Ttail=float(days.next_tail_target_soc_kwh.iloc[-1])
    act=execute_slot(Q=Atail,load_kwh=Ltail,pv_kwh=Gtail,S0=float(state.soc),target_soc=Ttail,allow_emergency_charging=cfg.allow_emergency_charging)
    sett=settlement_components(Btail,Atail,ctail,act.eL+act.eB)
    tail={'timestamp':'2026-01-01T00:00:00','source_plan_date':'2025-12-31','plan_j':143,'B_kwh':Btail,'A_kwh':Atail,'price_yuan_per_kwh':ctail,
          'load_kwh':Ltail,'pv_kwh':Gtail,'S0_kwh':act.S0,'target_soc_kwh':Ttail,'S1_kwh':act.S1,'qL':act.qL,'qB':act.qB,'u':act.u,
          'gL':act.gL,'gB':act.gB,'kappa':act.kappa,'eL':act.eL,'eB':act.eB,'x':act.x,'y':act.y,'I_kwh':act.I,
          'regular_fee_yuan':sett.regular_fee,'emergency_fee_yuan':sett.emergency_fee,'cash_fee_yuan':sett.total_cash,'physics_residual':act.residual_max,
          'schema_hash':act.schema_hash,'scope':'committed +1 tail bridge; excluded from 334-natural-day formal KPI but required for template/continuity audit'}
    (bdir/'tail_bridge.json').write_text(json.dumps(tail,ensure_ascii=False,indent=2),encoding='utf-8')
    summary={'branch':branch,'selected':freeze['selected'],'config':asdict(cfg),'formal':day_metrics(days),'tail_bridge':tail,
             'row_counts':{'slots':len(slots),'events':len(events),'contracts':len(contracts),'days':len(days)},
             'formal_state_start':freeze['state_end'],'formal_state_end':_state_dict(state)}
    (bdir/'formal_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--branch',choices=['q4_2','q4_3','both'],default='both'); ap.add_argument('--no-resume',action='store_true'); a=ap.parse_args()
    targets=('q4_2','q4_3') if a.branch=='both' else (a.branch,)
    out={b:run_branch(b,not a.no_resume) for b in targets}
    print(json.dumps({b:{'selected':v['selected'],'cash':v['formal']['cash_total_yuan'],'emergency':v['formal']['emergency_total_kwh']} for b,v in out.items()},ensure_ascii=False),flush=True)
if __name__=='__main__': main()
