# -*- coding: utf-8 -*-
"""Resume only unfinished E13 and assemble E0--E15 after the long partial run."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json, numpy as np, pandas as pd
import q4_sim
from q4_data import load_q4_inputs
from q4_price import PriceForecaster,PriceConfig
from q4_sim import SimConfig,SimState,simulate_range
from q4_metrics import day_metrics
ROOT=Path(__file__).resolve().parents[2]; CODE=Path(__file__).resolve().parent; AD=CODE/'ablations'; DATA=load_q4_inputs(ROOT)
FREEZE=json.loads((CODE/'january_frozen_selection.json').read_text(encoding='utf-8')); LAD=json.loads((CODE/'baseline_ladder.json').read_text(encoding='utf-8'))
def pc(br,h):
 d=dict(FREEZE[br]['config']['price_config']);d['history_days']=h;return PriceConfig(**d)
def st(br):
 d=FREEZE[br]['state_end'];return SimState(float(d['soc']),np.array(d['prev_B'],float),np.array(d['prev_A'],float))
def compact(m):return {k:v for k,v in m.items() if k in ('cash_total_yuan','cash_mean_yuan_per_day','cash_cvar95_yuan_per_day','cash_worst_day_yuan','emergency_total_kwh','emergency_days','curtailment_total_kwh','unused_contract_total_kwh','soc_end_kwh','max_physics_residual')}
def readj(n):return json.loads((AD/n).read_text(encoding='utf-8'))
def run():
 # E13 only: slightly relaxed absolute lex tie tolerance to avoid HiGHS infeasibility in sensitivity; main formal outputs remain untouched.
 orig=q4_sim.solve_event
 def stable(**kw):
  kw.setdefault('lex_abs_tol',1e-4);return orig(**kw)
 q4_sim.solve_event=stable
 e13={}
 try:
  for br in ('q4_2','q4_3'):
   main=day_metrics(pd.read_csv(CODE/br/'daily_ledger.csv'));e13[br]={'42_frozen_main':compact(main)}
   for h in (28,56):
    cfg=SimConfig(branch=br,level='B0',price_config=pc(br,h),history_days=h,label=f'E13_history_{h}')
    r=simulate_range(DATA,PriceForecaster(DATA),cfg,31,365,st(br));e13[br][str(h)]=compact(day_metrics(r.days));print(br,h,e13[br][str(h)]['cash_total_yuan'],flush=True)
 finally:q4_sim.solve_event=orig
 (AD/'E13_history_window.json').write_text(json.dumps(e13,ensure_ascii=False,indent=2),encoding='utf-8')
 out={'frozen_main':{br:compact(day_metrics(pd.read_csv(CODE/br/'daily_ledger.csv'))) for br in ('q4_2','q4_3')},'experiments':{},'formal_retuning':False}
 out['experiments']['E0']={'status':'run','artifact':'e0_fixed_price_bridge.json','role':'diagnostic fixed-vs-variable tariff bridge'}
 pm=pd.read_csv(ROOT/'问题四'/'output'/'02_model_selection'/'price_model_ablation.csv');out['experiments']['E1']={'status':'run','artifact':'output/02_model_selection/price_model_ablation.csv','best_by_mae':pm.sort_values('mae').iloc[0].to_dict(),'frozen':'full_ridge/global/alpha10; six-block failed stability; lag7 predictive superiority retained as negative evidence'}
 out['experiments']['E2']=readj('E2_joint_structure.json')
 out['experiments']['E3']={br:{'B1_pass':LAD['branches'][br]['B1_pass'],'B1':LAD['branches'][br]['B1']} for br in ('q4_2','q4_3')}
 out['experiments']['E4']={br:{'B2_pass':LAD['branches'][br]['B2_pass'],'B2_best':LAD['branches'][br]['B2_best']} for br in ('q4_2','q4_3')}
 out['experiments']['E5']={br:LAD['branches'][br]['B2_candidates'] for br in ('q4_2','q4_3')}
 for e,fn in [('E6','E6_information_permission.json'),('E7','E7_event_contribution.json'),('E8','E8_terminal_horizon.json'),('E9','E9_measurement_timing.json'),('E10','E10_settlement_exposure.json'),('E11','E11_efficiency.json'),('E12','E12_emergency_charging.json')]:out['experiments'][e]=readj(fn)
 out['experiments']['E13']={**e13,'numerical_note':'28/56-day frozen-after-January sensitivity only; event lex absolute tie tolerance=1e-4 yuan to avoid HiGHS stage-3 numerical infeasibility. Frozen main formal replay remains 1e-7 and unchanged.'}
 oracle=json.loads((CODE/'oracle_audit.json').read_text(encoding='utf-8'));out['experiments']['E14']={'artifact':'oracle_audit.json','q4_2':oracle['q4_2']['audit'],'q4_3':oracle['q4_3']['audit']}
 e15={}
 for br in ('q4_2','q4_3'):
  ev=pd.read_csv(ROOT/'问题四'/'output'/'02_model_selection'/br/'B0'/'events.csv');e15[br]={'events':len(ev),'max_eq_residual':float(ev.max_eq_residual.max()),'lex_order':'cash/risk -> throughput -> PV curtailment','active_numeric_acceptance':'T27-T29 in q4_validate.py'}
 out['experiments']['E15']=e15
 exact=AD/'E10_settlement_exact_local.json'
 if exact.exists():out['experiments']['E10_exact_local_reoptimization']=readj('E10_settlement_exact_local.json')
 (AD/'ablation_matrix_E0_E15.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,default=lambda x:float(x) if isinstance(x,np.floating) else int(x) if isinstance(x,np.integer) else str(x)),encoding='utf-8')
 print('ASSEMBLED',sorted(out['experiments']),flush=True)
if __name__=='__main__':run()
