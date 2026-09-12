# -*- coding: utf-8 -*-
"""Frozen-after-January robustness/ablation matrix E0--E15.

No result in this script is allowed to alter january_frozen_selection.json.
Formal sensitivities start from the exact frozen Jan-31 state of each branch.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from datetime import datetime,timedelta
import json, contextlib
import numpy as np, pandas as pd

import q4_flow, q4_control, q4_sim
from q4_data import load_q4_inputs
from q4_price import PriceForecaster,PriceConfig
from q4_sim import SimConfig,SimState,simulate_range
from q4_metrics import day_metrics
from q4_scenarios import build_joint_scenarios
from q4_opt import solve_event

ROOT=Path(__file__).resolve().parents[2]; CODE=Path(__file__).resolve().parent
ADIR=CODE/'ablations'; ADIR.mkdir(parents=True,exist_ok=True)
DATA=load_q4_inputs(ROOT)
FREEZE=json.loads((CODE/'january_frozen_selection.json').read_text(encoding='utf-8'))
SPEC_DATES=['2025-03-20','2025-06-21','2025-09-23','2025-12-21']

def jsave(name,obj):
 (ADIR/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda x: float(x) if isinstance(x,np.floating) else int(x) if isinstance(x,np.integer) else x.tolist() if isinstance(x,np.ndarray) else str(x)),encoding='utf-8')

def pcfg(br,hist=None):
 d=dict(FREEZE[br]['config']['price_config']);
 if hist is not None:d['history_days']=int(hist)
 return PriceConfig(**d)

def state0(br):
 d=FREEZE[br]['state_end']; return SimState(float(d['soc']),np.asarray(d['prev_B'],float),np.asarray(d['prev_A'],float))

def clone(s): return SimState(float(s.soc),s.prev_B.copy(),s.prev_A.copy())

def frozen_cfg(br,**overrides):
 """Exact January-frozen formal configuration, with one explicit sensitivity override."""
 d=dict(FREEZE[br]['config']); d['price_config']=PriceConfig(**d['price_config'])
 d['disabled_vintage_hours']=tuple(d.get('disabled_vintage_hours') or ())
 d.update(overrides)
 return SimConfig(**d)

def runformal(cfg):
 r=simulate_range(DATA,PriceForecaster(DATA),cfg,31,365,clone(state0(cfg.branch)))
 return day_metrics(r.days),r

def main_result(br):
 return day_metrics(pd.read_csv(CODE/br/'daily_ledger.csv'))

def compact(m): return {k:v for k,v in m.items() if k in ('cash_total_yuan','cash_mean_yuan_per_day','cash_cvar95_yuan_per_day','cash_worst_day_yuan','emergency_total_kwh','emergency_days','curtailment_total_kwh','unused_contract_total_kwh','soc_end_kwh','max_physics_residual')}

def start_state_for_day(br,day_idx):
 # exact state/lead contracts at natural 00:00 from authoritative main ledger
 slots=pd.read_csv(CODE/br/'physical_10min.csv'); ctr=pd.read_csv(CODE/br/'contract_ledger.csv')
 row=slots[(slots.day_index==day_idx)&(slots.slot==0)].iloc[0]
 prev=ctr[ctr.day_index==day_idx-1].sort_values('plan_j')
 if len(prev)!=144: raise RuntimeError('missing previous contract row')
 return SimState(float(row.S0_kwh),prev.B_kwh.to_numpy(float),prev.A_kwh.to_numpy(float))


def run():
 out={'frozen_main':{br:compact(main_result(br)) for br in ('q4_2','q4_3')},'experiments':{},'formal_retuning':False}
 # E0/E1/E3/E4/E5 are immutable January/frozen artifacts already produced.
 # Add a matched causal no-price-adaptation benchmark to resolve the weak R0
 # comparison: same Q4-2 state/physics/information, only contract optimization
 # ignores variable-price forecasts; realized settlement remains Attachment 4.
 c0m,c0r=runformal(frozen_cfg('q4_2',decision_fixed_price=True,label='E0_matched_no_price_adaptation'))
 main2=compact(main_result('q4_2')); c0=compact(c0m)
 c0r.days.to_csv(ADIR/'E0_q4_2_matched_no_price_adaptation_daily.csv',index=False,encoding='utf-8-sig')
 formal_daily=pd.read_csv(CODE/'q4_2'/'daily_ledger.csv').sort_values('date').reset_index(drop=True)
 c0_daily=c0r.days.sort_values('date').reset_index(drop=True)
 if list(formal_daily.date)!=list(c0_daily.date): raise AssertionError('E0 paired-date mismatch')
 diff=formal_daily.cash_fee_yuan.to_numpy(float)-c0_daily.cash_fee_yuan.to_numpy(float)
 # Preserve serial dependence with a deterministic 7-day moving-block
 # bootstrap. Positive values mean the price-adaptive policy costs more.
 rng=np.random.default_rng(2026); n=len(diff); L=7; starts=np.arange(n-L+1); boots=np.empty(5000)
 for bi in range(len(boots)):
  idx=[]
  while len(idx)<n:
   s=int(rng.choice(starts)); idx.extend(range(s,s+L))
  boots[bi]=float(np.mean(diff[np.asarray(idx[:n],int)]))
 ci=np.quantile(boots,[.025,.975])
 e0={'artifact':'e0_fixed_price_bridge.json','role':'diagnostic fixed-vs-variable tariff bridge plus matched causal policy benchmark',
     'matched_no_price_adaptation':c0,'formal_price_adaptive':main2,
     'adaptive_cash_gain_pct':100.0*(c0['cash_total_yuan']-main2['cash_total_yuan'])/c0['cash_total_yuan'],
     'paired_daily_cash_difference_adaptive_minus_C0':{'mean_yuan_per_day':float(np.mean(diff)),'median_yuan_per_day':float(np.median(diff)),
       'weekly_block_bootstrap_95pct_CI_yuan_per_day':[float(ci[0]),float(ci[1])],'bootstrap_seed':2026,'block_days':7,'replicates':5000,
       'interpretation':'interpret the paired weekly-block bootstrap interval by sign: entirely positive means adaptive is more expensive; entirely negative means adaptive is cheaper; crossing zero means no robust cash dominance'},
     'claim_boundary':'matched policy benchmark; unlike R0 it shares the Q4 state, physics and information process and changes only the decision tariff'}
 out['experiments']['E0']=e0; jsave('E0_matched_policy_benchmark.json',e0)
 pm=pd.read_csv(ROOT/'问题四'/'output'/'02_model_selection'/'price_model_ablation.csv')
 out['experiments']['E1']={'status':'run','artifact':'output/02_model_selection/price_model_ablation.csv',
                          'best_by_mae':pm.sort_values(['mae','rmse']).iloc[0].to_dict(),
                          'frozen':FREEZE['q4_2']['config']['price_config'],
                          'selection_rule':'minimum January causal walk-forward MAE; RMSE/q95 tie-break; no privileged Ridge class'}
 ladder=json.loads((CODE/'baseline_ladder.json').read_text(encoding='utf-8'))
 out['experiments']['E3']={br:{'B1_pass':ladder['branches'][br]['B1_pass'],'B1':ladder['branches'][br]['B1']} for br in ('q4_2','q4_3')}
 out['experiments']['E4']={br:{'B2_pass':ladder['branches'][br]['B2_pass'],'B2_best':ladder['branches'][br]['B2_best']} for br in ('q4_2','q4_3')}
 out['experiments']['E5']={br:ladder['branches'][br]['B2_candidates'] for br in ('q4_2','q4_3')}
 # E2 same-origin vs path-preserving cross-day mismatch; January only, because this is mechanism evidence, not formal selection.
 e2={}
 for br in ('q4_2','q4_3'):
  s=state0(br) # state at Feb1 is not appropriate for Jan; use saved Jan common warmup state instead
  wd=json.loads((ROOT/'问题四'/'output'/'02_model_selection'/br/'common_warmup_state.json').read_text(encoding='utf-8'))
  ws=SimState(float(wd['soc']),np.asarray(wd['prev_B'],float),np.asarray(wd['prev_A'],float))
  base=SimConfig(branch=br,level='B1',price_config=pcfg(br),scenario_k=3,label='E2_same_origin')
  rs=simulate_range(DATA,PriceForecaster(DATA),base,8,31,clone(ws))
  rm=simulate_range(DATA,PriceForecaster(DATA),SimConfig(**{**asdict(base),'price_config':base.price_config,'mismatch':True,'label':'E2_cross_day_mismatch'}),8,31,clone(ws))
  e2[br]={'same_origin':compact(day_metrics(rs.days)),'cross_day_mismatch':compact(day_metrics(rm.days)),'internal_path_order_preserved':True}
 out['experiments']['E2']=e2; jsave('E2_joint_structure.json',e2)
 # E6 Q4-3 information x permission, full formal sensitivity at the exact
 # January-frozen scenario/risk level; only information/permission toggles.
 e6={}
 for tag,pvu,rev in [('A_none',False,False),('B_PV_only',True,False),('C_revision_only',False,True),('D_both',True,True)]:
  m,r=runformal(frozen_cfg('q4_3',use_pv_updates=pvu,allow_contract_revision=rev,label='E6_'+tag))
  e6[tag]=compact(m); r.days.to_csv(ADIR/f'E6_{tag}_daily.csv',index=False,encoding='utf-8-sig')
 out['experiments']['E6']=e6; jsave('E6_information_permission.json',e6)
 # E7 disable each legal update version independently.
 e7={'main_all_versions':compact(main_result('q4_3'))}
 for h in (6,12,18):
  m,r=runformal(frozen_cfg('q4_3',disabled_vintage_hours=(h,),label=f'E7_disable_{h:02d}'))
  e7[f'disable_{h:02d}']=compact(m); r.days.to_csv(ADIR/f'E7_disable_{h:02d}_daily.csv',index=False,encoding='utf-8-sig')
 out['experiments']['E7']=e7; jsave('E7_event_contribution.json',e7)
 # E8 simple terminal vs zero terminal and dynamic causal terminal-value diagnostic.
 e8={}
 orig_solve=q4_sim.solve_event
 def dyn_solve(**kw):
  sc=kw['scenarios']; # causal price support only
  tail=sc.price[:,-min(12,sc.price.shape[1]):]
  kw['terminal_value']=float(np.median(tail))
  return orig_solve(**kw)
 for br in ('q4_2','q4_3'):
  m0,r0=runformal(frozen_cfg(br,terminal_value=0.0,label='E8_no_terminal_value'))
  q4_sim.solve_event=dyn_solve
  try: md,rd=runformal(frozen_cfg(br,terminal_value=.8,label='E8_dynamic_nu'))
  finally: q4_sim.solve_event=orig_solve
  e8[br]={'simple_frozen':compact(main_result(br)),'zero_terminal_value':compact(m0),'dynamic_nu_causal_tail_median':compact(md),
          'execution_horizon_note':'execution remains 10-min one-step; event horizons are inherited Q2/Q3 information horizons and therefore are not tuned on formal data'}
 out['experiments']['E8']=e8; jsave('E8_terminal_horizon.json',e8)
 # E9 feedback timing. First run the full formal one-slot-lag sensitivity,
 # then retain the four specified-date current/lagged/nominal drill-down.
 e9={}; original_exec=q4_sim.execute_slot
 from q4_forecast import historical_forecast_kwh
 for br in ('q4_2','q4_3'):
  # At formal start the most recent controller observation is Jan-31 23:50,
  # which is already released.  The delayed controller sees only the previous
  # 10-min L/PV; current actual L/PV enters the second call solely as automatic
  # physical balancing/recourse so the source-flow ledger remains exact.
  lag={'prevL':float(DATA.load_cal_kwh[30,143]),'prevG':float(DATA.pv_cal_kwh[30,143])}
  def full_lag_wrapper(*,Q,load_kwh,pv_kwh,S0,target_soc,allow_emergency_charging=False,_c=lag):
   tentative=original_exec(Q=Q,load_kwh=_c['prevL'],pv_kwh=_c['prevG'],S0=S0,target_soc=target_soc,
                           allow_emergency_charging=allow_emergency_charging)
   physical=original_exec(Q=Q,load_kwh=load_kwh,pv_kwh=pv_kwh,S0=S0,target_soc=tentative.S1,
                          allow_emergency_charging=allow_emergency_charging)
   _c['prevL']=float(load_kwh);_c['prevG']=float(pv_kwh)
   return physical
  q4_sim.execute_slot=full_lag_wrapper
  try:
   lm,lr=runformal(frozen_cfg(br,label='E9_formal_lag1_measurement'))
  finally:q4_sim.execute_slot=original_exec
  current=compact(main_result(br)); lagged=compact(lm)
  e9[br]={'formal_current':current,'formal_lag1_measurement':lagged,
          'cash_delta_pct':100.0*(lagged['cash_total_yuan']-current['cash_total_yuan'])/current['cash_total_yuan'],
          'interpretation':'one-slot delayed controller measurement; current realized L/PV is used only for automatic physical balancing/emergency recourse, not for the delayed control target',
          'specified_dates':{}}
  lr.days.to_csv(ADIR/f'E9_{br}_formal_lag1_daily.csv',index=False,encoding='utf-8-sig')
  for ds in SPEC_DATES:
   day_idx=(datetime.fromisoformat(ds).date()-datetime(2025,1,1).date()).days; st=start_state_for_day(br,day_idx)
   times=[DATA.dates[day_idx]+timedelta(minutes=10*i) for i in range(144)]
   Lnom=historical_forecast_kwh(DATA,DATA.dates[day_idx],times,'load'); Gnom=historical_forecast_kwh(DATA,DATA.dates[day_idx],times,'pv')
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
    try:r=simulate_range(DATA,PriceForecaster(DATA),frozen_cfg(br,label='E9_'+mode),day_idx,day_idx+1,clone(st))
    finally:q4_sim.execute_slot=original_exec
    vals[mode]=compact(day_metrics(r.days))
   e9[br]['specified_dates'][ds]=vals
 out['experiments']['E9']=e9; jsave('E9_measurement_timing.json',e9)
 # E10 fixed-main-strategy settlement exposure over formal period + explicit label (not reoptimized).
 e10={}
 for br in ('q4_2','q4_3'):
  slots=pd.read_csv(CODE/br/'physical_10min.csv'); B=slots.B_kwh.to_numpy(float); A=slots.A_kwh.to_numpy(float); c=slots.price_yuan_per_kwh.to_numpy(float); em=slots.e_kwh.to_numpy(float)
  original=float(slots.cash_fee_yuan.sum())
  sunk=float(np.sum(c*B + .5*c*np.maximum(B-A,0)+1.5*c*np.maximum(A-B,0)+5*c*em))
  # sequential one-step exposure is deliberately not called optimized: without intermediate A-event vectors the authoritative ledger cannot reconstruct a sequential reoptimization.
  e10[br]={'original_anchor_main_yuan':original,'sunk_plan_fee_exposure_yuan':sunk,'fixed_strategy_only':True,
           'claim_boundary':'exposure only; alternative settlement requires reoptimization and is not used to rank/freeze the main model'}
 out['experiments']['E10']=e10; jsave('E10_settlement_exposure.json',e10)
 # E11 efficiency: patch the single canonical flow kernel and executor together, then replay full formal sensitivity.
 e11={}; old=(q4_flow.ETA_C,q4_flow.ETA_D,q4_control.ETA_C,q4_control.ETA_D)
 try:
  q4_flow.ETA_C=q4_flow.ETA_D=q4_control.ETA_C=q4_control.ETA_D=.9
  for br in ('q4_2','q4_3'):
   m,r=runformal(frozen_cfg(br,label='E11_two_oneway_90pct'))
   e11[br]={'RTE90_symmetric_main':compact(main_result(br)),'two_oneway90_sensitivity':compact(m)}
 finally:q4_flow.ETA_C,q4_flow.ETA_D,q4_control.ETA_C,q4_control.ETA_D=old
 out['experiments']['E11']=e11; jsave('E11_efficiency.json',e11)
 # E12 emergency charging allowed, full formal.
 e12={}
 for br in ('q4_2','q4_3'):
  m,r=runformal(frozen_cfg(br,allow_emergency_charging=True,label='E12_emergency_charge'))
  e12[br]={'main_eB0':compact(main_result(br)),'sensitivity_eB_allowed':compact(m),'sensitivity_eB_total_kwh':float(r.slots.eB.sum())}
 out['experiments']['E12']=e12; jsave('E12_emergency_charging.json',e12)
 # E13 formal sensitivity only; never feed back into selection.
 e13={}
 for br in ('q4_2','q4_3'):
  e13[br]={'42_frozen_main':compact(main_result(br))}
  for H in (28,56):
   m,r=runformal(frozen_cfg(br,price_config=pcfg(br,H),history_days=H,label=f'E13_history_{H}'))
   e13[br][str(H)]=compact(m)
 out['experiments']['E13']=e13; jsave('E13_history_window.json',e13)
 # E14 is generated by q4_oracle_run.py; record linkage now, actual values merged by run_q4/metrics when job completes.
 out['experiments']['E14']={'status':'separate_job','artifact':'oracle_audit.json','claim':'PO diagnostic only; matched FI only strict cash lower bound'}
 # E15 real-event lex preservation from saved selected/January event optimizer evidence.
 # q4_opt stores the exact stage-1 optimum and tolerances; final source-flow ledger verifies no simultaneous charge/discharge and stage-3 source identity.
 e15={}
 for br in ('q4_2','q4_3'):
  selected=FREEZE[br]['selected']; ev=pd.read_csv(ROOT/'问题四'/'output'/'02_model_selection'/br/selected/'events.csv')
  e15[br]={'events':int(len(ev)),'max_eq_residual':float(ev.max_eq_residual.max()),'stage1_objective_min':float(ev.level1.min()),'stage1_objective_max':float(ev.level1.max()),
           'selected_level':selected,'lex_order':'cash/risk -> throughput -> PV curtailment','note':'event solver enforces stage1 <= f1+tol1 then stage2 <= f2+tol2 before stage3; separate T27/T29 numerical tests check preservation on real events'}
 out['experiments']['E15']=e15; jsave('E15_lexicographic.json',e15)
 jsave('ablation_matrix_E0_E15.json',out)
 print(json.dumps({k:'done' for k in out['experiments']},ensure_ascii=False),flush=True)
if __name__=='__main__': run()
