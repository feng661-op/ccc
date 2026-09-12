from pathlib import Path
import json,numpy as np,pandas as pd
import q4_sim
from q4_data import load_q4_inputs
from q4_price import PriceForecaster,PriceConfig
from q4_sim import SimConfig,SimState,simulate_range
from q4_metrics import day_metrics
ROOT=Path(__file__).resolve().parents[2]; CODE=Path(__file__).resolve().parent; AD=CODE/'ablations'; D=load_q4_inputs(ROOT); F=json.loads((CODE/'january_frozen_selection.json').read_text(encoding='utf-8'))
def st(br):
 d=F[br]['state_end'];return SimState(float(d['soc']),np.array(d['prev_B']),np.array(d['prev_A']))
def pc(br,h):
 d=dict(F[br]['config']['price_config']);d['history_days']=h;return PriceConfig(**d)
def comp(m):return {k:v for k,v in m.items() if k in ('cash_total_yuan','cash_mean_yuan_per_day','cash_cvar95_yuan_per_day','cash_worst_day_yuan','emergency_total_kwh','emergency_days','curtailment_total_kwh','unused_contract_total_kwh','soc_end_kwh','max_physics_residual')}
def run():
 e={'q4_2':{'42_frozen_main':comp(day_metrics(pd.read_csv(CODE/'q4_2'/'daily_ledger.csv'))),'28':{'cash_total_yuan':19322679.58851721,'cash_mean_yuan_per_day':57852.33379795572},'56':{'cash_total_yuan':19312466.06672262,'cash_mean_yuan_per_day':57821.754690785096}},'q4_3':{'42_frozen_main':comp(day_metrics(pd.read_csv(CODE/'q4_3'/'daily_ledger.csv')))}}
 orig=q4_sim.solve_event
 def stable(**kw):kw.setdefault('lex_abs_tol',1e-2);return orig(**kw)
 q4_sim.solve_event=stable
 try:
  for h in (28,56):
   r=simulate_range(D,PriceForecaster(D),SimConfig(branch='q4_3',level='B0',price_config=pc('q4_3',h),history_days=h,label=f'E13_{h}'),31,365,st('q4_3'));e['q4_3'][str(h)]=comp(day_metrics(r.days));print(h,e['q4_3'][str(h)]['cash_total_yuan'],flush=True)
 finally:q4_sim.solve_event=orig
 e['numerical_note']='Frozen-after-January sensitivity only. q4_2 values are from the completed preceding E13 run before its q4_3 failure; q4_3 rerun uses 0.01 yuan absolute lexicographic tie tolerance solely to avoid HiGHS stage-3 numerical infeasibility. Main formal replay remains 1e-7 and unchanged.'
 (AD/'E13_history_window.json').write_text(json.dumps(e,ensure_ascii=False,indent=2),encoding='utf-8')
 print('DONE')
if __name__=='__main__':run()
