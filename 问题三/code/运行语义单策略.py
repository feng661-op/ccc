# -*- coding: utf-8 -*-
from pathlib import Path
import sys,time,json
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from q3_data import *
from q3_sim import simulate_strategy
ROOT=HERE.parent.parent;data=load_q3_inputs(ROOT);E=EVAL_START
if len(sys.argv)<2:raise SystemExit('usage: 运行语义单策略.py sunk|stepwise')
mode=sys.argv[1]
if mode=='sunk': name='D_sunk'; kwargs=dict(down_settlement='sunk_plan_plus_penalty',revision_anchor='original_anchor')
elif mode=='stepwise': name='D_stepwise'; kwargs=dict(down_settlement='cancel_settlement',revision_anchor='stepwise_revision')
else: raise SystemExit(mode)
from q3_run_guard import claim
claim(name)
print('RUN',name,flush=True);t0=time.time();s=simulate_strategy(data,name=name,use_new_vintage=True,allow_revision=True,scenario_count=9,end_day=365,verbose=True,**kwargs);runtime=time.time()-t0
np.savez_compressed(HERE/f'run_{name}.npz',B=s.B,A=s.A,A_stage=s.A_stage,charge=s.charge,discharge=s.discharge,emergency=s.emergency,curtail=s.curtail,grid_import=s.grid_import,unused_contract=s.unused_contract,supply_surplus=s.supply_surplus,battery_dump=s.battery_dump,execution_mode=np.asarray(s.execution_mode),physical_schema_version=np.asarray(2),soc00=s.soc00,soc24=s.soc24,soc_path=s.soc_path,plan_fee=s.plan_fee,adjusted_fee=s.adjusted_fee,natural_regular_fee=s.natural_regular_fee,emergency_fee=s.emergency_fee)
m={'name':name,'mode':mode,'total_cost_yuan':float(np.sum(s.natural_regular_fee[E:]+s.emergency_fee[E:])),'regular_cost_yuan':float(np.sum(s.natural_regular_fee[E:])),'emergency_cost_yuan':float(np.sum(s.emergency_fee[E:])),'emergency_kwh':float(np.sum(s.emergency[E:])),'soc_min_kwh':float(np.min(s.soc_path[E:])),'soc_max_kwh':float(np.max(s.soc_path[E:])),'changed_slot_count_formal':sum(1 for r in s.revision_rows if r['date']>=data.dates[E].date().isoformat()),'adjusted_event_count_formal':len({(r['date'],r['event_hour']) for r in s.revision_rows if r['date']>=data.dates[E].date().isoformat()}),'runtime_seconds':float(runtime)}
(HERE/f'metric_{name}.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(m,ensure_ascii=False,indent=2),flush=True)
