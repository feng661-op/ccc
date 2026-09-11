# -*- coding: utf-8 -*-
from pathlib import Path
import sys,time,json,csv
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from q3_data import *
from q3_sim import simulate_strategy
ROOT=HERE.parent.parent;data=load_q3_inputs(ROOT);E=EVAL_START

def save_csv(path,rows):
    if not rows:return
    with open(path,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)

def summarize(s,runtime):
    sl=slice(E,365);comp=settlement_components(s.B[sl],s.A[sl],data.price_plan[None,:],s.down_settlement)
    formal_start=data.dates[E].date().isoformat();evf=[r for r in s.event_rows if r['date']>=formal_start];evw=[r for r in s.event_rows if r['date']<formal_start];rvf=[r for r in s.revision_rows if r['date']>=formal_start];rvw=[r for r in s.revision_rows if r['date']<formal_start];adf={(r['date'],r['event_hour']) for r in rvf};adw={(r['date'],r['event_hour']) for r in rvw};dpf=[r for r in s.dispatch_rows if r['date']>=formal_start];dpw=[r for r in s.dispatch_rows if r['date']<formal_start]
    return {'execution_mode':s.execution_mode,'measurement_model':'ideal_piecewise_constant_current_slot','curtail_semantics':'pv_only','grid_import_kwh':float(s.grid_import[sl].sum()),'unused_contract_kwh':float(s.unused_contract[sl].sum()),'supply_surplus_kwh':float(s.supply_surplus[sl].sum()),'battery_dump_kwh':float(s.battery_dump[sl].sum()),'name':s.name,'use_new_vintage':s.use_new_vintage,'allow_revision':s.allow_revision,'scenario_count':s.scenario_count,'disabled_vintage_hours':list(s.disabled_vintage_hours),'down_settlement':s.down_settlement,'formal_days':334,'total_cost_yuan':float(np.sum(s.natural_regular_fee[sl]+s.emergency_fee[sl])),'regular_cost_yuan':float(np.sum(s.natural_regular_fee[sl])),'emergency_cost_yuan':float(np.sum(s.emergency_fee[sl])),'planned_B_kwh':float(np.sum(s.B[sl])),'final_A_kwh':float(np.sum(s.A[sl])),'down_adjust_kwh':float(np.sum(comp['down_kwh'])),'up_adjust_kwh':float(np.sum(comp['up_kwh'])),'charge_kwh':float(np.sum(s.charge[sl])),'discharge_kwh':float(np.sum(s.discharge[sl])),'emergency_kwh':float(np.sum(s.emergency[sl])),'curtail_kwh':float(np.sum(s.curtail[sl])),'soc_min_kwh':float(np.min(s.soc_path[E:])),'soc_max_kwh':float(np.max(s.soc_path[E:])),'max_soc_continuity_error_kwh':float(np.max(np.abs(s.soc24[:-1]-s.soc00[1:]))),'event_count_total':len(s.event_rows),'warmup_event_count':len(evw),'formal_event_count':len(evf),'changed_slot_count_total':len(s.revision_rows),'warmup_changed_slot_count':len(rvw),'formal_changed_slot_count':len(rvf),'warmup_adjusted_event_count':len(adw),'formal_adjusted_event_count':len(adf),'formal_adjusted_event_rate':len(adf)/(334*3) if s.allow_revision else 0.0,'dispatch_count_total':len(s.dispatch_rows),'warmup_dispatch_count':len(dpw),'formal_dispatch_count':len(dpf),'solver_max_eq_residual':float(max(r['max_eq_residual'] for r in s.event_rows)),'solver_max_ub_violation':float(max(r['max_ub_violation'] for r in s.event_rows)),'max_tree_nodes':int(max(r['node_count'] for r in s.event_rows)),'max_scenarios':int(max(r['scenario_count'] for r in s.event_rows)),'scenario_weight_sum_max_error':float(max(abs(r['scenario_weight_sum']-1) for r in s.event_rows)),'scenario_clip_rate_mean':float(np.mean([r['clip_rate'] for r in s.event_rows])),'scenario_clip_rate_max':float(max(r['clip_rate'] for r in s.event_rows)),'terminal_shortfall_binding_event_count':int(sum(r['terminal_shortfall_binding_scenarios']>0 for r in s.event_rows)),'terminal_shortfall_binding_event_rate':float(np.mean([r['terminal_shortfall_binding_scenarios']>0 for r in s.event_rows])),'terminal_shortfall_expected_kwh_mean':float(np.mean([r['terminal_shortfall_expected_kwh'] for r in s.event_rows])),'runtime_seconds':float(runtime)}

if len(sys.argv)<2:raise SystemExit('usage: 运行单策略.py A|B|C|D|D_point|D_no6|D_no12|D_no18|D_K5')
name=sys.argv[1]
configs={
'A':dict(use_new_vintage=False,allow_revision=False,scenario_count=3),
'B':dict(use_new_vintage=True,allow_revision=False,scenario_count=3),
'C':dict(use_new_vintage=False,allow_revision=True,scenario_count=3),
'D':dict(use_new_vintage=True,allow_revision=True,scenario_count=3),
'D_point':dict(use_new_vintage=True,allow_revision=True,scenario_count=1),
'D_no6':dict(use_new_vintage=True,allow_revision=True,scenario_count=3,disabled_vintage_hours=(6,)),
'D_no12':dict(use_new_vintage=True,allow_revision=True,scenario_count=3,disabled_vintage_hours=(12,)),
'D_no18':dict(use_new_vintage=True,allow_revision=True,scenario_count=3,disabled_vintage_hours=(18,)),
'D_K5':dict(use_new_vintage=True,allow_revision=True,scenario_count=5),
}
if name not in configs:raise SystemExit(name)
print('RUN',name,configs[name],flush=True);t0=time.time();s=simulate_strategy(data,name=name,end_day=365,verbose=True,**configs[name]);runtime=time.time()-t0
np.savez_compressed(HERE/f'run_{name}.npz',B=s.B,A=s.A,A_stage=s.A_stage,charge=s.charge,discharge=s.discharge,emergency=s.emergency,curtail=s.curtail,grid_import=s.grid_import,unused_contract=s.unused_contract,supply_surplus=s.supply_surplus,battery_dump=s.battery_dump,execution_mode=np.asarray(s.execution_mode),physical_schema_version=np.asarray(2),soc00=s.soc00,soc24=s.soc24,soc_path=s.soc_path,plan_fee=s.plan_fee,adjusted_fee=s.adjusted_fee,natural_regular_fee=s.natural_regular_fee,emergency_fee=s.emergency_fee)
if name=='D':save_csv(HERE/'event_audit.csv',s.event_rows);save_csv(HERE/'revision_log.csv',s.revision_rows);save_csv(HERE/'dispatch_audit.csv',s.dispatch_rows)
m=summarize(s,runtime);(HERE/f'metric_{name}.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(m,ensure_ascii=False,indent=2),flush=True)
