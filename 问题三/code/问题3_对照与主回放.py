# -*- coding: utf-8 -*-
from pathlib import Path
import sys,time,json,csv
import numpy as np
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
from q3_data import load_q3_inputs,EVAL_START,settlement_components
from q3_sim import simulate_strategy
ROOT=HERE.parent.parent; OUT=ROOT/'问题三'/'code'; data=load_q3_inputs(ROOT)

def save_csv(path,rows):
    if not rows:return
    with open(path,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)

def save_sim(s):
    np.savez_compressed(OUT/f'run_{s.name}.npz',B=s.B,A=s.A,A_stage=s.A_stage,charge=s.charge,discharge=s.discharge,emergency=s.emergency,curtail=s.curtail,grid_import=s.grid_import,unused_contract=s.unused_contract,supply_surplus=s.supply_surplus,battery_dump=s.battery_dump,execution_mode=np.asarray(s.execution_mode),physical_schema_version=np.asarray(2),soc00=s.soc00,soc24=s.soc24,soc_path=s.soc_path,plan_fee=s.plan_fee,adjusted_fee=s.adjusted_fee,natural_regular_fee=s.natural_regular_fee,emergency_fee=s.emergency_fee)
    if s.name=='D':
        save_csv(OUT/'event_audit.csv',s.event_rows); save_csv(OUT/'revision_log.csv',s.revision_rows); save_csv(OUT/'dispatch_audit.csv',s.dispatch_rows)
    sl=slice(EVAL_START,365); comp=settlement_components(s.B[sl],s.A[sl],data.price_plan[None,:],s.down_settlement)
    formal_dates={data.dates[d].date().isoformat() for d in range(EVAL_START,365)}
    ev_formal=[r for r in s.event_rows if r['date'] in formal_dates]; ev_warm=[r for r in s.event_rows if r['date'] not in formal_dates]
    rev_formal=[r for r in s.revision_rows if r['date'] in formal_dates]; rev_warm=[r for r in s.revision_rows if r['date'] not in formal_dates]
    adj_events_formal={(r['date'],r['event_hour']) for r in rev_formal}; adj_events_warm={(r['date'],r['event_hour']) for r in rev_warm}
    disp_formal=[r for r in s.dispatch_rows if r['date'] in formal_dates]; disp_warm=[r for r in s.dispatch_rows if r['date'] not in formal_dates]
    return {
      'name':s.name,'use_new_vintage':s.use_new_vintage,'allow_revision':s.allow_revision,'scenario_count':s.scenario_count,'disabled_vintage_hours':list(s.disabled_vintage_hours),'down_settlement':s.down_settlement,
      'formal_days':334,'total_cost_yuan':float(np.sum(s.natural_regular_fee[sl]+s.emergency_fee[sl])),
      'regular_cost_yuan':float(np.sum(s.natural_regular_fee[sl])),'emergency_cost_yuan':float(np.sum(s.emergency_fee[sl])),
      'planned_B_kwh':float(np.sum(s.B[sl])),'final_A_kwh':float(np.sum(s.A[sl])),'down_adjust_kwh':float(np.sum(comp['down_kwh'])),'up_adjust_kwh':float(np.sum(comp['up_kwh'])),
      'charge_kwh':float(np.sum(s.charge[sl])),'discharge_kwh':float(np.sum(s.discharge[sl])),'emergency_kwh':float(np.sum(s.emergency[sl])),'curtail_kwh':float(np.sum(s.curtail[sl])),
      'soc_min_kwh':float(np.min(s.soc_path[EVAL_START:])),'soc_max_kwh':float(np.max(s.soc_path[EVAL_START:])),
      'max_soc_continuity_error_kwh':float(np.max(np.abs(s.soc24[:-1]-s.soc00[1:]))),
      'event_count_total':len(s.event_rows),'warmup_event_count':len(ev_warm),'formal_event_count':len(ev_formal),
      'changed_slot_count_total':len(s.revision_rows),'warmup_changed_slot_count':len(rev_warm),'formal_changed_slot_count':len(rev_formal),
      'warmup_adjusted_event_count':len(adj_events_warm),'formal_adjusted_event_count':len(adj_events_formal),'formal_adjusted_event_rate':len(adj_events_formal)/(334*3) if s.allow_revision else 0.0,
      'dispatch_count_total':len(s.dispatch_rows),'warmup_dispatch_count':len(disp_warm),'formal_dispatch_count':len(disp_formal),
      'solver_max_eq_residual':float(max(r['max_eq_residual'] for r in s.event_rows)),'solver_max_ub_violation':float(max(r['max_ub_violation'] for r in s.event_rows)),
      'max_tree_nodes':int(max(r['node_count'] for r in s.event_rows)),'max_scenarios':int(max(r['scenario_count'] for r in s.event_rows)),
      'scenario_weight_sum_max_error':float(max(abs(r['scenario_weight_sum']-1.0) for r in s.event_rows)),
      'scenario_clip_rate_mean':float(np.mean([r['clip_rate'] for r in s.event_rows])),'scenario_clip_rate_max':float(max(r['clip_rate'] for r in s.event_rows)),
      'terminal_shortfall_binding_event_count':int(sum(r['terminal_shortfall_binding_scenarios']>0 for r in s.event_rows)),
      'terminal_shortfall_binding_event_rate':float(np.mean([r['terminal_shortfall_binding_scenarios']>0 for r in s.event_rows])),
      'terminal_shortfall_expected_kwh_mean':float(np.mean([r['terminal_shortfall_expected_kwh'] for r in s.event_rows])),
    }

variants=[('A',False,False,3),('B',True,False,3),('C',False,True,3),('D',True,True,3),('D_point',True,True,1)]
metrics={}; t_all=time.time()
for name,new,flex,k in variants:
    t=time.time(); print(f'RUN {name} start',flush=True)
    s=simulate_strategy(data,name=name,use_new_vintage=new,allow_revision=flex,scenario_count=k,end_day=365,verbose=True)
    m=save_sim(s); m['runtime_seconds']=time.time()-t; metrics[name]=m
    with open(OUT/'metrics_partial.json','w',encoding='utf-8') as f: json.dump(metrics,f,ensure_ascii=False,indent=2)
    print(f'RUN {name} done cost={m["total_cost_yuan"]:.3f} emg={m["emergency_kwh"]:.3f} sec={m["runtime_seconds"]:.1f}',flush=True)
JA,JB,JC,JD=[metrics[x]['total_cost_yuan'] for x in ('A','B','C','D')]
metrics['factorial']={'realized_forecast_update_effect_B_minus_A':JB-JA,'delta_info_cost_B_minus_A':JB-JA,'delta_flex_cost_C_minus_A':JC-JA,'delta_interaction_cost':JD-JB-JC+JA,'info_saving_A_minus_B':JA-JB,'flex_saving_A_minus_C':JA-JC,'joint_saving_A_minus_D':JA-JD,'interaction_saving':-(JD-JB-JC+JA)}
metrics['scenario_comparison']={'interpretation':'realized policy effect; not theoretical VSS/VOI','D_point_total_cost_yuan':metrics['D_point']['total_cost_yuan'],'D_total_cost_yuan':JD,'D_point_minus_D_realized_cost':metrics['D_point']['total_cost_yuan']-JD}
metrics['total_runtime_seconds']=time.time()-t_all
with open(OUT/'metrics.json','w',encoding='utf-8') as f:json.dump(metrics,f,ensure_ascii=False,indent=2)
print('ALL DONE',json.dumps(metrics['factorial'],ensure_ascii=False),flush=True)
