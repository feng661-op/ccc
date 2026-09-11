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
    np.savez_compressed(OUT/f'run_{s.name}.npz',B=s.B,A=s.A,A_stage=s.A_stage,charge=s.charge,discharge=s.discharge,emergency=s.emergency,curtail=s.curtail,soc00=s.soc00,soc24=s.soc24,soc_path=s.soc_path,plan_fee=s.plan_fee,adjusted_fee=s.adjusted_fee,natural_regular_fee=s.natural_regular_fee,emergency_fee=s.emergency_fee)
    if s.name=='D':
        save_csv(OUT/'event_audit.csv',s.event_rows); save_csv(OUT/'revision_log.csv',s.revision_rows)
    sl=slice(EVAL_START,365); comp=settlement_components(s.B[sl],s.A[sl],data.price_plan[None,:],s.down_settlement)
    return {
      'name':s.name,'use_new_vintage':s.use_new_vintage,'allow_revision':s.allow_revision,'scenario_count':s.scenario_count,'down_settlement':s.down_settlement,
      'formal_days':334,'total_cost_yuan':float(np.sum(s.natural_regular_fee[sl]+s.emergency_fee[sl])),
      'regular_cost_yuan':float(np.sum(s.natural_regular_fee[sl])),'emergency_cost_yuan':float(np.sum(s.emergency_fee[sl])),
      'planned_B_kwh':float(np.sum(s.B[sl])),'final_A_kwh':float(np.sum(s.A[sl])),'down_adjust_kwh':float(np.sum(comp['down_kwh'])),'up_adjust_kwh':float(np.sum(comp['up_kwh'])),
      'charge_kwh':float(np.sum(s.charge[sl])),'discharge_kwh':float(np.sum(s.discharge[sl])),'emergency_kwh':float(np.sum(s.emergency[sl])),'curtail_kwh':float(np.sum(s.curtail[sl])),
      'soc_min_kwh':float(np.min(s.soc_path[EVAL_START:])),'soc_max_kwh':float(np.max(s.soc_path[EVAL_START:])),
      'max_soc_continuity_error_kwh':float(np.max(np.abs(s.soc24[:-1]-s.soc00[1:]))),
      'event_count':len(s.event_rows),'revision_count':len(s.revision_rows),
      'solver_max_eq_residual':float(max(r['max_eq_residual'] for r in s.event_rows)),'solver_max_ub_violation':float(max(r['max_ub_violation'] for r in s.event_rows)),
      'max_tree_nodes':int(max(r['node_count'] for r in s.event_rows)),'max_scenarios':int(max(r['scenario_count'] for r in s.event_rows)),
    }

variants=[('A',False,False,3),('B',True,False,3),('C',False,True,3),('D',True,True,3),('D_det',True,True,1)]
metrics={}; t_all=time.time()
for name,new,flex,k in variants:
    t=time.time(); print(f'RUN {name} start',flush=True)
    s=simulate_strategy(data,name=name,use_new_vintage=new,allow_revision=flex,scenario_count=k,end_day=365,verbose=True)
    m=save_sim(s); m['runtime_seconds']=time.time()-t; metrics[name]=m
    with open(OUT/'metrics_partial.json','w',encoding='utf-8') as f: json.dump(metrics,f,ensure_ascii=False,indent=2)
    print(f'RUN {name} done cost={m["total_cost_yuan"]:.3f} emg={m["emergency_kwh"]:.3f} sec={m["runtime_seconds"]:.1f}',flush=True)
JA,JB,JC,JD=[metrics[x]['total_cost_yuan'] for x in ('A','B','C','D')]
metrics['factorial']={'delta_info_cost_B_minus_A':JB-JA,'delta_flex_cost_C_minus_A':JC-JA,'delta_interaction_cost':JD-JB-JC+JA,'info_saving_A_minus_B':JA-JB,'flex_saving_A_minus_C':JA-JC,'joint_saving_A_minus_D':JA-JD,'interaction_saving':-(JD-JB-JC+JA)}
metrics['scenario_value']={'D_minus_D_det_cost':JD-metrics['D_det']['total_cost_yuan'],'D_det_minus_D_saving':metrics['D_det']['total_cost_yuan']-JD}
metrics['total_runtime_seconds']=time.time()-t_all
with open(OUT/'metrics.json','w',encoding='utf-8') as f:json.dump(metrics,f,ensure_ascii=False,indent=2)
print('ALL DONE',json.dumps(metrics['factorial'],ensure_ascii=False),flush=True)
