# -*- coding: utf-8 -*-
from pathlib import Path
from datetime import datetime
from dataclasses import replace
import sys,csv,json
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import q3_opt
from q3_data import *
ROOT=HERE.parent.parent;data=load_q3_inputs(ROOT);z=np.load(HERE/'run_D.npz');B=z['B'];AS=z['A_stage'];sp=z['soc_path'];A=z['A']
idx={d.date():i for i,d in enumerate(data.dates)}; orig=q3_opt.build_scenarios

def event_args(d,eh):
    if eh==12:return float(sp[d,72]),AS[d,1]
    if eh==18:return float(sp[d,108]),AS[d,2]
    raise ValueError(eh)

def shifted_objective(d,eh,target_t,delta):
    def wrapper(data0,day_idx,event_hour,horizon=None,use_new_vintage=True,max_scenarios=3,history_window=56,disabled_vintage_hours=()):
        b=orig(data0,day_idx,event_hour,horizon,use_new_vintage,max_scenarios,history_window,disabled_vintage_hours=disabled_vintage_hours)
        sc=b.scenarios.copy();pt=b.point_net.copy();sc[:,target_t]+=delta;pt[target_t]+=delta
        return replace(b,point_net=pt,scenarios=sc)
    q3_opt.build_scenarios=wrapper
    try:
        soc,ab=event_args(d,eh);lead=0 if d==0 else float(A[d-1,143])
        sol=q3_opt.solve_event_lp(data,d,eh,soc,B[d],ab,lead_contract=lead,use_new_vintage=True,allow_revision=True,scenario_count=3,revision_anchor='original_anchor')
        return float(sol.objective),sol.current_contract
    finally:q3_opt.build_scenarios=orig

rows=[];eps=1.0
for ds in ('2025-03-20','2025-06-21','2025-09-23','2025-12-21'):
    d=idx[datetime.fromisoformat(ds).date()]
    for eh,stage_idx,(jlo,jhi) in ((12,2,(71,107)),(18,3,(107,144))):
        # Choose the most actively revised not-yet-delivered slot so the audit
        # illustrates an actual contract response instead of an arbitrary zero-PV slot.
        gap=np.abs(AS[d,stage_idx,jlo:jhi]-B[d,jlo:jhi]); j=int(jlo+np.argmax(gap)); target_t=int(j-jlo)
        jm,cm=shifted_objective(d,eh,target_t,-eps);jp,cp=shifted_objective(d,eh,target_t,eps);mu=(jp-jm)/(2*eps);p=float(data.price_plan[j]);lo=.5*p;hi=1.5*p
        accepted=float((cm[j]+cp[j])/2);b=float(B[d,j]);delta=accepted-b
        actual='down' if delta<-1e-5 else ('up' if delta>1e-5 else 'deadzone')
        ktol=1e-4
        if abs(mu-lo)<=ktol: predicted='down_boundary'
        elif abs(mu-hi)<=ktol: predicted='up_boundary'
        elif mu<lo: predicted='down'
        elif mu>hi: predicted='up'
        else: predicted='deadzone'
        match=(actual=='down' and predicted in ('down','down_boundary')) or (actual=='up' and predicted in ('up','up_boundary')) or (actual=='deadzone' and predicted=='deadzone')
        rows.append({'date':ds,'event_hour':eh,'target_offset_10min':target_t,'slot_j':j,'slot_label':data.plan_headers[j],'price':p,'mu_fd':mu,'deadzone_low_0.5p':lo,'deadzone_high_1.5p':hi,'B_kwh':b,'A_event_kwh':accepted,'A_minus_B_kwh':delta,'predicted_direction':predicted,'actual_direction':actual,'direction_match':match,'J_minus':jm,'J_plus':jp})
with open(HERE/'marginal_value_audit.csv','w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
summary={'epsilon_kwh':eps,'rows':len(rows),'direction_match_count':sum(r['direction_match'] for r in rows),'within_deadzone_count':sum(r['predicted_direction']=='deadzone' for r in rows),'rows_data':rows}
(HERE/'marginal_deadzone_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(summary,ensure_ascii=False,indent=2))
