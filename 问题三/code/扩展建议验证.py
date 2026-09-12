# -*- coding: utf-8 -*-
from pathlib import Path
from datetime import datetime,timedelta
import sys,json,time
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from q3_data import *
from q3_opt import build_scenarios
from q3_sim import simulate_strategy
ROOT=HERE.parent.parent;data=load_q3_inputs(ROOT);E=EVAL_START

def legacy_profile_kw(issue,horizon):
    arr=data.load_cal_kwh;dcur=(issue.date()-data.dates[0].date()).days;out=np.zeros(horizon)
    for k in range(horizon):
        t=issue+timedelta(minutes=10*k);si=(t.hour*60+t.minute)//10;vals=[];same=[]
        for lag in range(1,8):
            j=dcur-lag
            if 0<=j<365 and np.isfinite(arr[j,si]):vals.append(float(arr[j,si]))
        for j in range(max(0,dcur-35),dcur):
            if data.dates[j].weekday()==t.weekday() and np.isfinite(arr[j,si]):same.append(float(arr[j,si]))
        base=float(np.median(vals)) if vals else (float(np.median(same)) if same else 5000*DT)
        if same:base=.75*base+.25*float(np.median(same[-4:]))
        out[k]=max(0,base/DT)
    return out

def load_benchmark(a,b):
    out={}
    for name in ('legacy_robust_history','seasonal_naive_7d'):
        es=[]
        for d in range(a,b):
            issue=data.dates[d];pred=legacy_profile_kw(issue,144)*DT if name.startswith('legacy') else historical_profile_kw(data,issue,'load',144)*DT
            y=data.load_cal_kwh[d];mask=np.isfinite(pred)&np.isfinite(y);es.extend((pred[mask]-y[mask]).tolist())
        e=np.asarray(es,float);out[name]={'MAE_kwh':float(np.mean(np.abs(e))),'RMSE_kwh':float(np.sqrt(np.mean(e*e))),'n':int(len(e))}
    return out

metrics=json.loads((HERE/'metrics.json').read_text(encoding='utf-8'))
out={'protocol':{'selection':'Q2 causal ensemble remains the production load forecast; old 7-day method is diagnostic only; Q2 parameters inherited directly; no parameter selected from revised formal-period results','formal_period':'2025-02-01..2025-12-31','no_formal_retuning':True}}
out['load_forecast']={'historical_diagnostic_january':load_benchmark(7,31),'historical_diagnostic_formal':load_benchmark(31,365),'production':'unchanged Q2 causal ensemble'}
idx={d.date():i for i,d in enumerate(data.dates)};diag=[];point_err=[]
for ds in ('2025-03-20','2025-06-21','2025-09-23','2025-12-21'):
    d=idx[datetime.fromisoformat(ds).date()]
    for eh in (0,6,12,18):
        b1=build_scenarios(data,d,eh,use_new_vintage=True,max_scenarios=1);b3=build_scenarios(data,d,eh,use_new_vintage=True,max_scenarios=9)
        point_err.append(float(np.max(np.abs(b1.scenarios[0]-b1.point_net))))
        diag.append({'date':ds,'event_hour':eh,'weights':[float(x) for x in b3.weights],'weight_sum':float(b3.weights.sum()),'clip_rate':float(b3.clip_rate),'source_days':b3.source_days})
out['scenario_structure']={'K1_max_point_error_kwh':max(point_err),'representative_K9':diag,'max_weight_sum_error':max(abs(x['weight_sum']-1) for x in diag),'max_clip_rate':max(x['clip_rate'] for x in diag)}
runs={name:json.loads((HERE/f'metric_{name}.json').read_text(encoding='utf-8')) for name in ('D_no6','D_no12','D_no18','D_K5')}
Dcost=float(metrics['D']['total_cost_yuan'])
for m in runs.values():m['delta_vs_D_yuan']=m['total_cost_yuan']-Dcost
out['full_year_realized_policy_sensitivity']=runs
out['scenario_count']={'K1_point':metrics['D_point'],'K9_main':metrics['D'],'K5':runs['D_K5'],'interpretation':'formal-period realized-policy stability only; K=9 directly inherits Q2 and is not selected by these formal results; not theoretical VSS/VOI'}
# terminal value sensitivity is intentionally limited to January to avoid formal-period retuning.
jan={}
for tv in (.4,.8,1.2):
    print('JAN terminal',tv,flush=True);t0=time.time();s=simulate_strategy(data,name=f'Jan_vT_{tv}',use_new_vintage=True,allow_revision=True,scenario_count=9,end_day=31,verbose=False,terminal_value=tv)
    jan[str(tv)]={'jan_total_cost_yuan':float(np.sum(s.natural_regular_fee+s.emergency_fee)),'jan_emergency_kwh':float(np.sum(s.emergency)),'jan_soc_end_kwh':float(s.soc24[-1]),'terminal_binding_event_rate':float(np.mean([r['terminal_shortfall_binding_scenarios']>0 for r in s.event_rows])),'runtime_seconds':time.time()-t0}
out['terminal_value_january_only']=jan
(HERE/'extended_experiments.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2),flush=True)
