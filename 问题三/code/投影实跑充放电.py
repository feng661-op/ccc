# -*- coding: utf-8 -*-
from pathlib import Path
import sys,json,csv
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from q3_data import *
ROOT=HERE.parent.parent;data=load_q3_inputs(ROOT);E=EVAL_START
names=['A','B','C','D','D_point','D_no6','D_no12','D_no18','D_K5','D_sunk','D_stepwise']
if (HERE/'run_D.npz').exists() and 'physical_schema_version' in np.load(HERE/'run_D.npz').files:
    raise SystemExit('Schema-2 replay already uses one-way actions and split physical flows; legacy projection refused to overwrite it.')

def project(x,y):
    delta=ETA_C*x-y/ETA_D
    xp=np.where(delta>=-1e-10,np.maximum(0,delta/ETA_C),0.0)
    yp=np.where(delta<-1e-10,np.maximum(0,-delta*ETA_D),0.0)
    return xp,yp,delta

def recompute(z):
    B=z['B'];A=z['A'];x0=z['charge'];y0=z['discharge'];xp,yp,delta=project(x0,y0)
    soc_res=float(np.max(np.abs((ETA_C*xp-yp/ETA_D)-delta)))
    if soc_res>1e-7:raise RuntimeError(f'SOC increment changed {soc_res}')
    em=np.zeros_like(xp);cur=np.zeros_like(xp)
    for d in range(A.shape[0]):
        for i in range(144):
            actual=float(data.net_cal_kwh[d,i]) if d<365 else np.nan
            if not np.isfinite(actual):continue
            q=float(A[d-1,143]) if i==0 and d>0 else (float(A[d,i-1]) if i>0 else 0.0)
            im=actual+float(xp[d,i])-q-float(yp[d,i]);em[d,i]=max(0,im);cur[d,i]=max(0,-im)
    ef=np.sum(em*5*data.price_calendar[None,:],axis=1)
    return xp,yp,em,cur,ef,soc_res

for name in names:
    p=HERE/f'run_{name}.npz'
    if not p.exists():continue
    z=np.load(p);d={k:z[k] for k in z.files};xp,yp,em,cur,ef,res=recompute(z);d.update(charge=xp,discharge=yp,emergency=em,curtail=cur,emergency_fee=ef)
    np.savez_compressed(p,**d)
    mp=HERE/f'metric_{name}.json'
    if mp.exists():
        m=json.loads(mp.read_text(encoding='utf-8'));sl=slice(E,365)
        m['charge_kwh']=float(np.sum(xp[sl]));m['discharge_kwh']=float(np.sum(yp[sl]));m['emergency_kwh']=float(np.sum(em[sl]));m['curtail_kwh']=float(np.sum(cur[sl]));m['emergency_cost_yuan']=float(np.sum(ef[sl]));m['total_cost_yuan']=float(m['regular_cost_yuan']+m['emergency_cost_yuan']);m['projection_soc_increment_max_error']=res;m['max_simultaneous_charge_discharge_kwh']=float(np.max(np.minimum(xp,yp)))
        mp.write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
    print(name,'projected','socerr',res,'simult',float(np.max(np.minimum(xp,yp))),'emg formal',float(np.sum(em[E:])))
# Update D dispatch audit to match projected execution.
dp=HERE/'dispatch_audit.csv'
if dp.exists():
    rows=list(csv.DictReader(open(dp,encoding='utf-8-sig')))
    for r in rows:
        xr=float(r['charge_kwh']);yr=float(r['discharge_kwh']);xx,yy,_=project(np.asarray([xr]),np.asarray([yr]));x=float(xx[0]);y=float(yy[0]);q=float(r['contract_kwh']);act=float(r['actual_net_kwh']);pred=float(r.get('forecast_net_kwh') or act)
        r['charge_kwh']=x;r['discharge_kwh']=y;r['predicted_emergency_kwh']=max(0,pred+x-q-y);im=act+x-q-y;r['emergency_kwh']=max(0,im);r['curtail_kwh']=max(0,-im)
    with open(dp,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
    print('dispatch audit updated',len(rows))
