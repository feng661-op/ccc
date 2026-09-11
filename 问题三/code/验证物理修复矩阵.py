"""Independent whole-year physical audit for every repaired Q3 replay."""
from pathlib import Path
import json
import hashlib
import numpy as np
from q3_data import load_q3_inputs, EVAL_START, ETA_C, ETA_D, SOC_MIN, SOC_MAX, XMAX

HERE=Path(__file__).resolve().parent
DATA=load_q3_inputs(HERE.parent.parent)
NAMES=['A','B','C','D','D_point','D_no6','D_no12','D_no18','D_K5','D_sunk','D_stepwise']
rows=[]
for name in NAMES:
    p=HERE/f'run_{name}.npz'; z=np.load(p)
    if int(z['physical_schema_version'])!=2:
        raise AssertionError(f'{name}: stale schema')
    A=z['A']; q=np.zeros_like(A);q[:,1:]=A[:,:143];q[1:,0]=A[:-1,143]
    sel=slice(EVAL_START,None)
    I,u,cur,e,x,y,w,dump=[z[k][sel] for k in ['grid_import','unused_contract','curtail','emergency','charge','discharge','supply_surplus','battery_dump']]
    pv=DATA.pv_cal_kwh[sel]; n=DATA.net_cal_kwh[sel]; s=z['soc_path'][sel]
    residual=I+pv-cur+y+e-DATA.load_cal_kwh[sel]-x
    checks={
        'all_finite':all(np.all(np.isfinite(v)) for v in [I,u,cur,e,x,y,w,dump,s]),
        'positive_flows':min(float(v.min()) for v in [I,u,cur,e,x,y,w])>=-1e-6,
        'physical_balance':float(np.max(np.abs(residual)))<1e-6,
        'contract_split':float(np.max(np.abs(I+u-q[sel])))<1e-6,
        'import_bound':float(np.max(I-q[sel]))<1e-6,
        'pv_bound':float(np.max(cur-pv))<1e-6,
        'no_night_curtail':float(np.max(cur[pv<1e-7],initial=0))<1e-6,
        'no_battery_dump':float(np.max(np.abs(dump)))<1e-6,
        'no_wasted_discharge':not bool(np.any((y>1e-6)&(w>1e-6))),
        'no_emergency_charge':not bool(np.any((x>1e-6)&(e>1e-6))),
        'soc_conservation':float(np.max(np.abs(s[:,1:]-s[:,:-1]-ETA_C*x+y/ETA_D)))<1e-6,
        'soc_bounds':float(s.min())>=SOC_MIN-1e-6 and float(s.max())<=SOC_MAX+1e-6,
        'soc_continuity':float(np.max(np.abs(z['soc24'][:-1]-z['soc00'][1:])))<1e-6,
        'power_bound':max(float(x.max()),float(y.max()))<=XMAX+1e-6,
        'one_way':float(np.max(np.minimum(x,y)))<1e-6,
        'execution_mode':str(z['execution_mode'])=='measured_current_mpc',
    }
    rows.append({'name':name,'all_pass':all(checks.values()),'checks':checks,
        'slots':int(e.size),'total_cost_yuan':float((z['natural_regular_fee'][sel]+z['emergency_fee'][sel]).sum()),
        'emergency_kwh':float(e.sum()),'pv_curtail_kwh':float(cur.sum()),
        'actual_import_kwh':float(I.sum()),'unused_contract_kwh':float(u.sum()),
        'supply_surplus_kwh':float(w.sum()),'battery_dump_kwh':float(dump.sum()),
        'max_physical_residual_kwh':float(np.max(np.abs(residual))),
        'run_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
legacy=json.loads((HERE/'physical_fix_evidence'/'legacy_physical_audit.json').read_text(encoding='utf-8'))
out={'all_pass':all(r['all_pass'] for r in rows),'strategy_count':len(rows),
     'check_count':sum(len(r['checks']) for r in rows),'pass_count':sum(sum(r['checks'].values()) for r in rows),
     'current_measurement_assumption':'ideal_piecewise_constant_current_slot; not an observed zero-delay sensor claim',
     'contract_scope':'past completed observations and released forecasts only',
     'current_measurement_scope':'lower-level controller only; first horizon entry',
     'legacy':legacy,'strategies':rows}
(HERE/'physical_fix_evidence'/'physical_matrix_validation.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
if not out['all_pass']:raise SystemExit(2)
