"""Full-year Q2 -> Q3 degeneration, unified cost/flow audit, protected scope."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
from q3_data import load_q3_inputs, settlement_components

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]; OUT=HERE/'cross_problem_evidence'
data=load_q3_inputs(ROOT); ref=np.load(ROOT/'问题二'/'code'/'run_data.npz')
closed=np.load(HERE/'run_A.npz'); main=np.load(HERE/'run_D.npz'); sel=slice(31,None)
zero=np.load(HERE/'run_Q3_zero_only.npz')
checks={}; evidence={}
def ck(name,yes,detail):
    checks[name]=bool(yes); evidence[name]=detail

for key,reference in {'B':'plan_q','A':'plan_q','charge':'charge','discharge':'discharge','emergency':'emergency','soc00':'soc00','soc24':'soc24','plan_fee':'plan_cost','emergency_fee':'emergency_cost','supply_surplus':'curtailment'}.items():
    err=float(np.max(np.abs(closed[key]-ref[reference])))
    ck('all_year_'+key,err<1e-6,{'max_error':err,'entries':int(closed[key].size)})
same_cost=float((ref['plan_cost'][sel]+ref['emergency_cost'][sel]).sum())
closed_cost=float((closed['adjusted_fee'][sel]+closed['emergency_fee'][sel]).sum())
ck('published_q2_cost_reproduced',abs(same_cost-closed_cost)<1e-5,{'q2':same_cost,'closed_q3':closed_cost,'difference':closed_cost-same_cost})
bridge=float(data.price_plan[-1]*(ref['plan_q'][30,143]-ref['plan_q'][364,143]))
natural=float((closed['natural_regular_fee'][sel]+closed['emergency_fee'][sel]).sum())
ck('natural_period_bridge',abs(natural-same_cost-bridge)<1e-5,{'bridge':bridge,'natural_total':natural})
ck('all_stages_contract_unchanged',np.max(np.abs(closed['A_stage']-closed['B'][:,None,:]))<1e-7,{'max_error':float(np.max(np.abs(closed['A_stage']-closed['B'][:,None,:])))})
zero_cost=float((zero['natural_regular_fee'][sel]+zero['emergency_fee'][sel]).sum())
zero_frozen=float(np.max(np.abs(zero['A_stage']-zero['B'][:,None,:])))
zero_res=float(np.max(np.abs(zero['grid_import'][sel]+data.pv_cal_kwh[sel]-zero['curtail'][sel]+zero['discharge'][sel]+zero['emergency'][sel]-data.load_cal_kwh[sel]-zero['charge'][sel])))
zero_soc=float(np.max(np.abs(np.diff(zero['soc_path'],axis=1)-np.sqrt(.9)*zero['charge']+zero['discharge']/np.sqrt(.9))))
ck('zero_forecast_only_control',zero_frozen<1e-7 and zero_res<1e-6 and zero_soc<1e-6,
   {'total_cost':zero_cost,'delta_vs_q2_natural':zero_cost-natural,'contract_freeze_error':zero_frozen,
    'physical_residual':zero_res,'soc_residual':zero_soc,'information':'midnight official PV retained; all intraday releases and contract revision disabled'})
hashes=json.loads((OUT/'protected_hashes.json').read_text(encoding='utf-8'))
changed=[p for p,h in hashes.items() if not (ROOT/p).is_file() or hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
ck('q1_q2_q4_attachments_unchanged',not changed,{'files':len(hashes),'changed':changed})
rows=[]
for label,z in [('第二问同口径',closed),('第三问完整模型',main)]:
    B=z['B'];A=z['A'];b=np.zeros_like(B);a=np.zeros_like(A)
    b[:,1:]=B[:,:143]; b[1:,0]=B[:-1,143]
    a[:,1:]=A[:,:143]; a[1:,0]=A[:-1,143]
    c=settlement_components(b[sel],a[sel],data.price_calendar)
    row={'策略':label,'原计划电量':float(B[sel].sum()),'最终合同计划行电量':float(A[sel].sum()),
         '生效合同自然日电量':float(a[sel].sum()),'上调电量':float(c['up_kwh'].sum()),'下调电量':float(c['down_kwh'].sum()),
         '实际普通取电':float(z['grid_import'][sel].sum()),'未利用合同':float(z['unused_contract'][sel].sum()),
         '充电':float(z['charge'][sel].sum()),'放电':float(z['discharge'][sel].sum()),
         '紧急购电':float(z['emergency'][sel].sum()),'真正弃光':float(z['curtail'][sel].sum()),
         '储电量下限':float(z['soc_path'][sel].min()),'储电量上限':float(z['soc_path'][sel].max()),
         '正式期起点储电量':float(z['soc00'][31]),'正式期终点储电量':float(z['soc24'][-1]),
         '保留计划费用':float(c['F_plan'].sum()),'下调费用':float(c['F_cancel'].sum()),'上调费用':float(c['F_add'].sum()),
         '紧急费用':float(z['emergency_fee'][sel].sum()),'总费用':float((z['natural_regular_fee'][sel]+z['emergency_fee'][sel]).sum())}
    err=abs(row['总费用']-sum(row[k] for k in ('保留计划费用','下调费用','上调费用','紧急费用')))
    ck('fee_decomposition_'+label,err<1e-5,{'error':err})
    rows.append(row)
with (OUT/'q2_q3_comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
out={'all_pass':all(checks.values()),'pass_count':sum(checks.values()),'check_count':len(checks),
     'checks':checks,'evidence':evidence,'comparison':rows,'original_audit':json.loads((OUT/'before_comparison.json').read_text(encoding='utf-8')),
     'old_closed_replay':json.loads((OUT/'old_closed_replay.json').read_text(encoding='utf-8'))}
(OUT/'cross_problem_validation.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
if not out['all_pass']:raise SystemExit(2)
