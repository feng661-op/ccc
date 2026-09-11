# -*- coding: utf-8 -*-
from pathlib import Path
import json,hashlib,math,csv
HERE=Path(__file__).resolve().parent;Q3=HERE.parent;FIG=Q3/'figures'
base=json.loads((HERE/'validation.json').read_text(encoding='utf-8'))
metrics=json.loads((HERE/'metrics_final.json').read_text(encoding='utf-8'))
sem=json.loads((HERE/'semantic_robustness.json').read_text(encoding='utf-8'))
marg=json.loads((HERE/'marginal_deadzone_summary.json').read_text(encoding='utf-8'))
checks={};evidence={}
def ck(k,v,e):checks[k]='PASS' if bool(v) else 'FAIL';evidence[k]=e
ck('base_validation_all_pass',base.get('all_pass') is True,{'pass_count':base.get('pass_count'),'check_count':base.get('check_count')})
for name in ('A','B','C','D'):
    v=metrics['factorial'][name]['total_cost_yuan'];ck(f'factorial_{name}_finite',math.isfinite(v) and v>0,v)
ck('semantic_sunk_full_run',(HERE/'run_D_sunk.npz').exists() and math.isfinite(sem['sunk_plan_plus_penalty']['total_cost_yuan']),sem['sunk_plan_plus_penalty']['total_cost_yuan'])
ck('semantic_stepwise_full_run',(HERE/'run_D_stepwise.npz').exists() and math.isfinite(sem['stepwise_revision']['total_cost_yuan']),sem['stepwise_revision']['total_cost_yuan'])
aud=sem['revision_time_exact_milp_audit'];milp_ok=len(aud)==4 and all('Optimal' in x['status'] and x['nonconvex_delivery_slots']>0 for x in aud)
ck('semantic_revision_time_nonconvex_milp',milp_ok,{'rows':len(aud),'nonconvex_slots':[x['nonconvex_delivery_slots'] for x in aud],'statuses':[x['status'] for x in aud]})
ck('marginal_deadzone_direction_consistency',marg['rows']==marg['direction_match_count'] and marg['rows']>0,{'match':marg['direction_match_count'],'rows':marg['rows']})
figs=['fig_q3_typical_day.png','fig_q3_factorial_abcd.png','fig_q3_fee_decomposition.png','fig_q3_representative_days.png','fig_q3_marginal_deadzone.png','fig_q3_semantic_robustness.png']
fig_info={f:(FIG/f).stat().st_size if (FIG/f).exists() else 0 for f in figs};ck('figures_complete',all(v>20000 for v in fig_info.values()),fig_info)
paper=Q3/'问题3论文材料.md';ck('paper_material_complete',paper.exists() and paper.stat().st_size>10000,{'bytes':paper.stat().st_size if paper.exists() else 0})
result=Q3/'result3.xlsx';ck('result3_exists',result.exists() and result.stat().st_size>100000,{'bytes':result.stat().st_size if result.exists() else 0})
# Cost equality survives every aggregation layer.
D=float(metrics['main_D']['total_cost_yuan']); led=0.0
with open(HERE/'five_ledger.csv',encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):led+=float(r['F_total'])
ck('final_cost_identity',abs(D-led)<1e-5,{'metrics_D':D,'five_ledger':led,'diff':led-D})
# Hash final user-facing artifacts for traceability.
def sha(p):
    h=hashlib.sha256();
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
art={'result3_sha256':sha(result),'paper_sha256':sha(paper),'metrics_final_sha256':sha(HERE/'metrics_final.json')}
out={'all_pass':all(v=='PASS' for v in checks.values()),'pass_count':sum(v=='PASS' for v in checks.values()),'check_count':len(checks),'checks':checks,'evidence':evidence,'artifact_hashes':art}
(HERE/'final_acceptance.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(out,ensure_ascii=False,indent=2))
if not out['all_pass']:raise SystemExit(2)
