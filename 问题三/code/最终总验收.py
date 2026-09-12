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
figs=['fig_q3_typical_day.png','fig_q3_factorial_abcd.png','fig_q3_fee_decomposition.png','fig_q3_representative_days.png','fig_q3_marginal_deadzone.png','fig_q3_semantic_robustness.png','fig_q3_cross_problem.png']
fig_info={f:(FIG/f).stat().st_size if (FIG/f).exists() else 0 for f in figs};ck('figures_complete',all(v>20000 for v in fig_info.values()),fig_info)
paper=Q3/'问题3论文材料.md';ck('paper_material_complete',paper.exists() and paper.stat().st_size>10000,{'bytes':paper.stat().st_size if paper.exists() else 0})
result=Q3/'result3.xlsx';ck('result3_exists',result.exists() and result.stat().st_size>100000,{'bytes':result.stat().st_size if result.exists() else 0})
# Cost equality survives every aggregation layer.
D=float(metrics['main_D']['total_cost_yuan']); led=0.0
with open(HERE/'five_ledger.csv',encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):led+=float(r['F_total'])
ck('final_cost_identity',abs(D-led)<1e-5,{'metrics_D':D,'five_ledger':led,'diff':led-D})
# Physical semantics, whole matrix, and executed targeted test suite.
phys=json.loads((HERE/'physical_fix_evidence'/'physical_matrix_validation.json').read_text(encoding='utf-8'))
ck('physical_matrix_all_11_pass',phys.get('all_pass') and phys.get('strategy_count')==11 and phys.get('pass_count')==phys.get('check_count'),{'pass':phys['pass_count'],'count':phys['check_count']})
import xml.etree.ElementTree as ET
jt=ET.parse(HERE/'physical_fix_evidence'/'targeted_pytest.xml').getroot()
suites=list(jt.iter('testsuite')); nt=sum(int(x.attrib.get('tests',0)) for x in suites);nf=sum(int(x.attrib.get('failures',0))+int(x.attrib.get('errors',0)) for x in suites)
ck('physical_targeted_regressions_pass',nt>=12 and nf==0,{'tests':nt,'failures_or_errors':nf})
cross=json.loads((HERE/'cross_problem_evidence'/'cross_problem_validation.json').read_text(encoding='utf-8'))
ck('q2_to_q3_full_year_degeneration',cross.get('all_pass') and cross.get('pass_count')==cross.get('check_count'),{'pass':cross['pass_count'],'count':cross['check_count'],'cost':cross['evidence']['published_q2_cost_reproduced']})
ck('inheritance_regressions_executed',nt>=27 and nf==0,{'tests':nt,'failures_or_errors':nf})
midnight=json.loads((HERE/'cross_problem_evidence'/'midnight_solver_validation.json').read_text(encoding='utf-8'))
ck('midnight_q2_native_solver_full_audit',len(midnight)==4 and all(r['midnight_solves']==365 and r['max_contract_reproduction_error']<1e-6 and max(r['max_eq_residual'],r['max_ub_violation'])<1e-7 for r in midnight),midnight)
protocol=metrics['protocol']
ck('same_q2_base_parameters',protocol['main_scenario_count']==9 and protocol['terminal_value']==0.8 and protocol['risk_lambda']==0.02 and protocol['cvar_alpha']==0.8 and protocol['history_days']==42 and protocol['floor_quantile']==0.25,protocol)
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
