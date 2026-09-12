# -*- coding: utf-8 -*-
from pathlib import Path
import json
HERE=Path(__file__).resolve().parent
m=json.loads((HERE/'metrics.json').read_text(encoding='utf-8'))
s=json.loads((HERE/'semantic_robustness.json').read_text(encoding='utf-8'))
d=json.loads((HERE/'marginal_deadzone_summary.json').read_text(encoding='utf-8'))
e=json.loads((HERE/'extended_experiments.json').read_text(encoding='utf-8'))
v=json.loads((HERE/'validation.json').read_text(encoding='utf-8'))
out={
  'main_semantics':{
    'down_settlement':'cancel_settlement','revision_anchor':'original_anchor','price_basis':'delivery_time',
    'no_export':True,'eta_rt':0.9,'eta_c_eta_d':'sqrt(0.9)'
  },
  'formal_period':{'start':'2025-02-01','end':'2025-12-31','days':334,'warmup':'2025-01-01..2025-01-31','soc_reset_on_feb1':False},
  'protocol':{
    'no_formal_retuning':True,
    'main_scenario_count':9,
    'terminal_value':0.8,
    'load_forecast':'unchanged Q2 causal ensemble',
    'dispatch':'unchanged Q2 current-measurement feedback with scenario SOC 25 percent reserve',
    'measurement_model':'ideal_piecewise_constant_current_slot; not actual zero-delay sensor validation',
    'contract_information':'completed history and released forecasts only',
    'physical_schema_version':2,
    'curtail_semantics':'PV-only; unused paid contract is separate',
    'event_storage_recoursing':'Q2 48-hour valuation recourse; future purchases never issued; not globally nonanticipative optimal control',
    'risk_lambda':0.02,'cvar_alpha':0.8,'history_days':42,'floor_quantile':0.25,
    'revision_context':'structural correction after old results observed; no revised formal-cost parameter search'
  },
  'factorial':{k:m[k] for k in ('A','B','C','D')},
  'factorial_effects':m['factorial_effects'],
  'scenario_comparison':m['scenario_comparison'],
  'extended_experiments':e,
  'semantic_robustness':s,
  'marginal_deadzone':{'epsilon_kwh':d['epsilon_kwh'],'rows':d['rows'],'direction_match_count':d['direction_match_count']},
  'validation_summary':{'all_pass':v['all_pass'],'pass_count':v['pass_count'],'check_count':v['check_count']},
  'main_D':m['D'],
  'cross_problem':json.loads((HERE/'cross_problem_evidence'/'cross_problem_validation.json').read_text(encoding='utf-8')),
  'physical_fix':json.loads((HERE/'physical_fix_evidence'/'physical_matrix_validation.json').read_text(encoding='utf-8')),
}
(HERE/'metrics_final.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
# metrics_partial is a tracked progress artifact; keep it synchronized rather than leaving stale aborted-run values.
(HERE/'metrics_partial.json').write_text(json.dumps({k:m[k] for k in ('A','B','C','D','D_point')},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'D_total':m['D']['total_cost_yuan'],'A_minus_D':m['factorial_effects']['joint_saving_A_minus_D'],'marginal_match':f"{d['direction_match_count']}/{d['rows']}",'validation':f"{v['pass_count']}/{v['check_count']}",'no12_delta':e['full_year_realized_policy_sensitivity']['D_no12']['delta_vs_D_yuan']},ensure_ascii=False,indent=2))
