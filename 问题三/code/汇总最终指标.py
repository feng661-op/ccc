# -*- coding: utf-8 -*-
from pathlib import Path
import json
HERE=Path(__file__).resolve().parent
m=json.loads((HERE/'metrics.json').read_text(encoding='utf-8'))
s=json.loads((HERE/'semantic_robustness.json').read_text(encoding='utf-8'))
d=json.loads((HERE/'marginal_deadzone_summary.json').read_text(encoding='utf-8'))
out={
  'main_semantics':{'down_settlement':'cancel_settlement','revision_anchor':'original_anchor','price_basis':'delivery_time'},
  'formal_period':{'start':'2025-02-01','end':'2025-12-31','days':334,'warmup':'2025-01-01..2025-01-31','soc_reset_on_feb1':False},
  'factorial':{k:m[k] for k in ('A','B','C','D')},
  'factorial_effects':m['factorial'],
  'scenario_value':m['scenario_value'],
  'semantic_robustness':s,
  'marginal_deadzone':{'epsilon_kwh':d['epsilon_kwh'],'rows':d['rows'],'direction_match_count':d['direction_match_count']},
  'main_D':m['D'],
}
(HERE/'metrics_final.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'D_total':m['D']['total_cost_yuan'],'A_minus_D':m['factorial']['joint_saving_A_minus_D'],'marginal_match':f"{d['direction_match_count']}/{d['rows']}",'sunk':s['sunk_plan_plus_penalty']['total_cost_yuan'],'stepwise':s['stepwise_revision']['total_cost_yuan'],'revision_time':s['revision_time']['total_cost_yuan']},ensure_ascii=False,indent=2))
