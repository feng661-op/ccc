# -*- coding: utf-8 -*-
from pathlib import Path
import json
HERE=Path(__file__).resolve().parent
names=['A','B','C','D','D_point']
m={n:json.loads((HERE/f'metric_{n}.json').read_text(encoding='utf-8')) for n in names}
JA,JB,JC,JD=[m[n]['total_cost_yuan'] for n in 'ABCD']
m['factorial_effects']={
 'realized_forecast_update_effect_B_minus_A':JB-JA,
 'delta_info_cost_B_minus_A':JB-JA,
 'delta_flex_cost_C_minus_A':JC-JA,
 'delta_interaction_cost':JD-JB-JC+JA,
 'info_saving_A_minus_B':JA-JB,
 'flex_saving_A_minus_C':JA-JC,
 'joint_saving_A_minus_D':JA-JD,
 'interaction_saving':-(JD-JB-JC+JA),
 'interpretation':'annual realized policy effects under a fixed predeclared protocol; not theoretical VOI',
}
m['scenario_comparison']={
 'interpretation':'realized-policy stability comparison only; K=1 is pure point forecast; not theoretical VSS/VOI',
 'D_point_total_cost_yuan':m['D_point']['total_cost_yuan'],
 'D_total_cost_yuan':JD,
 'D_point_minus_D_realized_cost':m['D_point']['total_cost_yuan']-JD,
 'D_point_emergency_kwh':m['D_point']['emergency_kwh'],
 'D_K3_emergency_kwh':m['D']['emergency_kwh'],
}
(HERE/'metrics.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(m,ensure_ascii=False,indent=2))
