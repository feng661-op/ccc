# -*- coding: utf-8 -*-
"""Assemble final E0--E15 matrix from independently produced frozen artifacts.
No formal-period output is allowed to feed back into January model selection.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import q4_ablations as qa

CODE=Path(__file__).resolve().parent
ADIR=CODE/'ablations'
ROOT=CODE.parents[1]

def loadj(name):
    p=ADIR/name
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding='utf-8'))

def main():
    out={'frozen_main':{br:qa.compact(qa.main_result(br)) for br in ('q4_2','q4_3')},'experiments':{},'formal_retuning':False}
    ex=out['experiments']
    ex['E0']=loadj('E0_matched_policy_benchmark.json')
    pm=pd.read_csv(ROOT/'问题四'/'output'/'02_model_selection'/'price_model_ablation.csv')
    ex['E1']={'status':'run','artifact':'output/02_model_selection/price_model_ablation.csv',
              'best_by_mae':pm.sort_values(['mae','rmse']).iloc[0].to_dict(),
              'frozen':qa.FREEZE['q4_2']['config']['price_config'],
              'selection_rule':'minimum January causal walk-forward MAE; RMSE/q95 tie-break; no privileged Ridge class'}
    ex['E2']=loadj('E2_joint_structure.json')
    ladder=json.loads((CODE/'baseline_ladder.json').read_text(encoding='utf-8'))
    ex['E3']={br:{'B1_pass':ladder['branches'][br]['B1_pass'],'B1':ladder['branches'][br]['B1']} for br in ('q4_2','q4_3')}
    ex['E4']={br:{'B2_pass':ladder['branches'][br]['B2_pass'],'B2_best':ladder['branches'][br]['B2_best']} for br in ('q4_2','q4_3')}
    ex['E5']={br:ladder['branches'][br]['B2_candidates'] for br in ('q4_2','q4_3')}
    ex['E6']=loadj('E6_information_permission.json')
    ex['E7']=loadj('E7_event_contribution.json')
    ex['E8']=loadj('E8_terminal_horizon.json')
    ex['E9']=loadj('E9_measurement_timing.json')
    # E10: current formal fixed-policy settlement exposure; explicitly not re-optimized.
    e10={}
    for br in ('q4_2','q4_3'):
        slots=pd.read_csv(CODE/br/'physical_10min.csv')
        B=slots.B_kwh.to_numpy(float); A=slots.A_kwh.to_numpy(float)
        c=slots.price_yuan_per_kwh.to_numpy(float); em=slots.e_kwh.to_numpy(float)
        original=float(slots.cash_fee_yuan.sum())
        sunk=float(np.sum(c*B + .5*c*np.maximum(B-A,0)+1.5*c*np.maximum(A-B,0)+5*c*em))
        e10[br]={'original_anchor_main_yuan':original,'sunk_plan_fee_exposure_yuan':sunk,'fixed_strategy_only':True,
                 'claim_boundary':'exposure only; alternative settlement requires reoptimization and is not used to rank/freeze the main model'}
    ex['E10']=e10
    (ADIR/'E10_settlement_exposure.json').write_text(json.dumps(e10,ensure_ascii=False,indent=2),encoding='utf-8')
    ex['E11']=loadj('E11_efficiency.json')
    ex['E12']=loadj('E12_emergency_charging.json')
    ex['E13']=loadj('E13_history_window.json')
    oracle=json.loads((CODE/'oracle_audit.json').read_text(encoding='utf-8'))
    ex['E14']={'status':'complete','artifact':'oracle_audit.json','claim':'Price Oracle diagnostic only; matched Full-information Oracle is the only strict cash lower bound',
               'q4_2':oracle['q4_2']['audit'],'q4_3':oracle['q4_3']['audit']}
    e15={}
    for br in ('q4_2','q4_3'):
        selected=qa.FREEZE[br]['selected']
        ev=pd.read_csv(ROOT/'问题四'/'output'/'02_model_selection'/br/selected/'events.csv')
        e15[br]={'events':int(len(ev)),'max_eq_residual':float(ev.max_eq_residual.max()),
                 'stage1_objective_min':float(ev.level1.min()),'stage1_objective_max':float(ev.level1.max()),
                 'selected_level':selected,'lex_order':'cash/risk -> throughput -> PV curtailment',
                 'note':'event solver preserves stage-1 then stage-2 tolerances before stage-3; validation T27/T29 independently verifies numerical preservation'}
    ex['E15']=e15
    (ADIR/'E15_lexicographic.json').write_text(json.dumps(e15,ensure_ascii=False,indent=2),encoding='utf-8')
    p=ADIR/'ablation_matrix_E0_E15.json'
    p.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':'merged','experiments':list(ex),'path':str(p)},ensure_ascii=False))

if __name__=='__main__': main()
