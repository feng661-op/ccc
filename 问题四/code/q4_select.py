# -*- coding: utf-8 -*-
"""January-only V1.2 model-selection ladder: B0 -> B1 -> B2.

This script is intentionally unable to touch Feb-Dec data.  Every candidate
starts from the same Jan-09 causal B0 warm-up state within its branch.  It saves
all January evidence required to freeze the formal model before any formal
replay is allowed to run.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
import pandas as pd

from q4_data import load_q4_inputs
from q4_price import PriceForecaster, PriceConfig, structure_audit
from q4_scenarios import calibrate_epsilon_january, validate_transport_duality, frozen_distance_scales
from q4_sim import SimConfig, SimState, simulate_range, replay_summary
from q4_metrics import day_metrics, paired_block_bootstrap, relative_improvement
from q4_control import isomorphism_audit
from q4_settlement import hand_check
from q4_tree import validate_tree

ROOT=Path(__file__).resolve().parents[2]
CODE=Path(__file__).resolve().parent
OUT=ROOT/'问题四'/'output'/'02_model_selection'
OUT.mkdir(parents=True,exist_ok=True)
SEL_START=8   # Jan-09
SEL_END=31    # end-exclusive Feb-01


def _j(obj):
    if isinstance(obj,(np.floating,)): return float(obj)
    if isinstance(obj,(np.integer,)): return int(obj)
    if isinstance(obj,np.ndarray): return obj.tolist()
    raise TypeError(type(obj))


def _clone(s:SimState)->SimState:
    return SimState(float(s.soc),np.asarray(s.prev_B,float).copy(),np.asarray(s.prev_A,float).copy())


def _save_candidate(branch:str,name:str,res):
    p=OUT/branch/name; p.mkdir(parents=True,exist_ok=True)
    res.days.to_csv(p/'days.csv',index=False,encoding='utf-8-sig')
    res.events.to_csv(p/'events.csv',index=False,encoding='utf-8-sig')
    # Contract/slot evidence is useful for B0/B1/B2 selection reproducibility.
    res.contracts.to_csv(p/'contracts.csv',index=False,encoding='utf-8-sig')
    res.slots.to_csv(p/'physical_10min.csv',index=False,encoding='utf-8-sig')
    summ={'replay':replay_summary(res),'metrics':day_metrics(res.days),'config':asdict(res.config),
          'state_end':{'soc':float(res.state_end.soc),'prev_B':res.state_end.prev_B.tolist(),'prev_A':res.state_end.prev_A.tolist()}}
    (p/'summary.json').write_text(json.dumps(summ,ensure_ascii=False,indent=2,default=_j),encoding='utf-8')
    return summ


def _candidate_row(name,res,base=None):
    m=day_metrics(res.days)
    row={'name':name,**m,'risk_lambda':res.config.risk_lambda,'alpha':res.config.alpha,'epsilon':res.config.epsilon,'level':res.config.level}
    if base is not None:
        bm=day_metrics(base.days)
        row['mean_cash_improvement_pct']=100*relative_improvement(bm['cash_mean_yuan_per_day'],m['cash_mean_yuan_per_day'])
        row['cvar95_improvement_pct']=100*relative_improvement(bm['cash_cvar95_yuan_per_day'],m['cash_cvar95_yuan_per_day'])
        row['paired']=paired_block_bootstrap(base.days,res.days,block_days=7,reps=2000)
    return row


def run():
    data=load_q4_inputs(ROOT); pf=PriceForecaster(data)
    # E1 price evidence and structural evidence are both January-only.
    price_rows,price_metrics,price_cfg,price_decision=pf.january_walkforward()
    # The same fair January predictive winner must feed every downstream
    # January decision candidate; no model class receives a manual override.
    PC=price_cfg
    price_rows.to_csv(CODE/'price_forecast_walkforward.csv',index=False,encoding='utf-8-sig')
    price_metrics.to_csv(OUT/'price_model_ablation.csv',index=False,encoding='utf-8-sig')
    (CODE/'price_structure_audit.json').write_text(json.dumps(structure_audit(data),ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'price_freeze_decision.json').write_text(json.dumps(price_decision,ensure_ascii=False,indent=2),encoding='utf-8')

    eps={b:calibrate_epsilon_january(data,pf,PC,b) for b in ('q4_2','q4_3')}
    (CODE/'epsilon_calibration.json').write_text(json.dumps(eps,ensure_ascii=False,indent=2,default=_j),encoding='utf-8')
    static_checks={'transport_duality':validate_transport_duality(),'isomorphism':isomorphism_audit(),
                   'settlement_hand_check':hand_check(),'tree':validate_tree()}
    (OUT/'preselection_static_checks.json').write_text(json.dumps(static_checks,ensure_ascii=False,indent=2,default=_j),encoding='utf-8')

    ladder={'selection_period':'2025-01-09..2025-01-31','formal_period_examined':False,
            'price_config':asdict(PC),'price_decision':price_decision,'branches':{}}
    frozen={}
    for branch in ('q4_2','q4_3'):
        # Common B0 warm-up, reused exactly by every candidate in the branch.
        warm=simulate_range(data,pf,SimConfig(branch=branch,level='B0',price_config=PC,label='common_B0_warmup'),0,SEL_START)
        warm_state=_clone(warm.state_end)
        bp=OUT/branch; bp.mkdir(parents=True,exist_ok=True)
        warm.days.to_csv(bp/'common_warmup_days.csv',index=False,encoding='utf-8-sig')
        (bp/'common_warmup_state.json').write_text(json.dumps({'soc':warm_state.soc,'prev_B':warm_state.prev_B.tolist(),'prev_A':warm_state.prev_A.tolist()},indent=2),encoding='utf-8')

        b0=simulate_range(data,pf,SimConfig(branch=branch,level='B0',price_config=PC,label='B0'),SEL_START,SEL_END,_clone(warm_state))
        _save_candidate(branch,'B0',b0)
        b1=simulate_range(data,pf,SimConfig(branch=branch,level='B1',price_config=PC,scenario_k=3,risk_lambda=0.0,epsilon=0.0,label='B1'),SEL_START,SEL_END,_clone(warm_state))
        _save_candidate(branch,'B1',b1)
        row0=_candidate_row('B0',b0); row1=_candidate_row('B1',b1,b0)
        b1mean=row1['mean_cash_improvement_pct']; b1tail=row1['cvar95_improvement_pct']; pb1=row1['paired']
        mean_sig=pb1['mean_cash_gain_80pct_interval'][0]>0
        tail_sig=pb1['cvar95_gain_80pct_interval'][0]>0
        b1_pass=bool((b1mean>=1.0 and mean_sig) or (b1tail>=2.0 and tail_sig))

        q50=float(eps[branch]['q50']); q75=float(eps[branch]['q75'])
        b2rows=[]; b2results={}
        # epsilon=0 degeneracy diagnostic is explicitly separate from candidate ranking.
        b2zero=simulate_range(data,pf,SimConfig(branch=branch,level='B2',price_config=PC,scenario_k=3,risk_lambda=0.0,alpha=.8,epsilon=0.0,label='B2_eps0'),SEL_START,SEL_END,_clone(warm_state))
        _save_candidate(branch,'B2_eps0_diagnostic',b2zero)
        eps0={'cash_abs_diff_B1':abs(float(b2zero.days.cash_fee_yuan.sum()-b1.days.cash_fee_yuan.sum())),
              'day_cash_max_abs_diff':float(np.max(np.abs(b2zero.days.cash_fee_yuan.to_numpy()-b1.days.cash_fee_yuan.to_numpy())))}
        for alpha in (.8,.9):
            for lam in (.05,.1):
                for ename,ev in (('q50',q50),('q75',q75)):
                    name=f'B2_a{alpha:.1f}_l{lam:.2f}_{ename}'
                    cfg=SimConfig(branch=branch,level='B2',price_config=PC,scenario_k=3,alpha=alpha,risk_lambda=lam,epsilon=ev,label=name)
                    rr=simulate_range(data,pf,cfg,SEL_START,SEL_END,_clone(warm_state))
                    b2results[name]=rr; _save_candidate(branch,name,rr)
                    b2rows.append(_candidate_row(name,rr,b1))
        # Freeze B2 candidate using Jan tail objective only, with prespecified <=1% mean-cash penalty.
        eligible=[]
        for r in b2rows:
            mean_penalty=-r['mean_cash_improvement_pct']
            tail_gain=r['cvar95_improvement_pct']
            tail_lb=r['paired']['cvar95_gain_80pct_interval'][0]
            r['mean_cash_penalty_pct']=mean_penalty
            r['eligible_tail_candidate']=bool(mean_penalty<=1.0 and tail_gain>0 and tail_lb>0)
            if r['eligible_tail_candidate']: eligible.append(r)
        best_b2=max(eligible,key=lambda r:(r['cvar95_improvement_pct'],r['mean_cash_improvement_pct'])) if eligible else None
        b2_pass=bool(best_b2 is not None and best_b2['cvar95_improvement_pct']>=2.0)

        selected_name='B0'; selected=b0; reason='B1 did not pass January joint-model practical/bootstrap gate.'
        if b1_pass:
            selected_name='B1'; selected=b1; reason='B1 passed January joint-model practical/bootstrap gate; B2 did not establish required tail gain.'
            if b2_pass:
                selected_name=best_b2['name']; selected=b2results[selected_name]; reason='B2 passed January tail-gain/bootstrap gate with <=1% mean cash penalty.'
        selected_summary=_save_candidate(branch,'SELECTED_JAN_FREEZE',selected)
        branch_record={'common_warmup_end_soc':warm_state.soc,'B0':row0,'B1':row1,'B1_pass':b1_pass,
                       'B1_gate':{'mean>=1pct_and_80CI_positive':bool(b1mean>=1 and mean_sig),'cvar95>=2pct_and_80CI_positive':bool(b1tail>=2 and tail_sig)},
                       'B2_epsilon0_degeneracy':eps0,'B2_candidates':b2rows,'B2_best':best_b2,'B2_pass':b2_pass,
                       'selected':selected_name,'selection_reason':reason,'selected_config':asdict(selected.config),
                       'selected_end_state':selected_summary['state_end']}
        ladder['branches'][branch]=branch_record
        frozen[branch]={'config':asdict(selected.config),'state_end':selected_summary['state_end'],'selected':selected_name}
        (bp/'selection_decision.json').write_text(json.dumps(branch_record,ensure_ascii=False,indent=2,default=_j),encoding='utf-8')
    (CODE/'baseline_ladder.json').write_text(json.dumps(ladder,ensure_ascii=False,indent=2,default=_j),encoding='utf-8')
    (CODE/'january_frozen_selection.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2,default=_j),encoding='utf-8')
    print(json.dumps({'selected':{b:x['selected'] for b,x in frozen.items()},
                      'q4_2_B1_pass':ladder['branches']['q4_2']['B1_pass'],'q4_2_B2_pass':ladder['branches']['q4_2']['B2_pass'],
                      'q4_3_B1_pass':ladder['branches']['q4_3']['B1_pass'],'q4_3_B2_pass':ladder['branches']['q4_3']['B2_pass']},ensure_ascii=False))

if __name__=='__main__': run()
