# -*- coding: utf-8 -*-
"""Regenerate the matched Q4-2 no-price-adaptation benchmark with paired inference.

This is a frozen-model sensitivity only: it never feeds Feb-Dec information back
into model selection.  The only policy change is decision_fixed_price=True;
realized settlement remains Attachment-4 variable price.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

from q4_ablations import CODE, ADIR, frozen_cfg, runformal, compact, main_result


def run() -> None:
    c0m, c0r = runformal(frozen_cfg('q4_2', decision_fixed_price=True,
                                    label='E0_matched_no_price_adaptation'))
    main2 = compact(main_result('q4_2'))
    c0 = compact(c0m)
    c0r.days.to_csv(ADIR/'E0_q4_2_matched_no_price_adaptation_daily.csv',
                    index=False, encoding='utf-8-sig')

    formal_daily = pd.read_csv(CODE/'q4_2'/'daily_ledger.csv').sort_values('date').reset_index(drop=True)
    c0_daily = c0r.days.sort_values('date').reset_index(drop=True)
    if list(formal_daily.date) != list(c0_daily.date):
        raise AssertionError('E0 paired-date mismatch')
    diff = formal_daily.cash_fee_yuan.to_numpy(float) - c0_daily.cash_fee_yuan.to_numpy(float)

    # Deterministic moving-block bootstrap: a 7-day block retains short serial
    # dependence in daily costs. Positive diff means adaptive is more expensive.
    rng = np.random.default_rng(2026)
    n, block_days, reps = len(diff), 7, 5000
    starts = np.arange(n-block_days+1)
    boots = np.empty(reps)
    for bi in range(reps):
        idx = []
        while len(idx) < n:
            s = int(rng.choice(starts))
            idx.extend(range(s, s+block_days))
        boots[bi] = float(np.mean(diff[np.asarray(idx[:n], int)]))
    ci = np.quantile(boots, [.025, .975])
    if ci[0] > 0:
        conclusion = 'price-adaptive policy is slightly but robustly more expensive than matched C0 over the formal period; do not claim Q4-2 savings from price adaptation'
    elif ci[1] < 0:
        conclusion = 'price-adaptive policy is robustly cheaper than matched C0 over the formal period'
    else:
        conclusion = 'paired block-bootstrap interval crosses zero; the two causal policies are not robustly distinguishable in formal-period cash'

    e0 = {
        'artifact': 'e0_fixed_price_bridge.json',
        'role': 'diagnostic fixed-vs-variable tariff bridge plus matched causal policy benchmark',
        'matched_no_price_adaptation': c0,
        'formal_price_adaptive': main2,
        'adaptive_cash_gain_pct': 100.0*(c0['cash_total_yuan']-main2['cash_total_yuan'])/c0['cash_total_yuan'],
        'paired_daily_cash_difference_adaptive_minus_C0': {
            'mean_yuan_per_day': float(np.mean(diff)),
            'median_yuan_per_day': float(np.median(diff)),
            'weekly_block_bootstrap_95pct_CI_yuan_per_day': [float(ci[0]), float(ci[1])],
            'bootstrap_seed': 2026,
            'block_days': block_days,
            'replicates': reps,
            'interpretation': 'interpret the paired weekly-block bootstrap interval by sign: entirely positive means adaptive is more expensive; entirely negative means adaptive is cheaper; crossing zero means no robust cash dominance',
            'conclusion': conclusion,
        },
        'claim_boundary': 'matched policy benchmark; unlike R0 it shares the Q4 state, physics and information process and changes only the decision tariff',
    }
    p = ADIR/'E0_matched_policy_benchmark.json'
    p.write_text(json.dumps(e0, ensure_ascii=False, indent=2), encoding='utf-8')

    matrix_path = ADIR/'ablation_matrix_E0_E15.json'
    if matrix_path.exists():
        matrix = json.loads(matrix_path.read_text(encoding='utf-8'))
        matrix.setdefault('experiments', {})['E0'] = e0
        matrix_path.write_text(json.dumps(matrix, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(e0, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
