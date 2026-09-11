# -*- coding: utf-8 -*-
"""问题2鲁棒性/消融独立验收：保护冻结主结果并核对新增证据。"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
Q2 = HERE.parent
ROB = Q2 / 'output' / 'robustness'
EXPECTED_HASHES = {
    Q2 / 'result2.xlsx': '0d7f3b3cf6f3159b6a69b4c30d39ff517fe0f0dd2ea276eefdf8209e9b94e884',
    HERE / 'metrics.json': 'b2d78713b420db25557a250989646a7140c2102ca894e746810ffa7831c1cf91',
    HERE / 'run_data.npz': '209dd342ab5963250cda6875cd20403281c71cbde5986bc222f6097835ea5d17',
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_csv(name: str):
    with (ROB / name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def main() -> None:
    frozen = {str(p.relative_to(Q2)): sha256(p) == expected for p, expected in EXPECTED_HASHES.items()}
    pred = read_csv('prediction_baselines.csv')
    kvals = read_csv('k_sensitivity.csv')
    floors = read_csv('soc_reserve_ablation.csv')
    vt = read_csv('terminal_value_sensitivity.csv')
    windows = read_csv('window_sensitivity.csv')
    budgets = read_csv('risk_budget_reuse.csv')
    payload = json.loads((ROB / 'robustness_results.json').read_text(encoding='utf-8'))

    cvar = float(payload['exact_cvar_example_k9_alpha08']['self_test'])
    expected_cvar = (9.0 + 0.8 * 8.0) / 1.8
    main_match = payload['main_reproduction_check']

    checks = {
        'frozen_main_files_unchanged': all(frozen.values()),
        'prediction_3models_x_3targets': len(pred) == 9 and {r['model'] for r in pred} == {'persistence','seasonal_naive_7d','causal_ensemble'},
        'k_sensitivity_5_7_9_12_15': [int(float(r['k_target'])) for r in kvals] == [5,7,9,12,15],
        'soc_reserve_4_policies': {r['floor_policy'] for r in floors} == {'hard_only','Q0.10','Q0.25','Q0.50'},
        'terminal_values_0_04_08_12': [round(float(r['terminal_value']),1) for r in vt] == [0.0,0.4,0.8,1.2],
        'window_28_42_56': [int(float(r['window_days'])) for r in windows] == [28,42,56],
        'risk_budget_reuse_3_levels': len(budgets) == 3,
        'exact_fractional_cvar': abs(cvar - expected_cvar) < 1e-12,
        'main_reproduction_total_cost': float(main_match['total_cost_abs_diff_yuan']) < 1e-4,
        'main_reproduction_emergency': float(main_match['emergency_abs_diff_kwh']) < 1e-4,
        'paper_markdown_exists': (Q2 / 'output' / '09_鲁棒性与消融验证.md').exists(),
    }

    for image_name in ['图13_鲁棒性与消融验证.png','表6_预测基线对比.png']:
        p = ROB / image_name
        ok = p.exists() and p.stat().st_size > 30000
        if ok:
            with Image.open(p) as im:
                im.verify()
        checks[f'image_{image_name}'] = ok

    report = {
        'status': 'PASS' if all(checks.values()) else 'FAIL',
        'checks': checks,
        'frozen_hashes': frozen,
        'diagnostics': {
            'exact_cvar_test': cvar,
            'expected_cvar_test': expected_cvar,
            'main_reproduction_check': main_match,
        },
    }
    (ROB / 'validation_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
