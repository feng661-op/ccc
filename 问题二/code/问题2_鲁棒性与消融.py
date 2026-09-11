# -*- coding: utf-8 -*-
"""问题2：冻结主模型后的鲁棒性/消融证据加固。

原则
----
1. 不改写 result2.xlsx、metrics.json、run_data.npz 或主参数；
2. 2—12月只用于事后鲁棒性验证，不反向调参；
3. 所有基线共享严格0:00信息边界；
4. 输出独立写入 output/robustness/，正文摘要写入 output/09_鲁棒性与消融验证.md。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
Q2 = HERE.parent
REPO = Q2.parent
OUT = Q2 / 'output'
ROB = OUT / 'robustness'
ROB.mkdir(parents=True, exist_ok=True)
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from q2_core import (  # noqa: E402
    CVAR_ALPHA, DT, ETA_C, ETA_D, REALTIME_SOC_FLOOR_QUANTILE,
    SCENARIO_QUANTILES, SCENARIO_WINDOW_DAYS, SOC0, SOC_MAX, SOC_MIN,
    TERMINAL_SHORTAGE_VALUE, XMAX,
    InputData, ForecastBundle, _masked_previous_profile, build_net_scenarios,
    empirical_cvar_equal_prob, execute_interval, load_official_inputs,
    precompute_causal_forecasts, solve_stochastic_mpc,
)

EVAL_START = 31
MAIN_LAMBDA = 0.02
NEAR_FLOOR_KWH = 1500.0
K_VALUES = (5, 7, 9, 12, 15)
FLOOR_POLICIES: Tuple[Union[str, float], ...] = ('hard', 0.10, 0.25, 0.50)
TERMINAL_VALUES = (0.0, 0.4, 0.8, 1.2)
WINDOW_VALUES = (28, 42, 56)

# K=9 必须原样复现冻结主模型；其余K才使用对称的近均匀0.05—0.95分位网格。
def quantiles_for_k(k: int) -> Tuple[float, ...]:
    if k == 9:
        return tuple(float(x) for x in SCENARIO_QUANTILES)
    return tuple(float(x) for x in np.linspace(0.05, 0.95, int(k)))


@dataclass
class VariantSummary:
    name: str
    k_target: int
    window_days: int
    terminal_value: float
    floor_policy: str
    total_cost_yuan: float
    plan_cost_yuan: float
    emergency_cost_yuan: float
    plan_kwh: float
    emergency_kwh: float
    emergency_share: float
    curtailed_kwh: float
    end_soc_kwh: float
    realtime_soc_min_kwh: float
    realtime_soc_max_kwh: float
    hard_floor_interval_count: int
    near_floor_interval_count: int
    near_floor_hours: float
    high_risk_emergency_kwh: float
    solve_time_median_s: float
    solve_time_p90_s: float
    mean_scenario_count: float


def _metrics(actual: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    a = actual[EVAL_START:]
    p = pred[EVAL_START:]
    err = p - a
    mae_e = float(np.mean(np.abs(err)))
    rmse_e = float(np.sqrt(np.mean(err ** 2)))
    nmae = float(np.sum(np.abs(err)) / max(1e-12, np.sum(np.abs(a))))
    return {
        'mae_kw': mae_e / DT,
        'rmse_kw': rmse_e / DT,
        'nmae': nmae,
    }


def prediction_baselines(data: InputData, fc: ForecastBundle) -> List[Dict[str, object]]:
    n = len(data.dates)
    persistence_load = np.zeros_like(data.load)
    persistence_pv = np.zeros_like(data.pv)
    seasonal_load = np.zeros_like(data.load)
    seasonal_pv = np.zeros_like(data.pv)

    for d in range(n):
        # persistence严格复用当前主模型对“上一行最后跨日10分钟”的掩码规则。
        pl = _masked_previous_profile(data.load, d)
        pp = _masked_previous_profile(data.pv, d)
        if pl is None:
            pl = np.full(144, 5000.0 * DT)
        if pp is None:
            pp = np.zeros(144)
        persistence_load[d] = pl
        persistence_pv[d] = pp

        # seasonal naive：d-7整行在d日0:00前已经完整发生；正式期d>=31始终可用。
        if d >= 7:
            seasonal_load[d] = data.load[d - 7]
            seasonal_pv[d] = data.pv[d - 7]
        else:
            seasonal_load[d] = pl
            seasonal_pv[d] = pp

    models = {
        'persistence': (persistence_load, persistence_pv),
        'seasonal_naive_7d': (seasonal_load, seasonal_pv),
        'causal_ensemble': (fc.load_current, fc.pv_current),
    }
    rows: List[Dict[str, object]] = []
    for model, (lp, pp) in models.items():
        for target, actual, pred in (
            ('load', data.load, lp),
            ('pv', data.pv, pp),
            ('net', data.net, lp - pp),
        ):
            row: Dict[str, object] = {'model': model, 'target': target}
            row.update(_metrics(actual, pred))
            rows.append(row)
    return rows


def natural_price(price: np.ndarray) -> np.ndarray:
    return np.concatenate(([price[-1]], price[:143]))


def high_risk_days(data: InputData, fc: ForecastBundle) -> Tuple[np.ndarray, np.ndarray]:
    """用冻结预测定义外生高风险日：高价时段正净负荷预测误差加权分数最高10%。"""
    cal_price = natural_price(data.price)
    scores = np.zeros(365, dtype=float)
    for d in range(EVAL_START, 365):
        actual = np.concatenate(([data.net[d - 1, 143]], data.net[d, :143]))
        pred = np.concatenate(([fc.net_current[d - 1, 143]], fc.net_current[d, :143]))
        scores[d] = float(np.dot(cal_price, np.maximum(actual - pred, 0.0)))
    formal = np.arange(EVAL_START, 365)
    n_top = int(math.ceil(0.10 * formal.size))
    order = formal[np.argsort(scores[formal])]
    top = np.sort(order[-n_top:])
    return top, scores


def precompute_scenarios(
    data: InputData,
    fc: ForecastBundle,
    quantiles: Sequence[float],
    window_days: int,
) -> List[np.ndarray]:
    return [
        build_net_scenarios(
            d, data, fc,
            scenario_quantiles=quantiles,
            window_days=window_days,
        )
        for d in range(365)
    ]


def _floor_after(sol, state_index: int, policy: Union[str, float]) -> float:
    if policy == 'hard':
        return SOC_MIN
    q = float(policy)
    return float(np.quantile(sol.scenario_soc[:, state_index], q))


def simulate_variant(
    name: str,
    data: InputData,
    scenarios_by_day: Sequence[np.ndarray],
    top_risk_days: np.ndarray,
    terminal_value: float = TERMINAL_SHORTAGE_VALUE,
    floor_policy: Union[str, float] = REALTIME_SOC_FLOOR_QUANTILE,
    window_days: int = SCENARIO_WINDOW_DAYS,
    k_target: int = 9,
) -> Tuple[VariantSummary, Dict[str, np.ndarray]]:
    cal_price = natural_price(data.price)
    top_set = set(int(x) for x in top_risk_days)
    soc = float(SOC0)
    prev_q_last = 0.0

    daily_plan_cost = np.zeros(365)
    daily_emg_cost = np.zeros(365)
    daily_plan_energy = np.zeros(365)
    daily_emg = np.zeros(365)
    daily_curtail = np.zeros(365)
    daily_soc00 = np.zeros(365)
    daily_soc24 = np.zeros(365)
    scenario_counts = np.zeros(365, dtype=int)
    solve_times: List[float] = []
    realtime_min = float('inf')
    realtime_max = float('-inf')
    hard_floor_count = 0
    near_floor_count = 0
    high_risk_emg = 0.0

    for d in range(365):
        daily_soc00[d] = soc
        realtime_min = min(realtime_min, soc)
        realtime_max = max(realtime_max, soc)
        scenarios = scenarios_by_day[d]
        t0 = time.perf_counter()
        sol = solve_stochastic_mpc(
            scenarios, data.price, soc,
            prev_q_last if d >= 1 else 0.0,
            risk_lambda=MAIN_LAMBDA,
            alpha=CVAR_ALPHA,
            terminal_value=float(terminal_value),
        )
        elapsed = time.perf_counter() - t0
        if d >= EVAL_START:
            solve_times.append(elapsed)
        scenario_counts[d] = sol.scenario_count
        q = np.maximum(sol.q_current, 0.0)
        q[np.abs(q) < 1e-9] = 0.0
        daily_plan_energy[d] = float(q.sum())
        daily_plan_cost[d] = float(np.dot(data.price, q))

        emg_vec = np.zeros(144, dtype=float)
        curtail_vec = np.zeros(144, dtype=float)

        if d >= 1:
            floor = _floor_after(sol, 1, floor_policy)
            soc, x, y, e, w = execute_interval(soc, prev_q_last, data.net[d - 1, 143], floor)
            emg_vec[0] = e
            curtail_vec[0] = w
            realtime_min = min(realtime_min, soc)
            realtime_max = max(realtime_max, soc)
            if d >= EVAL_START:
                hard_floor_count += int(soc <= SOC_MIN + 1e-6)
                near_floor_count += int(soc <= NEAR_FLOOR_KWH + 1e-9)

        for j in range(143):
            floor = _floor_after(sol, 2 + j, floor_policy)
            soc, x, y, e, w = execute_interval(soc, q[j], data.net[d, j], floor)
            emg_vec[j + 1] = e
            curtail_vec[j + 1] = w
            realtime_min = min(realtime_min, soc)
            realtime_max = max(realtime_max, soc)
            if d >= EVAL_START:
                hard_floor_count += int(soc <= SOC_MIN + 1e-6)
                near_floor_count += int(soc <= NEAR_FLOOR_KWH + 1e-9)

        if not (SOC_MIN - 1e-6 <= soc <= SOC_MAX + 1e-6):
            raise RuntimeError(f'{name}: {data.dates[d].date()} SOC越界 {soc}')
        soc = float(np.clip(soc, SOC_MIN, SOC_MAX))
        daily_soc24[d] = soc
        prev_q_last = float(q[143])
        daily_emg[d] = float(emg_vec.sum())
        daily_emg_cost[d] = float(np.dot(5.0 * cal_price, emg_vec))
        daily_curtail[d] = float(curtail_vec.sum())
        if d in top_set:
            high_risk_emg += daily_emg[d]

    s = slice(EVAL_START, 365)
    plan_cost = float(daily_plan_cost[s].sum())
    emg_cost = float(daily_emg_cost[s].sum())
    plan_kwh = float(daily_plan_energy[s].sum())
    emg_kwh = float(daily_emg[s].sum())
    total_grid = plan_kwh + emg_kwh
    summary = VariantSummary(
        name=name,
        k_target=int(k_target),
        window_days=int(window_days),
        terminal_value=float(terminal_value),
        floor_policy='hard_only' if floor_policy == 'hard' else f'Q{float(floor_policy):.2f}',
        total_cost_yuan=plan_cost + emg_cost,
        plan_cost_yuan=plan_cost,
        emergency_cost_yuan=emg_cost,
        plan_kwh=plan_kwh,
        emergency_kwh=emg_kwh,
        emergency_share=emg_kwh / total_grid if total_grid > 0 else 0.0,
        curtailed_kwh=float(daily_curtail[s].sum()),
        end_soc_kwh=float(daily_soc24[-1]),
        realtime_soc_min_kwh=float(realtime_min),
        realtime_soc_max_kwh=float(realtime_max),
        hard_floor_interval_count=int(hard_floor_count),
        near_floor_interval_count=int(near_floor_count),
        near_floor_hours=float(near_floor_count * 10.0 / 60.0),
        high_risk_emergency_kwh=float(high_risk_emg),
        solve_time_median_s=float(np.median(solve_times)),
        solve_time_p90_s=float(np.quantile(solve_times, 0.90)),
        mean_scenario_count=float(np.mean(scenario_counts[EVAL_START:])),
    )
    detail = {
        'daily_plan_cost': daily_plan_cost,
        'daily_emergency_cost': daily_emg_cost,
        'daily_emergency_kwh': daily_emg,
        'daily_soc00': daily_soc00,
        'daily_soc24': daily_soc24,
    }
    return summary, detail


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def budget_reuse(main_metrics: Dict) -> List[Dict[str, float]]:
    rows = main_metrics['january_risk_calibration']
    out = []
    for budget in (0.001, 0.002, 0.005):
        candidates = [
            r for r in rows
            if r['risk_lambda'] > 0
            and r['cost_premium_vs_neutral_yuan'] > 1e-9
            and r['cost_premium_vs_neutral_pct'] <= budget + 1e-12
            and r['emergency_reduction_vs_neutral_kwh'] > 1e-9
        ]
        if candidates:
            best = max(candidates, key=lambda r: (
                r['emergency_reduction_per_premium_kwh_per_yuan'],
                r['emergency_reduction_vs_neutral_kwh'],
                -r['risk_lambda'],
            ))
        else:
            best = next(r for r in rows if abs(r['risk_lambda']) < 1e-12)
        out.append({
            'insurance_budget_pct': budget * 100,
            'selected_lambda': float(best['risk_lambda']),
            'january_cost_premium_pct': float(best['cost_premium_vs_neutral_pct']) * 100,
            'january_emergency_reduction_kwh': float(best['emergency_reduction_vs_neutral_kwh']),
        })
    return out


def _fmt_million(x: float) -> str:
    return f'{x / 1e6:.3f}'


def build_conclusions(
    prediction_rows: List[Dict[str, object]],
    k_rows: List[Dict[str, object]],
    floor_rows: List[Dict[str, object]],
    vt_rows: List[Dict[str, object]],
    window_rows: List[Dict[str, object]],
) -> Dict[str, str]:
    pred = pd.DataFrame(prediction_rows)
    net = pred[pred.target == 'net'].set_index('model')
    ens = net.loc['causal_ensemble']
    pers = net.loc['persistence']
    sea = net.loc['seasonal_naive_7d']
    best_naive = min(float(pers.nmae), float(sea.nmae))
    rel = (best_naive - float(ens.nmae)) / best_naive * 100 if best_naive > 0 else 0.0
    if rel >= 0:
        pred_text = f'现有因果集成的净负荷NMAE为{float(ens.nmae):.4f}，较最佳简单基线改善{rel:.2f}%。'
    else:
        pred_text = (
            f'现有因果集成的净负荷NMAE为{float(ens.nmae):.4f}，而7日季节朴素基线为{float(sea.nmae):.4f}，'
            f'前者较最佳简单基线高{-rel:.2f}%。这说明当前预测层并非误差意义上的最优设计；该结论应作为模型局限如实报告，'
            '但由于它来自主模型冻结后的正式期检验，不据此事后重选主模型或改写result2。'
        )

    kdf = pd.DataFrame(k_rows).set_index('k_target')
    tail = kdf.loc[[k for k in (9, 12, 15) if k in kdf.index]]
    cost_span = (tail.total_cost_yuan.max() - tail.total_cost_yuan.min()) / kdf.loc[9, 'total_cost_yuan'] * 100
    emg_span = (tail.emergency_kwh.max() - tail.emergency_kwh.min()) / kdf.loc[9, 'emergency_kwh'] * 100
    if cost_span <= 1.0 and emg_span <= 5.0:
        k_text = f'K≥9时总费用跨度{cost_span:.3f}%，紧急购电量跨度{emg_span:.3f}%，可视为已进入相对稳定区间。'
    else:
        k_text = f'K≥9时总费用跨度{cost_span:.3f}%，紧急购电量跨度{emg_span:.3f}%，场景密度仍有可见影响，因此正文不宣称K=9“最优”。'

    fdf = pd.DataFrame(floor_rows).set_index('floor_policy')
    q25 = fdf.loc['Q0.25']
    hard = fdf.loc['hard_only']
    q10 = fdf.loc['Q0.10']
    floor_text = (
        f'Q0.25相对仅硬下限策略把≤{NEAR_FLOOR_KWH:.0f}kWh低SOC累计时长从{float(hard.near_floor_hours):.2f}h降至{float(q25.near_floor_hours):.2f}h，'
        f'但紧急购电量增加{(float(q25.emergency_kwh)/float(hard.emergency_kwh)-1)*100:.3f}%。'
        f'事后结果中Q0.10的总费用更低（{float(q10.total_cost_yuan):.2f}元），说明Q0.25体现的是更保守的储能状态保护，而非正式期经济最优；'
        '由于本轮只做冻结后的消融，不据此反向修改主参数。'
    )

    vdf = pd.DataFrame(vt_rows).set_index('terminal_value')
    vt0, vt08 = vdf.loc[0.0], vdf.loc[0.8]
    vt_text = (
        f'v_T=0时年末SOC={float(vt0.end_soc_kwh):.1f}kWh，v_T=0.8时仍为{float(vt08.end_soc_kwh):.1f}kWh，'
        f'总费用仅变化{(float(vt08.total_cost_yuan)/float(vt0.total_cost_yuan)-1)*100:+.4f}%。'
        '因此在当前48小时滚动视野下，0—1.2元/kWh的软终端价值几乎不改变正式期决策；它是防护性continuation-value项，'
        '而不是主结果的驱动因素，也不能再把年末未放空主要归因于v_T=0.8。'
    )

    wdf = pd.DataFrame(window_rows)
    wrange = (wdf.total_cost_yuan.max() - wdf.total_cost_yuan.min()) / wdf[wdf.window_days == 42].iloc[0].total_cost_yuan * 100
    wemg = (wdf.emergency_kwh.max() - wdf.emergency_kwh.min()) / wdf[wdf.window_days == 42].iloc[0].emergency_kwh * 100
    window_text = f'残差窗口28/42/56天的总费用跨度为{wrange:.3f}%，紧急购电量跨度为{wemg:.3f}%；费用结论较稳定，风险量存在一定但有限的窗口敏感性。'
    return {
        'prediction': pred_text,
        'k': k_text,
        'floor': floor_text,
        'terminal': vt_text,
        'window': window_text,
    }


def make_figures(
    prediction_rows: List[Dict[str, object]],
    k_rows: List[Dict[str, object]],
    floor_rows: List[Dict[str, object]],
    vt_rows: List[Dict[str, object]],
    window_rows: List[Dict[str, object]],
) -> None:
    for f in ['Microsoft YaHei', 'SimHei', 'SimSun']:
        if any(x.name == f for x in font_manager.fontManager.ttflist):
            plt.rcParams['font.sans-serif'] = [f]
            break
    plt.rcParams['axes.unicode_minus'] = False

    # 表6：预测基线
    pdf = pd.DataFrame(prediction_rows)
    wide = []
    model_labels = {'persistence':'前一可见日', 'seasonal_naive_7d':'7日季节朴素', 'causal_ensemble':'当前因果集成'}
    for model in ['persistence', 'seasonal_naive_7d', 'causal_ensemble']:
        row = [model_labels[model]]
        for target in ['load', 'pv', 'net']:
            r = pdf[(pdf.model == model) & (pdf.target == target)].iloc[0]
            row.extend([f'{r.mae_kw:.1f}', f'{r.rmse_kw:.1f}', f'{r.nmae:.4f}'])
        wide.append(row)
    headers = ['模型','Load MAE','Load RMSE','Load NMAE','PV MAE','PV RMSE','PV NMAE','Net MAE','Net RMSE','Net NMAE']
    fig, ax = plt.subplots(figsize=(14, 3.2))
    ax.axis('off')
    ax.set_title('表6  严格因果信息边界下的预测基线对比（MAE/RMSE单位：kW）', fontsize=12, pad=10)
    tb = ax.table(cellText=wide, colLabels=headers, cellLoc='center', loc='center')
    tb.auto_set_font_size(False); tb.set_fontsize(8.5); tb.scale(1, 1.7)
    for (r,c), cell in tb.get_celld().items():
        cell.set_edgecolor('#bbbbbb')
        if r == 0:
            cell.set_facecolor('#f2f2f2'); cell.set_text_props(weight='bold')
    fig.tight_layout()
    fig.savefig(ROB / '表6_预测基线对比.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # 图13：四组核心鲁棒性证据
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    kdf = pd.DataFrame(k_rows).sort_values('k_target')
    ax = axes[0,0]
    ax.plot(kdf.k_target, kdf.total_cost_yuan/1e6, marker='o', label='总费用(百万元)')
    ax2 = ax.twinx(); ax2.plot(kdf.k_target, kdf.emergency_kwh/1e3, marker='s', linestyle='--', label='紧急购电(MWh)')
    ax.set_xlabel('场景数 K'); ax.set_ylabel('总费用 (百万元)'); ax2.set_ylabel('紧急购电量 (MWh)')
    ax.set_title('(a) 场景密度敏感性')
    l1,la1=ax.get_legend_handles_labels(); l2,la2=ax2.get_legend_handles_labels(); ax.legend(l1+l2,la1+la2,frameon=False,fontsize=8)

    fdf = pd.DataFrame(floor_rows)
    ax = axes[0,1]
    x=np.arange(len(fdf)); ax.bar(x, fdf.emergency_kwh/1e3, label='紧急购电(MWh)')
    ax.set_xticks(x); ax.set_xticklabels(fdf.floor_policy); ax.set_ylabel('紧急购电量 (MWh)')
    ax2=ax.twinx(); ax2.plot(x, fdf.near_floor_hours, marker='o', color='#d62728', label=f'≤{NEAR_FLOOR_KWH:.0f}kWh时长(h)')
    ax2.set_ylabel('低SOC累计时长 (h)'); ax.set_title('(b) 实时SOC储备策略消融')
    l1,la1=ax.get_legend_handles_labels(); l2,la2=ax2.get_legend_handles_labels(); ax.legend(l1+l2,la1+la2,frameon=False,fontsize=8)

    vdf = pd.DataFrame(vt_rows).sort_values('terminal_value')
    base_vt_cost = float(vdf.iloc[0].total_cost_yuan)
    delta_vt_cost = vdf.total_cost_yuan - base_vt_cost
    ax=axes[1,0]; ax.plot(vdf.terminal_value, delta_vt_cost, marker='o', label='相对v_T=0费用变化(元)')
    ax.set_xlabel('终端缺口价值 v_T (元/kWh)'); ax.set_ylabel('费用变化 (元)')
    ax2=ax.twinx(); ax2.plot(vdf.terminal_value, vdf.end_soc_kwh, marker='s', linestyle='--', color='#2ca02c', label='12月31日24:00 SOC')
    ax2.set_ylabel('年末SOC (kWh)'); ax.set_title('(c) 软终端价值敏感性')
    l1,la1=ax.get_legend_handles_labels(); l2,la2=ax2.get_legend_handles_labels(); ax.legend(l1+l2,la1+la2,frameon=False,fontsize=8)

    wdf=pd.DataFrame(window_rows).sort_values('window_days')
    ax=axes[1,1]; ax.plot(wdf.window_days,wdf.total_cost_yuan/1e6,marker='o',label='总费用(百万元)')
    ax.set_xlabel('历史残差窗口 (天)'); ax.set_ylabel('总费用 (百万元)')
    ax2=ax.twinx(); ax2.plot(wdf.window_days,wdf.emergency_kwh/1e3,marker='s',linestyle='--',color='#ff7f0e',label='紧急购电(MWh)')
    ax2.set_ylabel('紧急购电量 (MWh)'); ax.set_title('(d) 残差窗口附录级检查')
    l1,la1=ax.get_legend_handles_labels(); l2,la2=ax2.get_legend_handles_labels(); ax.legend(l1+l2,la1+la2,frameon=False,fontsize=8)

    for ax in axes.flat:
        ax.grid(axis='y', alpha=0.15)
    fig.suptitle('图13  问题2冻结主模型后的鲁棒性与消融验证', y=0.995, fontsize=13)
    fig.tight_layout(rect=(0,0,1,0.975))
    fig.savefig(ROB / '图13_鲁棒性与消融验证.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def write_markdown(
    prediction_rows: List[Dict[str, object]],
    k_rows: List[Dict[str, object]],
    floor_rows: List[Dict[str, object]],
    vt_rows: List[Dict[str, object]],
    window_rows: List[Dict[str, object]],
    budget_rows: List[Dict[str, float]],
    conclusions: Dict[str, str],
    main_match: Dict[str, float],
) -> None:
    pred = pd.DataFrame(prediction_rows)
    def pred_table() -> str:
        lines = ['| 方法 | 对象 | MAE/kW | RMSE/kW | NMAE |','|---|---|---:|---:|---:|']
        names={'persistence':'前一可见日 persistence','seasonal_naive_7d':'7日季节朴素','causal_ensemble':'当前因果集成'}
        for _,r in pred.iterrows():
            lines.append(f"| {names[r['model']]} | {r['target']} | {r['mae_kw']:.2f} | {r['rmse_kw']:.2f} | {r['nmae']:.4f} |")
        return '\n'.join(lines)

    def variant_table(rows: List[Dict[str, object]], cols: List[Tuple[str,str,str]]) -> str:
        head='| '+' | '.join(x[1] for x in cols)+' |'
        sep='|'+ '|'.join(['---:' if i else '---' for i in range(len(cols))])+'|'
        out=[head,sep]
        for r in rows:
            vals=[]
            for key,_,fmt in cols:
                v=r[key]
                vals.append(format(v,fmt) if fmt else str(v))
            out.append('| '+' | '.join(vals)+' |')
        return '\n'.join(out)

    text=f'''# 问题2模型合理性与鲁棒性验证

> 本节属于**冻结主模型之后的事后鲁棒性/消融验证**。正式 `result2.xlsx`、主模型 $K=9$、$\\lambda=0.02$、$\\alpha=0.8$ 及2—12月主结果均未因本节实验重新选择或改写。正式期数据仅用于检验结论是否脆弱，不用于事后调参。

## 1. 预测基线：复杂度是否真的有必要

所有方法使用完全相同的0:00严格信息边界。persistence同样屏蔽上一行最后一个跨日10分钟未来点；7日季节朴素仅使用已经完整发生的 $d-7$ 曲线。

{pred_table()}

**实测结论：** {conclusions['prediction']}

## 2. 场景数 $K$ 敏感性

主模型 $K=9$ 的经验分位点保持冻结版本 `(0.05,0.15,...,0.95)` 原样不动；其他 $K$ 采用关于0.5对称、覆盖0.05—0.95的近均匀分位网格。除 $K$ 外，其余主参数均固定。求解耗时只统计正式期 `solve_stochastic_mpc` 本身，并报告中位数与P90。

{variant_table(k_rows,[('k_target','K','d'),('total_cost_yuan','总费用/元','.2f'),('emergency_kwh','紧急购电/kWh','.2f'),('solve_time_median_s','求解中位数/s','.3f'),('solve_time_p90_s','P90/s','.3f')])}

**实测结论：** {conclusions['k']}

## 3. 实时SOC储备策略消融

“hard_only”是真正的无额外风险储备基线：只守题设硬下限 $S^{{min}}=1200$ kWh；其余方案分别采用场景SOC的10%、25%、50%分位作为实时储备线。低SOC统计定义为实时SOC不高于 {NEAR_FLOOR_KWH:.0f} kWh。高风险日固定为冻结预测下“高价时段正净负荷预测误差加权分数”最高10%的日期，各策略在完全相同日期集合上比较。

{variant_table(floor_rows,[('floor_policy','策略',''),('total_cost_yuan','总费用/元','.2f'),('emergency_kwh','紧急购电/kWh','.2f'),('near_floor_hours','≤1500kWh时长/h','.2f'),('hard_floor_interval_count','触及1200段数','d'),('high_risk_emergency_kwh','高风险日紧急购电/kWh','.2f')])}

**实测结论：** {conclusions['floor']}

这里的“储备线”应解释为**储能状态安全/跨时段备用水平**，而不是“减少紧急购电”的机制。实测中更高分位储备显著减少了电池长时间贴近下限的现象，但会更早使用外网紧急购电，因此形成了清晰的“储能深度利用—外网兜底”权衡。

## 4. 软终端价值 $v_T$ 敏感性

软储备参考值固定为 $S^{{tar}}=6000$ kWh：它既是安全区间 $[1200,10800]$ 的中点，也是题设初始SOC，因此作为“中性库存参考水平”具有直接物理解释。本实验只扰动缺口价值 $v_T$，不做二维调参。

{variant_table(vt_rows,[('terminal_value','v_T','0.1f'),('total_cost_yuan','总费用/元','.2f'),('emergency_kwh','紧急购电/kWh','.2f'),('end_soc_kwh','12/31末SOC/kWh','.2f'),('near_floor_hours','≤1500kWh时长/h','.2f')])}

**实测结论：** {conclusions['terminal']}

因此论文只需把 $v_T$ 说明为防止有限视野极端行为的保险性设计；本题数据下它基本不绑定，真正限制年末行为的是48小时滚动视野、后续购电成本和储能连续状态的共同作用。

## 5. 历史残差窗口附录级检查

{variant_table(window_rows,[('window_days','窗口/天','d'),('total_cost_yuan','总费用/元','.2f'),('emergency_kwh','紧急购电/kWh','.2f'),('end_soc_kwh','年末SOC/kWh','.2f')])}

**实测结论：** {conclusions['window']}

## 6. 0.20%风险保险预算的定位

该比例是建模者事前设置的“可靠性保险预算”，不是题设常数。无需利用2—12月重新调参，直接复用1月原校准曲线可得到：

{variant_table(budget_rows,[('insurance_budget_pct','保险预算/%','.2f'),('selected_lambda','1月规则选择λ','.3f'),('january_cost_premium_pct','1月成本溢价/%','.4f'),('january_emergency_reduction_kwh','1月紧急购电减少/kWh','.2f')])}

## 7. 冻结主模型复现检查

鲁棒性脚本中 `K=9 + 42天窗口 + v_T=0.8 + Q0.25` 是主模型的独立复现，不用于重写正式结果。与冻结 `metrics.json` 的差异：

- 总费用绝对差：{main_match['total_cost_abs_diff_yuan']:.6f} 元；
- 紧急购电量绝对差：{main_match['emergency_abs_diff_kwh']:.6f} kWh；
- 年末SOC：{main_match['replicate_end_soc_kwh']:.6f} kWh。

## 8. 论文建议

正文只需保留“表6预测基线 + 图13鲁棒性/消融”以及上述五条实测结论；完整CSV/JSON放入 `output/robustness/` 作为可复核附件。论述重点不是寻找正式期最优参数，而是说明：**冻结主策略的结论在合理参数扰动下是否稳定，以及每个风险机制是否具有可解释的现实贡献。**
'''
    (OUT / '09_鲁棒性与消融验证.md').write_text(text, encoding='utf-8')


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--only', choices=['prediction','all'], default='all')
    args=ap.parse_args()

    print('[1/8] 读取冻结主数据与严格因果预测...')
    data=load_official_inputs(REPO)
    fc=precompute_causal_forecasts(data)
    main_metrics=json.loads((HERE/'metrics.json').read_text(encoding='utf-8'))

    print('[2/8] 预测baseline...')
    prediction_rows=prediction_baselines(data,fc)
    write_csv(ROB/'prediction_baselines.csv',prediction_rows)
    if args.only=='prediction':
        print(json.dumps(prediction_rows,ensure_ascii=False,indent=2))
        return

    top_days,risk_scores=high_risk_days(data,fc)
    pd.DataFrame({
        'date':[data.dates[d].date().isoformat() for d in range(EVAL_START,365)],
        'risk_score':[risk_scores[d] for d in range(EVAL_START,365)],
        'top10pct':[d in set(top_days) for d in range(EVAL_START,365)],
    }).to_csv(ROB/'fixed_high_risk_days.csv',index=False,encoding='utf-8-sig')

    print('[3/8] 预计算场景集...')
    scenario_cache: Dict[Tuple[Tuple[float,...],int],List[np.ndarray]]={}
    def scenarios(qs:Sequence[float],window:int):
        key=(tuple(float(x) for x in qs),int(window))
        if key not in scenario_cache:
            scenario_cache[key]=precompute_scenarios(data,fc,key[0],key[1])
        return scenario_cache[key]

    variant_cache: Dict[Tuple[Tuple[float,...],int,float,str],Tuple[VariantSummary,Dict[str,np.ndarray]]]={}
    def run_cached(name:str,qs:Sequence[float],window:int,vt:float,floor:Union[str,float],k_target:int):
        floor_key='hard' if floor=='hard' else f'{float(floor):.6f}'
        key=(tuple(float(x) for x in qs),int(window),float(vt),floor_key)
        if key not in variant_cache:
            print(f'  -> {name}')
            variant_cache[key]=simulate_variant(name,data,scenarios(qs,window),top_days,vt,floor,window,k_target)
        return variant_cache[key]

    print('[4/8] K敏感性...')
    k_summ=[]
    for k in K_VALUES:
        s,_=run_cached(f'K={k}',quantiles_for_k(k),42,0.8,0.25,k)
        k_summ.append(s)

    print('[5/8] SOC储备消融 + v_T敏感性...')
    main_q=quantiles_for_k(9)
    floor_summ=[]
    for fp in FLOOR_POLICIES:
        label='hard_only' if fp=='hard' else f'Q{float(fp):.2f}'
        s,_=run_cached(label,main_q,42,0.8,fp,9)
        floor_summ.append(s)
    vt_summ=[]
    for vt in TERMINAL_VALUES:
        s,_=run_cached(f'vT={vt:.1f}',main_q,42,vt,0.25,9)
        vt_summ.append(s)

    print('[6/8] 28/42/56天残差窗口检查...')
    window_summ=[]
    for w in WINDOW_VALUES:
        s,_=run_cached(f'window={w}',main_q,w,0.8,0.25,9)
        window_summ.append(s)

    # 统一字典格式
    k_rows=[asdict(x) for x in k_summ]
    floor_rows=[asdict(x) for x in floor_summ]
    vt_rows=[asdict(x) for x in vt_summ]
    window_rows=[asdict(x) for x in window_summ]
    write_csv(ROB/'k_sensitivity.csv',k_rows)
    write_csv(ROB/'soc_reserve_ablation.csv',floor_rows)
    write_csv(ROB/'terminal_value_sensitivity.csv',vt_rows)
    write_csv(ROB/'window_sensitivity.csv',window_rows)

    budgets=budget_reuse(main_metrics)
    write_csv(ROB/'risk_budget_reuse.csv',budgets)

    # 主模型独立复现一致性：K=9主场景、42天、vT=.8、Q=.25。
    main_rep=next(x for x in k_summ if x.k_target==9)
    main_match={
        'total_cost_abs_diff_yuan':abs(main_rep.total_cost_yuan-float(main_metrics['total_purchase_cost_yuan'])),
        'emergency_abs_diff_kwh':abs(main_rep.emergency_kwh-float(main_metrics['emergency_purchase_kwh'])),
        'replicate_end_soc_kwh':main_rep.end_soc_kwh,
    }

    conclusions=build_conclusions(prediction_rows,k_rows,floor_rows,vt_rows,window_rows)
    payload={
        'frozen_main':{
            'risk_lambda':MAIN_LAMBDA,'k':9,'alpha':CVAR_ALPHA,'window_days':42,
            'terminal_value':0.8,'floor_quantile':0.25,
            'note':'2-12月仅事后鲁棒性验证，不反向调参',
        },
        'prediction_baselines':prediction_rows,
        'k_sensitivity':k_rows,
        'soc_reserve_ablation':floor_rows,
        'terminal_value_sensitivity':vt_rows,
        'window_sensitivity':window_rows,
        'risk_budget_reuse':budgets,
        'main_reproduction_check':main_match,
        'conclusions':conclusions,
        'exact_cvar_example_k9_alpha08':{
            'tail_mass_scenarios':(1-CVAR_ALPHA)*9,
            'formula':'(worst + 0.8*second_worst)/1.8',
            'self_test':empirical_cvar_equal_prob([9,8,7,6,5,4,3,2,1],0.8),
        },
    }
    (ROB/'robustness_results.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')

    print('[7/8] 生成论文图表与文字...')
    make_figures(prediction_rows,k_rows,floor_rows,vt_rows,window_rows)
    write_markdown(prediction_rows,k_rows,floor_rows,vt_rows,window_rows,budgets,conclusions,main_match)

    print('[8/8] 核心实测结论')
    print(json.dumps({'main_match':main_match,'conclusions':conclusions},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
