# -*- coding: utf-8 -*-
"""
问题2 绘图
==========
风格与问题1_绘图.py统一：Microsoft YaHei/SimHei、Matplotlib默认Tab10配色、12英寸宽图、150 dpi、中文标题与单位。
图号承接问题1的图1~图3，表号承接问题1的表1~表2。

输出到 output/图表/：
  图4_问题2因果滚动调度框架.png
  图5_一月风险参数校准与敏感性.png
  图6_全年SOC轨迹与安全边界.png
  图7_月度计划购电与紧急购电.png
  图8_负荷光伏净负荷预测效果.png
  图9_典型日12月21日实时调度.png
  图10_计划购电零值率日内分布.png
  图11_风险方案与风险中性方案对比.png
  图12_全年每日紧急购电与高风险日.png
  表3_指定日期计划购电量.png
  表4_指定日期储能充放电量.png
  表5_指定日期紧急购电量.png
"""
from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
Q2 = HERE.parent
REPO = Q2.parent
OUT = Q2 / 'output'
FIG = OUT / '图表'
FIG.mkdir(parents=True, exist_ok=True)
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from q2_core import (  # noqa: E402
    DT, ETA_C, ETA_D, SOC_MIN, SOC_MAX, TERMINAL_RESERVE,
    grouped_emergency_periods, load_official_inputs,
)

# 与问题1统一的中文字体
for _f in ['Microsoft YaHei', 'SimHei', 'SimSun']:
    if any(f.name == _f for f in font_manager.fontManager.ttflist):
        plt.rcParams['font.sans-serif'] = [_f]
        break
plt.rcParams['axes.unicode_minus'] = False

# 与问题1统一的配色
C_RED = '#d62728'
C_BLUE = '#1f77b4'
C_ORANGE = '#ff7f0e'
C_GREEN = '#2ca02c'
C_PURPLE = '#9467bd'
C_GRAY = '#7f7f7f'

NPZ = np.load(HERE / 'run_data.npz')
METRICS = json.loads((HERE / 'metrics.json').read_text(encoding='utf-8'))
DAILY = pd.read_csv(HERE / 'daily_summary.csv', encoding='utf-8-sig')
DAILY['date'] = pd.to_datetime(DAILY['date'])
DATA = load_official_inputs(REPO)
DATES = pd.to_datetime([d.date() for d in DATA.dates])
EVAL_START = 31
SELECTED_DATES = ['2025-03-20', '2025-06-21', '2025-09-23', '2025-12-21']


def save(fig, filename):
    fig.tight_layout()
    fig.savefig(FIG / filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def set_hour_axis(ax, step=2):
    ax.set_xlim(0, 24)
    ticks = np.arange(0, 25, step)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{int(h):02d}:00' for h in ticks])
    ax.set_xlabel('时间')


def fig4_flowchart():
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    boxes = [
        (0.03, 0.67, 0.19, 0.17, '0:00 信息集', '完整历史≤d−2\n昨日已发生143点\n末10 min强制掩码', C_BLUE),
        (0.27, 0.67, 0.19, 0.17, '因果预测', '负荷 / 光伏分别预测\n上一日+7日滞后\n同星期+近7日', C_ORANGE),
        (0.51, 0.67, 0.19, 0.17, '经验场景', '历史预测残差\n9个等概率分位场景\n保留日内相关结构', C_GREEN),
        (0.75, 0.67, 0.22, 0.17, '两阶段随机MPC', '289段滚动视野\nCVaR: α=0.80\nλ=0.02', C_PURPLE),
        (0.75, 0.27, 0.22, 0.17, '发布计划购电', '当天144段计划\n0:10开始生效\n昨日末段仍先执行', C_BLUE),
        (0.51, 0.27, 0.19, 0.17, '日内实时执行', '每10 min读取实际净负荷\n富余优先充电\n缺口优先放电', C_ORANGE),
        (0.27, 0.27, 0.19, 0.17, '风险修正与结算', 'SOC风险底线\n不足按5×电价紧急购电\n计划电量照常计费', C_RED),
        (0.03, 0.27, 0.19, 0.17, '状态更新', 'SOC跨日连续\n实际数据进入历史\n滚动到d+1', C_GREEN),
    ]
    centers = []
    for x, y, w, h, title, body, color in boxes:
        patch = FancyBboxPatch((x, y), w, h,
                               boxstyle='round,pad=0.012,rounding_size=0.02',
                               linewidth=1.3, edgecolor=color, facecolor='white')
        ax.add_patch(patch)
        ax.text(x + w/2, y + h*0.72, title, ha='center', va='center',
                fontsize=11, fontweight='bold', color=color)
        ax.text(x + w/2, y + h*0.34, body, ha='center', va='center', fontsize=8.6, linespacing=1.25)
        centers.append((x+w/2, y+h/2))
    for a, b in [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7)]:
        ax.add_patch(FancyArrowPatch(centers[a], centers[b], arrowstyle='-|>',
                                     mutation_scale=12, lw=1.1, color='#555555',
                                     shrinkA=55, shrinkB=55))
    ax.add_patch(FancyArrowPatch((0.12, 0.35), (0.12, 0.70), arrowstyle='-|>',
                                 mutation_scale=12, lw=1.1, color='#555555',
                                 connectionstyle='arc3,rad=-0.55'))
    ax.text(0.5, 0.94, '图4  问题2严格因果滚动调度框架', ha='center', va='center', fontsize=14)
    ax.text(0.5, 0.08,
            '边界口径：当天0:00–0:10执行昨日计划最后一段；当天0:00新计划从0:10开始生效。',
            ha='center', fontsize=9.5)
    save(fig, '图4_问题2因果滚动调度框架.png')


def fig5_risk_calibration():
    rows = pd.DataFrame(METRICS['january_risk_calibration'])
    x = rows['risk_lambda'].to_numpy(float)
    premium = rows['cost_premium_vs_neutral_pct'].to_numpy(float) * 100
    reduction = rows['emergency_reduction_vs_neutral_kwh'].to_numpy(float)

    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(x, premium, color=C_BLUE, marker='o', lw=1.8, label='成本增幅(%)')
    ax1.axhline(0.2, color=C_GRAY, ls='--', lw=1.0, label='1月风险保险预算0.2%')
    ax1.set_xlabel('CVaR风险权重 λ')
    ax1.set_ylabel('相对风险中性成本增幅 (%)', color=C_BLUE)
    ax1.tick_params(axis='y', labelcolor=C_BLUE)
    ax1.grid(axis='y', alpha=0.18)

    ax2 = ax1.twinx()
    ax2.plot(x, reduction, color=C_GREEN, marker='s', lw=1.8, label='紧急购电减少量(kWh)')
    ax2.set_ylabel('紧急购电减少量 (kWh)', color=C_GREEN)
    ax2.tick_params(axis='y', labelcolor=C_GREEN)

    sel = int(np.argmin(np.abs(x - METRICS['risk_lambda'])))
    ax1.scatter([x[sel]], [premium[sel]], s=90, color=C_RED, zorder=5)
    ax1.annotate(f"最终 λ={x[sel]:.02f}\n增幅 {premium[sel]:.3f}%\n减少 {reduction[sel]:.1f} kWh",
                 xy=(x[sel], premium[sel]), xytext=(0.055, max(0.4, premium[sel] + 0.7)),
                 arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.0), fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#bbbbbb'))
    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc='upper left', ncol=3, frameon=False)
    ax1.set_title('图5  1月风险参数校准：经济性与可靠性权衡')
    save(fig, '图5_一月风险参数校准与敏感性.png')


def reconstruct_soc(start=EVAL_START, end=365):
    charge = NPZ['charge']
    discharge = NPZ['discharge']
    soc00 = NPZ['soc00']
    times, vals = [], []
    for d in range(start, end):
        s = float(soc00[d])
        t0 = pd.Timestamp(DATES[d])
        if not times:
            times.append(t0); vals.append(s)
        for h in range(144):
            s += ETA_C * float(charge[d, h]) - float(discharge[d, h]) / ETA_D
            times.append(t0 + timedelta(minutes=10 * int(h + 1)))
            vals.append(s)
    return pd.to_datetime(times), np.asarray(vals)


def fig6_soc():
    times, soc = reconstruct_soc()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times, soc, color=C_PURPLE, lw=0.75, label='SOC(kWh)')
    ax.axhline(SOC_MIN, color=C_RED, ls='--', lw=0.9, label='SOC下限1200kWh')
    ax.axhline(SOC_MAX, color=C_RED, ls='--', lw=0.9, label='SOC上限10800kWh')
    ax.axhline(TERMINAL_RESERVE, color=C_GRAY, ls=':', lw=1.0, label='软储备目标6000kWh')
    ax.set_ylabel('储电量 SOC (kWh)')
    ax.set_xlabel('日期')
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m月'))
    ax.set_ylim(0, 12000)
    ax.legend(loc='upper left', ncol=4, frameon=False)
    ax.grid(axis='y', alpha=0.15)
    ax.set_title('图6  正式评价期储能SOC连续轨迹与安全边界')
    ax.text(0.01, 0.03,
            f'10分钟重构范围：{soc.min():.1f}–{soc.max():.1f} kWh；12月31日24:00：{NPZ["soc24"][-1]:.1f} kWh',
            transform=ax.transAxes, fontsize=8.8,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#cccccc'))
    save(fig, '图6_全年SOC轨迹与安全边界.png')


def fig7_monthly_energy():
    df = DAILY.iloc[EVAL_START:].copy()
    df['month'] = df['date'].dt.to_period('M')
    g = df.groupby('month').agg(plan_kwh=('plan_kwh','sum'), emergency_kwh=('emergency_kwh','sum'))
    x = np.arange(len(g))
    labels = [f'{p.month}月' for p in g.index]

    fig, ax1 = plt.subplots(figsize=(12, 4))
    bars = ax1.bar(x, g['plan_kwh'].to_numpy()/1e6, width=0.62, color=C_BLUE, label='计划购电量(GWh)')
    ax1.set_ylabel('计划购电量 (GWh)', color=C_BLUE)
    ax1.tick_params(axis='y', labelcolor=C_BLUE)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel('月份')
    ax1.grid(axis='y', alpha=0.15)

    ax2 = ax1.twinx()
    line = ax2.plot(x, g['emergency_kwh'].to_numpy()/1e3, color=C_RED, marker='o', lw=1.8,
                    label='紧急购电量(MWh)')
    ax2.set_ylabel('紧急购电量 (MWh)', color=C_RED)
    ax2.tick_params(axis='y', labelcolor=C_RED)
    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc='upper left', ncol=2, frameon=False)
    ax1.set_title('图7  2025年2—12月计划购电与紧急购电月度分布')
    save(fig, '图7_月度计划购电与紧急购电.png')


def fig8_forecast():
    actual = [
        DATA.load_kw[EVAL_START:].mean(axis=0),
        DATA.pv_kw[EVAL_START:].mean(axis=0),
        (DATA.net[EVAL_START:] / DT).mean(axis=0),
    ]
    pred = [
        (NPZ['forecast_load'][EVAL_START:] / DT).mean(axis=0),
        (NPZ['forecast_pv'][EVAL_START:] / DT).mean(axis=0),
        (NPZ['forecast_net'][EVAL_START:] / DT).mean(axis=0),
    ]
    names = ['负荷', '光伏', '净负荷']
    maes = [METRICS['forecast_metrics']['load_mae_kw'], METRICS['forecast_metrics']['pv_mae_kw'],
            METRICS['forecast_metrics']['net_mae_kw']]
    colors = [C_BLUE, C_ORANGE, C_GREEN]
    x = (np.arange(144) + 1) * 10

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for ax, a, p, n, mae, c in zip(axes, actual, pred, names, maes, colors):
        ax.plot(x, a, color=c, lw=1.7, label=f'{n}实际均值')
        ax.plot(x, p, color=C_RED, ls='--', lw=1.4, label=f'{n}预测均值')
        ax.set_ylabel('功率 (kW)')
        ax.legend(loc='upper right', frameon=False)
        ax.grid(axis='y', alpha=0.15)
        ax.text(0.01, 0.88, f'MAE={mae:.1f} kW', transform=ax.transAxes, fontsize=9)
    ticks = list(range(0, 1441, 120))
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels([f'{t//60:02d}:{t%60:02d}' for t in ticks])
    axes[-1].set_xlim(0, 1440)
    axes[-1].set_xlabel('时间')
    fig.suptitle('图8  因果预测与实际曲线对比（正式期平均日内形态）', y=0.995, fontsize=12)
    fig.tight_layout(rect=(0,0,1,0.98))
    fig.savefig(FIG / '图8_负荷光伏净负荷预测效果.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def natural_plan(d):
    lead = NPZ['plan_q'][d-1, 143] if d > 0 else 0.0
    return np.concatenate(([lead], NPZ['plan_q'][d, :143]))


def natural_net(d):
    lead = DATA.net[d-1, 143] if d > 0 else 0.0
    return np.concatenate(([lead], DATA.net[d, :143]))


def day_soc(d):
    s = float(NPZ['soc00'][d])
    vals = [s]
    for h in range(144):
        s += ETA_C * float(NPZ['charge'][d,h]) - float(NPZ['discharge'][d,h]) / ETA_D
        vals.append(s)
    return np.asarray(vals)


def fig9_typical_day():
    d = int(np.where(DATES == pd.Timestamp('2025-12-21'))[0][0])
    mins = np.arange(144) * 10
    q = natural_plan(d)
    net = natural_net(d)
    ch = NPZ['charge'][d]
    dis = NPZ['discharge'][d]
    emg = NPZ['emergency'][d]
    soc = day_soc(d)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True,
                             gridspec_kw={'height_ratios':[1.1,1.0,0.85]})
    ax = axes[0]
    ax.step(mins, net, where='post', color=C_BLUE, lw=1.5, label='实际净负荷(kWh/10min)')
    ax.step(mins, q, where='post', color=C_ORANGE, lw=1.4, label='已承诺计划购电(kWh/10min)')
    ax.set_ylabel('电量 (kWh/10min)')
    ax.legend(loc='upper left', ncol=2, frameon=False)
    ax.grid(axis='y', alpha=0.15)

    ax = axes[1]
    ax.bar(mins, ch, width=8, color=C_GREEN, label='充电量(kWh)')
    ax.bar(mins, -dis, width=8, color=C_RED, label='放电量(kWh)')
    ax.bar(mins, emg, width=8, color=C_ORANGE, alpha=0.75, label='紧急购电量(kWh)')
    ax.axhline(0, color='black', lw=0.6)
    ax.set_ylabel('实时调节量 (kWh/10min)')
    ax.legend(loc='upper left', ncol=3, frameon=False)
    ax.grid(axis='y', alpha=0.15)

    ax = axes[2]
    smins = np.arange(145) * 10
    ax.plot(smins, soc, color=C_PURPLE, lw=1.8, label='SOC(kWh)')
    ax.axhline(SOC_MIN, color=C_GRAY, ls='--', lw=0.8)
    ax.axhline(SOC_MAX, color=C_GRAY, ls='--', lw=0.8)
    ax.set_ylabel('SOC (kWh)')
    ax.legend(loc='upper left', frameon=False)
    ax.grid(axis='y', alpha=0.15)
    ticks = list(range(0, 1441, 120))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{t//60:02d}:{t%60:02d}' for t in ticks])
    ax.set_xlim(0, 1440)
    ax.set_xlabel('时间')
    fig.suptitle(f'图9  典型高压力日调度（2025-12-21，紧急购电{emg.sum():.1f} kWh）', y=0.995, fontsize=12)
    fig.tight_layout(rect=(0,0,1,0.98))
    fig.savefig(FIG / '图9_典型日12月21日实时调度.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def fig10_zero_rate():
    q = NPZ['plan_q'][EVAL_START:]
    zero = np.isclose(q, 0.0, atol=1e-9).mean(axis=0) * 100
    mean_q = q.mean(axis=0)
    mins = (np.arange(144) + 1) * 10

    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(mins, zero, color=C_BLUE, lw=1.8, label='购电量为0的日期比例(%)')
    ax1.fill_between(mins, 0, zero, color=C_BLUE, alpha=0.08)
    ax1.set_ylabel('零值比例 (%)', color=C_BLUE)
    ax1.tick_params(axis='y', labelcolor=C_BLUE)
    ax1.set_ylim(0, 105)
    ax1.axvspan(480, 960, color=C_ORANGE, alpha=0.05)
    ticks = list(range(0, 1441, 120))
    ax1.set_xticks(ticks)
    ax1.set_xticklabels([f'{t//60:02d}:{t%60:02d}' for t in ticks])
    ax1.set_xlim(0, 1440)
    ax1.set_xlabel('计划时段起点')
    ax1.grid(axis='y', alpha=0.15)

    ax2 = ax1.twinx()
    ax2.plot(mins, mean_q, color=C_RED, ls='--', lw=1.4, label='平均计划购电量(kWh)')
    ax2.set_ylabel('平均计划购电量 (kWh)', color=C_RED)
    ax2.tick_params(axis='y', labelcolor=C_RED)
    l1, lab1 = ax1.get_legend_handles_labels(); l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1+l2, lab1+lab2, loc='upper right', ncol=2, frameon=False)
    ax1.set_title('图10  计划购电零值率的日内分布（白天0值为最优调度结果）')
    save(fig, '图10_计划购电零值率日内分布.png')


def fig11_risk_compare():
    main_plan = METRICS['plan_purchase_cost_yuan']/1e6
    main_emg = METRICS['emergency_purchase_cost_yuan']/1e6
    base_plan = METRICS['risk_neutral_plan_cost_yuan']/1e6
    base_emg = METRICS['risk_neutral_emergency_cost_yuan']/1e6
    emg_energy = [METRICS['risk_neutral_emergency_kwh']/1e4, METRICS['emergency_purchase_kwh']/1e4]
    labels = ['风险中性\nλ=0', '最终方案\nλ=0.02']
    x = np.arange(2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    ax.bar(x, [base_plan, main_plan], color=C_BLUE, label='计划购电费')
    ax.bar(x, [base_emg, main_emg], bottom=[base_plan, main_plan], color=C_ORANGE, label='紧急购电费')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('购电费用 (百万元)')
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=0.15)
    ax.set_title('(a) 总费用构成')

    ax = axes[1]
    bars = ax.bar(x, emg_energy, color=[C_GRAY, C_GREEN])
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('紧急购电量 (10⁴ kWh)')
    ax.grid(axis='y', alpha=0.15)
    ax.set_title('(b) 紧急购电需求')
    for b, v in zip(bars, emg_energy):
        ax.text(b.get_x()+b.get_width()/2, b.get_height(), f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    ax.text(0.5, max(emg_energy)*0.78,
            f"成本 +{METRICS['risk_cost_change_vs_neutral_pct']*100:.3f}%\n紧急购电 −{METRICS['risk_emergency_reduction_pct']*100:.3f}%",
            ha='center', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#bbbbbb'))
    fig.suptitle('图11  CVaR风险方案与风险中性基线对比', y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / '图11_风险方案与风险中性方案对比.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def fig12_daily_emergency():
    df = DAILY.iloc[EVAL_START:].copy()
    df['rolling14'] = df['emergency_kwh'].rolling(14, min_periods=1).mean()
    top = df.nlargest(5, 'emergency_kwh')
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(df['date'], df['emergency_kwh'], width=1.0, color=C_ORANGE, alpha=0.45, label='每日紧急购电量')
    ax.plot(df['date'], df['rolling14'], color=C_RED, lw=1.6, label='14日滑动均值')
    for _, r in top.iterrows():
        ax.annotate(f"{r['date']:%m-%d}\n{r['emergency_kwh']:.0f}",
                    xy=(r['date'], r['emergency_kwh']), xytext=(0, 5), textcoords='offset points',
                    ha='center', fontsize=7.5)
    ax.set_ylabel('紧急购电量 (kWh)')
    ax.set_xlabel('日期')
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m月'))
    ax.legend(loc='upper left', ncol=2, frameon=False)
    ax.grid(axis='y', alpha=0.15)
    ax.set_title('图12  正式评价期每日紧急购电量及高风险日')
    save(fig, '图12_全年每日紧急购电与高风险日.png')


def render_table(filename, title, header, rows, figsize=(12, 4.8), fontsize=9.2, scale_y=1.55):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    ax.set_title(title, fontsize=12, pad=10)
    tb = ax.table(cellText=rows, colLabels=header, cellLoc='center', loc='center')
    tb.auto_set_font_size(False)
    tb.set_fontsize(fontsize)
    tb.scale(1, scale_y)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor('#bbbbbb')
        if r == 0:
            cell.set_facecolor('#f2f2f2')
            cell.set_text_props(weight='bold')
    fig.tight_layout()
    fig.savefig(FIG / filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def table3_plan():
    starts = [600, 720, 840, 960, 1080, 1200]
    idxs = [m//10 - 1 for m in starts]
    header = ['日期','10:00-10:10','12:00-12:10','14:00-14:10','16:00-16:10','18:00-18:10','20:00-20:10','全天购电量','全天购电费']
    rows = []
    for ds in SELECTED_DATES:
        d = int(np.where(DATES == pd.Timestamp(ds))[0][0])
        q = NPZ['plan_q'][d]
        vals = [f'{q[i]:.3f}' for i in idxs]
        rows.append([pd.Timestamp(ds).strftime('%Y.%m.%d'), *vals, f'{q.sum():.3f}', f'{NPZ["plan_cost"][d]:.3f}'])
    render_table('表3_指定日期计划购电量.png', '表3  指定日期、指定时段的计划购电量及全天购电量/购电费', header, rows,
                 figsize=(15, 3.8), fontsize=8.2, scale_y=1.7)


def table4_storage():
    header = ['日期','时间段','充电量(kWh)','放电量(kWh)','0:00 SOC(kWh)','24:00 SOC(kWh)']
    blocks = ['0:00-4:00','4:00-8:00','8:00-12:00','12:00-16:00','16:00-20:00','20:00-24:00']
    rows = []
    for ds in SELECTED_DATES:
        d = int(np.where(DATES == pd.Timestamp(ds))[0][0])
        for b, label in enumerate(blocks):
            sl = slice(24*b, 24*(b+1))
            rows.append([
                pd.Timestamp(ds).strftime('%Y.%m.%d') if b == 0 else '', label,
                f'{NPZ["charge"][d,sl].sum():.3f}', f'{NPZ["discharge"][d,sl].sum():.3f}',
                f'{NPZ["soc00"][d]:.3f}' if b == 0 else '', f'{NPZ["soc24"][d]:.3f}' if b == 0 else ''
            ])
    render_table('表4_指定日期储能充放电量.png', '表4  指定日期储能设备分时段充放电量及0:00/24:00储电量', header, rows,
                 figsize=(11.5, 11.2), fontsize=8.8, scale_y=1.35)


def table5_emergency():
    header = ['日期','紧急购电时间段','紧急购电量(kWh)']
    rows = []
    for ds in SELECTED_DATES:
        d = int(np.where(DATES == pd.Timestamp(ds))[0][0])
        events = grouped_emergency_periods(NPZ['emergency'][d])
        if not events:
            events = [('无', 0.0)]
        for k, (period, amount) in enumerate(events):
            rows.append([pd.Timestamp(ds).strftime('%Y.%m.%d') if k == 0 else '', period, f'{amount:.3f}'])
        rows.append(['', '合计', f'{NPZ["emergency"][d].sum():.3f}'])
    render_table('表5_指定日期紧急购电量.png', '表5  指定日期紧急购电时间段及购电量', header, rows,
                 figsize=(9, 10.5), fontsize=9.0, scale_y=1.4)


def write_index():
    text = f'''# 问题2图表索引与图注建议

> 本问题图表与问题1统一：使用 `output/图表/` 目录、相同中文字体/Tab10配色/150 dpi/单位写法；图号承接问题1图1—图3，表号承接问题1表1—表2。储能统一采用 $\\eta_c=\\eta_d=\\sqrt{{0.9}}$、$S^{{min}}=1200$ kWh、$S^{{max}}=10800$ kWh、$P^{{max}}=5000$ kW。

| 编号 | 文件 | 推荐图注/表注 | 建议位置 |
|---|---|---|---|
| 图4 | `图表/图4_问题2因果滚动调度框架.png` | 问题2严格因果滚动调度总体框架 | 问题2模型建立开头 |
| 图5 | `图表/图5_一月风险参数校准与敏感性.png` | 1月风险权重校准及经济性—可靠性权衡 | 参数确定/敏感性分析 |
| 图6 | `图表/图6_全年SOC轨迹与安全边界.png` | 正式评价期储能SOC连续轨迹及安全边界 | 结果分析 |
| 图7 | `图表/图7_月度计划购电与紧急购电.png` | 2025年2—12月计划购电与紧急购电月度分布 | 结果分析 |
| 图8 | `图表/图8_负荷光伏净负荷预测效果.png` | 负荷、光伏、净负荷因果预测的平均日内形态 | 预测误差分析 |
| 图9 | `图表/图9_典型日12月21日实时调度.png` | 2025-12-21高压力日计划、实时修正与SOC变化 | 典型日分析 |
| 图10 | `图表/图10_计划购电零值率日内分布.png` | 计划购电零值率及平均购电量的日内分布 | 解释result2白天大量0值 |
| 图11 | `图表/图11_风险方案与风险中性方案对比.png` | CVaR风险方案与风险中性基线对比 | 模型评价 |
| 图12 | `图表/图12_全年每日紧急购电与高风险日.png` | 正式评价期每日紧急购电量及高风险日 | 结果分析/附录 |
| 表3 | `图表/表3_指定日期计划购电量.png` | 指定日期、指定时段计划购电量及全天费用 | 题目要求结果 |
| 表4 | `图表/表4_指定日期储能充放电量.png` | 指定日期储能分时段充放电量与SOC | 题目要求结果 |
| 表5 | `图表/表5_指定日期紧急购电量.png` | 指定日期紧急购电时间段与购电量 | 题目要求结果 |

## 正文优先级

正文页数有限时，建议优先保留 **图4、图5、图6、图9、图11 + 表3—表5**。图7、图8、图10、图12可视篇幅放入结果分析或附录。

## 最终口径

- 正式评价：2025-02-01—2025-12-31，共334天；
- 9个等概率经验残差场景；$\\alpha=0.80$，$\\lambda=0.02$；
- 总购电费用：{METRICS['total_purchase_cost_yuan']:.2f}元；
- 紧急购电量：{METRICS['emergency_purchase_kwh']:.2f} kWh；
- 相对风险中性：成本 +{METRICS['risk_cost_change_vs_neutral_pct']*100:.3f}%，紧急购电量 −{METRICS['risk_emergency_reduction_pct']*100:.3f}%。
'''
    (OUT / '08_图表索引与图注.md').write_text(text, encoding='utf-8')


def main():
    fig4_flowchart()
    fig5_risk_calibration()
    fig6_soc()
    fig7_monthly_energy()
    fig8_forecast()
    fig9_typical_day()
    fig10_zero_rate()
    fig11_risk_compare()
    fig12_daily_emergency()
    table3_plan()
    table4_storage()
    table5_emergency()
    write_index()
    files = sorted(FIG.glob('*.png'))
    print('已生成', len(files), '张PNG图表到', FIG)
    for p in files:
        print(p.name, p.stat().st_size)


if __name__ == '__main__':
    main()
