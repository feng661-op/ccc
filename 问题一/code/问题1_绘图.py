# -*- coding: utf-8 -*-
"""
问题1 绘图
==========
生成 5 张图，存入 output/图表/：
  图1_电价与负载光伏.png      电价曲线 + 负载/光伏功率
  图2_储能调度与SOC轨迹.png   充放电柱状图 + SOC 轨迹
  图3_灵敏度对比.png          灵敏度分析费用对比
  表1_指定时段购电量.png
  表2_储能充放电量.png
"""
import os
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from 数据读取与预处理 import load_data           # noqa: E402
from 问题1_模型求解 import build_and_solve       # noqa: E402

BASE = r'D:\数学建模\26C题'
CODE = os.path.join(BASE, 'C题交付', 'code')
FIG = os.path.join(BASE, 'C题交付', 'output', '图表')

# 中文字体
for _f in ['Microsoft YaHei', 'SimHei', 'SimSun']:
    if any(f.name == _f for f in font_manager.fontManager.ttflist):
        plt.rcParams['font.sans-serif'] = [_f]
        break
plt.rcParams['axes.unicode_minus'] = False


def mm_to_label(m):
    h = int(m) // 60
    mm = int(m) % 60
    return f'{h:02d}:{mm:02d}'


def set_time_axis(ax):
    """x 轴每 2 小时一个刻度（0:00 ~ 24:00）。"""
    ticks = list(range(0, 1441, 120))
    ax.set_xticks(ticks)
    ax.set_xticklabels([mm_to_label(t) for t in ticks], rotation=0)
    ax.set_xlim(0, 1440)


def fig1(data):
    times = np.array(data['times'])
    price = np.array(data['price'])
    load = np.array(data['load_kw'])
    pv = np.array(data['pv_kw'])

    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(times, price, color='#d62728', lw=1.8, label='电价(元/kWh)')
    ax1.set_ylabel('电价 (元/kWh)', color='#d62728')
    ax1.tick_params(axis='y', labelcolor='#d62728')
    ax1.set_xlabel('时间')
    set_time_axis(ax1)

    ax2 = ax1.twinx()
    ax2.plot(times, load, color='#1f77b4', lw=1.3, label='小区负载(kW)')
    ax2.plot(times, pv, color='#ff7f0e', lw=1.3, label='光伏预测功率(kW)')
    ax2.set_ylabel('功率 (kW)')
    ax2.set_ylim(0, max(load.max(), pv.max()) * 1.1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', ncol=3, frameon=False)
    ax1.set_title('图1  电价与负载、光伏预测功率')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, '图1_电价与负载光伏.png'), dpi=150)
    plt.close(fig)


def fig2(data, sol):
    x = np.array(sol['x'])
    y = np.array(sol['y'])
    s = np.array(sol['s'])
    # 柱状图横轴用区间起点 0,10,...,1430 分钟；SOC 对应时间点 0:00..24:00（145个点）
    bar_times = np.arange(0, 1440, 10)
    s_times = np.linspace(0, 1440, len(s))

    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.bar(bar_times, x, width=8, color='#2ca02c', label='充电量(kWh)')
    ax1.bar(bar_times, -y, width=8, color='#d62728', label='放电量(kWh)')
    ax1.set_ylabel('充放电量 (kWh/10min)')
    ax1.set_xlabel('时间')
    set_time_axis(ax1)
    ax1.axhline(0, color='black', lw=0.6)

    ax2 = ax1.twinx()
    ax2.plot(s_times, s, color='#9467bd', lw=1.8, label='SOC(kWh)')
    ax2.axhline(6000, color='gray', lw=0.8, ls='--')
    ax2.set_ylabel('储电量 SOC (kWh)')
    ax2.set_ylim(0, 12000)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', ncol=3, frameon=False)
    ax1.set_title('图2  储能充放电调度与 SOC 轨迹')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, '图2_储能调度与SOC轨迹.png'), dpi=150)
    plt.close(fig)


def render_table(filename, title, header, rows, col_widths=None):
    fig, ax = plt.subplots(figsize=(8, 0.6 * (len(rows) + 1)))
    ax.axis('off')
    ax.set_title(title, fontsize=12, pad=10)
    tb = ax.table(cellText=rows, colLabels=header, cellLoc='center', loc='center')
    tb.auto_set_font_size(False)
    tb.set_fontsize(10)
    tb.scale(1, 1.6)
    for key, cell in tb.get_celld().items():
        cell.set_edgecolor('#bbbbbb')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, filename), dpi=150, bbox_inches='tight')
    plt.close(fig)


def table1(metrics):
    sel = metrics['表1_指定时段购电量_kWh']
    header = ['时间段', '购电量(kWh)']
    rows = [[k, f"{v:.4f}"] for k, v in sel.items()]
    rows.append(['全天购电量', f"{metrics['全天购电量_kWh']:.4f}"])
    rows.append(['全天购电费(元)', f"{metrics['全天购电费_元']:.4f}"])
    render_table('表1_指定时段购电量.png', '表1  微网在指定时段的购电量及全天购电量/购电费', header, rows)


def table2(metrics):
    blk = metrics['表2_充放电量_kWh']
    header = ['时间段', '充电量(kWh)', '放电量(kWh)', '时刻', '储电量(kWh)']
    rows = []
    for i, (k, v) in enumerate(blk.items()):
        if i == 0:
            rows.append([k, f"{v['充电量']:.4f}", f"{v['放电量']:.4f}", '0:00', '6000.0000'])
        elif i == 1:
            rows.append([k, f"{v['充电量']:.4f}", f"{v['放电量']:.4f}", '24:00', '6000.0000'])
        else:
            rows.append([k, f"{v['充电量']:.4f}", f"{v['放电量']:.4f}", '', ''])
    render_table('表2_储能充放电量.png', '表2  储能设备在指定时段的充放电量及0:00/24:00储电量', header, rows)


def fig3_sensitivity():
    p = os.path.join(CODE, 'sensitivity.json')
    if not os.path.exists(p):
        return
    with open(p, encoding='utf-8') as f:
        sd = json.load(f)
    cases = [r['case'] for r in sd['results']]
    costs = [r['cost'] for r in sd['results']]
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ['#1f77b4'] + ['#ff7f0e'] * 2 + ['#2ca02c'] * 2 + ['#9467bd'] * 2
    ax.bar(range(len(cases)), costs, color=colors)
    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels(cases, rotation=15, ha='right', fontsize=8)
    ax.set_ylabel('全天购电费 (元)')
    ax.set_title('图3  灵敏度分析：参数扰动对购电费的影响')
    for i, c in enumerate(costs):
        ax.text(i, c, f'{c:.0f}', ha='center', va='bottom', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, '图3_灵敏度对比.png'), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    data = load_data()
    sol = build_and_solve(data)
    with open(os.path.join(CODE, 'metrics.json'), encoding='utf-8') as f:
        metrics = json.load(f)

    fig1(data)
    fig2(data, sol)
    table1(metrics)
    table2(metrics)
    fig3_sensitivity()
    print('图表已生成到', FIG)


if __name__ == '__main__':
    main()
