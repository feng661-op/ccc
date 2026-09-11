# -*- coding: utf-8 -*-
"""
问题1 灵敏度分析
================
对两个关键参数做单因素扰动，观察全天购电费变化：
  1. 充放电效率（往返90%）：对称√0.9(基准) / 单边0.9/1.0 / 81%(充放各90%)
  2. 储能容量：12000(基准) / 9600 / 14400 kWh（SOC 上限随容量变化）
  3. 充放电功率上限：5000(基准) / 4000 / 6000 kW

输出：code/sensitivity.json
"""
import os
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from 数据读取与预处理 import load_data           # noqa: E402
from 问题1_模型求解 import build_and_solve, XMAX, SOC_MIN  # noqa: E402

BASE = r'D:\数学建模\26C题'
CODE = os.path.join(BASE, '问题一', 'code')


def run():
    data = load_data()
    results = []

    # 基准
    base = build_and_solve(data)
    base_cost = base['cost']
    results.append({'case': '基准(往返90%对称√0.9, 容量12000, 功率5000)',
                    'cost': round(base_cost, 2), 'delta_%': 0.0})

    def add(name, **kw):
        r = build_and_solve(data, **kw)
        results.append({'case': name, 'cost': round(r['cost'], 2),
                        'delta_%': round(100 * (r['cost'] - base_cost) / base_cost, 3)})

    # 1) 效率（往返90%的不同拆分 + 往返81%）
    add('效率 单边拆分(充0.9放1.0)', eta_c=0.9, eta_d=1.0)
    add('效率 往返81%(充放各90%)', eta_c=0.9, eta_d=0.9)

    # 2) 容量 ±20%（SOC上限 = 容量 - 1200）
    add('容量 80% (9600 kWh)', soc_max=9600 - 1200)
    add('容量 120% (14400 kWh)', soc_max=14400 - 1200)

    # 3) 功率 ±20%
    add('功率 80% (4000 kW)', xmax=4000 / 6.0)
    add('功率 120% (6000 kW)', xmax=6000 / 6.0)

    out = {'基准费用_元': round(base_cost, 2), 'results': results}
    with open(os.path.join(CODE, 'sensitivity.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f'基准费用 = {base_cost:.2f} 元')
    for r in results:
        print(f'  {r["case"]:32s}  费用={r["cost"]:>10.2f}  变化={r["delta_%"]:+.3f}%')


if __name__ == '__main__':
    run()
