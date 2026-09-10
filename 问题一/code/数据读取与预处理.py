# -*- coding: utf-8 -*-
"""
数据读取与预处理
================
- 读取附件1（某天的电价、小区负载、光伏发电预测功率）
- 统一解析时间列（datetime.time 与字符串混合：0:10, ..., 23:50, 0:00+1）
- 功率(kW) -> 每10分钟能量(kWh)：除以 6（10分钟 = 1/6 小时）

区间约定：段 i = [i*10min, (i+1)*10min]，i = 0..143。
数据点（样本）时刻 T 表示区间 [T, T+10min]（样本=区间起点，与 result1 模板标签一致）：
  样本 0:10        -> 区间 [0:10, 0:20]
  样本 0:00+1(=24:00) -> 区间 [24:00, 24:10] = [0:00, 0:10]（日周期 24:00≡0:00）
故 段 i 使用样本 (i-1) mod 144：求解前对 load/pv/price 做一次循环右移(np.roll(·, 1))。
"""
import os
import datetime
import openpyxl

BASE = r'D:\数学建模\26C题'
ATTACH1 = os.path.join(BASE, '附件', '附件1.xlsx')


def parse_time_to_minutes(t):
    """把 时间 单元格解析为 分钟数 (0:00=0, 24:00=1440)。"""
    if isinstance(t, datetime.time):
        return t.hour * 60 + t.minute
    if isinstance(t, datetime.datetime):
        return t.hour * 60 + t.minute
    if isinstance(t, str):
        s = t.strip()
        if s == '0:00+1':
            return 1440
        hh, mm = s.split(':')
        return int(hh) * 60 + int(mm)
    raise ValueError(f'无法解析时间: {t!r}')


def load_data(path=ATTACH1):
    """返回字典：times(分钟), price(元/kWh), load_kw/pv_kw(kW), load/pv(kWh/段)。"""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb['Sheet1']
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    wb.close()

    times, price, load_kw, pv_kw = [], [], [], []
    for r in rows:
        times.append(parse_time_to_minutes(r[0]))
        price.append(float(r[1]))
        load_kw.append(float(r[2]))
        pv_kw.append(float(r[3]))

    n = len(times)
    return {
        'n': n,
        'times': times,                # 分钟数
        'price': price,                # 元/kWh
        'load_kw': load_kw,            # kW
        'pv_kw': pv_kw,                # kW (光伏预测)
        'load': [v / 6.0 for v in load_kw],   # kWh/段
        'pv': [v / 6.0 for v in pv_kw],        # kWh/段
    }


if __name__ == '__main__':
    d = load_data()
    print(f'段数 = {d["n"]}')
    print(f'电价   min={min(d["price"]):.4f} max={max(d["price"]):.4f} 元/kWh')
    print(f'负载   min={min(d["load_kw"]):.1f} max={max(d["load_kw"]):.1f} kW')
    print(f'光伏预测 max={max(d["pv_kw"]):.1f} kW')
    print(f'光伏>负载 段数 = {sum(1 for p, l in zip(d["pv_kw"], d["load_kw"]) if p > l)}')
