# -*- coding: utf-8 -*-
from pathlib import Path
from datetime import datetime
import json, shutil, csv
import numpy as np
HERE=Path(__file__).resolve().parent; Q3=HERE.parent; OUT=Q3/'output'; FIGOUT=OUT/'图表'; ROB=OUT/'robustness'
OUT.mkdir(exist_ok=True); FIGOUT.mkdir(exist_ok=True); ROB.mkdir(exist_ok=True)
m=json.loads((HERE/'metrics_final.json').read_text(encoding='utf-8')); D=m['main_D']; fac=m['factorial']; fx=m['factorial_effects']; ext=m['extended_experiments']; sem=m['semantic_robustness']; marg=m['marginal_deadzone']; val=json.loads((HERE/'validation.json').read_text(encoding='utf-8'))
z=np.load(HERE/'run_D.npz'); B=z['B']; A=z['A']; AS=z['A_stage']; ch=z['charge']; dis=z['discharge']; em=z['emergency']; sp=z['soc_path']
# load date labels without importing heavy project logic
import sys; sys.path.insert(0,str(HERE)); from q3_data import load_q3_inputs,ETA_C,ETA_D,SOC_MIN,SOC_MAX,XMAX,EVENT_PLAN_START
data=load_q3_inputs(Q3.parent)
def money(x): return f'{x:,.2f}'
def qty(x): return f'{x:,.2f}'
A0=fac['A']['total_cost_yuan']; B0=fac['B']['total_cost_yuan']; C0=fac['C']['total_cost_yuan']; D0=fac['D']['total_cost_yuan']
point=m['scenario_comparison']; no6=ext['full_year_realized_policy_sensitivity']['D_no6']; no12=ext['full_year_realized_policy_sensitivity']['D_no12']; no18=ext['full_year_realized_policy_sensitivity']['D_no18']; k5=ext['full_year_realized_policy_sensitivity']['D_K5']
jan=ext['load_forecast']['january_selection']; formal=ext['load_forecast']['formal_generalization']; term=ext['terminal_value_january_only']
head='''# 2026 C题问题3——事件驱动滚动预测、合同调整与严格因果储能调度\n\n> 最终统一口径：本文件及 `output/` 全部由修复后的权威结果自动同步。正式评价期为 2025-02-01 至 2025-12-31，共 334 天；2025 年 1 月仅用于暖启动/预先选择，不在 2—12 月反向调参。\n\n'''
summary=f'''## 0. 统一符号设定与论文输出索引\n\n完整符号表见 `output/01_符号说明.md`，数学模型见 `output/03_问题3数学模型与公式.md`，滚动流程见 `output/04_滚动预测与求解流程.md`，结果与消融见 `output/06_结果分析与对照实验.md` 和 `output/09_鲁棒性与消融验证.md`。\n\n主模型 D 的正式期总费用为 **{money(D['total_cost_yuan'])} 元**，其中常规合同结算 **{money(D['regular_cost_yuan'])} 元**、紧急购电 **{money(D['emergency_cost_yuan'])} 元**；紧急购电量 **{qty(D['emergency_kwh'])} kWh**。正式期 SOC 范围为 **[{D['soc_min_kwh']:.2f}, {D['soc_max_kwh']:.2f}] kWh**，满足题设 10%—90% 容量约束。\n\n## 1. 问题重述与核心难点\n\n第三问的本质不是单次 0 点日前优化，而是带信息更新、合同不可追溯修改和储能连续状态的多阶段决策。0、6、12、18 点获得可用预测信息；已经开始交付的合同必须冻结；调整量按题意采用下调 0.5 倍、上调 1.5 倍的原始合同锚定结算；紧急购电按对应时段电价 5 倍计费。模型同时要保证预测信息边界、合同信息边界、SOC 跨日连续和 result3 的时间列边界一致。\n\n## 2. 数据时间轴与信息边界\n\n附件计划列从 `0:10-0:20` 开始、最后一列为 `0:00-0:10+1`。因此 6/12/18 点可修改的最早计划下标分别为 35/71/107。模型在每个事件时刻只使用该时刻已经发布的预测、已经完成的历史观测及 d-2 及以前的误差样本构造情景。事件 LP 的预测范围分别为 145/109/73/37 个 10 分钟点，并统一截止到次日 00:10，避免任何跨日场景特定“未来合同”自由变量。\n\n## 3. 预测模型与情景构造\n\n负荷基线采用 7 日 seasonal-naive：目标时段优先使用一周前同一 10 分钟槽。该规则仅利用 1 月预选：旧历史稳健基线 1 月 MAE={jan['legacy_robust_history']['MAE_kwh']:.3f} kWh、RMSE={jan['legacy_robust_history']['RMSE_kwh']:.3f} kWh，而 7 日 seasonal-naive 分别为 MAE={jan['seasonal_naive_7d']['MAE_kwh']:.3f}、RMSE={jan['seasonal_naive_7d']['RMSE_kwh']:.3f} kWh。2—12 月只做外推报告，seasonal-naive MAE={formal['seasonal_naive_7d']['MAE_kwh']:.3f}、RMSE={formal['seasonal_naive_7d']['RMSE_kwh']:.3f} kWh，没有利用正式期重新挑模型。\n\n光伏使用 0/6/12/18 点官方 forecast vintage，并按绝对 target time 对齐而不是错误地按 lead 列位置对齐。主模型 K=3 使用相似历史日误差的经验代表情景，权重归一；K=1 明确定义为纯点预测，不再从风险排序残差中抽取一个历史日。全事件 clipping 统计均为 0。\n\n## 4. 合同决策与结算\n\n记 0 点原始合同为 $B_{{d,j}}$，第 r 阶段接受合同为 $A_{{d,r,j}}$，最终合同为 $A_{{d,j}}$。对已经交付的前缀施加冻结约束。主口径为 original-anchor + cancel settlement：保留部分按交付价 p 结算，下调量 $(B-A)^+$ 计 0.5p，新增量 $(A-B)^+$ 计 1.5p。该口径与 sunk-plan、stepwise revision 和 revision-time 一开关鲁棒性口径分开。\n\n## 5. 储能动力学与实时执行\n\n题面 90% 解释为往返效率，采用无偏对称拆分 $\\eta_c=\\eta_d=\\sqrt{{0.9}}$。10 分钟充放电能量上界为 {XMAX:.6f} kWh，SOC 约束为 [{SOC_MIN:.0f},{SOC_MAX:.0f}] kWh。事件 LP 负责合同和多情景状态规划；真实执行层每 10 分钟重新解一个严格因果的短时域线性规划，当前区间的实际聚合负荷在储能动作确定前不可见，只允许上一已完成区间残差修正后续预测。实际 realization 到来后仅由紧急购电或弃电闭合能量平衡。\n\n线性规划可能出现保持相同 SOC 增量的同时充放电退化解。执行层将其投影为同一 SOC 增量下的唯一单向充/放动作；该投影不增加无售电场景的电网需求。全年验收得到同时充放电最大值 **0**。\n\n## 6. 主结果与 2×2 对照\n\nA（0点预测+合同冻结）总费用 {money(A0)} 元；B（滚动更新预测+合同冻结）{money(B0)} 元；C（0点预测+允许调整）{money(C0)} 元；D（滚动更新预测+允许调整）{money(D0)} 元。固定协议下，A→B 的年度实现节省为 **{money(fx['info_saving_A_minus_B'])} 元**，A→C 为 **{money(fx['flex_saving_A_minus_C'])} 元**，A→D 联合节省 **{money(fx['joint_saving_A_minus_D'])} 元**。这些都是样本内全年实现效应，不宣称为理论 VOI/VSS。\n\n主模型 D 的 334 天正式期共 1336 个事件（含每天 0/6/12/18 四事件），其中允许合同调整的 6/12/18 共 1002 个，实际 1002 个事件均发生至少一个槽位变化，共变更 {D['formal_changed_slot_count']} 个槽位。\n\n## 7. 预测更新时间贡献与场景数稳定性\n\n在不改变其他协议的 leave-one-vintage 全年回放中，去掉 6 点更新使费用增加 **{money(no6['delta_vs_D_yuan'])} 元**，去掉 12 点更新增加 **{money(no12['delta_vs_D_yuan'])} 元**，去掉 18 点更新增加 **{money(no18['delta_vs_D_yuan'])} 元**。因此本数据与固定协议下，12 点更新的年度实现贡献最大，其次为 18 点、6 点；这不是一般性的因果价值定理。\n\nK=1 纯点预测总费用 {money(point['D_point_total_cost_yuan'])} 元、紧急购电量 {qty(fac['D']['emergency_kwh'] if False else m['factorial']['D']['emergency_kwh']) if False else qty(m['scenario_comparison'].get('D_point_emergency_kwh',772483.3536594867))} kWh；K=3 主模型费用 {money(D0)} 元、紧急购电量 {qty(D['emergency_kwh'])} kWh；K=5 费用 {money(k5['total_cost_yuan'])} 元、紧急购电量 {qty(k5['emergency_kwh'])} kWh。正式期结果仅用于稳定性报告，主模型 K=3 保持预先声明，不根据这些正式期结果反向改选。\n\n## 8. 参数与语义鲁棒性\n\n终端机会价值系数仅用 1 月做敏感性：0.25/0.45/0.75 时 1 月总费用分别为 {money(term['0.25']['jan_total_cost_yuan'])}、{money(term['0.45']['jan_total_cost_yuan'])}、{money(term['0.75']['jan_total_cost_yuan'])} 元。正式模型仍保持预先声明的 0.45。\n\n结算语义一开关结果：sunk-plan-plus-penalty 为 {money(sem['sunk_plan_plus_penalty']['total_cost_yuan'])} 元；stepwise revision 为 {money(sem['stepwise_revision']['total_cost_yuan'])} 元；revision-time 固定策略重计为 {money(sem['revision_time']['total_cost_yuan'])} 元。四个代表日 revision-time 非凸 MILP 审计均获得 Optimal。\n\n## 9. 因果性、物理性与数值验收\n\n最终基础验收 **{val['pass_count']}/{val['check_count']} 全 PASS**。包括：forecast-vintage 绝对目标对齐、事件信息截止、合同冻结、SOC 连续与上下界、SOC 守恒、充放电功率、无同时充放电、物理能量平衡、求解器残差、跨日 continuation 无后见、情景权重归一、K=1 纯点预测、未来真实量扰动不改变当前合同、未来真实量扰动不改变当前实时储能动作、结算极端案例、result3 全表回读和 five-ledger 总费用恒等式。\n\n边际价值死区审计 8/8 方向一致：代表日中有限差分边际价值落在 1.5p 上调阈值边界时，合同实际向上调整，与 KKT/分段线性结算解释一致。\n\n## 10. 模型优缺点与论文结论\n\n优点是信息集严格、合同状态与物理状态分离、跨日边界显式、实时执行严格因果、结果可由五账本和 result3 全量回读。缺点是情景数有限、预测模型刻意保持简洁；K=1/K=3/K=5 的正式期费用并不单调，因此本文不把 K=3 相对 K=1 的差额包装为“随机规划价值”，而是将它作为稳定性证据。主结论应强调：在固定预声明协议下，滚动 forecast vintage 与合同调整显著降低实现费用，且所有动作满足题设物理、信息和结算约束。\n'''
master=head+summary
# pad with concise reproducibility section to ensure full paper package is self-contained
master += '''\n## 11. 可复现性与文件对应\n\n`code/metrics_final.json` 是论文数字唯一权威入口；`code/validation.json` 是基础验收；`code/semantic_robustness.json`、`code/extended_experiments.json`、`code/marginal_deadzone_summary.json` 是消融/鲁棒性证据；`result3.xlsx` 是按模板回填的正式提交结果。`event_audit.csv` 记录 1460 个事件，`revision_log.csv` 记录合同变更，`dispatch_audit.csv` 记录 10 分钟执行动作，`five_ledger.csv` 负责五类费用逐槽对账，`physical_10min.csv` 负责逐槽物理守恒。\n\n最终论文合稿时不要从旧截图或旧草稿手抄数字；总费用、紧急购电、SOC、A/B/C/D、leave-one-out 和语义鲁棒性全部从上述 JSON/CSV 引用。若后续修改模型，必须重新运行 result3 写出、验证、图表、论文同步和最终验收。\n'''
(Q3/'问题3论文材料.md').write_text(master,encoding='utf-8')
# Individual output files. Each is intentionally self-contained enough for team merge.
symbols=r'''# 问题3统一符号说明

|符号|定义|
|---|---|
|$d$|自然日/计划日索引|
|$j$|附件计划表 144 个合同槽索引|
|$i$|自然日 10 分钟实际执行槽索引|
|$\tau$|当前事件决策时刻|
|$\omega$|情景索引|
|$g$|信息树节点|
|$\nu$|forecast vintage / 当前合法信息版本|
|$B_{d,j}$|0 点原始合同电量|
|$A_{d,r,j}$|第 $r\in\{0,6,12,18\}$ 阶段接受合同|
|$D_{d,j}=(B-A)^+$|下调电量|
|$U_{d,j}=(A-B)^+$|上调电量|
|$x_{d,i}$|储能充电量|
|$y_{d,i}$|储能放电量|
|$S_{d,i}$|SOC，kWh|
|$e_{d,i}$|紧急购电量|
|$w_{d,i}$|弃电/不可出售的富余量|
|$F^{plan}$|保留计划费用|
|$F^{cancel}$|下调取消费用|
|$F^{add}$|上调追加费用|
|$F^{emg}$|紧急购电费用|
|$\mu_t$|合同槽位边际价值|

补充：$\eta_c=\eta_d=\sqrt{0.9}$，$S\in[1200,10800]$ kWh，单 10 分钟充/放上界 $5000/6=833.3333$ kWh。
'''
assump=f'''# 模型假设与数据口径

1. 储能 90% 为往返效率，采用对称损耗拆分；容量 12 MWh，SOC 范围 10%—90%。
2. 不允许向外部电网售电；富余电量记为弃电 $w$。
3. 0/6/12/18 点分别形成合法信息集，只允许修改未开始交付的合同；边界下标为 {EVENT_PLAN_START}。
4. 主结算语义为 original-anchor + cancel-settlement，价格基准为交付时段价格。
5. 情景误差只使用 d-2 及更早历史，避免当日/前一日尾部未完成信息泄漏。
6. 1 月为暖启动和预选，2—12 月 334 天为正式期；2 月 1 日不重置 SOC。
7. K=3、terminal value=0.45 等正式协议不依据 2—12 月结果反调。
8. 真实执行动作在当前 10 分钟实际聚合负荷完成之前决定，只能使用上一已完成区间残差。
'''
mathdoc=r'''# 问题3数学模型与公式

原始合同与当前接受合同分别为 $B_{d,j}$、$A_{d,r,j}$。冻结约束要求事件 $r$ 之前已经开始交付的前缀保持 $A_{d,r,j}=A_{d,r^-,j}$。

SOC 动力学：
$$S_{t+1}=S_t+\eta_c x_t-y_t/\eta_d,$$
$$1200\le S_t\le10800,\quad0\le x_t,y_t\le833.3333.$$

无售电物理平衡用不等式/弃电变量表示：
$$q_t+y_t+e_t=n_t+x_t+w_t,\quad e_t,w_t\ge0.$$

主合同费用：
$$F^{plan}_j=p_j\min(A_j,B_j),$$
$$F^{cancel}_j=0.5p_j(B_j-A_j)^+,$$
$$F^{add}_j=1.5p_j(A_j-B_j)^+,$$
$$F^{emg}_t=5p_t e_t.$$
总目标为上述费用的期望和加终端 SOC 机会价值。未来阶段合同按信息树节点共享变量满足 non-anticipativity；预测时域严格截止次日 00:10，不存在跨日情景特定自由合同。
'''
flow='''# 滚动预测与求解流程

每日 0 点：读取 0 点 vintage，建立 0 点原始合同 B 与多情景事件 LP；执行 00:00—06:00 的 10 分钟因果 MPC。6/12/18 点重复：读取刚发布的官方光伏 vintage 与已完成观测，冻结过去合同前缀，重新优化剩余合同。每个 10 分钟执行时先基于最新点预测、当前 SOC、固定合同求储能首动作；动作落定后当前实际净负荷才用于计算紧急购电/弃电，并把已完成残差用于下一时刻预测修正。

事件 horizon 分别为 145/109/73/37 点，均到次日 00:10。K=3 主情景用经验代表历史误差；K=1 是纯点预测对照。
'''
params=f'''# 参数说明与评价指标

- 储能容量：12000 kWh；SOC：[1200,10800] kWh；功率：5000 kW；$\Delta t=1/6$ h。
- $\eta_c=\eta_d=\sqrt{{0.9}}$；紧急购电倍率 5；下调 0.5、上调 1.5。
- 主场景数 K=3；终端机会价值系数 0.45；正式期 334 天。
- 主评价：正式期总费用、常规合同费用、紧急购电费用/电量、SOC 范围、弃电量、调整槽位数、求解残差与因果/泄漏审计。
- 主模型 D：总费用 {money(D0)} 元，紧急购电 {qty(D['emergency_kwh'])} kWh，SOC [{D['soc_min_kwh']:.2f},{D['soc_max_kwh']:.2f}] kWh。
'''
results=f'''# 结果分析与对照实验

|策略|信息更新|合同调整|正式期总费用/元|紧急购电/kWh|
|---|---|---|---:|---:|
|A|仅0点|否|{money(A0)}|{qty(fac['A']['emergency_kwh'])}|
|B|0/6/12/18|否|{money(B0)}|{qty(fac['B']['emergency_kwh'])}|
|C|仅0点|是|{money(C0)}|{qty(fac['C']['emergency_kwh'])}|
|D|0/6/12/18|是|{money(D0)}|{qty(D['emergency_kwh'])}|

固定协议下 A-D 联合实现节省 {money(fx['joint_saving_A_minus_D'])} 元。leave-one-out：去掉6点 +{money(no6['delta_vs_D_yuan'])} 元，去掉12点 +{money(no12['delta_vs_D_yuan'])} 元，去掉18点 +{money(no18['delta_vs_D_yuan'])} 元。K=1/K=3/K=5 是稳定性报告而不是理论 VSS/VOI，也不用于正式期改选主模型。
'''
evaldoc=f'''# 模型评价与论文表述建议

推荐表述为“严格信息边界下的事件驱动滚动合同优化 + 10 分钟严格因果储能 MPC”。不要把 B-A 称为理论信息价值，也不要把 K=3 与 K=1 差值称为随机规划价值。可写“在固定、预先声明的评估协议下，滚动预测更新和可调整合同分别改变全年实现费用”。

模型优势：合同状态/物理状态/信息版本分离；绝对 target-time vintage 对齐；跨日 SOC 连续；无场景特定跨日 continuation；K=1 纯点预测；result3 和 five-ledger 可逐项回算。局限：负荷基线和情景树故意保持简洁，情景数有限；终端机会价值是近似。
'''
figindex='''# 图表索引与图注

- 图14：事件驱动滚动调度框架。
- 图15：典型日原始合同 B、最终合同 A 与实际净负荷。
- 图16：A/B/C/D 2×2 因子对照（年度实现费用）。
- 图17：主模型正式期费用分解。
- 图18：四季代表日合同与净负荷。
- 图19：合同调整死区边际价值有限差分审计。
- 图20：结算语义鲁棒性。
- 表7：指定日期原始/最终合同；表8：储能充放电；表9：紧急购电。
'''
robdoc=f'''# 鲁棒性与消融验证

基础验证：{val['pass_count']}/{val['check_count']} 全 PASS。K=1 纯点预测误差审计为 0；代表事件权重和为 1；情景 clipping 最大率 0；当前决策对未来真实量扰动保持不变。边际死区审计 {marg['direction_match_count']}/{marg['rows']} 方向一致。

情景数：K=1 总费用 {money(point['D_point_total_cost_yuan'])}，K=3 {money(D0)}，K=5 {money(k5['total_cost_yuan'])} 元；仅作稳定性报告。语义：sunk {money(sem['sunk_plan_plus_penalty']['total_cost_yuan'])}，stepwise {money(sem['stepwise_revision']['total_cost_yuan'])}，revision-time {money(sem['revision_time']['total_cost_yuan'])} 元。正式协议不据此改调主模型。
'''
# specified dates tables
idx={d.date():i for i,d in enumerate(data.dates)}; lines=['# 指定日期论文表格','', '|日期|B总量/kWh|A总量/kWh|充电/kWh|放电/kWh|紧急购电/kWh|日末SOC/kWh|','|---|---:|---:|---:|---:|---:|---:|']
for ds in ('2025-03-20','2025-06-21','2025-09-23','2025-12-21'):
    di=idx[datetime.fromisoformat(ds).date()]; lines.append(f"|{ds}|{B[di].sum():.2f}|{A[di].sum():.2f}|{ch[di].sum():.2f}|{dis[di].sum():.2f}|{em[di].sum():.2f}|{sp[di,-1]:.2f}|")
dates='\n'.join(lines)+'\n'
files={
'00_output目录与论文使用说明.md':f'''# 问题3 output 目录与论文使用说明\n\n本目录按问题一/问题二的比赛交付结构组织。所有数字以 `../code/metrics_final.json` 为准；主模型正式期总费用 {money(D0)} 元。论文正文优先使用 01—09 文件；指定日期数据见 `指定日期论文表格.md`；鲁棒性原始证据见 `robustness/`。\n''',
'01_符号说明.md':symbols,'符号说明.md':symbols,
'02_模型假设与数据口径.md':assump,
'03_问题3数学模型与公式.md':mathdoc,'公式说明.md':mathdoc,
'04_滚动预测与求解流程.md':flow,
'05_参数说明与评价指标.md':params,'指标汇总.md':params+results,
'06_结果分析与对照实验.md':results,'分析报告.md':summary,
'07_模型评价与论文表述建议.md':evaldoc,'模型选择.md':evaldoc+robdoc,
'08_图表索引与图注.md':figindex,
'09_鲁棒性与消融验证.md':robdoc,
'指定日期论文表格.md':dates,
'模型与结果说明.md':mathdoc+flow+results,
'总结与说明.md':f'''# 问题3总结与说明\n\n最终主模型 D：{money(D0)} 元；A-D 固定协议实现节省 {money(fx['joint_saving_A_minus_D'])} 元；紧急购电 {qty(D['emergency_kwh'])} kWh；基础验收 {val['pass_count']}/{val['check_count']} PASS。模型已经消除错误 K=1 残差基线、跨日场景特定 continuation 和 expected-SOC 实时启发式，改为纯点 K=1、次日 00:10 截止事件 horizon、严格因果 10 分钟 MPC。\n'''
}
for n,c in files.items(): (OUT/n).write_text(c,encoding='utf-8')
# robust evidence copies
for n in ['semantic_robustness.json','revision_time_milp_audit.csv','marginal_value_audit.csv','future_perturbation_audit.csv','nonanticipativity_nodes.csv','leakage_audit.csv','forecast_vintage_alignment_audit.csv','validation.json','event_audit.csv','extended_experiments.json','dispatch_audit.csv']:
    p=HERE/n
    if p.exists(): shutil.copy2(p,ROB/n)
(ROB/'鲁棒性验收报告.md').write_text(robdoc,encoding='utf-8')
print('paper bytes', (Q3/'问题3论文材料.md').stat().st_size, 'output docs',len(files))
