"""Generate every Q3 paper view from the corrected model and current evidence."""
from pathlib import Path
from datetime import datetime
import csv
import json
import shutil
import numpy as np
from q3_data import load_q3_inputs, grouped_emergency_periods

HERE=Path(__file__).resolve().parent;Q3=HERE.parent;OUT=Q3/'output';ROB=OUT/'robustness'
m=json.loads((HERE/'metrics_final.json').read_text(encoding='utf-8'))
D=m['main_D'];fac=m['factorial'];fx=m['factorial_effects'];ext=m['extended_experiments'];sem=m['semantic_robustness']
cross=m['cross_problem'];old=cross['original_audit'];v=m['validation_summary'];phys=m['physical_fix']
z=np.load(HERE/'run_D.npz');data=load_q3_inputs(Q3.parent)
money=lambda x:f'{x:,.2f}'
names={'A':'继承第二问，关闭新增预报和调整','B':'使用新增预报，合同保持不变','C':'沿用历史预报，允许合同调整','D':'使用新增预报并允许合同调整'}
table='|策略|定义|总费用／元|紧急购电／千瓦时|\n|---|---|---:|---:|\n'
for n in 'ABCD':table+=f"|{n}|{names[n]}|{money(fac[n]['total_cost_yuan'])}|{money(fac[n]['emergency_kwh'])}|\n"
comparison='|指标|第二问同口径|第三问完整模型|\n|---|---:|---:|\n'
r0,r1=cross['comparison']
for k in list(r0)[1:]:comparison+=f'|{k}|{money(r0[k])}|{money(r1[k])}|\n'
index=f'''## 0. 统一符号设定与论文输出索引

符号表见 `output/01_符号说明.md`，数学模型见 `output/03_问题3数学模型与公式.md`，两问同口径对照见 `output/10_跨问题一致性审计.md`。正式评价期为2025年2月1日至12月31日，共334天；储能从1月1日给定的6000千瓦时开始连续运行。

第三问正式期自然日总费用为 **{money(D['total_cost_yuan'])}元**，其中普通合同结算{money(D['regular_cost_yuan'])}元、紧急购电{money(D['emergency_cost_yuan'])}元。紧急购电量为{money(D['emergency_kwh'])}千瓦时，实际储电量位于[{money(D['soc_min_kwh'])},{money(D['soc_max_kwh'])}]千瓦时。
'''
audit=rf'''## 1. 跨问题独立审计与修订理由

修订前第二问公布费用为{money(old['Q2']['published_total'])}元，第三问完整模型为{money(old['old_D']['natural_total'])}元。逐项重算表明，两问物理设备、效率、无售电条件及五倍紧急电价一致；第三问新增了零点及日内官方光伏预报、尚未交付合同的调整权。将第二问动作原样代入第三问的实际取电、未利用合同、弃光等式后，最大电量平衡误差仅为{old['Q2']['physical_error']:.3g}千瓦时。因此第二问策略并未被题目物理或结算条件排除。

旧第三问程序的关闭模式仍使用三场景、七日负荷预测、截止次日零点十分的事件视野及最长六小时的另一个实时执行器，没有尾部风险项，也没有第二问的场景储电储备规则。重新运行旧程序全年后，其关闭模式费用为{money(cross['old_closed_replay']['total_cost'])}元，原计划、最终合同、充电、放电、紧急购电和储电轨迹与旧结果逐项完全一致。该关闭模式不是第二问的数值复现，不能用它衡量相对于第二问的新增信息收益。

旧完整第三问的普通取电{money(old['old_D']['grid_import'])}千瓦时、未利用合同{money(old['old_D']['unused_contract'])}千瓦时；第二问对应为{money(old['Q2']['actual_regular_import_kwh'])}和{money(old['Q2']['unused_contract_kwh'])}千瓦时。旧第三问充电{money(old['old_D']['charge'])}千瓦时、放电{money(old['old_D']['discharge'])}千瓦时，约为第二问的一半。低价时段储能不足、后续更多追加合同与未利用额度共同体现了实现路径差异；这些伴随变化不能拆成互不交互的单因素因果贡献。

第二问按计划行统计普通费用，第三问按自然日统计普通费用。统一到自然日后，第二问仅需作{old['Q2']['natural_bridge']:.8f}元的首尾十分钟桥接，总费用为{money(old['Q2']['natural_total'])}元。约34元的口径差异不能解释原约416万元的差距。独立结论是：问题约束本身允许继承，但旧程序与对照设计缺少继承关系，必须修复。修订不是因为“新增信息后的实际费用理论上绝不许增加”，而是因为原对照并未保持基础能力一致。

## 2. 策略集合包含关系与退化证明

设第二问在决策时刻可见的信息为 $\mathcal F_t$，第三问增加已发布光伏预报后的信息为 $\mathcal G_t$，则 $\mathcal F_t\subseteq\mathcal G_t$。任意第二问合法因果策略 $\pi_2$，都可在第三问定义一个扩展策略：对额外预报不作响应，零点发布同一原计划，日内所有合同均保持原计划，并使用同一储能反馈规则。第二问策略对较小信息集可测，故对较大信息集同样可测。

用原计划 $B$ 和各阶段合同 $A_r$ 表示该嵌入，有
$$A_r=B\quad(r=0,6,12,18),\qquad (B-A_r)^+=(A_r-B)^+=0.$$
两种已保留的结算解释在无调整时都变为 $pB$；紧急费用仍为 $5pe$。状态方程、充放电上界和储电边界完全相同，所以相同初始状态下，逐段执行由数学归纳法得到相同储电状态、充放电和紧急购电。统一统计区间后费用亦完全相同，即 $\Pi_2\subseteq\Pi_3$。

在相同概率模型、相同评价目标和真正求得策略集合上的最优解时，才可进一步推出 $\inf_{{\pi\in\Pi_3}}J(\pi)\le\inf_{{\pi\in\Pi_2}}J(\pi)$。该结论不能替代对一个指定近似控制器的逐年实际回测，也不能保证更换预测分布后每个样本路径都更便宜。本文保留完整退化路径，并报告实现费用，不将两组不同预测分布下的差额称为一般性理论信息价值。

本轮退化运行从1月1日起重新调用第二问原求解器，不复制其结果数组。全年计划、充放电、紧急购电、储电量、费用和总富余逐时段对照；同时核对每个日内阶段合同均未改变。关闭新增信息和合同调整后，按第二问原公布口径费用为{money(cross['evidence']['published_q2_cost_reproduced']['closed_q3'])}元；费用误差为{cross['evidence']['published_q2_cost_reproduced']['difference']:.10f}元。
'''
zero=cross['evidence']['zero_forecast_only_control']
audit+=f"\n另外单独保留零点官方光伏预报、关闭六点十二点十八点更新且禁止合同调整，全年费用为{money(zero['total_cost'])}元，较同口径第二问变化{money(zero['delta_vs_q2_natural'])}元。该方案仍含第二问没有的零点预报，因此并不要求数值完全相等；只有额外零点预报也关闭时，才要求精确退化。零点预报单独对照也检查了合同冻结、逐槽物理平衡及储电守恒。\n"
assumptions=r'''## 3. 共同数据、时间轴与预测机制

两问使用相同附件电价、负荷和实际光伏，功率统一乘以六分之一小时转为十分钟电量。计划日从零点十分开始，最后一段属于次日零点到零点十分；自然日首段由昨日已承诺合同供给。零点时，最晚完整历史行只能到前两天，昨日行的跨日末段必须屏蔽。两个问题均不把当天未来实际量用于零点合同，也不每日重置电池。

第三问的负荷预测直接继承第二问的历史加权集成：昨天可见部分、七日滞后、最近四个同星期历史均值及最近七日中位数。历史光伏预测亦作为始终可选的退化路径。使用新增信息时，仅把已发布预报覆盖的光伏时段替换为按绝对目标时刻插值的官方预报；未覆盖的后续时段继续使用第二问因果历史预测。不采用旧第三问另选的负荷预测器。

关闭新增信息时，场景构造逐元素调用第二问原方法：最近42天完整历史残差，按高价正净负荷误差风险排序，取九个经验分位代表日，保持完整时序误差形状和相同去重规则。启用官方预报后，零点残差使用相应零点预报；日内同时更新点预测和历史同发布时间的残差，避免把旧预报误差直接当作新预报误差。历史样本均已完整实现，正式期未来实际量不可进入场景生成。

九场景沿用第二问已有参数，不由本轮第三问正式期费用重选。单场景对照明确为纯点预测，五场景只作为场景密度敏感性检查。历史样本不足时只使用实际可得代表场景，不通过重复样本虚构更多历史信息。
'''
model=r'''## 4. 继承第二问的滚动数学模型

零点窗口包含已经承诺的首个十分钟、当日新计划144段和下一日辅助144段，共289段。六点、十二点和十八点的剩余窗口分别为253、217和181段，均截止当前日后第二天零点十分。下一日购电变量仍仅是第二问已有的情景续期估值，不能直接发布，也不能修改当前原计划。下一日到零点再使用真实可得信息重新制定计划。

当前已接受合同在所有场景中共同；尚未执行的后续本日合同可以在合法事件重新选择。已经开始交付的前缀保持不变。在当前求解中，不把尚未发生的后续日内调整当成可以预先兑现的采购能力，因此不会为想象中的后续调整而削弱零点基础计划。

沿用往返效率90%的对称拆分 $\eta_c=\eta_d=\sqrt{0.9}$，储能动力学与边界为
$$S_{\omega,t+1}=S_{\omega,t}+\eta_c x_{\omega,t}-y_{\omega,t}/\eta_d,$$
$$1200\le S_{\omega,t}\le10800,\qquad0\le x_{\omega,t},y_{\omega,t}\le5000/6.$$
每个场景满足普通合同、光伏、储能和紧急购电可以覆盖负载与充电。事件层净负荷不等式是与第二问相同的估值近似；真正执行必须满足后述物理流等式。

原始合同锚定的主结算费用为
$$c(B,A)=p\min(A,B)+0.5p(B-A)^++1.5p(A-B)^+.$$
等价地，$c(B,A)=\max\{0.5pA+0.5pB,\ 1.5pA-0.5pB\}$，可用两个线性上界约束构成分段线性费用。保持原合同 $A=B$ 总是允许的；针对相同事件、相同场景与储电状态，验证器额外求解固定合同与允许调整两个问题，检查允许调整问题的目标不会更差。

从当前事件起，本日剩余紧急损失为 $Z_\omega=\sum_{t\in\text{本日合同剩余范围}}5p_te_{\omega,t}$。沿用第二问的风险与终端项：
$$\operatorname{CVaR}_{0.8}(Z)=\zeta+\frac1{0.2K}\sum_\omega u_\omega,\qquad u_\omega\ge Z_\omega-\zeta,\quad u_\omega\ge0,$$
$$R_\omega\ge6000-S_{\omega,H},\qquad R_\omega\ge0.$$
零点直接使用第二问求解器，目标为当天计划费、下一日辅助购电费、窗口期望紧急费、$0.02\operatorname{CVaR}_{0.8}(Z)$和$0.8\mathbb E[R]$之和。日内只把剩余本日计划费替换为合法合同调整后的分段费用，其他核心项与参数保持一致。

允许按场景变化的下一日采购及储能轨迹是第二问的同一种续期近似，不能宣称完整非预知多阶段最优控制。保留这一近似是为了公平继承基础能力，并非证明它等于真实未来最优反馈。通过未来实际值扰动不变性检查确认它没有读取未来真实数据；二者是不同层面的检查。
'''
execution=r'''## 5. 真实执行和物理流分账

执行器直接调用第二问的实时反馈规则。以当前普通合同减去当前实际净负荷作为余量：有富余时在功率和容量允许范围内充电；不足时，放电不能超过功率限额，也不能穿越场景储电量25%分位形成的备用线；剩余缺口按五倍电价紧急补购。备用线是执行限额，并不把实际储电量重置为预测值。

关闭全部新增信息和合同调整时，六点、十二点、十八点仅继续使用零点已有的储备轨迹，不偷偷重新求解另一套控制器。开启新信息或调整权限时，可在相应事件重新估计剩余窗口的储备轨迹。所有情况下，储电状态都由实际发生的充放电递推。

与第二问一样，反馈回放把十分钟聚合量近似为区间内分段常值的当前测量。这是给定数据分辨率下的建模假设，不等于已经验证真实传感器没有延迟。当前聚合量只进入执行反馈，不能回写同刻已经制定的合同；未来真实量仍不可见。

付费普通合同 $A_t$ 与实际普通取电 $I_t$ 分开。优先使用光伏，定义未利用合同 $u_t=A_t-I_t$、真正弃光 $\kappa_t$，满足
$$I_t+PV_t-\kappa_t+y_t+e_t=L_t+x_t,$$
$$0\le I_t\le A_t,\qquad u_t=A_t-I_t\ge0,\qquad0\le\kappa_t\le PV_t.$$
未使用已付合同额度不产生退费；主动下调生效合同才按照调整规则结算。不存在售电收入、隐藏反送或电池无效耗散。总富余为 $u_t+\kappa_t$，第二问旧“剩余电量”必须按这个等式拆分后才能与第三问比较。
'''
results=f'''## 6. 两问同口径主结果与信息、权限对照

{table}
以上四组都使用继承后的同一基础模型。这里新增预报维度包含零点官方预报及日内更新，基准组完全不使用附件三，正好退化为第二问。因此这四组定义与旧第三问的对照定义不同，不能把旧组名对应结果混入本轮表格。

新增预报在合同冻结时的实现节省为{money(fx['info_saving_A_minus_B'])}元；仅增加调整权限的实现节省为{money(fx['flex_saving_A_minus_C'])}元；完整策略相对退化基准的实现节省为{money(fx['joint_saving_A_minus_D'])}元。正值表示节省，负值表示增加费用。它们是本年度与固定参数下的实现差额，存在交互，不能简单相加或解释为所有数据分布下的普遍收益。

电量单位均为千瓦时，费用单位为元。原计划和最终合同计划行量按照官方结果模板的计划日统计；生效合同自然日量、实际流量和费用按统一自然日统计，两者首尾十分钟单独桥接。

{comparison}
正式期开始和结束的储电量如表所示。不同策略的期初状态来自各自一月连续运行，不能重置以美化对照；全年退化检验则要求相同路径从一月起逐段一致。
'''
results+=f"\n在合同已经可调整时，完整官方预报方案相对历史预报方案的费用变化为{money(fac['D']['total_cost_yuan']-fac['C']['total_cost_yuan'])}元。正值不支持新增预报在该条件下独立降本；这与完整方案相对第二问整体降本是不同的比较。本文报告这一负面或正面结果，不按正式期费用反向更换主方案。\n"
sensitivity='## 7. 预报发布时间、场景数和结算解释敏感性\n\n'
for h in (6,12,18):
    r=ext['full_year_realized_policy_sensitivity'][f'D_no{h}']
    sensitivity+=f"关闭{h}点新预报后的正式期费用为{money(r['total_cost_yuan'])}元，减去完整模型的差额为{money(r['delta_vs_D_yuan'])}元。该检查保留其他时刻的新预报，不能把一个关闭实验的差额解释为独立、可加的单次预测价值。\n\n"
sensitivity+='|场景设置|费用／元|紧急购电／千瓦时|\n|---|---:|---:|\n'
for label,r in [('单场景纯点预测',ext['scenario_count']['K1_point']),('九场景继承主模型',D),('五场景敏感性',ext['scenario_count']['K5'])]:
    sensitivity+=f"|{label}|{money(r['total_cost_yuan'])}|{money(r['emergency_kwh'])}|\n"
sensitivity+='\n终端价值只在一月检查0.4、0.8和1.2三个系数，主模型直接继承第二问的0.8，不根据本轮正式期结果重新挑选。\n\n'
for tv,r in ext['terminal_value_january_only'].items():sensitivity+=f"系数{tv}的一月费用为{money(r['jan_total_cost_yuan'])}元，月末储电量{money(r['jan_soc_end_kwh'])}千瓦时。\n\n"
sensitivity+=f"保留原计划费再加取消罚金的解释下，全年重新优化费用为{money(sem['sunk_plan_plus_penalty']['total_cost_yuan'])}元；逐次调整解释下为{money(sem['stepwise_revision']['total_cost_yuan'])}元。以调整时刻价格重新计价主轨迹的费用为{money(sem['revision_time']['total_cost_yuan'])}元；该项是固定策略重计，并附四个代表日的局部非凸优化检查，不能写成第三种语义已完成全年重新优化。\n"
validation=f'''## 8. 验证结果和证据边界

基础验收{v['pass_count']}/{v['check_count']}项通过，覆盖全表回读、物理供需、储电递推、连续性、功率和能量边界、合同前缀冻结、发布时间与目标时刻对齐、五类账本和统计区间桥接。十一组完整策略的物理检查为{phys['pass_count']}/{phys['check_count']}项通过；跨问题退化及保护范围检查为{cross['pass_count']}/{cross['check_count']}项通过。

回归测试独立检查关闭新增信息后的预测与场景逐元素等同第二问；保留合同候选可行、放开调整后的同模型目标不劣；只返回当天144段合同；未来实际数据及尚未发布预报的扰动不影响当前合同或当前储备线；夜间不得产生虚假弃光、不能无去处放电、不得通过紧急电力专门充电。原物理回归中的随机状态检查继续保留。

跨问题验证读取第二问原始结果，与第三问重新计算的退化结果比较，不以某个总费用相近替代逐段一致。输入附件以及第一、第二、第四问全部以本轮开始时的文件指纹复核，任何改动都会使保护范围检查失败。

合同边际费用在下调侧为半倍交付价、上调侧为一点五倍交付价；折点处允许边际值落在两者之间。代表日有限差分检查仅解释选定合同槽的局部最优条件，不能据八个选定点推广为全部场景或全部年份的最优性证明。
'''
limitations='''## 9. 模型评价、修订时点与论文结论

这次修订发生在已经看到旧全年结果之后，属于基于结构审计的模型修正。风险权重、置信水平、历史窗口、储备分位数、终端目标与价值系数均直接继承第二问，没有按照新正式期费用逐日选优或搜索参数。新年度结果仍应称为修订后的历史回测证据，不能声称是事前未接触过的外部独立测试。

模型的主要改善是消除了跨问题比较中的基础能力替换：第二问的日前计划、长视野及实时储能能力可以在第三问完整保留。新增官方预报和合同调整成为显式可关闭的扩展，合同不变策略始终可行。现有回测结果应客观说明费用、紧急购电、未利用合同和弃光如何共同改变，不能只报告某个有利指标。

仍存在三个主要局限。第一，有限场景是经验残差的近似，并不等于准确估计未来全部不确定性；单场景、五场景和九场景的实现费用不必单调。第二，下一日辅助采购与场景储能是续期估值近似，并未证明全局非预知多阶段最优。第三，十分钟当前测量是分段常值近似，没有额外高频数据来验证区间内响应延迟。完整物理与因果检查通过，也不能消除这些统计和控制层面的限制。

结论应分别陈述：题目策略集合的包含关系成立；旧代码不能数值退化，因此旧两组费用不可用来直接衡量新增权限价值；修订后的程序通过全年退化检验，并在同一基础能力上报告新增预报和调整权限的实现效果。不要将其夸大成“任何新增预报都必然省钱”或“已证明所有可能策略中的最优解”。

## 10. 可复现性与交付对应

官方结果表保留原来的四张工作表：计划购电、调整购电、充放电与紧急购电。前两张表按计划日填写全量334乘144段数据；充放电按自然日四小时块汇总，紧急购电按连续十分钟事件汇总；各表都必须完整回读到同一轮数值结果。

费用账本按保留计划费用、下调费用、上调费用、紧急购电费用和总费用逐段对账；物理账本逐段保存合同、实际取电、未利用额度、弃光、实际充放电和储电量。跨问题证据目录保留修订前完整快照、旧关闭模式重跑结果、本轮对照、退化检查与受保护文件指纹。

重现本轮先运行十一组策略的全年回放，再运行日前求解诊断和跨问题一致性验证，最后生成结果表、基础验证、扩展实验、结算解释检查、图表、论文材料和最终验收。任何模型改动都必须重新生成相关结果，不能把修订前的验证文件或截图当作修订后证据。
'''
symbols=r'''# 问题3统一符号说明

|符号|定义|
|---|---|
|$d$|自然日与计划日索引|
|$j$|计划合同的十分钟时段|
|$i$|自然日十分钟执行时段|
|$\tau$|当前合同决策时刻|
|$\omega$|历史残差场景|
|$g$|当前共同决策信息集；下一日场景变量仅作估值|
|$\nu$|已发布光伏预报版本|
|$B_{d,j}$|零点原始计划合同|
|$A_{d,r,j}$|第阶段生效合同|
|$D_{d,j}=(B-A)^+$|下调电量|
|$U_{d,j}=(A-B)^+$|上调电量|
|$x_{d,i}$|实际充电量|
|$y_{d,i}$|实际放电量|
|$S_{d,i}$|储电量|
|$e_{d,i}$|紧急购电量|
|$I_{d,i}$|实际普通取电|
|$u_{d,i}$|未利用的付费合同额度|
|$\kappa_{d,i}$|真正弃光|
|$F^{plan}$|保留计划费用|
|$F^{cancel}$|合同下调费用|
|$F^{add}$|合同上调费用|
|$F^{emg}$|紧急购电费用|
|$\mu_t$|当前合同槽边际价值|
|$R_\omega$|终端储备缺口|
|$\alpha,\lambda$|尾部风险置信水平0.8及权重0.02|

电量均为千瓦时，费用为元，十分钟充放电上限为5000/6千瓦时。
'''
dates=['# 指定日期论文表格','']
for ds in ('2025-03-20','2025-06-21','2025-09-23','2025-12-21'):
    d=next(i for i,t in enumerate(data.dates) if t.date()==datetime.fromisoformat(ds).date())
    dates += [f'## {ds}','', '### 表1 指定时段与全天合同','|时段|原计划／千瓦时|最终合同／千瓦时|','|---|---:|---:|']
    for h in (10,12,14,16,18,20):dates.append(f"|{h}:00—{h}:10|{money(z['B'][d,h*6-1])}|{money(z['A'][d,h*6-1])}|")
    dates += [f"|全天计划行|{money(z['B'][d].sum())}|{money(z['A'][d].sum())}|",'',f"原计划费{money(z['plan_fee'][d])}元；最终合同计划行费{money(z['adjusted_fee'][d])}元。自然日总费用{money(z['natural_regular_fee'][d]+z['emergency_fee'][d])}元，其中紧急费用{money(z['emergency_fee'][d])}元。",'', '### 表2 储能','|时段|充电／千瓦时|放电／千瓦时|','|---|---:|---:|']
    for b in range(6):dates.append(f"|{b*4}:00—{(b+1)*4}:00|{money(z['charge'][d,b*24:(b+1)*24].sum())}|{money(z['discharge'][d,b*24:(b+1)*24].sum())}|")
    dates += ['',f"零点储电量{money(z['soc00'][d])}千瓦时；二十四点储电量{money(z['soc24'][d])}千瓦时。",'', '### 表3 紧急购电','|时间段|电量／千瓦时|','|---|---:|']
    for t,e in grouped_emergency_periods(z['emergency'][d]) or [('无',0.0)]:dates.append(f'|{t}|{money(e)}|')
    dates.append(f"|合计|{money(z['emergency'][d].sum())}|")
master='# 2026年第三问：继承第二问的滚动预报与合同调整模型\n\n'+index+audit+assumptions+model+execution+results+sensitivity+validation+limitations
figindex='''# 图表索引与图注

图14说明继承第二问的模型流程；图15展示代表日的原合同与最终合同；图16展示四组信息和权限对照；图17展示费用分解；图18展示四个指定日；图19展示局部边际价值检查；图20展示结算解释敏感性；表7至表9分别给出指定日合同、储能和紧急购电。新增附图为两问同口径的费用与物理电量对照。所有图只使用本轮重新计算后的数组和费用账本。
'''
params='''# 参数说明与评价指标

沿用第二问：九场景、42天残差窗口、尾部风险置信水平0.8、权重0.02、实时储备25%分位、终端参考6000千瓦时、缺口价值每千瓦时0.8元。储能初始6000千瓦时、范围1200—10800千瓦时、功率5000千瓦、往返效率90%。零点视野289个十分钟段；日内保留相同终点。主指标同时报告费用、合同、实际取电、未利用合同、充放电、紧急购电和真正弃光。
'''
files={'00_output目录与论文使用说明.md':index,'01_符号说明.md':symbols,'符号说明.md':symbols,
       '02_模型假设与数据口径.md':assumptions+execution,'03_问题3数学模型与公式.md':model,
       '公式说明.md':model,'04_滚动预测与求解流程.md':assumptions+execution,'05_参数说明与评价指标.md':params,
       '06_结果分析与对照实验.md':results,'07_模型评价与论文表述建议.md':limitations,
       '08_图表索引与图注.md':figindex,'09_鲁棒性与消融验证.md':sensitivity+validation,
       '10_跨问题一致性审计.md':audit+comparison+validation,'指定日期论文表格.md':'\n'.join(dates),
       '模型与结果说明.md':index+model+results,'指标汇总.md':params+results,'模型选择.md':audit+limitations,
       '分析报告.md':audit+results+sensitivity,'总结与说明.md':index+limitations,
       '物理语义修复说明.md':execution+validation}
OUT.mkdir(exist_ok=True);ROB.mkdir(exist_ok=True)
(Q3/'问题3论文材料.md').write_text(master,encoding='utf-8')
for name,text in files.items():(OUT/name).write_text(text.rstrip()+'\n',encoding='utf-8')
for name in ['semantic_robustness.json','revision_time_milp_audit.csv','marginal_value_audit.csv','future_perturbation_audit.csv','nonanticipativity_nodes.csv','leakage_audit.csv','forecast_vintage_alignment_audit.csv','validation.json','event_audit.csv','extended_experiments.json','dispatch_audit.csv','semantic_robustness.csv','marginal_deadzone_summary.json']:
    shutil.copy2(HERE/name,ROB/name)
for name in ['cross_problem_validation.json','q2_q3_comparison.csv','midnight_solver_validation.json']:
    shutil.copy2(HERE/'cross_problem_evidence'/name,ROB/name)
(ROB/'鲁棒性验收报告.md').write_text(sensitivity+validation,encoding='utf-8')
print('Updated',len(files),'paper views;',len(master.encode('utf-8')),'bytes')
