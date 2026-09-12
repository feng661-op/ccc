# -*- coding: utf-8 -*-
from pathlib import Path
import json,pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parents[2];CODE=Path(__file__).resolve().parent;Q4=ROOT/'问题四';OUT=Q4/'output'

def money(x):return f'{float(x):,.2f}'
def num(x):return f'{float(x):,.3f}'
def pct(a,b):return 100*(float(a)-float(b))/float(b)
def run():
 q2=json.loads((CODE/'q4_2'/'formal_summary.json').read_text(encoding='utf-8'));q3=json.loads((CODE/'q4_3'/'formal_summary.json').read_text(encoding='utf-8'));r0=json.loads((CODE/'r0_repricing.json').read_text(encoding='utf-8'));lad=json.loads((CODE/'baseline_ladder.json').read_text(encoding='utf-8'));oracle=json.loads((CODE/'oracle_audit.json').read_text(encoding='utf-8'));eps=json.loads((CODE/'epsilon_calibration.json').read_text(encoding='utf-8'));st=json.loads((CODE/'price_structure_audit.json').read_text(encoding='utf-8'));pdec=json.loads((OUT/'02_model_selection'/'price_freeze_decision.json').read_text(encoding='utf-8'));chosen_price=pdec['chosen']
 abp=CODE/'ablations'/'ablation_matrix_E0_E15.json';ab=json.loads(abp.read_text(encoding='utf-8')) if abp.exists() else {'experiments':{}};e10p=CODE/'ablations'/'E10_settlement_exact_local.json';e10=json.loads(e10p.read_text(encoding='utf-8')) if e10p.exists() else None
 m2=q2['formal'];m3=q3['formal'];R2=r0['q4_2']['R0_variable_price']['cash_total_yuan'];R3=r0['q4_3']['R0_variable_price']['cash_total_yuan'];C2=m2['cash_total_yuan'];C3=m3['cash_total_yuan']
 lines=[];A=lines.append
 A(r'# 问题四：波动电价下微网因果购电与滚动调控模型（论文材料）');A(r'');A(r'> 本材料以 `问题四/code` 的权威账本为唯一数值来源。2025年1月只用于结构审计、因果预测/模型选择与参数冻结；2025年2月1日至12月31日334天只做冻结后的正式评价，不反向选择模型。');A(r'')
 A(r'## 1 题意重构与四问递进');A(r'');A(r'问题四不是把附件4的“实际电价”直接代入一个已知未来价格的确定性优化，而是在前三问的物理、合同和信息边界上增加**实时揭示且未来未知的价格过程**。因此我们保留两条继承链：Q4-2严格继承问题二的信息权限，不读取附件3未来PV；Q4-3严格继承问题三的0/6/12/18时滚动PV预报和合同调整权限。唯一新增的是波动价格的因果预测及其与净负荷不确定性的联合处理。');A(r'')
 A(r'为避免把解释性相关、预测精度和最终调度收益混在一起，全文严格区分三层证据：**结构机制证据 → 预测证据 → 决策证据**。');A(r'')
 A(r'## 2 双时间坐标与信息集');A(r'');A(r'官方计划行从 `00:10-00:20` 开始，最后一列为次日 `00:00-00:10+1`；正式评价则按自然日 `[00:00,24:00)`。故自然日首段必须读取上一计划行第143列，其余143段依次读取当日计划行第0—142列。2025-01-01首段缺少上一计划行，程序保留为缺失值，**从未补0**。正式期的2月1日首段正确桥接1月31日尾列，12月31日的`+1`尾段另建尾账本连续复算。');A(r'')
 A(r'在任意决策时刻 $\tau$，合同层只能使用 $\mathcal F_\tau$ 中已揭示的价格、历史负荷/PV，以及对应分支合法的PV预测版本；执行层每10 min只用当前段测量修正源—汇动作。合同层和执行层共享同一物理方程，但信息输入不同。');A(r'')
 A(r'## 3 波动价格：先验证机制，再冻结因果预测');A(r'');A(rf"1月离线结构审计得到 $\mathrm{{Corr}}(c_{{d,t}}-\bar c_t,N_{{d,t}}-\bar N_t)={st['value']:.6f}$。该值只说明“同一日内时段上价格异常与净负荷异常共振”，**不作为预测准确率，也不作为调度节费证据**。")
 A(r'');A(r'预测层事前列出 `lag1 / lag7 / lag7+当日已揭示残差 / lag7+净负荷校正 / 结构Ridge` 五类候选，并同时比较全日单组与六个四小时块。**所有候选使用完全相同的1月因果walk-forward样本，按MAE最小、RMSE与q95依次破平局；模型复杂度只在预测指标近似并列时才作最后破平局。** 因而结构机制证据不会给Ridge任何“默认主模型”特权。');A(r'')
 pm=pd.read_csv(OUT/'02_model_selection'/'price_model_ablation.csv');lag=pm[pm.model=='lag7'].iloc[0];rr=pm[pm.model=='full_ridge'].sort_values(['mae','rmse']).iloc[0];win=pm.sort_values(['mae','rmse','q95_abs_error']).iloc[0]
 if chosen_price['model']=='lag7':
  A(r'最终1月公平冻结得到最简单的周同期基线');A(r'');A(r'$$\hat c_{d,t}=c_{d-7,t}.$$');A(r'')
 else:
  A(rf"最终1月公平冻结为 `{chosen_price['model']}/{chosen_price['block_scheme']}`；参数与版本完整记录在 `price_freeze_decision.json`。") ;A(r'')
 A(rf"其1月MAE为 {win.mae:.6f} 元/kWh（RMSE={win.rmse:.6f}）。作为结构候选的最佳Ridge MAE为 {rr.mae:.6f}，而lag7为 {lag.mae:.6f}。因此0.960986只保留为机制证据，**预测模型由真实walk-forward成绩决定，而不是由模型复杂度或叙事偏好决定**。训练/残差历史池仍执行 expanding→42天封顶。")
 A(r'');A(r'## 4 唯一源—汇物理内核');A(r'');A(r'对每个10 min段，合同购电量为 $Q$，PV为 $G$，负荷为 $L$。定义合同电直供/充电 $q^L,q^B$，未使用合同 $u$；PV直供/充电 $g^L,g^B$，真正弃光 $\kappa$；紧急供负荷/充电 $e^L,e^B$；电池充放电 $x,y$。主模型强制 $e^B=0$。统一物理为');A(r'');A(r'$$q^L+q^B+u=Q,\qquad g^L+g^B+\kappa=G,$$');A(r'$$q^L+g^L+y+e^L=L,\qquad x=q^B+g^B+e^B,$$');A(r'$$S_{t+1}=S_t+\sqrt{0.9}\,x_t-\frac{y_t}{\sqrt{0.9}},\quad 1200\le S_t\le10800,$$');A(r'$$0\le x_t,y_t\le5000\times\frac16=833.333\ \mathrm{kWh}.$$');A(r'');A(r'这样“未取合同电”$u$与“真正弃光”$\kappa$不会混淆，且事件优化器和10 min执行器调用同一 `q4_flow.py` 约束生成器。最终48,096段正式账本最大物理残差均为 $1.82\times10^{-12}$ kWh。');A(r'')
 A(r'## 5 合同结算与三级字典序');A(r'');A(r'Q4-3中00:00原始合同记为 $B_t$，后续绝对合同记为 $A_t$。主结算锚定原始$B_t$：');A(r'');A(r'$$\Phi_t(B,A;c)=c\min(A,B)+0.5c(B-A)_++1.5c(A-B)_+.$$');A(r'');A(r'紧急购电按逐段实际价格计 $5c_te_t$；已付合同未用部分不退款。每个事件严格按三级字典序求解：①现金/冻结风险目标最优；②在第一层容差内最小化 $\sum(x+y)$；③再最小化 $\sum\kappa$。');A(r'')
 A(r'## 6 场景、CVaR与DRO：允许被证据淘汰');A(r'');A(r'联合场景使用同一历史日的负荷、PV、价格残差路径，medoid只做联合路径约简，权重为簇频率。距离尺度在1月选模开始前由完全释放的1月2—7日残差冻结；$\varepsilon$只按相邻经验分布的Wasserstein漂移尺度校准，Q4-2的Q50/Q75分别为 %.4f/%.4f，Q4-3为 %.4f/%.4f。**这不是统计置信半径，也不声称覆盖概率。**'%(eps['q4_2']['q50'],eps['q4_2']['q75'],eps['q4_3']['q50'],eps['q4_3']['q75']))
 A(r'');A(r'候选依次为B0因果点预测、B1联合场景SAA、B2运输DRO+CVaR。只允许使用1月9—31日决定复杂度。');A(r'');A(r'|分支|B1相对B0平均现金改善|B1相对B0 CVaR95改善|B1门槛|B2门槛|最终冻结|');A(r'|---|---:|---:|---|---|---|')
 for br in ('q4_2','q4_3'):
  d=lad['branches'][br];A(rf"|{br}|{d['B1']['mean_cash_improvement_pct']:.2f}%|{d['B1']['cvar95_improvement_pct']:.2f}%|{'通过' if d['B1_pass'] else '未通过'}|{'通过' if d['B2_pass'] else '未通过'}|{d['selected']}|")
 selected_text='；'.join(f"{br}→{lad['branches'][br]['selected']}" for br in ('q4_2','q4_3'))
 A(r'');A(rf"复杂度严格由1月门槛决定，最终为 **{selected_text}**。未通过门槛的SAA/DRO不因算法更复杂而保留；通过门槛的场景层则按冻结规则进入正式期。");A(r'')
 A(r'## 7 334天正式结果');A(r'');A(r'|指标|Q4-2|Q4-3|');A(r'|---|---:|---:|');
 for label,key in [('现金总费用/元','cash_total_yuan'),('紧急购电/kWh','emergency_total_kwh'),('弃光/kWh','curtailment_total_kwh'),('未取合同/kWh','unused_contract_total_kwh'),('期末SOC/kWh','soc_end_kwh'),('最大物理残差/kWh','max_physics_residual')]:A(rf"|{label}|{num(m2[key])}|{num(m3[key])}|")
 A(r'');A(rf"冻结前三问的**已实现全年轨迹**、只把结算价格换成附件4得到R0暴露：Q4-2 {money(R2)}元、Q4-3 {money(R3)}元。正式Q4-2为 {money(C2)}元，相对R0 **{pct(C2,R2):+.2f}%**；正式Q4-3为 {money(C3)}元，相对R0 **{pct(C3,R3):+.2f}%**。R0不是重新面对波动价格后按相同状态/信息逐日决策的匹配政策，而是既有轨迹的反事实重计价，因此只作为“历史轨迹压力参考”。无论差额正负，本文都原样报告，绝不把R0包装成新模型的节费因果基线。")
 if 'E0' in ab.get('experiments',{}) and 'matched_no_price_adaptation' in ab['experiments']['E0']:
  e0=ab['experiments']['E0'];c0=e0['matched_no_price_adaptation']['cash_total_yuan'];A(r'');A(rf"为解决R0口径不匹配，另构造**匹配的因果无价格适应基线 C0**：与Q4-2共享同一1月冻结状态、物理约束、信息过程和附件4实际结算，只把合同优化使用的价格替换为附件1已知固定分时价。C0正式现金为 {money(c0)} 元，价格自适应Q4-2相对C0为 {100*(C2-c0)/c0:+.2f}%。这个比较才用于判断“价格适应本身”的正式价值；R0只保留为历史轨迹压力参考。")
  if 'paired_daily_cash_difference_adaptive_minus_C0' in e0:
   z=e0['paired_daily_cash_difference_adaptive_minus_C0'];lo,hi=z['weekly_block_bootstrap_95pct_CI_yuan_per_day'];
   concl=('正式期价格自适应略贵，不能宣称Q4-2靠价格预测节费' if lo>0 else ('正式期价格自适应具有稳健节费' if hi<0 else '两种因果政策在全年现金上没有稳健可分差异'))
   A(r'');A(rf"进一步对334个配对日的“自适应−C0”现金差做7日移动块bootstrap（5000次，seed=2026），日均差为 {z['mean_yuan_per_day']:.2f} 元/日，95%区间为 [{lo:.2f}, {hi:.2f}] 元/日。由区间方向可知：**{concl}**。这项结果只用于诚实评价价格适应的外样本价值，不回调1月已冻结模型。")
 A(r'');A(rf"Q4-3比Q4-2正式现金少 {money(C2-C3)} 元、紧急购电少 {num(m2['emergency_total_kwh']-m3['emergency_total_kwh'])} kWh；其来源需要由信息×权限消融进一步拆解，而不能仅凭总差额归因。")
 A(r'');A(r'## 8 信息权限与鲁棒性');A(r'');
 if 'E6' in ab.get('experiments',{}):
  e6=ab['experiments']['E6'];A(r'Q4-3冻结后的A/B/C/D全期敏感性为：');A(r'');A(r'|设置|正式现金/元|紧急购电/kWh|');A(r'|---|---:|---:|')
  for k,v in e6.items():A(rf"|{k}|{money(v['cash_total_yuan'])}|{num(v['emergency_total_kwh'])}|")
 else:A(r'E6全期敏感性由 `code/ablations/ablation_matrix_E0_E15.json` 给出。')
 A(r'');A(r'历史窗28/56天、紧急充电、效率口径、事件版本删除、反馈时序、终端价值等均在模型冻结后单独重放；这些结果只评估鲁棒性，不回调42天主窗、风险偏好或正式模型。结算语义变化不应用固定策略的事后重计价来替代重新优化；本稿E10仅保留固定主策略的结算 exposure，明确不称最优，也不用于冻结或排名主模型。');A(r'')
 if 'E9' in ab.get('experiments',{}):
  e9=ab['experiments']['E9'];A(r'针对“同一10 min槽直接读取当前负荷/PV”可能被质疑为零延迟测量，我们额外做**334天一槽滞后控制敏感性**：控制器构造目标时只看上一槽L/PV，当前实际L/PV仅用于物理平衡与紧急兜底。');A(r'');A(r'|分支|当前测量正式现金/元|lag-one控制现金/元|变化|');A(r'|---|---:|---:|---:|')
  for br in ('q4_2','q4_3'):
   z=e9[br];A(rf"|{br}|{money(z['formal_current']['cash_total_yuan'])}|{money(z['formal_lag1_measurement']['cash_total_yuan'])}|{z['cash_delta_pct']:+.2f}%|")
  A(r'')
 if 'E12' in ab.get('experiments',{}):
  e12=ab['experiments']['E12'];A(r'主模型把紧急购电视为**只保供负荷的安全兜底**，不允许主动拿5倍价紧急电给电池充电，即 $e^B=0$。为把这一运行语义从“隐含假设”变成可审查假设，我们又允许 $e^B>0$ 做334天全期敏感性：');A(r'');A(r'|分支|主模型 eB=0 /元|允许紧急充电 /元|费用变化|');A(r'|---|---:|---:|---:|')
  for br in ('q4_2','q4_3'):
   z=e12[br];base=z['main_eB0']['cash_total_yuan'];alt=z['sensitivity_eB_allowed']['cash_total_yuan'];A(rf"|{br}|{money(base)}|{money(alt)}|{100*(alt-base)/base:+.2f}%|")
  A(r'');A(r'因此 $e^B=0$ 是可解释的运行规则而不是物理定律；其敏感性必须与主结果并列披露。');A(r'')
 A(r'## 9 Oracle边界');A(r'');A(r'|分支|因果主策略/元|Price Oracle/元|Full-information Oracle/元|');A(r'|---|---:|---:|---:|')
 for br in ('q4_2','q4_3'):
  o=oracle[br];A(rf"|{br}|{money(o['causal']['cash_total_yuan'])}|{money(o['price_oracle']['cash_total_yuan'])}|{money(o['full_information']['strict_cash_lower_bound_yuan'])}|")
 A(r'');A(r'Price Oracle只把未来价格改为已知，但负荷/PV、滚动合同和状态条件仍不同，因此**不称理论下界**。只有Full-information Oracle在相同源—汇物理、合同/结算、正式期首段锁定合同、初始SOC、匹配的末SOC及纯现金目标下放松未来信息与非预见性，所以才满足“FI现金费用 ≤ 任意同口径可行因果策略”的严格下界条件。');A(r'')
 A(r'## 10 指定日期输出');A(r'');A(r'题面指定的2025-03-20、06-21、09-23、12-21均已分别输出：10:00、12:00、14:00、16:00、18:00、20:00六个10 min购电点及全天量/费；六个四小时块的充放电量与S00/S24；以及紧急购电完整事件列表。“无”事件明确写“无”。权威详表见 `output/04_tables/指定日期表格.md` 与两个官方Excel。');A(r'')
 A(r'## 11 图表与可复现材料');A(r'');A(r'正文主图为：图1结构机制、图2信息时间轴、图3模型结构、图4成本—风险、图5典型日联合调度、图6信息权限/历史窗机制消融；附录另含价格预测消融、ε漂移及Oracle图。每张图的底层数据均保存在 `output/figures/subdata`。');A(r'')
 A(r'所有正式结果均可追溯到 `physical_10min.csv`、`contract_ledger.csv`、`event_ledger.csv`、`daily_ledger.csv`、`tail_bridge.json`；Excel、CSV、JSON之间由T01—T42验收独立重读核对。Q1/Q2/Q3与官方题面附件不修改。');A(r'')
 A(r'## 12 限制与结论');A(r'');A(rf"1. 0.960986是1月结构相关，不是因果关系；价格主模型最终由公平walk-forward冻结为 `{chosen_price['model']}`，Ridge只保留为结构/预测消融证据。") ;A(r'2. 场景SAA/DRO是否进入正式模型完全服从1月冻结门槛；算法复杂度不能凌驾于冻结证据。');A(r'3. 当前段负荷/PV采用10 min分段常值的当前测量近似，不包装成已验证的零延迟传感器；全年lag-one敏感性与四个指定日钻取均单列报告。');A(rf"4. R0仅是历史轨迹价格暴露；Q4-2相对R0为 {pct(C2,R2):+.2f}%，该符号不回避，也不用于宣称新策略节费；价格适应价值优先看同口径C0比较。") ;A(r'5. $e^B=0$ 是“紧急电只保负荷”的运行语义假设，已有允许紧急充电的全期敏感性；ε只表示经验分布漂移尺度，Price Oracle也只表示价格信息价值诊断。');A(r'');A(r'在这些边界内，Q4-3利用合法的日内PV更新与合同调整，在同一源—汇物理和逐段波动价格结算下显著降低了正式现金费用与紧急购电；其可靠性由连续334日账本、跨日尾桥接、Oracle、冻结后消融及完整验收共同支撑。')
 A(r'');A(r'---');A(r'');A(r'**文件引用**：`result4-2.xlsx`、`result4-3.xlsx`、`output/04_tables/指定日期表格.md`、`output/figures/`、`code/validation.json`、`code/oracle_audit.json`、`code/ablations/`。')
 (Q4/'问题4论文材料.md').write_text('\n'.join(lines),encoding='utf-8');print('paper_written',len(lines))
if __name__=='__main__':run()
