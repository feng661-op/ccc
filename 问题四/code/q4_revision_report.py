"""Render the current Q4 evidence only; never reuse stale V1.2 claims."""



from __future__ import annotations



import argparse,json,shutil,hashlib



from pathlib import Path



import numpy as np



import pandas as pd



import matplotlib



matplotlib.use('Agg')



import matplotlib.pyplot as plt



from q4_revision_run import CODE,ROOT,REV,RUNS,VERSION,load,dump,sha,fingerprint



OUT=CODE.parent/'output';FIG=OUT/'figures';ARCHIVE=REV/'legacy_pre_inheritance'



plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':11,'pdf.fonttype':42})











def archive(p):



    p=Path(p)



    if p.exists():



        target=ARCHIVE/'human_and_outputs'/p.relative_to(CODE.parent)



        if not target.exists():



            target.parent.mkdir(parents=True,exist_ok=True)



            if p.is_dir():shutil.copytree(p,target)



            else:shutil.copy2(p,target)







def write(p,text):



    p=Path(p);archive(p);p.parent.mkdir(parents=True,exist_ok=True);text=__import__('re').sub(r'\n{3,}', '\n\n', text.replace('\r\n','\n').replace('\r','\n'));text=__import__('re').sub(r'(?m)(^\|[^\n]*\|)\n(?:[ \t]*\n)+(?=\|)', r'\1\n', text);p.write_bytes(text.encode('utf-8'))







def records():



    fr=load(CODE/'january_frozen_selection.json');main={b:load(CODE/b/'formal_summary.json') for b in ('q4_2','q4_3')}



    for m in main.values():



        if m['provenance']['numerical_core_hash']!=fingerprint():raise AssertionError('stale formal input')



    matched={b:load(RUNS/f'{b}_delivery_matched_fixed_decision/formal_summary.json') for b in main}



    backup=load(RUNS/'q4_3_delivery_backup/formal_summary.json')



    return fr,main,matched,backup







def metric_table(items):



    head='|方案|334日现金费用/元|紧急购电/kWh|日费用CVaR95/元|末端SOC/kWh|\n|---|---:|---:|---:|---:|\n'



    return head+'\n'.join(f"|{label}|{x['cash_total_yuan']:,.2f}|{x['emergency_total_kwh']:,.2f}|{x['cash_cvar95_yuan_per_day']:,.2f}|{x['soc_end_kwh']:,.3f}|" for label,x in items)+'\n'







def figure_run():



    fr,main,matched,backup=records()



    for path in (FIG,OUT/'图表'):



        if path.exists() and not (path/'INHERITANCE_V2.txt').exists():



            archive(path)



            shutil.rmtree(path)



        path.mkdir(parents=True,exist_ok=True)



        (path/'INHERITANCE_V2.txt').write_text(VERSION,encoding='utf-8')



    registry=[]



    def save(fig,name,title,source,claim):



        fig.tight_layout()



        for ext in ('png','pdf'):



            p=FIG/(name+'.'+ext);fig.savefig(p,dpi=220,bbox_inches='tight');shutil.copy2(p,OUT/'图表'/p.name)



        plt.close(fig);registry.append({'id':name,'title':title,'source':source,'supported_claim':claim})



    fig,ax=plt.subplots(figsize=(11.5,5.0));ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)



    boxes=[(.02,'Q1：共同物理基础','容量 12,000 kWh\nSOC 1,200—10,800 kWh\n功率 5,000 kW；往返效率 90%'),(.27,'Q2：连续因果决策','48小时估值视野 + 跨日尾段\n9个历史净负荷场景\nSOC储备反馈；不每日重置'),(.52,'Q3：增加信息与权限','0 / 6 / 12 / 18 点PV预报\n新增合同调整权\n交付前可调整，已交付锁定'),(.77,'Q4：只扩展价格信息','lag7因果电价及配对残差\n保留Q2/Q3的视野与控制器\n固定价格时逐段回退')]



    for x,title,text in boxes:



        ax.text(x+.105,.72,title,ha='center',va='center',fontsize=12,fontweight='bold')



        ax.text(x+.105,.45,text,ha='center',va='center',fontsize=10.3,linespacing=1.7,bbox={'boxstyle':'round,pad=.8','fc':'none'})



        if x<.77:ax.annotate('',xy=(x+.245,.46),xytext=(x+.225,.46),arrowprops={'arrowstyle':'->'})



    ax.text(.5,.10,'继承的是物理与决策结构，不是把四个答案的全年费用强行设成相同。',ha='center',fontsize=11)



    save(fig,'fig01_四问连续继承','四问连续继承','q4_inheritance.py; bridge_q4_2.json; bridge_q4_3.json','固定电价下两条分支都精确回退到现有上游策略。')



    fig,ax=plt.subplots(figsize=(10.5,4.6));ax.set_ylim(0,4.8);ax.set_xlim(-1,25);ax.set_yticks([])



    ax.plot([0,24],[2,2],lw=2)



    for x in [0,6,12,18]:



        ax.scatter([x],[2],s=55);ax.text(x,1.60,f'{x}:00',ha='center',fontsize=12)



        ax.annotate('已发布PV\n当前价格已知' if x else '冻结今日原始合同B\n继承昨日末段合同',xy=(x,2),xytext=(x,3.45),ha='center',arrowprops={'arrowstyle':'->'})



    ax.annotate('10分钟物理执行：实际负荷/PV → 储备反馈 → SOC连续传递',xy=(12,1.75),xytext=(12,.9),ha='center')



    ax.text(12,.28,'未来电价仅作因果预测；第二天决策变量是估值追索，不是提前知道未来或提前执行合同。',ha='center',fontsize=10)



    ax.set_title('Q4-3预报与合同更新时间轴（Q4-2仅0点定约，不使用附件3）',fontsize=11)

    ax.set_xticks([0,6,12,18,24]);ax.set_xlabel('自然日时间 / h');ax.spines[['top','left','right']].set_visible(False)



    save(fig,'fig02_信息与跨日时间轴','因果信息与连续执行','q4_inheritance.make_bundle; q4_sim_v2.simulate_range','新增信息有发布时间；未知未来实际值不参与当前决策。')



    fig,ax=plt.subplots(figsize=(9.6,5.3));labels=['Q4-2\n固定价决策','Q4-2\n因果价格正式','Q4-3\n固定价决策','Q4-3\n因果价格正式']



    vals=[matched['q4_2']['formal']['cash_total_yuan'],main['q4_2']['formal']['cash_total_yuan'],matched['q4_3']['formal']['cash_total_yuan'],main['q4_3']['formal']['cash_total_yuan']]



    bars=ax.bar(labels,np.asarray(vals)/1e6)



    for bar,v in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.08,f'{v/1e6:.4f}',ha='center')



    ax.set_ylim(0,19);ax.set_ylabel('正式期现金费用 / 百万元');ax.set_title('匹配初始状态、实际波动电价结算：当前预测没有实现节费')



    save(fig,'fig03_匹配价格决策对照','匹配价格决策对照','inheritance_revision/runs/*delivery_selected & *matched_fixed_decision','对照的是整体价格适应包，不宣称识别单一预测器的独立因果贡献。')



    fig,ax=plt.subplots(figsize=(9.0,5.3));points=[]



    for i,br in enumerate(('q4_2','q4_3')):



        row=fr[br]['B1'];x=-row['mean_cash_improvement_pct'];y=row['cvar95_improvement_pct'];points.append({'branch':br,'cash_penalty_pct':x,'cvar95_gain_pct':y,'selected':fr[br]['selected']})



        ax.scatter([x],[y],s=90,marker=('o','s')[i]);ax.annotate(f"{br.upper().replace('_','-')}：B1\n现金 +{x:.3f}%，尾部改善 {y:.3f}%\n一月选择 {fr[br]['selected']}",(x,y),xytext=(-155,18) if i==0 else (-195,18),textcoords='offset points',fontsize=10)



    ax.axvline(1,ls='--',label='现金代价上限 1%');ax.axhline(2,ls=':',label='尾部改善门槛 2%');ax.set_xlim(-.45,1.25);ax.set_ylim(-2.5,5.1)



    ax.set_xlabel('B1相对B0的平均现金费用增幅 / %（越小越好）');ax.set_ylabel('日费用CVaR95改善 / %（越大越好）');ax.set_title('一月成本—风险统一晋级：同时展示代价与收益');ax.legend(loc='lower left',fontsize=9)



    pd.DataFrame(points).to_csv(FIG/'fig04_data.csv',index=False,encoding='utf-8-sig')



    save(fig,'fig04_一月成本风险权衡','一月成本与风险同时披露','january_frozen_selection.json; fig04_data.csv','Q4-3 B1现金代价在1%以内，且尾部改善及其80%区块区间通过门槛；Q4-2不通过。')



    fig,ax=plt.subplots(figsize=(10.5,4.8));sl=pd.read_csv(CODE/'q4_3/physical_10min.csv');days=pd.read_csv(CODE/'q4_3/daily_ledger.csv');target=days.iloc[(days.cash_fee_yuan-days.cash_fee_yuan.quantile(.95)).abs().argmin()].date;day=sl[sl.date==target]



    ax.step(np.r_[0.,(day.slot.to_numpy()+1)/6],np.r_[day.S0_kwh.iloc[0],day.S1_kwh.to_numpy()],where='post',label='实际SOC');ax.plot((day.slot+1)/6,day.target_soc_kwh,ls='--',label='继承的四分位储备目标');ax.axhline(1200,ls=':');ax.axhline(10800,ls=':')



    ax.set_xlim(0,24);ax.set_ylim(900,11500);ax.set_xlabel('时间 / h');ax.set_ylabel('SOC / kWh');ax.set_title(f'Q4-3正式策略：{target}（按日费用最接近P95规则选取）');ax.legend(loc='best')



    day.to_csv(FIG/'fig05_day_data.csv',index=False,encoding='utf-8-sig')



    save(fig,'fig05_连续SOC与储备反馈','实际SOC与场景储备反馈','q4_3/physical_10min.csv; fig05_day_data.csv','真实SOC满足边界，并按固定的代表日规则展示，不挑选最好的一天。')



    fig,ax=plt.subplots(figsize=(10.0,3.7));ax.axis('off');proof=[]



    for br in ('q4_2','q4_3'):



        b=load(REV/f'bridge_{br}.json');e=b['max_absolute_errors'];proof.append([br.upper().replace('_','-'),str(b['days_recomputed']),str(b['actual_slots_checked']),f"{e['B']:.1f}",f"{e['A']:.1f}",f"{e['SOC']:.1f}",f"{e['cash']:.2e}"])



    table=ax.table(cellText=proof,colLabels=['新分支','重算天数','实测时段数','原合同误差','最终合同误差','SOC误差','日现金误差/元'],cellLoc='center',loc='center');table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,2)



    ax.set_title('独立全年重算后的退化校验：合同和SOC逐段误差为0',pad=22);ax.text(.5,.05,'比较目标为只读的Q2/Q3冻结数组；并未把其结果作为新仿真的输入。',transform=ax.transAxes,ha='center',fontsize=10)



    save(fig,'fig06_全年继承回归证据','全年继承回归证据','inheritance_revision/bridge_q4_2.json; bridge_q4_3.json','以全年逐段一致性替代仅看总费用接近的弱证据。')



    dump(CODE/'figure_registry.json',{'backend':VERSION,'figures':registry})



    dump(CODE/'figure_manifest.json',{p.name:sha(p) for p in FIG.iterdir() if p.suffix in ('.png','.pdf','.csv')})



    print('CURRENT_FIGURES_DONE',len(registry),flush=True)











def config_run():



    text='''version: q4-inheritance-v2



base_q3_head: c2a58e9a4139a88ad18bbb7f191862fddae00802



periods:



  warmup: 2025-01-01..2025-01-08



  selection: 2025-01-09..2025-01-31



  formal: 2025-02-01..2025-12-31



  formal_days: 334



physical:



  capacity_kwh: 12000



  power_kw: 5000



  dt_hours: 0.16666666666666666



  soc_min_kwh: 1200



  soc_max_kwh: 10800



  initial_soc_kwh: 6000



  rte: 0.9



  eta_c: 0.9486832980505138



  eta_d: 0.9486832980505138



  efficiency_interpretation: inherited_symmetric_round_trip_assumption



  emergency_to_battery: false



  daily_soc_reset: false



inheritance:



  q2_source: ../../问题二/code/q2_core.py



  q3_source: ../../问题三/code/q3_inheritance.py



  midnight_horizon_intervals: 289



  source_scenarios_max: 9



  source_history_days: 42



  risk_lambda: 0.02



  optimization_cvar_alpha: 0.8



  execution_reserve_quantile: 0.25



  terminal_reserve_kwh: 6000



  terminal_shortage_value_yuan_per_kwh: 0.8



  planning_model: inherited_net_balance_relaxation



  execution_model: native_reserve_feedback_and_full_source_flow_audit



  future_day_variables: valuation_recourse_not_executable_contracts



price:



  formal_regressor: lag7



  scope: global



  ridge_alpha_metadata: 1.0



  historical_price_model_comparison: reused_unchanged_january_prediction_task



  current_price_revealed: true



  future_price_known: false



selection:



  B0: nine_net_load_scenarios_plus_point_price



  B1: same_net_load_scenarios_plus_paired_historical_price_errors



  B2: not_run_by_user_request



  uniform_gate_cash_gain_pct: 1.0



  uniform_gate_max_cash_penalty_pct: 1.0



  uniform_gate_tail_gain_pct: 2.0



  paired_block_days: 7



  bootstrap_repetitions: 2000



  confidence_level: 0.8



  formal_period_used_for_selection: false



  q4_2: B0



  q4_3: B1



settlement:



  primary: delivery



  primary_reason: explicit_interpretation_preserving_latest_Q3_semantics



  alternative: adjustment_time



  anchoring: original_midnight_B



  emergency_multiplier: 5.0



validation:



  essential: causal_rights_physics_365_day_bridge_334_day_ledgers_excel_hashes



  robustness: NOT_RUN_BY_USER_REQUEST



  old_ablations: STALE_NOT_RERUN



  old_price_oracle: STALE_NOT_RERUN



  full_information_oracle: NOT_RUN_NO_CURRENT_LOWER_BOUND_CLAIM



'''



    write(CODE/'config_frozen.yaml',text)











def docs_run():



    fr,main,matched,backup=records();b2=main['q4_2']['formal'];b3=main['q4_3']['formal'];jan=fr['q4_3']['B1']



    sem_path=RUNS/'q4_3_adjustment_time_selected/formal_summary.json'



    if not sem_path.exists():raise FileNotFoundError('Do not seal the paper before alternative settlement reoptimization finishes')



    sem=load(sem_path);sj=load(RUNS/'q4_3_adjustment_time_january/decision.json')



    if sem['provenance'].get('semantic_adapter_sha256')!=sha(CODE/'q4_semantics_fast.py'):raise AssertionError('stale alternative solver')



    main_table=metric_table([('Q4-2：B0正式',b2),('Q4-3：B1正式',b3),('Q4-3：B0备选',backup['formal'])])



    price_table=metric_table([('Q4-2：固定价决策、实际价结算',matched['q4_2']['formal']),('Q4-2：因果电价正式',b2),('Q4-3：固定价决策、实际价结算',matched['q4_3']['formal']),('Q4-3：因果电价正式',b3)])



    semantic_table=metric_table([('交付时段价格：'+main['q4_3']['selected'],b3),('调整事件价格：'+sem['selected'],sem['formal'])])



    change=100*(sem['formal']['cash_total_yuan']/b3['cash_total_yuan']-1)



    base='> 当前版本：`q4-inheritance-v2`。以下均来自本轮继承修复后的新账本；旧V1.2结果不作为当前证据。\n\n'



    intro=f'''## 1 四问的连续关系与本次修订







官方题面第二页要求在波动电价下重新计算问题2和问题3。因此Q4-2是问题2的价格扩展，Q4-3是问题3的价格扩展，不能另换规划视野或储能执行器。问题1提供共同储能物理模型；问题2引入历史信息下的连续决策；问题3加入已发布PV预报和日内合同调整；问题4只扩展价格信息及其场景。







当前第三问基线为`c2a58e9a4139a88ad18bbb7f191862fddae00802`。本轮沿用Q2原生48小时估值视野（含跨日首段共289个10分钟段）、最多9个历史净负荷误差场景、CVaR风险系数0.02、四分位SOC储备反馈及跨日合同。Q1的单日首末SOC相等属于该问边界，不被误用成Q2—Q4每日重置SOC。







固定价退化实验从1月1日SOC=6000独立运行365天，Q4-2与Q2原生/Q3-A、Q4-3与最新Q3-D的原始合同B、最终合同A和每个时段SOC误差均为0。两分支各比较52,559个已知实际时段，日费用误差仅为浮点舍入量级。这是结构继承证据，不是强迫四种不同问题得到一样的现金费用。



'''



    symbols=r'''## 2 符号与物理口径







|符号|含义|单位/范围|



|---|---|---|



|d,t,τ|日期、10分钟交付段、信息/决策时刻|自然时间|



|B,A,Q|0点原始合同、当前生效合同、实际交付合同量|kWh；交付时Q=A|



|L,G,N|实际负荷、光伏、净负荷L-G|kWh/段|



|S,x,y|储能电量、充电量、放电量|kWh|



|qL,qB,u|合同电供负荷、供电池、未取合同电|kWh|



|gL,gB,κ|光伏供负荷、供电池、弃光|kWh|



|e|紧急购电，仅直接补负荷|kWh|



|p,pτ|交付段电价、调整交易时刻电价|元/kWh|







题目90%效率在四问中统一解释为往返效率，并采用对称拆分ηc=ηd=√0.9。这是沿用的建模解释，不说成题面分别给出了两侧效率。S∈[1200,10800]，x,y∈[0,5000/6]。容量标称12000并不意味着可把SOC用到12000。







```math



S_{t+1}=S_t+\eta_c x_t-y_t/\eta_d.



```







实际执行逐段核对：







```math



q^L_t+q^B_t+u_t=Q_t,\qquad g^L_t+g^B_t+\kappa_t=G_t,



```



```math



q^L_t+g^L_t+y_t+e_t=L_t,\qquad x_t=q^B_t+g^B_t.



```







紧急电只补负荷是本模型的主运行语义，不是声称电能物理上不能充电；实际控制不会同时充放电。u与κ严格分列：合同电付费后未取，不应伪装成弃光。



'''



    information=r'''## 3 信息边界与因果预测







本文将实时波动电价解释为“当前时段实时揭示，未来价格尚未公布”；题面没有明确规定公布机制，所以这是需要写明的假设。lag7只访问对应时段7天前的价格，缺失时回退到已知的附件1周期价格。当前时段价格可用；未来当天负荷、光伏、真实价格不可读取。







Q4-2沿用问题2历史因果净负荷预测，不获得附件3的额外PV预报。Q4-3仅使用截至0/6/12/18点已经发布的预报。历史场景沿用Q3/Q2的同一完整来源日，B1的价格残差与净负荷残差按来源日配对，最近残差来源的后续日必须在当前决策前完整可用。实际正值价格裁剪与当前已知价格固定在场景构建内完成。







一月附件内按时段去均值的电价—净负荷相关系数为0.960986。它只作为供需耦合的结构证据，不是因果估计，也不是在线时刻可读取的一月全样本均值特征。该结构不自动意味着复杂预测器的外样本费用更低。







B0与B1都有最多9个净负荷场景：B0采用一条因果电价点预测；B1加入配对历史价格误差。它们不是旧版“单场景B0/三场景B1”。价格模型比较的旧一月数据任务本身未改变，故复用并记录哈希；调度策略的一月选择已按新继承内核重新运行。



'''



    math=r'''## 4 继承式随机滚动模型与合同结算







午夜按原生Q2变量顺序构造场景规划，今日合同对所有场景共同，昨日末段合同已经确定；第二天的分场景购电变量只用于估值追索，不被当作今日可以提前签订的承诺，更不是读取未来实际信息。日内仅在原模型允许的决策点更新未交付部分，原始B始终不变，已交付合同前缀锁定。







场景规划保留上游净能量不等式及储能动力学：







```math



Q_t+e_{k,t}+y_{k,t}-x_{k,t}\ge \widehat N_{k,t},\quad



S_{k,t+1}=S_{k,t}+\eta_c x_{k,t}-y_{k,t}/\eta_d.



```







目标为继承的预期现金估值、紧急购电风险和末端储备短缺罚：







```math



\min\ \mathbb E_k\!\left[\sum_{t\in今日}\Phi(B_t,A_t;p_{k,t})



+\sum_{t\in未来估值}p_{k,t}q_{k,t}



+\sum_t5p_{k,t}e_{k,t}+0.8(6000-S_{k,T})_+\right]



+0.02\,\operatorname{CVaR}_{0.8}(\hbox{当前承诺区间紧急购电费}).



```







CVaR使用ζ和非负超额变量的标准线性表达。该目标是风险调整的有限视野近似，不等于证明未知全年现金费用全局最优。实际执行沿用上游四分位SOC储备目标；规划是净平衡松弛，执行才做完整源流核验，不能再声称两者是“同一源流生成器”或严格同构。







为保持与最新Q3的经济规则连续，正式文件采用“交易标的交付时段电价”的解释：







```math



\Phi(B,A;p)=p\min(B,A)+0.5p(B-A)_++1.5p(A-B)_+.



```







紧急费为5pe，全部现金账为正常合同结算与紧急费之和。调整始终相对0点原始B，而不把上一次A当成新的免费基准。扩大合同调整可行域不可能使同一次、同一目标的优化值变差；单元测试比较可调整/锁合同问题验证这一点。但它不保证两个不同信息预测下的实际全年实现费用单调下降，因此不能用“信息更多”直接证明外样本现金一定更低。



'''



    selection=f'''## 5 统一一月选模，不按正式期反向挑选







1月1—8日作共同预热，B0/B1从同一1月8日末状态分别运行1月9—31日。复杂模型晋级必须满足：现金平均改善至少1%且80%配对区块区间下界为正；或者现金恶化不超过1%、日费用CVaR95改善至少2%且对应区间下界为正。两条复杂度晋级统一使用同一规则，区块长度7日、重采样2000次。1%是沿用的成本容忍纪律，不是题目规定。







Q4-2冻结为B0。Q4-3新B1的现金代价为+{-jan['mean_cash_improvement_pct']:.6f}%，CVaR95改善{jan['cvar95_improvement_pct']:.6f}%，其尾部收益80%区间为[{jan['paired']['cvar95_gain_80pct_interval'][0]:.3f}, {jan['paired']['cvar95_gain_80pct_interval'][1]:.3f}]元/日，因此按同一规则重新冻结为B1。







这不是沿用旧B1，也不能为了满足桌面审计对旧数据的B0预测而强行改选。旧审计中的+6.057%代价属于旧控制器；修复继承关系后必须以新一月证据为准。图4同时展示现金代价与尾部收益。







优化中的CVaR置信参数为0.8，正式报告的日费用CVaR95是另一评价量，两者不能混写。B2/DRO与参数扫描按本轮要求不运行，也不凭旧结果获得晋级资格。



'''



    results=f'''## 6 正式期与价格适应对照







正式期为2025年2月1日—12月31日，共334日、每分支48,096个自然时段；每个策略接自己的1月冻结末状态。Q4-2全年0点事件334次；Q4-3四个事件点共1,336次。







{main_table}



Q4-3正式策略相对Q4-2减少现金{b2['cash_total_yuan']-b3['cash_total_yuan']:,.2f}元，降幅{100*(1-b3['cash_total_yuan']/b2['cash_total_yuan']):.4f}%。这里包含PV信息、合同权限与各自冻结价格场景的综合差异；未重新运行的单机制消融不能用来断言“主要由某一种权限贡献”。







Q4-3的B0备选更便宜{b3['cash_total_yuan']-backup['formal']['cash_total_yuan']:,.2f}元，但正式期差异只用于事后披露成本—风险权衡，不能反向改动仅用一月作出的选择，也不把B1一律称为成本最优。







匹配对照固定正式期初始状态、物理控制器和真实波动价格结算，仅把决策用价格改回附件1周期价（价格场景关闭、净负荷场景保留）：







{price_table}



在这两个匹配对照下，本轮价格适应均没有实现现金节省。Q4-3比较的是点预测加价格场景的整体适应包，而不是只改变一个lag7系数的纯单因素实验。不得写成“价格预测显著节费”，也不能由一次负结果推断未来价格信息没有价值。



'''



    semantic=f'''## 7 交易时刻电价：独立重优化与主语义决定







官方Q3使用“交易时刻电价”，存在交付时段价格和调整事件时刻价格两种解释。本轮不是给既有合同重新计价，而是把调整部分的0.5/1.5倍价改为调整事件价格，重新运行一月选择及334日正式优化。保留部分仍按交付价；每个新的调整事件对尚可调整合同作原始B锚定的最新确认，不进行逐次差额叠加套利。







{semantic_table}



替代语义的一月选择为{sj['selected']}，正式总现金相对交付语义变化{change:+.4f}%。这是两个独立冻结政策及语义的联合比较，而不是纯结算效应因果分解。







本次重新审定后，主提交仍沿用最新Q3已经采用的交付时段口径，原因是题目要求“重新计算问题2和问题3”，不能只在Q4暗中改变Q3的合同经济规则。该选择不以哪组结果更便宜为依据。另一解释单独完整披露，不能据此声称全部结算语义下策略或结论完全不变；若统一改采事件口径，应同时重审Q3而不是把两问拼接成不同的经济系统。







事件口径中费用在A=B处可能非凸，使用具有派生有限上界的二元分段MILP。由于所有场景的断点B和合同A相同，预期正常合同费可精确合并到平均交付电价，紧急电与SOC仍保留各场景。该等价降维不改变目标或可行域。求解失败/超时不会成为正式验收结果，替代求解器另记源码哈希。



'''



    boundary=r'''## 8 0:00+1、现金账与交付文件







附件一行最后的0:00+1属于次日0:00时段，而不是本日末段的同义复制。自然日第0段来自上一计划日j=143；本日其余143段来自本日计划j=0..142。2月1日首段继承1月31日末合同；12月31日模板的j=143仍保留到2026年1月1日的已承诺尾段。尾段物理执行独立记录，不混入2—12月334日现金KPI。







正常合同现金对账恒等式为：自然日费用＝计划行费用合计−末计划行+1尾段费用＋2月1日首段费用。每日S24等于下一日S00，0点没有人工补能；终端SOC价值也不是现金费用。







正式提交`result4-2.xlsx`、`result4-3.xlsx`从权威CSV生成，计划表保留0点B、调整表保留交付前最终A，未取合同电不从已购合同费用中扣除。计划表末列为原始B的参考购电费；Q4-3调整表末列已经是按最终A和原始B计算的正常合同结算总费用，不是应在原计划费上再次相加的“增量费用”。正式现金总额取正常合同最终结算加紧急购电费，禁止把两张表费用重复相加。充放电表按六个四小时块汇总，紧急电按连续事件汇总。指定日期表使用题面10/12/14/16/18/20点，不能用每小时整点替换。Excel全单元格数据与账本独立重读核对。



'''



    limitations='''## 9 验证范围、保留意见与复现







本轮实际执行：新的B0/B1一月冻结、双分支334日连续回放、双分支365日固定价退化回归、匹配固定价决策对照、Q4-3备选策略、替代结算完整重新优化、因果性/权限/物理/结算单测、CSV/JSON/Excel对账与来源哈希。







本轮不执行鲁棒性参数扫描、DRO晋级、旧E0—E15全消融矩阵或Full-information Oracle。旧Price Oracle和旧FI Oracle因控制器/初态/语义不匹配，不能作为新结果直接引用；本版不报告严格最优下界、严格VOI或旧的最优性差额。旧结果保存在`code/inheritance_revision/legacy_pre_inheritance`，其他旧实验目录带STALE标记。SKIP不算PASS。







最终验证以`code/validation.json`为准；其中T42检查294份受保护上游与官方文件的SHA256，不再要求HEAD永远等于旧Q3提交。`run_manifest.json`分别记录base_q3_head、实际run_head、未提交状态、源文件/配置/结果/验证哈希，不能把未提交新产物冒充已有提交。







复现入口：`python -B q4_revision_run.py january --branch q4_2`及q4_3；然后`freeze`、`formal`、`publish`；导出用`q4_export.py`，图文用`q4_revision_report.py`，正确性验收用`q4_validate.py`。每一步只在输入指纹匹配时运行，旧月度断点不能静默恢复。参数/鲁棒性不是默认运行项。



'''



    figreg=load(CODE/'figure_registry.json')['figures'];figtext='## 10 图表索引\n\n'+'\n\n'.join(f"**{f['id']} {f['title']}**\n\n证据：`{f['source']}`。图注：{f['supported_claim']}" for f in figreg)+'\n'



    full='# 问题4论文材料：承接问题1—3的波动电价模型\n\n'+base+'\n'.join([intro,symbols,information,math,selection,results,semantic,boundary,limitations,figtext])



    write(CODE.parent/'问题4论文材料.md',full)



    mapping={'00_output目录与论文使用说明.md':intro+limitations,'01_符号说明.md':symbols,'02_模型假设与数据口径.md':information+boundary,'03_问题4数学模型与公式.md':math,'04_价格预测与滚动求解流程.md':information+selection,'05_参数说明与评价指标.md':selection,'06_结果分析与风险对比.md':results+semantic,'07_模型评价与论文表述建议.md':limitations,'08_图表索引与图注.md':figtext,'09_鲁棒性与消融验证.md':limitations,'模型与结果说明.md':intro+results,'鲁棒性证据说明.md':limitations,'10_四问连续性与修订验收.md':intro+selection+semantic+limitations}



    for name,text in mapping.items():write(OUT/name,'# '+name[:-3]+'\n\n'+base+text)



    write(CODE/'README.md','# 第四问复现入口\n\n'+base+limitations+'\n'+selection)



    write(OUT/'图表/README.md','# 当前图表\n\n'+base+figtext)



    tab=OUT/'04_tables/指定日期表格.md'



    if tab.exists():write(OUT/'指定日期论文表格.md',tab.read_text(encoding='utf-8'))



    write(OUT/'robustness/README.md','# 本轮未运行鲁棒性\n\n本目录旧实验不得视为继承版证据。'+limitations)



    mainrows=[]



    for label,m in [('q4_2_selected',main['q4_2']),('q4_3_selected',main['q4_3']),('q4_3_backup',backup),('q4_2_matched',matched['q4_2']),('q4_3_matched',matched['q4_3']),('q4_3_event_semantics',sem)]:mainrows.append({'scenario':label,'backend':VERSION,'selected':m['selected'],**m['formal']})



    pd.DataFrame(mainrows).to_csv(OUT/'04_tables/current_comparisons.csv',index=False,encoding='utf-8-sig')



    dump(CODE/'settlement_semantics_current.json',{'backend':VERSION,'primary':'delivery','reason':'preserve_latest_Q3_explicit_interpretation_not_choose_by_cheaper_result','delivery_selected':main['q4_3']['selected'],'alternative_selected':sem['selected'],'delivery_cash':b3['cash_total_yuan'],'alternative_cash':sem['formal']['cash_total_yuan'],'change_pct':change,'alternative_january_sha256':sha(RUNS/'q4_3_adjustment_time_january/decision.json'),'alternative_formal_sha256':sha(sem_path),'alternative_solver_sha256':sha(CODE/'q4_semantics_fast.py')})



    for fn,why in [('oracle_audit.json','旧Price/FI Oracle未按新控制器重跑；不支持新下界/VOI。'),('epsilon_calibration.json','用户不运行DRO或鲁棒性扫描。'),('r0_repricing.json','旧策略重计价不是新版同源匹配对照。')]:



        p=CODE/fn;archive(p);dump(p,{'backend':VERSION,'status':'STALE_NOT_RERUN','reason':why,'archive':str((ARCHIVE/'human_and_outputs'/p.relative_to(CODE.parent)).relative_to(CODE.parent))})



    for dr in [CODE/'ablations',OUT/'robustness']:



        if dr.exists():dump(dr/'STATUS.json',{'backend':VERSION,'status':'HISTORICAL_ONLY_NOT_CURRENT_VALIDATION','robustness_run':False})



    readme_run()


    refresh_public_evidence()
    print('CURRENT_PAPER_AND_HUMAN_OUTPUTS_DONE',flush=True)














def readme_run():


    fr,main,matched,backup=records()


    rows=metric_table([('Q4-2：'+main['q4_2']['selected'],main['q4_2']['formal']),('Q4-3：'+main['q4_3']['selected'],main['q4_3']['formal'])])


    text='# 问题四：四问连续继承修订版\n\n> 当前版本：`'+VERSION+'`。只改第四问，不改前三问与官方附件，不运行鲁棒性扫描。\n\n'


    text+='## 正式结果\n\n'+rows+'\n费用均为2025年2月1日—12月31日334个自然日的现金费用；末计划行的+1尾段单列桥接，不混入正式KPI。\n\n'


    text+='## 本次修复的核心\n\nQ4-2保留Q2原生48小时视野、9个净负荷场景、风险系数0.02和SOC储备反馈；Q4-3在同一基础上保留0/6/12/18点预报与合同调整权。增加的仅是价格信息及配对价格误差。固定电价下，两分支从1月1日重新运行365天，原始合同、最终合同及SOC逐段误差均为0。\n\n'


    j=fr['q4_3']['B1']


    text+=f"Q4-3新一月B1现金代价为+{-j['mean_cash_improvement_pct']:.6f}%，尾部CVaR95改善{j['cvar95_improvement_pct']:.6f}%，通过统一1%费用容忍与2%风险改善门槛及配对区间要求，因此重新冻结为B1。旧审计中+6.057%的判断属于旧模型，不能直接当新模型的证据。\n\n"


    text+='## 不能忽略的结果边界\n\n同初始状态、同实际波动价格结算的固定价决策对照比本轮价格适应更便宜；不宣称价格预测实现了节费。Q4-3的B0备选也保留，不能根据2—12月更低费用反向更换一月冻结的策略。主口径明确沿用最新Q3的交付时段价格；调整事件价格另作完整重新优化，见`output/06_结果分析与风险对比.md`。\n\n'


    text+='## 文件入口\n\n提交文件：`result4-2.xlsx`、`result4-3.xlsx`。完整论文材料：`问题4论文材料.md`。连续性与修订说明：`output/10_四问连续性与修订验收.md`。指定日期表：`output/指定日期论文表格.md`。六张当前图：`output/图表/`。\n\n'


    text+='权威账本为`code/q4_2/`、`code/q4_3/`；当前一月证据为`code/inheritance_revision/runs/*_delivery_january/`及`code/january_frozen_selection.json`。本版真实验收结果只看`code/validation.json`和`code/run_manifest.json`。旧42/42 PASS、旧Oracle和旧消融不属于当前验收，已归档或标记失效；未跑项目不能写成通过。\n'


    write(CODE.parent/'README.md',text)





def refresh_public_evidence():
    """Keep compatibility entry files reproducible; never leave old totals."""
    for name in ('e0_fixed_price_bridge.json','matched_no_price_adaptation_q4_2.json','matched_no_price_adaptation_q4_3.json','data_manifest.json','price_structure_audit.json'):
        archive(CODE/name)
    dump(CODE/'e0_fixed_price_bridge.json',{'backend':VERSION,'status':'PASS','scope':'fresh independent 365-day exact inheritance regression','branches':{b:load(REV/f'bridge_{b}.json') for b in ('q4_2','q4_3')}})
    for b in ('q4_2','q4_3'):
        m=load(RUNS/f'{b}_delivery_matched_fixed_decision/formal_summary.json')
        dump(CODE/f'matched_no_price_adaptation_{b}.json',{'backend':VERSION,'benchmark':'same formal initial state and physical policy; Attachment-1 decision price; Attachment-4 cash settlement','scope':'whole price adaptation package','formal_summary':m})
    m=load(CODE/'data_manifest.json');m.update(backend=VERSION,numerical_core_hash=fingerprint(),template_output_hashes={n:sha(CODE.parent/n) for n in ('result4-2.xlsx','result4-3.xlsx')});dump(CODE/'data_manifest.json',m)
    dump(CODE/'price_structure_audit.json',{'backend':VERSION,**load(REV/'january_structure_evidence.json')})


def main():



    p=argparse.ArgumentParser();p.add_argument('stage',nargs='?',choices=['figures','docs','all'],default='all');a=p.parse_args();config_run()



    if a.stage in ('figures','all'):figure_run()



    if a.stage in ('docs','all'):docs_run()



if __name__=='__main__':main()



