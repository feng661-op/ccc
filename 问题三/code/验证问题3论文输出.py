# -*- coding: utf-8 -*-
from pathlib import Path
import json, hashlib
ROOT=Path(__file__).resolve().parents[2]; Q3=ROOT/'问题三'; CODE=Q3/'code'; OUT=Q3/'output'; FIG=OUT/'图表'; ROB=OUT/'robustness'
checks={}; evidence={}
def ck(name,ok,ev=None): checks[name]='PASS' if ok else 'FAIL'; evidence[name]=ev
m=json.loads((CODE/'metrics_final.json').read_text(encoding='utf-8')); v=json.loads((CODE/'validation.json').read_text(encoding='utf-8')); fa=json.loads((CODE/'final_acceptance.json').read_text(encoding='utf-8')); D=m['main_D']; fac=m['factorial']; sr=m['semantic_robustness']
core=['00_output目录与论文使用说明.md','01_符号说明.md','02_模型假设与数据口径.md','03_问题3数学模型与公式.md','04_滚动预测与求解流程.md','05_参数说明与评价指标.md','06_结果分析与对照实验.md','07_模型评价与论文表述建议.md','08_图表索引与图注.md','09_鲁棒性与消融验证.md','指定日期论文表格.md','模型与结果说明.md','符号说明.md','公式说明.md','指标汇总.md','模型选择.md','分析报告.md','总结与说明.md']
missing=[f for f in core if not (OUT/f).is_file() or (OUT/f).stat().st_size<200]; ck('complete_output_documents',not missing,{'expected':len(core),'missing':missing})
sym=(OUT/'01_符号说明.md').read_text(encoding='utf-8')
needed=['$d$','$j$','$i$','$\\tau$','$\\omega$','$g$','$\\nu$','$B_{d,j}$','$A_{d,r,j}$','$D_{d,j}','$U_{d,j}','$x_{d,i}$','$y_{d,i}$','$S_{d,i}$','$e_{d,i}$','$F^{plan}$','$F^{cancel}$','$F^{add}$','$F^{emg}$','$\\mu_t$']
ck('symbol_table_core_coverage',all(x in sym for x in needed),{'symbols':len(needed),'missing':[x for x in needed if x not in sym]})
alltext='\n'.join((OUT/f).read_text(encoding='utf-8') for f in core if (OUT/f).exists())
nums=['14,786,522.56','16,189,702.09','16,222,490.01','15,109,651.55','21,512,141.84','20,967,445.42','1,428,411.00','883,714.57','5,990,848.22','5,394,838.83','2,643,667.52','15,119,524.44','14,802,426.83','14,720,004.19']
ck('final_metrics_referenced',all(x in alltext for x in nums),{'required_values':nums,'missing':[x for x in nums if x not in alltext]})
master=(Q3/'问题3论文材料.md').read_text(encoding='utf-8')
stale=['20,992,627.14','20,580,175.56','2,949,510.14','2,537,058.57','2,014,446.28','6,005,084.39','1,510,745.18','37274','3.64\\times10^{-12}','4.97\\times10^{-12}']
ck('master_paper_stale_values_removed',not any(x in master for x in stale),{'remaining':[x for x in stale if x in master]})
ck('master_paper_symbol_index', '## 0. 统一符号设定与论文输出索引' in master and 'output/01_符号说明.md' in master)
ck('model_validation_still_pass',v.get('all_pass') and v.get('pass_count')==21 and v.get('check_count')==21,{'pass':v.get('pass_count'),'count':v.get('check_count')})
ck('final_acceptance_still_pass',fa.get('all_pass') and fa.get('pass_count')==13 and fa.get('check_count')==13,{'pass':fa.get('pass_count'),'count':fa.get('check_count')})
figs=['图14_问题3事件驱动滚动调度框架.png','图15_问题3典型日合同滚动调整.png','图16_问题3ABCD因子对照.png','图17_问题3主模型费用分解.png','图18_问题3四季代表日合同与净负荷.png','图19_合同调整死区边际价值.png','图20_问题3结算语义鲁棒性.png','表7_指定日期原始与最终合同.png','表8_指定日期储能充放电.png','表9_指定日期紧急购电.png']
badfig=[f for f in figs if not (FIG/f).is_file() or (FIG/f).stat().st_size<20000]; ck('paper_figures_complete',not badfig,{'count':len(figs),'bad':badfig,'sizes':{f:(FIG/f).stat().st_size if (FIG/f).exists() else 0 for f in figs}})
robfiles=['semantic_robustness.json','revision_time_milp_audit.csv','marginal_value_audit.csv','future_perturbation_audit.csv','nonanticipativity_nodes.csv','leakage_audit.csv','forecast_vintage_alignment_audit.csv','validation.json','final_acceptance.json','event_audit.csv','鲁棒性验收报告.md']
missingrob=[f for f in robfiles if not (ROB/f).is_file()]; ck('robustness_evidence_complete',not missingrob,{'count':len(robfiles),'missing':missingrob})
for f in ['validation.json','final_acceptance.json']:
    if (ROB/f).exists(): ck('robustness_copy_'+f,hashlib.sha256((ROB/f).read_bytes()).digest()==hashlib.sha256((CODE/f).read_bytes()).digest())
ck('factorial_metric_identity',abs(fac['D']['total_cost_yuan']-D['total_cost_yuan'])<1e-9 and abs(sr['main_D_total_cost_yuan']-D['total_cost_yuan'])<1e-9)
ck('result3_exists',(Q3/'result3.xlsx').is_file() and (Q3/'result3.xlsx').stat().st_size>500000,{'bytes':(Q3/'result3.xlsx').stat().st_size})
out={'all_pass':all(x=='PASS' for x in checks.values()),'pass_count':sum(x=='PASS' for x in checks.values()),'check_count':len(checks),'checks':checks,'evidence':evidence}
(OUT/'论文输出验收.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(out,ensure_ascii=False,indent=2))
if not out['all_pass']: raise SystemExit(2)
