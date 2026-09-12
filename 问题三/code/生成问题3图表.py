# -*- coding: utf-8 -*-
from pathlib import Path
from datetime import datetime
import sys,json,csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
from q3_data import load_q3_inputs,EVAL_START
ROOT=HERE.parent.parent; FIG=ROOT/'问题三'/'figures'; FIG.mkdir(parents=True,exist_ok=True)
data=load_q3_inputs(ROOT); m=json.loads((HERE/'metrics.json').read_text(encoding='utf-8')); sem=json.loads((HERE/'semantic_robustness.json').read_text(encoding='utf-8')); marg=json.loads((HERE/'marginal_deadzone_summary.json').read_text(encoding='utf-8')); z=np.load(HERE/'run_D.npz')
B=z['B'];A=z['A'];AS=z['A_stage'];soc=z['soc_path']; em=z['emergency']
# Prefer a CJK font on Windows.
for f in ('Microsoft YaHei','SimHei','Noto Sans CJK SC'):
    if any(x.name==f for x in font_manager.fontManager.ttflist): plt.rcParams['font.sans-serif']=[f]; break
plt.rcParams['axes.unicode_minus']=False

def finish(name):
    plt.tight_layout(); plt.savefig(FIG/name,dpi=220,bbox_inches='tight'); plt.close()

# 1) Representative day: June 21 plan contract and actual net load.
d=next(i for i,x in enumerate(data.dates) if x.date()==datetime(2025,6,21).date()); x=np.arange(144)/6+1/6
plt.figure(figsize=(11,5.4)); plt.plot(x,data.net_plan_kwh[d],label='实际净负荷',linewidth=1.4); plt.plot(x,B[d],label='零点原计划',linewidth=1.15); plt.plot(x,A[d],label='最终生效合同',linewidth=1.15)
for h in (6,12,18): plt.axvline(h,linestyle='--',linewidth=.9,alpha=.7)
plt.xlabel('计划日时刻／小时'); plt.ylabel('十分钟电量／千瓦时'); plt.title('问题3代表日合同滚动调整（2025-06-21）'); plt.xlim(0,24); plt.xticks(range(0,25,3)); plt.legend(ncol=3); plt.grid(alpha=.2); finish('fig_q3_typical_day.png')

# 2) A/B/C/D cost comparison.
labels=['继承第二问\n关闭新增信息及调整','新增预报\n合同冻结','沿用历史预报\n允许调整','完整第三问\n新增预报及调整']; vals=np.array([m[k]['total_cost_yuan'] for k in ('A','B','C','D')])/1e6
plt.figure(figsize=(8.6,5.3)); bars=plt.bar(labels,vals); plt.ylabel('正式期总费用 / 百万元'); plt.title('2×2 因子对照：信息更新与合同灵活性'); plt.grid(axis='y',alpha=.2)
for b,v in zip(bars,vals): plt.text(b.get_x()+b.get_width()/2,v+.03,f'{v:.3f}',ha='center',va='bottom',fontsize=9)
plt.ylim(0,max(vals)*1.12); finish('fig_q3_factorial_abcd.png')

# 3) Main D five-ledger fee decomposition.
sums={'F_plan':0.,'F_cancel':0.,'F_add':0.,'F_emergency':0.}
with open(HERE/'five_ledger.csv',encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        for k in sums:sums[k]+=float(r[k])
keys=['保留计划电量','下调取消费用','上调追加费用','紧急购电']; vv=[sums[k]/1e6 for k in sums]
plt.figure(figsize=(8.3,5.3)); bars=plt.bar(keys,vv); plt.ylabel('费用 / 百万元'); plt.title('主模型 D 正式期费用分解'); plt.grid(axis='y',alpha=.2)
for b,v in zip(bars,vv): plt.text(b.get_x()+b.get_width()/2,v+.02,f'{v:.3f}',ha='center',fontsize=9)
finish('fig_q3_fee_decomposition.png')

# 4) Four representative days, B vs A vs actual net load.
fig,axs=plt.subplots(2,2,figsize=(12,7.5),sharex=True)
for ax,ds in zip(axs.ravel(),('2025-03-20','2025-06-21','2025-09-23','2025-12-21')):
    di=next(i for i,x0 in enumerate(data.dates) if x0.date()==datetime.fromisoformat(ds).date()); ax.plot(x,data.net_plan_kwh[di],label='实际净负荷',linewidth=1.0); ax.plot(x,B[di],label='原计划',linewidth=1.0); ax.plot(x,A[di],label='最终合同',linewidth=1.0)
    for h in (6,12,18):ax.axvline(h,linestyle='--',linewidth=.7,alpha=.55)
    ax.set_title(ds);ax.set_xlim(0,24);ax.grid(alpha=.18);ax.set_ylabel('千瓦时／十分钟')
axs[1,0].set_xlabel('时刻／小时');axs[1,1].set_xlabel('时刻／小时');axs[0,0].legend(ncol=3,fontsize=8)
fig.suptitle('四个季节代表日：原始合同、最终合同与实际净负荷',y=1.01); finish('fig_q3_representative_days.png')

# 5) Marginal value / dead-zone threshold audit.
rows=marg['rows_data']; xx=np.arange(len(rows)); mu=np.array([r['mu_fd'] for r in rows]); lo=np.array([r['deadzone_low_0.5p'] for r in rows]); hi=np.array([r['deadzone_high_1.5p'] for r in rows]); names=[r['date'][5:]+'\n'+str(r['event_hour'])+'h' for r in rows]
plt.figure(figsize=(10.5,5.2)); plt.plot(xx,mu,'o-',label='边际价值 μ'); plt.plot(xx,lo,'s--',label='0.5p 下调阈值'); plt.plot(xx,hi,'^--',label='1.5p 上调阈值'); plt.xticks(xx,names); plt.ylabel('元/kWh'); plt.title(f'合同调整死区边际价值审计（方向一致 {marg["direction_match_count"]}/{marg["rows"]}）'); plt.grid(alpha=.2); plt.legend(ncol=3); finish('fig_q3_marginal_deadzone.png')

# 6) Semantic robustness.
sv=[m['D']['total_cost_yuan'],sem['sunk_plan_plus_penalty']['total_cost_yuan'],sem['stepwise_revision']['total_cost_yuan'],sem['revision_time']['total_cost_yuan']]; sl=['主解释','原计划费另计','逐次调整计费','调整时刻价重计*']; sv=np.array(sv)/1e6
plt.figure(figsize=(8.8,5.2));bars=plt.bar(sl,sv);plt.ylabel('正式期总费用 / 百万元');plt.title('结算语义鲁棒性（一开关）');plt.grid(axis='y',alpha=.2)
for b,v in zip(bars,sv):plt.text(b.get_x()+b.get_width()/2,v+.025,f'{v:.3f}',ha='center',fontsize=9)
plt.figtext(.5,.01,'* 调整时刻价为全年固定策略重计，另附代表日局部非凸优化检查。',ha='center',fontsize=8);finish('fig_q3_semantic_robustness.png')
print('\n'.join(str(p) for p in sorted(FIG.glob('fig_q3_*.png'))))

# Common-period Q2/Q3 cost and physical flow comparison.
cross=json.loads((HERE/'cross_problem_evidence'/'cross_problem_validation.json').read_text(encoding='utf-8'))
a,b=cross['comparison']
fig,axs=plt.subplots(1,2,figsize=(13,5.3))
labels=['第二问','第三问']
keys=['保留计划费用','下调费用','上调费用','紧急费用'];bottom=np.zeros(2)
for key in keys:
    values=np.array([a[key],b[key]])/1e4;axs[0].bar(labels,values,bottom=bottom,label=key);bottom+=values
axs[0].set_ylabel('费用／万元');axs[0].set_title('相同自然日区间的费用分解');axs[0].legend(fontsize=8)
keys=['实际普通取电','未利用合同','充电','放电','紧急购电','真正弃光'];xx=np.arange(len(keys))
for shift,row,label in [(-.2,a,'第二问'),(.2,b,'第三问')]:
    axs[1].bar(xx+shift,[row[k]/1e4 for k in keys],width=.4,label=label)
axs[1].set_xticks(xx);axs[1].set_xticklabels(['实际普通\n取电','未利用\n合同','充电','放电','紧急\n购电','真正\n弃光']);axs[1].set_ylabel('电量／万千瓦时');axs[1].set_title('合同与实际物理流分开比较');axs[1].legend()
for ax in axs:ax.grid(axis='y',alpha=.18)
finish('fig_q3_cross_problem.png')
