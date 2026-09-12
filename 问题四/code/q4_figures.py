# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import json, math
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from q4_data import load_q4_inputs
ROOT=Path(__file__).resolve().parents[2]; CODE=Path(__file__).resolve().parent
FIG=ROOT/'问题四'/'output'/'figures'; SUB=FIG/'subdata'; FIG.mkdir(parents=True,exist_ok=True); SUB.mkdir(parents=True,exist_ok=True)
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','Arial Unicode MS','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False;plt.rcParams['figure.dpi']=150

def save(fig,name):
 fig.tight_layout();fig.savefig(FIG/f'{name}.png',bbox_inches='tight',dpi=220);fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight');plt.close(fig)

def f1(data):
 c=data.price_plan[:31]; n=data.net_plan_kwh[:31]/(1/6)/1000
 x=np.arange(144)*10/60+1/6
 cm=c.mean(0); nm=n.mean(0); cd=c-c.mean(0);nd=n-n.mean(0);corr=np.corrcoef(cd.ravel(),nd.ravel())[0,1]
 pd.DataFrame({'hour':x,'jan_mean_price':cm,'jan_mean_net_MW':nm}).to_csv(SUB/'fig1_structure.csv',index=False,encoding='utf-8-sig')
 fig,ax=plt.subplots(figsize=(9.2,4.4));ax.plot(x,cm,label='1月平均实时电价');ax.set_xlabel('日内时刻 / h');ax.set_ylabel(r'电价 / (元·kWh$^{-1}$)');ax.grid(alpha=.2)
 ax2=ax.twinx();ax2.plot(x,nm,linestyle='--',label='1月平均净负荷');ax2.set_ylabel('净负荷 / MW')
 h1,l1=ax.get_legend_handles_labels();h2,l2=ax2.get_legend_handles_labels();ax.legend(h1+h2,l1+l2,loc='upper left')
 ax.set_title(f'图1  附件结构证据：价格形状与净负荷共振（异常相关 r={corr:.3f}，仅结构证据）');save(fig,'fig1_structure_evidence')

def f2():
 fig,ax=plt.subplots(figsize=(10,4.0));ax.set_xlim(-.5,24.6);ax.set_ylim(-.2,4.2);ax.axis('off')
 ax.hlines(1,0,24,linewidth=2);events=[0,6,12,18,24]
 for h in events:ax.vlines(h,.85,1.15);ax.text(h,0.58,f'{h:02d}:00',ha='center')
 for h in (0,6,12,18):
  ax.add_patch(FancyBboxPatch((h+.15,2.7),5.5,.55,boxstyle='round,pad=.04',fill=False));ax.text(h+2.9,2.98,f'{h:02d}:00：获得新PV预报\nQ4-3可重算后续合同',ha='center',va='center',fontsize=9)
 ax.text(12,1.65,'实时电价：仅到当前决策时刻已揭示；未来价格必须因果预测',ha='center',bbox=dict(boxstyle='round',fc='white'))
 ax.text(12,.05,'执行层每10分钟使用当前负荷/PV测量，Q4-2始终禁止使用附件3未来PV',ha='center')
 ax.set_title('图2  信息时间轴与权限边界：事件层—执行层双时间坐标');save(fig,'fig2_information_timeline')

def f3():
 fig,ax=plt.subplots(figsize=(10,4.8));ax.set_xlim(0,10);ax.set_ylim(0,5);ax.axis('off')
 pdec=json.loads((ROOT/'问题四'/'output'/'02_model_selection'/'price_freeze_decision.json').read_text(encoding='utf-8'))
 pc=pdec['chosen']; price_label=f"1月公平冻结\n{pc['model']} / {pc['block_scheme']}"
 boxes=[(0.4,3.3,1.7,.9,price_label),(2.6,3.3,1.7,.9,'联合残差路径\n同源日 + medoid'),(4.8,3.3,1.7,.9,'事件层LP\nB0/B1/B2'),(7.0,3.3,2.1,.9,'源—汇物理内核\nqL,qB,gL,gB,u,κ,e,x,y,S'),(4.8,1.2,1.7,.9,'10 min执行层\n当前测量+SOC目标'),(7.0,1.2,2.1,.9,'现金/物理账本\nExcel·CSV·JSON')]
 for x,y,w,h,t in boxes:
  ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',fill=False,linewidth=1.5));ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=10)
 arrows=[((2.1,3.75),(2.6,3.75)),((4.3,3.75),(4.8,3.75)),((6.5,3.75),(7.0,3.75)),((8.0,3.3),(5.65,2.1)),((6.5,1.65),(7.0,1.65))]
 for a,b in arrows:ax.add_patch(FancyArrowPatch(a,b,arrowstyle='->',mutation_scale=13))
 ax.text(5,4.65,'结构证据 ≠ 预测提升 ≠ 调度收益；1月冻结后不得用2—12月回调',ha='center',fontsize=10)
 ax.set_title('图3  V1.2模型结构：预测—场景—优化—同构执行—账本');save(fig,'fig3_model_structure')

def f4():
 r0=json.loads((CODE/'r0_repricing.json').read_text(encoding='utf-8')); ladder=json.loads((CODE/'baseline_ladder.json').read_text(encoding='utf-8'))
 rows=[]
 for br in ('q4_2','q4_3'):
  fs=json.loads((CODE/br/'formal_summary.json').read_text(encoding='utf-8'));main=fs['formal']['cash_total_yuan']
  rr=r0[br]['R0_variable_price']['cash_total_yuan'];rows.append({'branch':br,'selected':fs['selected'],'R0_reprice':rr/1e6,'formal_selected':main/1e6})
 pd.DataFrame(rows).to_csv(SUB/'fig4_formal_cash.csv',index=False,encoding='utf-8-sig')
 fig,axs=plt.subplots(1,2,figsize=(10.5,4.3));x=np.arange(2);w=.35
 axs[0].bar(x-w/2,[r['R0_reprice'] for r in rows],w,label='R0历史轨迹仅重计价');axs[0].bar(x+w/2,[r['formal_selected'] for r in rows],w,label='1月冻结正式策略');axs[0].set_xticks(x,[f"Q4-2\n{rows[0]['selected']}",f"Q4-3\n{rows[1]['selected']}"]);axs[0].set_ylabel('334日现金费用 / 百万元');axs[0].legend(fontsize=8);axs[0].set_title('正式期：R0仅作历史暴露参照')
 e0p=CODE/'ablations'/'E0_matched_policy_benchmark.json'
 if e0p.exists():
  c0=json.loads(e0p.read_text(encoding='utf-8'))['matched_no_price_adaptation']['cash_total_yuan']/1e6
  axs[0].scatter([0],[c0],marker='D',s=48,label='Q4-2匹配无价格适应C0',zorder=5);axs[0].legend(fontsize=8)
 for br,xx in zip(('q4_2','q4_3'),[0,1]):
  d=ladder['branches'][br]; vals=[d['B0']['cash_cvar95_yuan_per_day'],d['B1']['cash_cvar95_yuan_per_day']]
  labels=['B0','B1'];
  if d['B2_candidates']:
   best=min(d['B2_candidates'],key=lambda z:z['cash_cvar95_yuan_per_day']);vals.append(best['cash_cvar95_yuan_per_day']);labels.append('B2最小CVaR候选')
  off=(xx-.5)*.08;axs[1].plot(labels,np.array(vals)/1000,marker='o',label=br)
 axs[1].set_ylabel('1月选模期日现金CVaR95 / 千元');axs[1].set_title('1月决策证据：复杂度按冻结门槛进入');axs[1].legend();axs[1].grid(alpha=.2)
 fig.suptitle(f"图4  成本—风险主结果（R0非算法节费基线；正式冻结 Q4-2={rows[0]['selected']}、Q4-3={rows[1]['selected']}）",fontsize=11);save(fig,'fig4_cost_risk')

def f5():
 br='q4_3';days=pd.read_csv(CODE/br/'daily_ledger.csv');q=float(days.cash_fee_yuan.quantile(.95));r=days.iloc[(days.cash_fee_yuan-q).abs().argsort()[:1]].iloc[0];ds=r.date
 s=pd.read_csv(CODE/br/'physical_10min.csv');s=s[s.date==ds].sort_values('slot').copy();s['hour']=s.slot/6
 s[['hour','price_yuan_per_kwh','load_kwh','pv_kwh','B_kwh','A_kwh','I_kwh','S1_kwh','e_kwh','kappa']].to_csv(SUB/'fig5_typical_day.csv',index=False,encoding='utf-8-sig')
 fig,axs=plt.subplots(3,1,figsize=(10,7.0),sharex=True)
 axs[0].plot(s.hour,s.price_yuan_per_kwh,label='实时电价');axs[0].set_ylabel('元/kWh');axs[0].legend();axs[0].grid(alpha=.2)
 axs[1].plot(s.hour,s.load_kwh-s.pv_kwh,label='实际净负荷');axs[1].plot(s.hour,s.B_kwh,label='B原始合同',alpha=.8);axs[1].plot(s.hour,s.A_kwh,label='A最终合同',alpha=.8);axs[1].plot(s.hour,s.I_kwh,label='实际合同电使用I',linestyle='--');axs[1].set_ylabel('10 min能量 / kWh');axs[1].legend(ncol=4,fontsize=8);axs[1].grid(alpha=.2)
 axs[2].plot(s.hour,s.S1_kwh,label='SOC');axs[2].fill_between(s.hour,0,s.e_kwh,alpha=.25,label='紧急购电e');axs[2].fill_between(s.hour,0,s.kappa,alpha=.18,label='弃光κ');axs[2].set_ylabel('kWh');axs[2].set_xlabel('时刻 / h');axs[2].legend(ncol=3,fontsize=8);axs[2].grid(alpha=.2)
 fig.suptitle(f'图5  Q4-3典型高成本日联合调度：{ds}（接近日费用95%分位）');save(fig,'fig5_typical_dispatch')

def f6():
 ab=json.loads((CODE/'ablations'/'ablation_matrix_E0_E15.json').read_text(encoding='utf-8'));e6=ab['experiments']['E6'];e13=ab['experiments']['E13']
 pd.DataFrame([{'variant':k,**v} for k,v in e6.items()]).to_csv(SUB/'fig6_E6.csv',index=False,encoding='utf-8-sig')
 fig,axs=plt.subplots(1,2,figsize=(10.5,4.2));ks=list(e6);cash=[e6[k]['cash_total_yuan']/1e6 for k in ks];axs[0].bar(range(len(ks)),cash);axs[0].set_xticks(range(len(ks)),['无更新/无调整','仅PV更新','仅合同调整','两者都有'],rotation=15);axs[0].set_ylabel('Q4-3正式现金 / 百万元');axs[0].set_title('信息×权限 A/B/C/D')
 for br,mark in [('q4_2','o'),('q4_3','s')]:
  z=e13[br];xs=[28,42,56];ys=[z['28']['cash_total_yuan']/1e6,z['42_frozen_main']['cash_total_yuan']/1e6,z['56']['cash_total_yuan']/1e6];axs[1].plot(xs,ys,marker=mark,label=br)
 axs[1].set_xticks(xs);axs[1].set_xlabel('历史窗 / 天');axs[1].set_ylabel('正式现金 / 百万元');axs[1].set_title('历史窗仅作冻结后敏感性');axs[1].legend();axs[1].grid(alpha=.2)
 fig.suptitle('图6  机制消融：信息权限与历史窗（均不回调主模型）');save(fig,'fig6_mechanism_ablation')

def appendices():
 pm=pd.read_csv(ROOT/'问题四'/'output'/'02_model_selection'/'price_model_ablation.csv');pdec=json.loads((ROOT/'问题四'/'output'/'02_model_selection'/'price_freeze_decision.json').read_text(encoding='utf-8'));fig,ax=plt.subplots(figsize=(9,4.3));g=pm.sort_values('mae');labels=[f"{r.model}/{r.block_scheme}/a={r.alpha:g}" for _,r in g.iterrows()];ax.bar(range(len(g)),g.mae);ax.set_xticks(range(len(g)),labels,rotation=70,ha='right',fontsize=7);ax.set_ylabel('1月walk-forward MAE /(元/kWh)');ax.set_title(f"附图A 价格预测公平消融：最终冻结 {pdec['chosen']['model']}");save(fig,'appendix_A_price_ablation')
 eps=json.loads((CODE/'epsilon_calibration.json').read_text(encoding='utf-8'));fig,ax=plt.subplots(figsize=(8.5,4.0));
 for br in ('q4_2','q4_3'):ax.plot(eps[br]['distances'],marker='o',markersize=3,label=br)
 ax.set_ylabel('相邻经验分布 W1漂移尺度');ax.set_xlabel('1月相邻walk-forward分布对');ax.set_title('附图B ε校准：经验分布漂移尺度（非置信半径）');ax.legend();ax.grid(alpha=.2);save(fig,'appendix_B_epsilon_drift')
 if (CODE/'oracle_audit.json').exists():
  o=json.loads((CODE/'oracle_audit.json').read_text(encoding='utf-8'));rows=[]
  for br in ('q4_2','q4_3'):rows.append({'branch':br,'Causal':o[br]['causal']['cash_total_yuan']/1e6,'Price Oracle':o[br]['price_oracle']['cash_total_yuan']/1e6,'Full-info Oracle':o[br]['full_information']['strict_cash_lower_bound_yuan']/1e6})
  fig,ax=plt.subplots(figsize=(8.5,4.2));x=np.arange(2);w=.25
  for j,k in enumerate(('Causal','Price Oracle','Full-info Oracle')):ax.bar(x+(j-1)*w,[r[k] for r in rows],w,label=k)
  ax.set_xticks(x,['Q4-2','Q4-3']);ax.set_ylabel('334日现金 / 百万元');ax.set_title('附图C Oracle：Price Oracle仅诊断；匹配FI才是严格现金下界');ax.legend(fontsize=8);save(fig,'appendix_C_oracle')

def run():
 data=load_q4_inputs(ROOT);f1(data);f2();f3();f4();f5();f6();appendices();
 m={p.name:__import__('hashlib').sha256(p.read_bytes()).hexdigest() for p in FIG.iterdir() if p.is_file()};(CODE/'figure_manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({'figures':len(m),'files':list(m)},ensure_ascii=False))
if __name__=='__main__':run()
