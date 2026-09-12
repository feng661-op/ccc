# -*- coding: utf-8 -*-
from pathlib import Path
from datetime import datetime
import shutil,sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
HERE=Path(__file__).resolve().parent; Q3=HERE.parent; SRC=Q3/'figures'; DST=Q3/'output'/'图表'; DST.mkdir(parents=True,exist_ok=True)
for f in ('Microsoft YaHei','SimHei','Noto Sans CJK SC'):
    if any(x.name==f for x in font_manager.fontManager.ttflist): plt.rcParams['font.sans-serif']=[f]; break
plt.rcParams['axes.unicode_minus']=False
mapping={
'fig_q3_typical_day.png':'图15_问题3典型日合同滚动调整.png',
'fig_q3_factorial_abcd.png':'图16_问题3ABCD因子对照.png',
'fig_q3_fee_decomposition.png':'图17_问题3主模型费用分解.png',
'fig_q3_representative_days.png':'图18_问题3四季代表日合同与净负荷.png',
'fig_q3_marginal_deadzone.png':'图19_合同调整死区边际价值.png',
'fig_q3_semantic_robustness.png':'图20_问题3结算语义鲁棒性.png',
'fig_q3_cross_problem.png':'附图_第二问第三问同口径对照.png'}
for a,b in mapping.items(): shutil.copy2(SRC/a,DST/b)
sys.path.insert(0,str(HERE)); from q3_data import load_q3_inputs
data=load_q3_inputs(Q3.parent); z=np.load(HERE/'run_D.npz');B=z['B'];A=z['A'];ch=z['charge'];dis=z['discharge'];em=z['emergency'];sp=z['soc_path']
idx={d.date():i for i,d in enumerate(data.dates)}; dates=('2025-03-20','2025-06-21','2025-09-23','2025-12-21')
def save_table(filename,title,cols,rows):
    fig,ax=plt.subplots(figsize=(13,3.8)); ax.axis('off'); ax.set_title(title,fontsize=15,pad=14)
    tab=ax.table(cellText=rows,colLabels=cols,loc='center',cellLoc='center');tab.auto_set_font_size(False);tab.set_fontsize(10);tab.scale(1,1.7)
    fig.tight_layout();fig.savefig(DST/filename,dpi=220,bbox_inches='tight');plt.close(fig)
rows=[]
for ds in dates:
 d=idx[datetime.fromisoformat(ds).date()]; rows.append([ds,f'{B[d].sum():.2f}',f'{A[d].sum():.2f}',f'{(A[d]-B[d]).sum():.2f}',f'{np.max(np.abs(A[d]-B[d])):.2f}'])
save_table('表7_指定日期原始与最终合同.png','表7 指定日期原始合同与最终合同',['日期','原计划／千瓦时','最终合同／千瓦时','净调整／千瓦时','最大单段调整／千瓦时'],rows)
rows=[]
for ds in dates:
 d=idx[datetime.fromisoformat(ds).date()];rows.append([ds,f'{ch[d].sum():.2f}',f'{dis[d].sum():.2f}',f'{sp[d].min():.2f}',f'{sp[d].max():.2f}',f'{sp[d,-1]:.2f}'])
save_table('表8_指定日期储能充放电.png','表8 指定日期储能运行',['日期','充电/kWh','放电/kWh','储电量最小／千瓦时','储电量最大／千瓦时','日末储电量／千瓦时'],rows)
rows=[]
for ds in dates:
 d=idx[datetime.fromisoformat(ds).date()]; nz=int(np.sum(em[d]>1e-8));rows.append([ds,f'{em[d].sum():.2f}',str(nz),f'{em[d].max():.2f}',f'{np.dot(em[d],5*data.price_calendar):.2f}'])
save_table('表9_指定日期紧急购电.png','表9 指定日期紧急购电',['日期','紧急购电/kWh','非零槽位数','最大单槽/kWh','日紧急购电费用/元'],rows)
# Regenerate the framework; the legacy static figure used the old timing.
fig=plt.figure(figsize=(13.5,7.5));ax=fig.add_axes([.03,.12,.94,.76]);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
boxes=[(.25,.84,'合同信息集\n已发布PV版本 + 已完成历史\n不读取未来/当前未完成汇总'),(.75,.84,'继承第二问日前规划\n九场景、四十八小时视野及尾部风险\n六点、十二点、十八点可调整未交付合同'),(.25,.49,'当前测量近似\n10分钟量视作段内分段常值\n仅进入下层控制，不回写B/A'),(.75,.49,'继承第二问实时反馈\n富余充电，按场景储备线放电\n关闭扩展时逐段复现第二问'),(.25,.14,'物理流独立核算\n实际取电I ≤ 付费合同A；未用u=A−I\n真正弃光κ ≤ PV；禁止电池无效耗散'),(.75,.14,'费用与跨日状态\n未用合同仍付费；紧急补购5倍\n真实SOC续接 + 自然日/计划行桥接')]
for x,y,txt in boxes:ax.text(x,y,txt,ha='center',va='center',fontsize=11,bbox={'boxstyle':'round,pad=.7','fill':False})
for start,end in [((.45,.84),(.54,.84)),((.75,.72),(.75,.61)),((.45,.49),(.54,.49)),((.75,.37),(.25,.26)),((.45,.14),(.54,.14))]:ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','lw':1.3})
fig.suptitle('图14 继承第二问基础能力的第三问调度流程',fontsize=17,y=.97)
fig.text(.5,.03,'测量近似不等于已验证零延迟传感器；未来真实数据不得参与当前合同或控制。',ha='center',fontsize=10)
fig.savefig(DST/'图14_问题3事件驱动滚动调度框架.png',dpi=220,bbox_inches='tight');plt.close(fig)
print('synced',len(mapping)+4,'figures/tables')

rob=Q3/'output'/'robustness'
shutil.copy2(SRC/'fig_q3_semantic_robustness.png',rob/'图20_结算语义鲁棒性.png')
shutil.copy2(SRC/'fig_q3_marginal_deadzone.png',rob/'图19_合同调整死区边际价值.png')
