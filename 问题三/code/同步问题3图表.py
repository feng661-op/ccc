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
'fig_q3_semantic_robustness.png':'图20_问题3结算语义鲁棒性.png'}
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
save_table('表7_指定日期原始与最终合同.png','表7 指定日期原始合同与最终合同',['日期','B总量/kWh','A总量/kWh','A-B/kWh','最大槽位改变量/kWh'],rows)
rows=[]
for ds in dates:
 d=idx[datetime.fromisoformat(ds).date()];rows.append([ds,f'{ch[d].sum():.2f}',f'{dis[d].sum():.2f}',f'{sp[d].min():.2f}',f'{sp[d].max():.2f}',f'{sp[d,-1]:.2f}'])
save_table('表8_指定日期储能充放电.png','表8 指定日期储能运行',['日期','充电/kWh','放电/kWh','SOC最小/kWh','SOC最大/kWh','日末SOC/kWh'],rows)
rows=[]
for ds in dates:
 d=idx[datetime.fromisoformat(ds).date()]; nz=int(np.sum(em[d]>1e-8));rows.append([ds,f'{em[d].sum():.2f}',str(nz),f'{em[d].max():.2f}',f'{np.dot(em[d],5*data.price_calendar):.2f}'])
save_table('表9_指定日期紧急购电.png','表9 指定日期紧急购电',['日期','紧急购电/kWh','非零槽位数','最大单槽/kWh','日紧急购电费用/元'],rows)
print('synced',len(mapping)+3,'figures/tables')
