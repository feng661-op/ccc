# -*- coding: utf-8 -*-
from pathlib import Path
from copy import copy
import sys,shutil,csv
import numpy as np,openpyxl
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
from q3_data import *
ROOT=HERE.parent.parent; Q3=ROOT/'问题三'; data=load_q3_inputs(ROOT); z=np.load(HERE/'run_D.npz')
B=z['B'];A=z['A'];ch=z['charge'];dis=z['discharge'];em=z['emergency'];cur=z['curtail'];sp=z['soc_path']
sl=range(EVAL_START,365); out=Q3/'result3.xlsx'; shutil.copy2(ROOT/'26C题'/'附件'/'附件5'/'result3.xlsx',out)
wb=openpyxl.load_workbook(out)
for sn,arr in [('计划购电量',B),('调整购电量',A)]:
    ws=wb[sn]
    for rr,d in enumerate(sl,start=2):
        ws.cell(rr,1).value=data.dates[d]
        for j in range(144): ws.cell(rr,2+j).value=float(arr[d,j])
        ws.cell(rr,146).value=float(arr[d].sum())
        if sn=='计划购电量': fee=float(np.dot(data.price_plan,B[d]))
        else: fee=float(settlement_components(B[d],A[d],data.price_plan)['F_regular'].sum())
        ws.cell(rr,147).value=fee
# Expand charge/discharge sheet to all 334 days x 6 blocks.
ws=wb['充放电量']; sample_styles=[[copy(ws.cell(2+(r%6),c)._style) for c in range(1,7)] for r in range(6)]
ws.delete_rows(2,ws.max_row-1)
row=2
blocks=[(0,24,'0:00-4:00'),(24,48,'4:00-8:00'),(48,72,'8:00-12:00'),(72,96,'12:00-16:00'),(96,120,'16:00-20:00'),(120,144,'20:00-24:00')]
for d in sl:
    for bi,(a,b,label) in enumerate(blocks):
        vals=[data.dates[d] if bi==0 else None,label,float(ch[d,a:b].sum()),float(dis[d,a:b].sum()),'0:00' if bi==0 else ('24:00' if bi==1 else None),float(sp[d,0] if bi==0 else sp[d,144]) if bi<2 else None]
        for c,v in enumerate(vals,start=1): ws.cell(row,c).value=v; ws.cell(row,c)._style=copy(sample_styles[bi][c-1])
        row+=1
# Expand emergency sheet: one or more rows/day, every contiguous actual emergency interval is retained.
ws=wb['紧急购电量']; style=[copy(ws.cell(2,c)._style) for c in range(1,4)]; ws.delete_rows(2,ws.max_row-1); row=2
for d in sl:
    groups=grouped_emergency_periods(em[d]); groups=groups if groups else [('',0.0)]
    for gi,(period,qty) in enumerate(groups):
        vals=[data.dates[d] if gi==0 else None,period if period else None,float(qty)]
        for c,v in enumerate(vals,start=1): ws.cell(row,c).value=v; ws.cell(row,c)._style=copy(style[c-1])
        row+=1
wb.save(out)
# Five-ledger settlement audit on the formal natural-day basis. 00:00-00:10
# mechanically bridges to the previous plan day's j=143, so this file sums
# exactly to the reported formal total cost.
with open(HERE/'five_ledger.csv','w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f); w.writerow(['date','natural_slot_i','contract_plan_date','contract_slot_j','price','B_kwh','A_final_kwh','retained_kwh','cancel_kwh','add_kwh','F_plan','F_cancel','F_add','F_emergency','F_total'])
    for d in sl:
        for i in range(144):
            pd,j=(d-1,143) if i==0 else (d,i-1); p=float(data.price_plan[j]); comp=settlement_components([B[pd,j]],[A[pd,j]],[p]); femg=5*p*float(em[d,i])
            w.writerow([data.dates[d].date().isoformat(),i,data.dates[pd].date().isoformat(),j,p,B[pd,j],A[pd,j],comp['keep_kwh'][0],comp['down_kwh'][0],comp['up_kwh'][0],comp['F_plan'][0],comp['F_cancel'][0],comp['F_add'][0],femg,float(comp['F_regular'][0]+femg)])
# Natural-day physical audit.
with open(HERE/'physical_10min.csv','w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f); w.writerow(['date','slot_i','start','price','contract_kwh','net_actual_kwh','charge_kwh','discharge_kwh','emergency_kwh','curtail_kwh','soc_start','soc_end','soc_balance_residual','power_balance_slack'])
    for d in sl:
        for i in range(144):
            q=float(A[d-1,143] if i==0 else A[d,i-1]); net=float(data.net_cal_kwh[d,i]); s0=float(sp[d,i]); s1=float(sp[d,i+1]); r=s1-s0-ETA_C*ch[d,i]+dis[d,i]/ETA_D; slack=q+dis[d,i]+em[d,i]-net-ch[d,i]
            t=calendar_slot_start(data.dates[d],i)
            w.writerow([data.dates[d].date().isoformat(),i,t.strftime('%H:%M'),data.price_calendar[i],q,net,ch[d,i],dis[d,i],em[d,i],cur[d,i],s0,s1,r,slack])
print(out)
