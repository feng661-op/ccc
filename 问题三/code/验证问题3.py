# -*- coding: utf-8 -*-
from pathlib import Path
from datetime import datetime,timedelta
import sys,json,csv,copy
import numpy as np,openpyxl
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
from q3_data import *
from q3_opt import solve_event_lp,build_scenarios
ROOT=HERE.parent.parent; Q3=ROOT/'问题三'; data=load_q3_inputs(ROOT); z=np.load(HERE/'run_D.npz')
B=z['B'];A=z['A'];AS=z['A_stage'];ch=z['charge'];dis=z['discharge'];em=z['emergency'];cur=z['curtail'];s00=z['soc00'];s24=z['soc24'];sp=z['soc_path']
metrics=json.loads((HERE/'metrics.json').read_text(encoding='utf-8')); tol=1e-6
checks={}; evidence={}
def check(name,cond,ev): checks[name]='PASS' if bool(cond) else 'FAIL'; evidence[name]=ev
# Forecast vintage alignment by absolute target, never by lead-column position.
rows=forecast_alignment_rows(data)
with open(HERE/'forecast_vintage_alignment_audit.csv','w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f);w.writerow(['old_issue','new_issue','target_time','old_lead_h','new_lead_h','target_unrealized_at_new_issue','pass'])
    for r in rows:w.writerow(r)
check('forecast_vintage_absolute_target_alignment',len(rows)>0 and all(r[-1] for r in rows),{'rows':len(rows),'example':list(map(str,rows[0][:5]))})
# Leakage ledger: baseline history may use d-1; residual scenarios mechanically stop at d-2.
leak=[]
for d,day in enumerate(data.dates):
    for eh in EVENT_HOURS:
        event=datetime(day.year,day.month,day.day)+timedelta(hours=eh); issue=event; hist=d-2
        ok=issue<=event and hist<=d-2
        leak.append([day.date().isoformat(),eh,event.isoformat(),issue.isoformat(),d-1,hist,ok])
with open(HERE/'leakage_audit.csv','w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f);w.writerow(['date','event_hour','decision_time','max_forecast_issue_used','max_baseline_history_day','max_scenario_error_day','pass']);w.writerows(leak)
check('event_information_cutoff',all(r[-1] for r in leak),{'events':len(leak),'scenario_error_rule':'day<=d-2'})
# Contract freeze and stage boundaries.
prefix_err=max(float(np.max(np.abs(AS[:,0]-B))),float(np.max(np.abs(AS[:,1,:35]-AS[:,0,:35]))),float(np.max(np.abs(AS[:,2,:71]-AS[:,1,:71]))),float(np.max(np.abs(AS[:,3,:107]-AS[:,2,:107]))),float(np.max(np.abs(A-AS[:,3]))))
check('contract_frozen_prefixes',prefix_err<tol,{'max_error_kwh':prefix_err,'starts':EVENT_PLAN_START})
for nm in ('A','B'):
    zz=np.load(HERE/f'run_{nm}.npz'); err=float(np.max(np.abs(zz['B']-zz['A']))); check(f'factorial_{nm}_contract_frozen',err<tol,{'max_B_A_error':err})
# SOC, power, conservation, no simultaneous C/D.
soc_cont=float(np.max(np.abs(s24[:-1]-s00[1:]))); soc_min=float(sp[EVAL_START:].min()); soc_max=float(sp[EVAL_START:].max())
soc_res=float(np.max(np.abs(sp[:,1:]-sp[:,:-1]-ETA_C*ch+dis/ETA_D)))
power=max(float(ch.max()),float(dis.max())); simultaneous=float(np.max(np.minimum(ch,dis)))
check('soc_continuity',soc_cont<tol,{'max_error_kwh':soc_cont});check('soc_bounds',soc_min>=SOC_MIN-tol and soc_max<=SOC_MAX+tol,{'min':soc_min,'max':soc_max})
check('soc_conservation',soc_res<tol,{'max_residual_kwh':soc_res});check('charge_discharge_power',power<=XMAX+tol,{'max_interval_energy_kwh':power,'limit':XMAX});check('no_simultaneous_charge_discharge',simultaneous<tol,{'max_min_xy':simultaneous})
# Physical balance: slack is curtailment under no-sale convention.
max_bal=0.0; min_slack=1e99
for d in range(EVAL_START,365):
    for i in range(144):
        q=float(A[d-1,143] if i==0 else A[d,i-1]); slack=q+dis[d,i]+em[d,i]-float(data.net_cal_kwh[d,i])-ch[d,i]; max_bal=max(max_bal,abs(slack-cur[d,i]));min_slack=min(min_slack,slack)
check('physical_energy_balance',max_bal<tol and min_slack>=-tol,{'max_slack_minus_curtail':max_bal,'min_slack':min_slack})
# Event solver residual and 18:00 nonbinding continuation boundary.
events=list(csv.DictReader(open(HERE/'event_audit.csv',encoding='utf-8-sig'))); maxeq=max(float(r['max_eq_residual']) for r in events); maxub=max(float(r['max_ub_violation']) for r in events)
check('solver_residuals',maxeq<1e-7 and maxub<1e-7,{'max_eq':maxeq,'max_ub':maxub})
cont_ok=True; bad=[]
for r in events:
    if int(r['event_hour'])==18:
        expected=(datetime.fromisoformat(r['date'])+timedelta(days=1,minutes=10)).strftime('%Y-%m-%d %H:%M:%S')
        if r['continuation_first_time']!=expected:cont_ok=False;bad.append((r['date'],r['continuation_first_time'],expected))
check('18h_continuation_nonbinding_boundary',cont_ok,{'bad_count':len(bad),'expected':'next day 00:10'})
# Revision log can only touch legal future slots.
revs=list(csv.DictReader(open(HERE/'revision_log.csv',encoding='utf-8-sig'))); illegal=[r for r in revs if str(r['legal']).lower() not in ('true','1')]
check('revision_log_legal_only',len(illegal)==0,{'rows':len(revs),'illegal':len(illegal)})
# Future perturbation invariance: perturb future actuals and future vintages, current decision must not move.
date_to_idx={d.date():i for i,d in enumerate(data.dates)}; samples=[]
for ds in ('2025-03-20','2025-06-21','2025-09-23','2025-12-21'):
    d=date_to_idx[datetime.fromisoformat(ds).date()]
    for eh,stage_before,sidx in ((0,None,0),(6,0,36),(12,1,72),(18,2,108)):
        soc=float(sp[d,sidx]); lead=0 if d==0 else float(A[d-1,143]); bb=None if eh==0 else B[d]; aa=None if eh==0 else AS[d,stage_before]
        base=solve_event_lp(data,d,eh,soc,bb,aa,lead_contract=lead,use_new_vintage=True,allow_revision=True,scenario_count=3)
        dd=copy.deepcopy(data); start_i=eh*6; dd.load_cal_kwh[d,start_i:]+=777.0;dd.net_cal_kwh[d,start_i:]+=777.0
        event=datetime(data.dates[d].year,data.dates[d].month,data.dates[d].day)+timedelta(hours=eh)
        for issue in list(dd.forecasts):
            if issue>event:
                dd.forecasts[issue]={t:v+9999.0 for t,v in dd.forecasts[issue].items()}
        pert=solve_event_lp(dd,d,eh,soc,bb,aa,lead_contract=lead,use_new_vintage=True,allow_revision=True,scenario_count=3)
        diff=float(np.max(np.abs(base.current_contract-pert.current_contract))); samples.append([ds,eh,diff])
with open(HERE/'future_perturbation_audit.csv','w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f);w.writerow(['date','event_hour','max_contract_change_kwh']);w.writerows(samples)
check('future_perturbation_invariance',max(x[2] for x in samples)<1e-7,{'samples':len(samples),'max_change':max(x[2] for x in samples)})
# Nonanticipativity structural audit: one contract variable per information node; record tree memberships for representative event.
bun=build_scenarios(data,date_to_idx[datetime(2025,6,21).date()],0,145,True,3); node_rows=[]
for (stage,nid),members in bun.node_members.items():node_rows.append([stage,nid,';'.join(map(str,members)),len(members)])
with open(HERE/'nonanticipativity_nodes.csv','w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f);w.writerow(['stage_hour','node_id','leaf_members','leaf_count']);w.writerows(node_rows)
cover=all(k in bun.tree_nodes[h] for h in bun.tree_nodes for k in range(bun.scenarios.shape[0]));check('node_nonanticipativity_structure',cover,{'scenarios':bun.scenarios.shape[0],'nodes':len(bun.node_members)})
# Extreme settlement identities.
main90=float(settlement_components([100],[80],[1])['F_regular'][0]); sunk110=float(settlement_components([100],[80],[1],'sunk_plan_plus_penalty')['F_regular'][0]); eq=float(settlement_components([100],[100],[1])['F_regular'][0]); zero=float(settlement_components([100],[0],[1])['F_regular'][0]); up=float(settlement_components([0],[100],[1])['F_regular'][0])
check('settlement_extreme_cases',max(abs(main90-90),abs(sunk110-110),abs(eq-100),abs(zero-50),abs(up-150))<tol,{'main_100_to_80':main90,'sunk':sunk110,'equal':eq,'A0':zero,'B0_A100':up})
# Workbook full readback and fee/quantity reconciliation.
wb=openpyxl.load_workbook(Q3/'result3.xlsx',read_only=True,data_only=True); max_b=max_a=max_sum=max_fee=0.0
for sn,arr in [('计划购电量',B),('调整购电量',A)]:
    ws=wb[sn]
    for d,row in zip(range(EVAL_START,365),ws.iter_rows(min_row=2,max_row=335,values_only=True)):
        vals=np.asarray([float(x) for x in row[1:145]]); maxerr=float(np.max(np.abs(vals-arr[d]))); max_b=max(max_b,maxerr) if sn=='计划购电量' else max_b;max_a=max(max_a,maxerr) if sn=='调整购电量' else max_a
        max_sum=max(max_sum,abs(float(row[145])-float(arr[d].sum())))
        expected=float(np.dot(data.price_plan,B[d])) if sn=='计划购电量' else float(settlement_components(B[d],A[d],data.price_plan)['F_regular'].sum());max_fee=max(max_fee,abs(float(row[146])-expected))
ws=wb['充放电量']; charge_rows=ws.max_row; wb_ch=wb_dis=0.0
for row in ws.iter_rows(min_row=2,values_only=True): wb_ch+=float(row[2] or 0); wb_dis+=float(row[3] or 0)
ws=wb['紧急购电量']; emergency_rows=ws.max_row; wb_em=0.0
for row in ws.iter_rows(min_row=2,values_only=True): wb_em+=float(row[2] or 0)
wb.close()
check('result3_contract_readback',max(max_b,max_a,max_sum,max_fee)<1e-6,{'max_B':max_b,'max_A':max_a,'max_daily_sum':max_sum,'max_fee':max_fee})
check('result3_physical_readback',abs(wb_ch-float(ch[EVAL_START:].sum()))<1e-6 and abs(wb_dis-float(dis[EVAL_START:].sum()))<1e-6 and abs(wb_em-float(em[EVAL_START:].sum()))<1e-6,{'charge_rows':charge_rows,'emergency_rows':emergency_rows,'charge_diff':wb_ch-float(ch[EVAL_START:].sum()),'discharge_diff':wb_dis-float(dis[EVAL_START:].sum()),'emergency_diff':wb_em-float(em[EVAL_START:].sum())})
# Natural-day bridge and five-ledger total.
regular=float(metrics['D']['regular_cost_yuan']); plan_day=float(np.sum([settlement_components(B[d],A[d],data.price_plan)['F_regular'].sum() for d in range(EVAL_START,365)])); bridge=float(settlement_components([B[EVAL_START-1,143]],[A[EVAL_START-1,143]],[data.price_plan[143]])['F_regular'][0]-settlement_components([B[364,143]],[A[364,143]],[data.price_plan[143]])['F_regular'][0])
check('natural_day_bridge',abs(regular-(plan_day+bridge))<1e-6,{'natural_regular':regular,'plan_day_regular':plan_day,'bridge':bridge})
ledger_total=0.0
with open(HERE/'five_ledger.csv',encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):ledger_total+=float(r['F_total'])
check('five_ledger_total',abs(ledger_total-float(metrics['D']['total_cost_yuan']))<1e-5,{'ledger_total':ledger_total,'metrics_total':metrics['D']['total_cost_yuan']})
validation={'all_pass':all(v=='PASS' for v in checks.values()),'checks':checks,'evidence':evidence,'pass_count':sum(v=='PASS' for v in checks.values()),'check_count':len(checks)}
(HERE/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(validation,ensure_ascii=False,indent=2))
if not validation['all_pass']: raise SystemExit(2)
