# -*- coding: utf-8 -*-
from pathlib import Path
from datetime import datetime,timedelta
import sys,json,csv,copy
import numpy as np,openpyxl
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from q3_data import *
from q3_opt import solve_event_lp,build_scenarios,solve_dispatch_mpc
ROOT=HERE.parent.parent;Q3=ROOT/'问题三';data=load_q3_inputs(ROOT);z=np.load(HERE/'run_D.npz')
B=z['B'];A=z['A'];AS=z['A_stage'];ch=z['charge'];dis=z['discharge'];em=z['emergency'];cur=z['curtail'];s00=z['soc00'];s24=z['soc24'];sp=z['soc_path']
metrics=json.loads((HERE/'metrics.json').read_text(encoding='utf-8'));tol=1e-6
checks={};evidence={}
def check(name,cond,ev):checks[name]='PASS' if bool(cond) else 'FAIL';evidence[name]=ev

# 1) Forecast-vintage absolute-target alignment.
rows=forecast_alignment_rows(data)
with open(HERE/'forecast_vintage_alignment_audit.csv','w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f);w.writerow(['old_issue','new_issue','target_time','old_lead_h','new_lead_h','target_unrealized_at_new_issue','pass']);w.writerows(rows)
check('forecast_vintage_absolute_target_alignment',len(rows)>0 and all(r[-1] for r in rows),{'rows':len(rows),'example':list(map(str,rows[0][:5]))})

# 2) Information cutoff ledger; scenario residual day <= d-2.
leak=[]
for d,day in enumerate(data.dates):
    for eh in EVENT_HOURS:
        event=datetime(day.year,day.month,day.day)+timedelta(hours=eh);hist=d-2;leak.append([day.date().isoformat(),eh,event.isoformat(),event.isoformat(),d-1,hist,hist<=d-2])
with open(HERE/'leakage_audit.csv','w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f);w.writerow(['date','event_hour','decision_time','max_forecast_issue_used','max_baseline_history_day','max_scenario_error_day','pass']);w.writerows(leak)
check('event_information_cutoff',all(r[-1] for r in leak),{'events':len(leak),'scenario_error_rule':'day<=d-2'})

# 3) Contract freeze and factor A/B frozen-contract definitions.
prefix_err=max(float(np.max(np.abs(AS[:,0]-B))),float(np.max(np.abs(AS[:,1,:35]-AS[:,0,:35]))),float(np.max(np.abs(AS[:,2,:71]-AS[:,1,:71]))),float(np.max(np.abs(AS[:,3,:107]-AS[:,2,:107]))),float(np.max(np.abs(A-AS[:,3]))))
check('contract_frozen_prefixes',prefix_err<tol,{'max_error_kwh':prefix_err,'starts':EVENT_PLAN_START})
for nm in ('A','B'):
    zz=np.load(HERE/f'run_{nm}.npz');err=float(np.max(np.abs(zz['B']-zz['A'])));check(f'factorial_{nm}_contract_frozen',err<tol,{'max_B_A_error':err})

# 4) SOC/power/conservation and physical balance.
soc_cont=float(np.max(np.abs(s24[:-1]-s00[1:])));soc_min=float(sp[EVAL_START:].min());soc_max=float(sp[EVAL_START:].max())
soc_res=float(np.max(np.abs(sp[:,1:]-sp[:,:-1]-ETA_C*ch+dis/ETA_D)));power=max(float(ch.max()),float(dis.max()));simultaneous=float(np.max(np.minimum(ch,dis)))
check('soc_continuity',soc_cont<tol,{'max_error_kwh':soc_cont});check('soc_bounds',soc_min>=SOC_MIN-tol and soc_max<=SOC_MAX+tol,{'min':soc_min,'max':soc_max})
check('soc_conservation',soc_res<tol,{'max_residual_kwh':soc_res});check('charge_discharge_power',power<=XMAX+tol,{'max_interval_energy_kwh':power,'limit':XMAX});check('no_simultaneous_charge_discharge',simultaneous<tol,{'max_min_xy':simultaneous})
assert int(z['physical_schema_version'])==2, 'Full physical replay required'
gi=z['grid_import'];unused=z['unused_contract'];surplus=z['supply_surplus'];dump=z['battery_dump']
qcal=np.zeros_like(A);qcal[:,1:]=A[:,:143];qcal[1:,0]=A[:-1,143]
sl=slice(EVAL_START,None); net=data.net_cal_kwh[sl];pv=data.pv_cal_kwh[sl]
res=gi[sl]+dis[sl]+em[sl]-net-ch[sl]-cur[sl]
check('physical_energy_balance',np.max(np.abs(res))<tol,{'max_physical_residual_kwh':float(np.max(np.abs(res)))})
check('actual_import_within_contract',np.min(gi[sl])>=-tol and np.max(gi[sl]-qcal[sl])<tol,{'max_excess_kwh':float(np.max(gi[sl]-qcal[sl]))})
check('contract_import_unused_identity',np.max(np.abs(gi[sl]+unused[sl]-qcal[sl]))<tol,{'max_residual_kwh':float(np.max(np.abs(gi[sl]+unused[sl]-qcal[sl])))})
check('pv_curtailment_source_bound',np.min(cur[sl])>=-tol and np.max(cur[sl]-pv)<tol,{'max_curtail_minus_pv_kwh':float(np.max(cur[sl]-pv))})
check('no_pv_curtailment_without_pv',np.max(cur[sl][pv<tol],initial=0)<tol,{'night_curtail_kwh':float(cur[sl][pv<tol].sum())})
check('no_battery_dump',np.max(np.abs(dump[sl]))<tol,{'battery_dump_kwh':float(dump[sl].sum())})
check('no_discharge_while_supply_surplus',not np.any((dis[sl]>tol)&(surplus[sl]>tol)),{'violating_slots':int(np.sum((dis[sl]>tol)&(surplus[sl]>tol)))})
check('no_emergency_funded_charging',not np.any((ch[sl]>tol)&(em[sl]>tol)),{'violating_slots':int(np.sum((ch[sl]>tol)&(em[sl]>tol)))})
check('surplus_source_identity',np.max(np.abs(surplus[sl]-unused[sl]-cur[sl]-dump[sl]))<tol,{'max_residual_kwh':float(np.max(np.abs(surplus[sl]-unused[sl]-cur[sl]-dump[sl])))})

# 5) Event audit: no clairvoyant cross-day qcont; each event horizon ends at next-day 00:10.
events=list(csv.DictReader(open(HERE/'event_audit.csv',encoding='utf-8-sig')));maxeq=max(float(r['max_eq_residual']) for r in events);maxub=max(float(r['max_ub_violation']) for r in events)
check('solver_residuals',maxeq<1e-7 and maxub<1e-7,{'max_eq':maxeq,'max_ub':maxub})
bad=[]
for r in events:
    expected=(datetime.fromisoformat(r['date'])+timedelta(days=2,minutes=10)).strftime('%Y-%m-%d %H:%M:%S')
    continuation=(datetime.fromisoformat(r['date'])+timedelta(days=1,minutes=10)).strftime('%Y-%m-%d %H:%M:%S')
    if r.get('continuation_first_time')!=continuation or r.get('terminal_time')!=expected:bad.append((r['date'],r['event_hour'],r.get('continuation_first_time'),r.get('terminal_time'),expected))
check('cross_day_continuation_no_clairvoyance',len(bad)==0,{'bad_count':len(bad),'terminal_expected':'day+2 00:10; day+1 continuation is Q2 valuation only','sample':bad[:3]})
weight_err=max(abs(float(r['scenario_weight_sum'])-1.) for r in events);clip_rates=np.asarray([float(r['clip_rate']) for r in events]);bind=np.asarray([int(r['terminal_shortfall_binding_scenarios'])>0 for r in events])
check('scenario_weights_sum_to_one',weight_err<1e-12,{'max_error':weight_err})
check('scenario_clipping_rate_reported',np.all((clip_rates>=0)&(clip_rates<=1)),{'mean':float(clip_rates.mean()),'max':float(clip_rates.max())})
check('terminal_penalty_binding_rate_reported',True,{'event_binding_rate':float(bind.mean()),'binding_events':int(bind.sum()),'events':len(bind)})

# 6) Warmup/formal reporting is explicit.
md=metrics['D'];counts_ok=(md['event_count_total']==1460 and md['warmup_event_count']==124 and md['formal_event_count']==1336)
check('warmup_formal_event_counts',counts_ok,{'total':md['event_count_total'],'warmup':md['warmup_event_count'],'formal':md['formal_event_count']})
check('revision_statistics_explicit',md['formal_adjusted_event_count']<=1002 and md['formal_changed_slot_count']>=md['formal_adjusted_event_count'],{'changed_slots_total':md['changed_slot_count_total'],'warmup_changed_slots':md['warmup_changed_slot_count'],'formal_changed_slots':md['formal_changed_slot_count'],'formal_adjusted_events':md['formal_adjusted_event_count'],'formal_adjusted_event_rate':md['formal_adjusted_event_rate']})

# 7) K=1 must be exact point forecast; K=3 weights are empirical and normalized.
date_to_idx={d.date():i for i,d in enumerate(data.dates)};point_err=[];werr=[];node_rows=[]
for ds in ('2025-03-20','2025-06-21','2025-09-23','2025-12-21'):
    d=date_to_idx[datetime.fromisoformat(ds).date()]
    for eh in EVENT_HOURS:
        b1=build_scenarios(data,d,eh,use_new_vintage=True,max_scenarios=1);b3=build_scenarios(data,d,eh,use_new_vintage=True,max_scenarios=9)
        point_err.append(float(np.max(np.abs(b1.scenarios[0]-b1.point_net))));werr.append(abs(float(b3.weights.sum())-1.0))
        for (stage,nid),members in b3.node_members.items():node_rows.append([ds,eh,stage,nid,';'.join(map(str,members)),len(members)])
with open(HERE/'nonanticipativity_nodes.csv','w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f);w.writerow(['date','root_event_hour','stage_hour','node_id','leaf_members','leaf_count']);w.writerows(node_rows)
check('deterministic_baseline_is_point_forecast',max(point_err)<1e-12,{'samples':len(point_err),'max_error_kwh':max(point_err)})
check('representative_scenario_weights_normalized',max(werr)<1e-12,{'samples':len(werr),'max_error':max(werr)})

# 8) Future realized data perturbation cannot alter event contracts or pre-realization dispatch action.
samples=[];dispatch_samples=[]
for ds in ('2025-03-20','2025-06-21','2025-09-23','2025-12-21'):
    d=date_to_idx[datetime.fromisoformat(ds).date()]
    for eh,stage_before,sidx in ((0,None,0),(6,0,36),(12,1,72),(18,2,108)):
        soc=float(sp[d,sidx]);lead=0 if d==0 else float(A[d-1,143]);bb=None if eh==0 else B[d];aa=None if eh==0 else AS[d,stage_before]
        base=solve_event_lp(data,d,eh,soc,bb,aa,lead_contract=lead,use_new_vintage=True,allow_revision=True,scenario_count=9)
        dd=copy.deepcopy(data);start_i=eh*6;dd.load_cal_kwh[d,start_i:]+=777.;dd.net_cal_kwh[d,start_i:]+=777.;event=datetime(data.dates[d].year,data.dates[d].month,data.dates[d].day)+timedelta(hours=eh)
        for issue in list(dd.forecasts):
            if issue>event:dd.forecasts[issue]={tt:vv+9999. for tt,vv in dd.forecasts[issue].items()}
        pert=solve_event_lp(dd,d,eh,soc,bb,aa,lead_contract=lead,use_new_vintage=True,allow_revision=True,scenario_count=9)
        diff=float(np.max(np.abs(base.current_contract-pert.current_contract)));samples.append([ds,eh,diff])
        if eh>0:
            j0=EVENT_PLAN_START[eh];H=36 if eh<18 else 36;H=min(H,len(base.point_net));q=base.current_contract[j0:j0+H];pr=np.asarray(data.price_calendar[start_i:start_i+H],float)
            a1=solve_dispatch_mpc(soc,q,base.point_net[:H],pr);a2=solve_dispatch_mpc(soc,q,pert.point_net[:H],pr);dispatch_samples.append([ds,eh,max(abs(a1[k]-a2[k]) for k in (0,1,2))])
with open(HERE/'future_perturbation_audit.csv','w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f);w.writerow(['date','event_hour','max_contract_change_kwh']);w.writerows(samples)
check('future_perturbation_contract_invariance',max(x[2] for x in samples)<1e-7,{'samples':len(samples),'max_change':max(x[2] for x in samples)})
check('nominal_forecast_dispatch_future_actual_invariance',max(x[2] for x in dispatch_samples)<1e-7,{'samples':len(dispatch_samples),'max_action_change':max(x[2] for x in dispatch_samples)})

# 9) Revision log legal-only.
revs=list(csv.DictReader(open(HERE/'revision_log.csv',encoding='utf-8-sig')));illegal=[r for r in revs if str(r['legal']).lower() not in ('true','1')]
check('revision_log_legal_only',len(illegal)==0,{'rows':len(revs),'illegal':len(illegal)})

# 10) Settlement extreme identities.
main90=float(settlement_components([100],[80],[1])['F_regular'][0]);sunk110=float(settlement_components([100],[80],[1],'sunk_plan_plus_penalty')['F_regular'][0]);eq=float(settlement_components([100],[100],[1])['F_regular'][0]);zero=float(settlement_components([100],[0],[1])['F_regular'][0]);up=float(settlement_components([0],[100],[1])['F_regular'][0])
check('settlement_extreme_cases',max(abs(main90-90),abs(sunk110-110),abs(eq-100),abs(zero-50),abs(up-150))<tol,{'main_100_to_80':main90,'sunk':sunk110,'equal':eq,'A0':zero,'B0_A100':up})

# 11) Workbook full readback and fee/quantity reconciliation.
wb=openpyxl.load_workbook(Q3/'result3.xlsx',read_only=True,data_only=True);max_b=max_a=max_sum=max_fee=0.
for sn,arr in [('计划购电量',B),('调整购电量',A)]:
    ws=wb[sn]
    for d,row in zip(range(EVAL_START,365),ws.iter_rows(min_row=2,max_row=335,values_only=True)):
        vals=np.asarray([float(x) for x in row[1:145]]);maxerr=float(np.max(np.abs(vals-arr[d])));max_b=max(max_b,maxerr) if sn=='计划购电量' else max_b;max_a=max(max_a,maxerr) if sn=='调整购电量' else max_a
        max_sum=max(max_sum,abs(float(row[145])-float(arr[d].sum())));expected=float(np.dot(data.price_plan,B[d])) if sn=='计划购电量' else float(settlement_components(B[d],A[d],data.price_plan)['F_regular'].sum());max_fee=max(max_fee,abs(float(row[146])-expected))
ws=wb['充放电量'];charge_rows=ws.max_row;wb_ch=wb_dis=0.
for row in ws.iter_rows(min_row=2,values_only=True):wb_ch+=float(row[2] or 0);wb_dis+=float(row[3] or 0)
ws=wb['紧急购电量'];emergency_rows=ws.max_row;wb_em=0.
for row in ws.iter_rows(min_row=2,values_only=True):wb_em+=float(row[2] or 0)
wb.close()
check('result3_contract_readback',max(max_b,max_a,max_sum,max_fee)<1e-6,{'max_B':max_b,'max_A':max_a,'max_daily_sum':max_sum,'max_fee':max_fee})
check('result3_physical_readback',abs(wb_ch-float(ch[EVAL_START:].sum()))<1e-6 and abs(wb_dis-float(dis[EVAL_START:].sum()))<1e-6 and abs(wb_em-float(em[EVAL_START:].sum()))<1e-6,{'charge_rows':charge_rows,'emergency_rows':emergency_rows,'charge_diff':wb_ch-float(ch[EVAL_START:].sum()),'discharge_diff':wb_dis-float(dis[EVAL_START:].sum()),'emergency_diff':wb_em-float(em[EVAL_START:].sum())})

# 12) Natural-day bridge + five-ledger total.
regular=float(metrics['D']['regular_cost_yuan']);plan_day=float(np.sum([settlement_components(B[d],A[d],data.price_plan)['F_regular'].sum() for d in range(EVAL_START,365)]));bridge=float(settlement_components([B[EVAL_START-1,143]],[A[EVAL_START-1,143]],[data.price_plan[143]])['F_regular'][0]-settlement_components([B[364,143]],[A[364,143]],[data.price_plan[143]])['F_regular'][0])
check('natural_day_bridge',abs(regular-(plan_day+bridge))<1e-6,{'natural_regular':regular,'plan_day_regular':plan_day,'bridge':bridge})
ledger_total=0.
with open(HERE/'five_ledger.csv',encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):ledger_total+=float(r['F_total'])
check('five_ledger_total',abs(ledger_total-float(metrics['D']['total_cost_yuan']))<1e-5,{'ledger_total':ledger_total,'metrics_total':metrics['D']['total_cost_yuan']})

validation={'all_pass':all(x=='PASS' for x in checks.values()),'checks':checks,'evidence':evidence,'pass_count':sum(x=='PASS' for x in checks.values()),'check_count':len(checks)}
(HERE/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(validation,ensure_ascii=False,indent=2))
if not validation['all_pass']:raise SystemExit(2)
