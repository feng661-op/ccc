# -*- coding: utf-8 -*-
"""T01--T42 evidence-driven acceptance suite for Q4 V1.2.

Every test records a concrete evidence payload.  A solver `success` flag alone
never constitutes acceptance.  The suite independently reloads final CSV/JSON
and XLSX artifacts and runs active non-anticipativity/counterexample tests.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime,timedelta
from pathlib import Path
import hashlib,json,subprocess,sys
import numpy as np,pandas as pd,openpyxl

from q4_data import *
from q4_data import _plan_to_calendar
from q4_flow import *
from q4_information import *
from q4_forecast import event_forecast
from q4_price import PriceForecaster,PriceConfig,structure_audit
from q4_scenarios import build_point_scenario,build_joint_scenarios,validate_transport_duality
from q4_opt import solve_event,epsilon_zero_equivalence,_discrete_cvar
from q4_control import execute_slot,isomorphism_audit
from q4_settlement import hand_check,settlement_components
from q4_tree import validate_tree,event_horizon
from q4_metrics import empirical_cvar

ROOT=Path(__file__).resolve().parents[2]; CODE=Path(__file__).resolve().parent; Q4=ROOT/'问题四'; OUT=Q4/'output'
def _frozen_price_config():
 fr=json.loads((CODE/'january_frozen_selection.json').read_text(encoding='utf-8'))
 a=PriceConfig(**fr['q4_2']['config']['price_config']); b=PriceConfig(**fr['q4_3']['config']['price_config'])
 if a.version!=b.version: raise AssertionError(f'branch price freeze mismatch: {a.version} != {b.version}')
 return a
PC=_frozen_price_config()
EXPECTED={
'C题.pdf':'2c098f6ae9dd47ae965aebdec3b9facf3de6c173f999783c1012c08fc5fb9d2d',
'附件1.xlsx':'66b87134f5ecccd68184d3539bb1293ef039f9e0fdd955a589b9bfa7f227c377',
'附件2.xlsx':'2e95fd446bfafa0d8c59577b5c2e2ea8b3f1def20dde54a3062556f4da9b4c72',
'附件3.xlsx':'8a61b06c52bd0d639a1cc37c61a7d9f5b75edcbca718f64c1bd3498ec9f9d843',
'附件4.xlsx':'20e9c93aeab5e8e21ae4dd15587f9e190f7408692504c1319598461cd654fe71',
'result4-2.xlsx':'1c26494cfc6d754e0bd9bff7e13e1126a73d2d2da6c5336eb251d89b9a1a1a47',
'result4-3.xlsx':'c59da470cabd0be23f602c95c8aa9d11ec224a0cdac216b3e1f218e65d006bdc'}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(o):return json.loads(json.dumps(o,default=lambda x: float(x) if isinstance(x,np.floating) else int(x) if isinstance(x,np.integer) else x.tolist() if isinstance(x,np.ndarray) else str(x)))
class Suite:
 def __init__(self):self.rows=[]
 def t(self,n,name,ok,evidence):self.rows.append({'id':f'T{n:02d}','name':name,'pass':bool(ok),'evidence':dump(evidence)});print(f'T{n:02d}', 'PASS' if ok else 'FAIL',name,flush=True)

def load_artifacts():
 d=load_q4_inputs(ROOT);arts={}
 for br in ('q4_2','q4_3'):
  p=CODE/br;arts[br]={'slots':pd.read_csv(p/'physical_10min.csv'),'events':pd.read_csv(p/'event_ledger.csv'),'contracts':pd.read_csv(p/'contract_ledger.csv'),'days':pd.read_csv(p/'daily_ledger.csv'),'summary':json.loads((p/'formal_summary.json').read_text(encoding='utf-8')),'tail':json.loads((p/'tail_bridge.json').read_text(encoding='utf-8'))}
 return d,arts

def solve_rep(data,branch='q4_3',day_idx=20,event_hour=0):
 pf=PriceForecaster(data);H={0:145,6:109,12:73,18:37}[event_hour];sc=build_point_scenario(data,pf,day_idx,event_hour,H,branch,PC)
 return solve_event(scenarios=sc,soc0=6000,day=data.dates[day_idx],event_hour=event_hour,branch=branch,original_B=np.zeros(144),current_A=np.zeros(144),lead_q=0,model_level='B0')

def xlsx_check(br,art):
 fn='result4-2.xlsx' if br=='q4_2' else 'result4-3.xlsx';p=Q4/fn;wb=openpyxl.load_workbook(p,read_only=True,data_only=True);ctr=art['contracts'];errs=[]
 plan=wb['计划购电量'];
 if (plan.max_row,plan.max_column)!=(335,147):errs.append('plan_geometry')
 for rr,d in ((2,31),(100,129),(335,364)):
  g=ctr[ctr.day_index==d].sort_values('plan_j');vals=np.asarray([plan.cell(rr,j+2).value for j in range(144)],float);errs.append(float(np.max(np.abs(vals-g.B_kwh.to_numpy(float)))))
 if br=='q4_3':
  ws=wb['调整购电量']
  for rr,d in ((2,31),(100,129),(335,364)):
   g=ctr[ctr.day_index==d].sort_values('plan_j');vals=np.asarray([ws.cell(rr,j+2).value for j in range(144)],float);errs.append(float(np.max(np.abs(vals-g.A_kwh.to_numpy(float)))))
 em=wb['紧急购电量'];total=0.;
 for r in range(2,em.max_row+1):
  v=em.cell(r,3).value
  if isinstance(v,(int,float)):total+=float(v)
 wb.close();return {'file':fn,'rows_plan':335,'contract_sample_errors':errs,'emergency_xlsx_sum':total,'emergency_ledger_sum':float(art['slots'].e_kwh.sum()),'sha256':sha(p)},(all(not isinstance(x,(float,int)) or abs(x)<1e-6 for x in errs) and abs(total-float(art['slots'].e_kwh.sum()))<1e-4)

def run():
 s=Suite();data,arts=load_artifacts();att=ROOT/'26C题'/'附件'
 # T01
 files={'C题.pdf':ROOT/'26C题'/'C题.pdf','附件1.xlsx':att/'附件1.xlsx','附件2.xlsx':att/'附件2.xlsx','附件3.xlsx':att/'附件3.xlsx','附件4.xlsx':att/'附件4.xlsx','result4-2.xlsx':att/'附件5'/'result4-2.xlsx','result4-3.xlsx':att/'附件5'/'result4-3.xlsx'};hashes={k:sha(v) for k,v in files.items()};s.t(1,'输入完整性',hashes==EXPECTED and data.price_plan.shape==(365,144) and len(data.forecasts)==1460,{'hashes':hashes,'dims':{'price':data.price_plan.shape,'forecast_issues':len(data.forecasts),'dates':[str(data.dates[0].date()),str(data.dates[-1].date())]}})
 # T02-T06
 s.t(2,'计划时段解析',data.plan_headers[0]=='0:10-0:20' and data.plan_headers[-1]=='0:00-0:10+1' and len(data.plan_headers)==144,{'first':data.plan_headers[0],'last':data.plan_headers[-1],'n':len(data.plan_headers)})
 m=[natural_to_plan_ref(31,i) for i in (0,1,143)];s.t(3,'自然日映射',m==[(30,143),(31,0),(31,142)] and data.price_cal[31,0]==data.price_plan[30,143],{'refs':m,'feb1_00_price':data.price_cal[31,0],'jan31_tail_price':data.price_plan[30,143]})
 tr=validate_tree();s.t(4,'事件边界',tr['pass'] and [EVENT_PLAN_START[h] for h in EVENT_HOURS]==[0,35,71,107],tr)
 bridge={br:{'first':arts[br]['slots'].iloc[0][['timestamp','plan_owner_day_index','plan_j']].to_dict(),'tail':arts[br]['tail'],'continuity':float(np.max(np.abs(arts[br]['days'].S00_kwh.to_numpy()[1:]-arts[br]['days'].S24_kwh.to_numpy()[:-1])))} for br in arts};s.t(5,'跨月/年度尾',all(x['first']['plan_owner_day_index']==30 and x['first']['plan_j']==143 and x['continuity']<1e-9 and x['tail']['plan_j']==143 for x in bridge.values()),bridge)
 s.t(6,'初始缺失段',np.isnan(data.load_cal_kwh[0,0]) and np.isnan(data.pv_cal_kwh[0,0]) and np.isnan(data.price_cal[0,0]),{'jan1_00_load':data.load_cal_kwh[0,0],'jan1_00_pv':data.pv_cal_kwh[0,0],'jan1_00_price':data.price_cal[0,0]})
 # T07-T08
 vint_ok=all(r.target_time==r.issue_time+timedelta(hours=r.lead_hour) and r.issue_time<=r.target_time for r in data.forecast_records);s.t(7,'vintage对齐',vint_ok and len(data.forecast_records)==35040,{'records':len(data.forecast_records),'sample':asdict(data.forecast_records[1234])})
 residual={'d20_e0':legal_residual_days(20,0),'d20_e6':legal_residual_days(20,6)};s.t(8,'残差可用性',max(residual['d20_e0'])==18 and max(residual['d20_e6'])==19,residual)
 # T09 active future perturbation
 d0=20;pf=PriceForecaster(data);sc=build_point_scenario(data,pf,d0,0,145,'q4_3',PC);a=solve_event(scenarios=sc,soc0=6000,day=data.dates[d0],event_hour=0,branch='q4_3',original_B=np.zeros(144),current_A=np.zeros(144),lead_q=0,model_level='B0');d2=deepcopy(data);d2.load_plan_kwh[d0:,50:]+=777;d2.pv_plan_kwh[d0:,50:]+=333;d2.net_plan_kwh=d2.load_plan_kwh-d2.pv_plan_kwh;d2.price_plan[d0:,50:]+=2.5;d2.load_cal_kwh=_plan_to_calendar(d2.load_plan_kwh);d2.pv_cal_kwh=_plan_to_calendar(d2.pv_plan_kwh);d2.net_cal_kwh=_plan_to_calendar(d2.net_plan_kwh);d2.price_cal=_plan_to_calendar(d2.price_plan);pf2=PriceForecaster(d2);sc2=build_point_scenario(d2,pf2,d0,0,145,'q4_3',PC);b=solve_event(scenarios=sc2,soc0=6000,day=d2.dates[d0],event_hour=0,branch='q4_3',original_B=np.zeros(144),current_A=np.zeros(144),lead_q=0,model_level='B0');diff=float(np.max(np.abs(a.contract_A-b.contract_A)));s.t(9,'合同未来扰动不变',diff<1e-6,{'max_contract_diff':diff})
 # T10 independently propagate the T09 future perturbation through the first
 # optimizer target and then the real first 10-min executor.  Future data are
 # materially different, while both the target and first action must be equal.
 L0=float(data.load_cal_kwh[d0,0]);G0=float(data.pv_cal_kwh[d0,0]);
 ta=float(a.expected_soc_end[0]);tb=float(b.expected_soc_end[0])
 ca=execute_slot(Q=0.0,load_kwh=L0,pv_kwh=G0,S0=6000,target_soc=ta);cb=execute_slot(Q=0.0,load_kwh=L0,pv_kwh=G0,S0=6000,target_soc=tb)
 avec=np.asarray([ca.qL,ca.qB,ca.u,ca.gL,ca.gB,ca.kappa,ca.eL,ca.eB,ca.x,ca.y,ca.S1]);bvec=np.asarray([cb.qL,cb.qB,cb.u,cb.gL,cb.gB,cb.kappa,cb.eL,cb.eB,cb.x,cb.y,cb.S1])
 actdiff=float(np.max(np.abs(avec-bvec)));targetdiff=abs(ta-tb);future_change=float(np.nanmax(np.abs(d2.price_plan[d0:,50:]-data.price_plan[d0:,50:])))
 s.t(10,'控制未来扰动不变',future_change>1 and targetdiff<1e-6 and actdiff<1e-6,{'future_price_perturbation_max':future_change,'first_target_diff':targetdiff,'first_action_max_diff':actdiff,'first_action':asdict(ca)})
 # T11 current measurement may change execution recourse, but execute_slot has
 # no authority to rewrite the already formed event contract.
 contract_before=a.contract_A.copy();c1=execute_slot(Q=800,load_kwh=900,pv_kwh=100,S0=6000,target_soc=6100);c2=execute_slot(Q=800,load_kwh=1300,pv_kwh=20,S0=6000,target_soc=6100)
 contract_rewrite=float(np.max(np.abs(a.contract_A-contract_before)));action_change=max(abs(c1.eL-c2.eL),abs(c1.y-c2.y),abs(c1.qL-c2.qL),abs(c1.x-c2.x))
 s.t(11,'当前测量隔离',contract_rewrite<1e-12 and action_change>1e-8,{'contract_rewrite_diff':contract_rewrite,'action_change_max':action_change,'action_before':asdict(c1),'action_after':asdict(c2)})
 # T12-T17
 _,_,net,times,_=event_forecast(data,20,6,37,'q4_3');pred,audit=PriceForecaster(data).predict(data.dates[20]+timedelta(hours=6),times,net,PC);s.t(12,'价格揭示边界',audit[0]['model_version']=='revealed' and audit[1]['model_version']!='revealed',{'now':audit[0],'future':audit[1]})
 d3=deepcopy(data);d3.forecasts={};x=event_forecast(data,20,0,10,'q4_2')[1];y=event_forecast(d3,20,0,10,'q4_2')[1];s.t(13,'Q4-2权限',float(np.max(np.abs(x-y)))==0.0,{'pv_forecast_diff_after_deleting_Att3':float(np.max(np.abs(x-y)))})
 versions={h:event_forecast(data,20,h,5,'q4_3')[4] for h in EVENT_HOURS};s.t(14,'Q4-3权限',versions=={0:'ATT3_VINTAGE_00',6:'ATT3_VINTAGE_06',12:'ATT3_VINTAGE_12',18:'ATT3_VINTAGE_18'},versions)
 h42=legal_history_days(100,cap=42);freeze=json.loads((CODE/'january_frozen_selection.json').read_text(encoding='utf-8'));s.t(15,'历史窗',len(h42)==42 and h42[0]==58 and all(freeze[b]['config']['history_days']==42 for b in freeze),{'day100_pool':[h42[0],h42[-1],len(h42)],'frozen':{b:freeze[b]['config']['history_days'] for b in freeze}})
 pm=pd.read_csv(OUT/'02_model_selection'/'price_model_ablation.csv');pf=PriceForecaster(data);decision=data.dates[20]+timedelta(hours=12);sub=pf._feature_table[pf._feature_table.target_time<decision];s.t(16,'价格分组与因果回归',set(pm.block_scheme)>= {'global','6block'} and sub.target_time.max()<decision,{'schemes':sorted(set(pm.block_scheme)),'six_blocks':list(range(6)),'max_training_target':str(sub.target_time.max()),'decision':str(decision)})
 models=set(pm.model);st=structure_audit(data);winner=pm.sort_values(['mae','rmse','q95_abs_error']).iloc[0];pdec=json.loads((OUT/'02_model_selection'/'price_freeze_decision.json').read_text(encoding='utf-8'));chosen=pdec['chosen'];selected_matches=(chosen['model']==winner.model and chosen['block_scheme']==winner.block_scheme and (winner.model in ('lag1','lag7','lag7_R') or abs(float(chosen['ridge_alpha'])-float(winner.alpha))<1e-12));s.t(17,'价格消融与证据分层',models>=set(('lag1','lag7','lag7_R','lag7_dN','full_ridge')) and selected_matches and abs(st['value']-.960986314486)<1e-9 and 'structural' in st['role'],{'models':sorted(models),'fair_winner':winner.to_dict(),'frozen_choice':chosen,'selection_rule':pdec.get('selection_rule'),'structure':st})
 # T18-T26 physics
 s.t(18,'单位与10分钟限值',abs(XMAX-833.3333333333334)<1e-9 and abs(ETA_C-np.sqrt(.9))<1e-12,{'DT':DT,'XMAX_kwh':XMAX,'eta':ETA_C})
 sl=pd.concat([arts[b]['slots'] for b in arts],ignore_index=True);r19=float(np.max(np.abs(sl.qL+sl.qB+sl.u-sl.Q_kwh)));s.t(19,'合同源流',r19<1e-6 and (sl[['qL','qB','u']]>=-1e-9).all().all(),{'max_residual':r19})
 r20=float(np.max(np.abs(sl.gL+sl.gB+sl.kappa-sl.pv_kwh)));night=float(sl.loc[sl.pv_kwh<1e-10,'kappa'].max());s.t(20,'PV源流',r20<1e-6 and night<1e-7,{'max_residual':r20,'night_kappa_max':night})
 r21=float(np.max(np.abs(sl.qL+sl.gL+sl.y+sl.eL-sl.load_kwh)));s.t(21,'负荷汇流',r21<=1e-6,{'max_residual':r21})
 r22=float(np.max(np.abs(sl.x-(sl.qB+sl.gB+sl.eB))));s.t(22,'充电汇流与eB语义',r22<1e-6 and float(sl.eB.max())<1e-9,{'max_residual':r22,'formal_eB_max':float(sl.eB.max())})
 r23=float(np.max(np.abs(sl.S1_kwh-(sl.S0_kwh+ETA_C*sl.x-sl.y/ETA_D))));s.t(23,'SOC递推与边界',r23<1e-6 and sl.S1_kwh.min()>=SOC_MIN-1e-6 and sl.S1_kwh.max()<=SOC_MAX+1e-6,{'max_residual':r23,'min':float(sl.S1_kwh.min()),'max':float(sl.S1_kwh.max())})
 simultaneous=int(((sl.x>1e-7)&(sl.y>1e-7)).sum());s.t(24,'充放电上限与无无意义循环',float(sl.x.max())<=XMAX+1e-6 and float(sl.y.max())<=XMAX+1e-6 and simultaneous==0,{'xmax':float(sl.x.max()),'ymax':float(sl.y.max()),'simultaneous_slots':simultaneous})
 mc=manual_counterexamples();s.t(25,'夜间反例',mc['night_unused_contract']['kappa']<1e-9 and mc['night_unused_contract']['unused_contract']>0 and mc['no_emergency_charge']['eB']<1e-9,mc)
 iso=isomorphism_audit();hashes=set(sl.schema_hash.astype(str));s.t(26,'事件/执行同一物理内核',iso['max_residual']<1e-9 and hashes=={structure_signature()}, {'isomorphism':iso,'formal_hashes':list(hashes)})
 # T27-T29 lex on real event
 rep=solve_rep(data);s.t(27,'字典序第一层保持',rep.success and rep.final_level1_objective<=rep.level1_objective+rep.lex_tolerance1+1e-8,{'f1':rep.level1_objective,'final_f1':rep.final_level1_objective,'tol':rep.lex_tolerance1})
 s.t(28,'字典序第二层最小化',rep.success and rep.final_level2_objective<=rep.level2_throughput+rep.lex_tolerance2+1e-8,{'f2':rep.level2_throughput,'final_f2':rep.final_level2_objective,'tol':rep.lex_tolerance2})
 stage3_match=abs(rep.level3_curtailment-rep.predicted_curtailment_kwh);gate1=rep.final_level1_objective-rep.level1_objective;gate2=rep.final_level2_objective-rep.level2_throughput;s.t(29,'字典序第三层弃光最小化',rep.success and rep.level3_curtailment>=-1e-8 and stage3_match<1e-7 and gate1<=rep.lex_tolerance1+1e-8 and gate2<=rep.lex_tolerance2+1e-8,{'stage3_objective':rep.level3_curtailment,'independent_weighted_curtailment':rep.predicted_curtailment_kwh,'stage3_match_abs':stage3_match,'level1_gate_slack':gate1,'level1_tol':rep.lex_tolerance1,'level2_gate_slack':gate2,'level2_tol':rep.lex_tolerance2,'solver':rep.message})
 # T30-T32 contract settlement
 pf=PriceForecaster(data);sc0=build_point_scenario(data,pf,20,0,145,'q4_3',PC);e0=solve_event(scenarios=sc0,soc0=6000,day=data.dates[20],event_hour=0,branch='q4_3',original_B=np.zeros(144),current_A=np.zeros(144),lead_q=0,model_level='B0');sc6=build_point_scenario(data,pf,20,6,109,'q4_3',PC);e6=solve_event(scenarios=sc6,soc0=e0.expected_soc_end[35],day=data.dates[20],event_hour=6,branch='q4_3',original_B=e0.original_B,current_A=e0.contract_A,lead_q=0,model_level='B0');s.t(30,'B锚定不变且A为绝对量',float(np.max(np.abs(e6.original_B-e0.original_B)))<1e-9 and np.all(e6.contract_A>=-1e-9),{'B_max_diff':float(np.max(np.abs(e6.original_B-e0.original_B))),'A_min':float(e6.contract_A.min())})
 hc=hand_check();s.t(31,'结算手算',hc['max_abs_diff']<1e-10,hc)
 er=float(np.max(np.abs(sl.emergency_fee_yuan-5*sl.price_yuan_per_kwh*sl.e_kwh)));s.t(32,'紧急购电逐槽5c e',er<1e-8,{'max_fee_residual':er})
 # T33-T40 scenario/risk/oracle
 joint=build_joint_scenarios(data,PriceForecaster(data),20,0,145,'q4_3',PC,k=3);s.t(33,'场景权重与同源路径',np.all(joint.weights>=0) and abs(joint.weights.sum()-1)<1e-12 and len(joint.source_days)==len(set(joint.source_days)),{'weights':joint.weights,'medoid_source_days':joint.medoid_source_days,'pool_source_days':joint.source_days})
 ladder=json.loads((CODE/'baseline_ladder.json').read_text(encoding='utf-8'));e0p=CODE/'ablations'/'E0_matched_policy_benchmark.json';e0=json.loads(e0p.read_text(encoding='utf-8')) if e0p.exists() else {};ok34=all('B0' in ladder['branches'][b] and 'B1' in ladder['branches'][b] for b in ('q4_2','q4_3')) and 'matched_no_price_adaptation' in e0 and 'formal_price_adaptive' in e0 and 'matched policy benchmark' in e0.get('claim_boundary','');s.t(34,'B0/B1联合价值与匹配价格基线',ok34,{'q4_2_B1_gain_pct':ladder['branches']['q4_2']['B1']['mean_cash_improvement_pct'],'q4_3_B1_gain_pct':ladder['branches']['q4_3']['B1']['mean_cash_improvement_pct'],'matched_C0':e0})
 eps=json.loads((CODE/'epsilon_calibration.json').read_text(encoding='utf-8'));s.t(35,'经验漂移ε校准',all(eps[b]['n']>0 and eps[b]['q50']>0 and eps[b]['q75']>=eps[b]['q50'] and 'NOT a confidence' in eps[b]['interpretation'] for b in eps),{b:{k:eps[b][k] for k in ('n','q50','q75','interpretation','frozen_scales')} for b in eps})
 dual=validate_transport_duality();eq=epsilon_zero_equivalence(scenarios=joint,soc0=6000,day=data.dates[20],event_hour=0,branch='q4_3',original_B=np.zeros(144),current_A=np.zeros(144),lead_q=0,terminal_reserve=6000,terminal_value=.8,allow_emergency_charging=False);s.t(36,'DRO原对偶与ε=0退化',dual['max_gap']<1e-8 and eq['contract_max_abs_diff']<1e-5 and eq['level1_abs_diff']<1e-5,{'duality_max_gap':dual['max_gap'],'epsilon0':eq})
 z=np.array([1.,2.,10.]);w=np.array([.2,.3,.5]);c1=weighted_cvar(z,w,.8);c2=_discrete_cvar(z,w,.8);s.t(37,'加权离散CVaR',abs(c1-c2)<1e-12 and abs(c1-10)<1e-12,{'flow_cvar':c1,'opt_cvar':c2})
 sel={b:freeze[b]['selected'] for b in freeze};ladsel={b:ladder['branches'][b]['selected'] for b in sel};s.t(38,'DRO不强行保留',sel==ladsel and all(v in ('B0','B1') for v in sel.values()) and all(not ladder['branches'][b]['B2_pass'] for b in sel),{'selected':sel,'ladder_selected':ladsel,'B2_pass':{b:ladder['branches'][b]['B2_pass'] for b in sel}})
 ablpath=CODE/'ablations'/'ablation_matrix_E0_E15.json';abl=json.loads(ablpath.read_text(encoding='utf-8')) if ablpath.exists() else {};e12=abl.get('experiments',{}).get('E12',{});ok39=bool(e12) and all('sensitivity_eB_total_kwh' in e12[b] for b in ('q4_2','q4_3'));s.t(39,'紧急充电独立敏感性',ok39,e12 if e12 else {'missing':str(ablpath)})
 oracle=json.loads((CODE/'oracle_audit.json').read_text(encoding='utf-8'))
 oracle_checks={}
 for b in ('q4_2','q4_3'):
  formal_cash=float(arts[b]['summary']['formal']['cash_total_yuan']); oc=float(oracle[b]['causal']['cash_total_yuan']); po=float(oracle[b]['price_oracle']['cash_total_yuan']); fi=float(oracle[b]['full_information']['strict_cash_lower_bound_yuan']); boundary=str(oracle[b].get('price_oracle_boundary',''))
  oracle_checks[b]={'causal_matches_formal':abs(oc-formal_cash)<1e-6,'full_info_le_price_oracle':fi<=po+1e-6,'full_info_le_causal':fi<=oc+1e-6,'boundary_mentions_tail':'Dec-31' in boundary and '+1 tail' in boundary,'price_oracle_claim':oracle[b]['audit']['Price_Oracle']['claim'],'strict_lower_bound_check_pass':oracle[b]['audit']['strict_lower_bound_check_pass']}
 ok40=all(v['causal_matches_formal'] and v['full_info_le_price_oracle'] and v['full_info_le_causal'] and v['boundary_mentions_tail'] and v['strict_lower_bound_check_pass'] and 'NOT a theoretical lower bound' in v['price_oracle_claim'] for v in oracle_checks.values());s.t(40,'Oracle边界与同口径一致性',ok40,oracle_checks)
 # T41 Excel/replay independent reread
 xc={br:xlsx_check(br,arts[br]) for br in ('q4_2','q4_3')};specified=OUT/'04_tables'/'指定日期表格.md';expm=json.loads((CODE/'export_manifest.json').read_text(encoding='utf-8'));hash_lock={'q4_2':expm['files']['result4-2']['sha256'],'q4_3':expm['files']['result4-3']['sha256']};hash_ok=all(xc[b][0]['sha256']==hash_lock[b] for b in xc);ok41=all(v[1] for v in xc.values()) and hash_ok and specified.exists() and all(len(arts[b]['days'])==334 and len(arts[b]['slots'])==48096 for b in arts);s.t(41,'连续回放/Excel/指定日期独立重读',ok41,{'xlsx':{b:v[0] for b,v in xc.items()},'export_manifest_hash_lock':hash_lock,'hash_lock_pass':hash_ok,'specified_table':str(specified),'row_counts':{b:{'days':len(arts[b]['days']),'slots':len(arts[b]['slots']),'events':len(arts[b]['events'])} for b in arts}})
 # T42 scope: no tracked/untracked modifications outside Q4 relative to clean inherited HEAD.
 cp=subprocess.run(['git','-c','core.quotepath=false','status','--porcelain=v1'],cwd=ROOT,text=True,capture_output=True,encoding='utf-8',errors='replace');lines=[x for x in cp.stdout.splitlines() if x.strip()];bad=[]
 for line in lines:
  p=line[3:].replace('\\','/')
  if ' -> ' in p:p=p.split(' -> ')[-1]
  if not (p.startswith('问题四/') or p=='问题四'):bad.append(line)
 head=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True).stdout.strip();s.t(42,'范围保护',cp.returncode==0 and not bad and head.startswith('c2864f3899685674419c9b209c800b6c89f4b8d6'),{'head':head,'git_status':lines,'outside_q4':bad})
 # Save manifests/results
 report={'suite':'Q4 V1.2 T01-T42','all_pass':all(r['pass'] for r in s.rows),'passed':sum(r['pass'] for r in s.rows),'failed':[r['id'] for r in s.rows if not r['pass']],'tests':s.rows}
 (CODE/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 # independent data/run manifests
 data_manifest={'official_inputs':{k:{'path':str(v),'sha256':sha(v),'bytes':v.stat().st_size} for k,v in files.items()},'formal_window':'2025-02-01..2025-12-31','natural_slots':48096,'template_output_hashes':{'result4-2.xlsx':sha(Q4/'result4-2.xlsx'),'result4-3.xlsx':sha(Q4/'result4-3.xlsx')}};(CODE/'data_manifest.json').write_text(json.dumps(data_manifest,ensure_ascii=False,indent=2),encoding='utf-8')
 run_manifest={'head':head,'python':sys.version,'price_config':asdict(PC),'frozen_selection':sel,'physical_schema_hash':structure_signature(),'formal':{b:arts[b]['summary']['formal'] for b in arts},'oracle_file':sha(CODE/'oracle_audit.json'),'validation_sha256':sha(CODE/'validation.json')};(CODE/'run_manifest.json').write_text(json.dumps(run_manifest,ensure_ascii=False,indent=2),encoding='utf-8')
 final={'status':'PASS' if report['all_pass'] else 'FAIL','T01_T42':{'passed':report['passed'],'failed':report['failed']},'required_next':'final_visual_then_user_review' if report['all_pass'] else 'fix_failures_and_rerun'};(CODE/'final_acceptance.json').write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(final,ensure_ascii=False),flush=True);sys.exit(0 if report['all_pass'] else 2)
if __name__=='__main__':run()
