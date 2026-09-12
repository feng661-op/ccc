"""Reproducible no-robustness revision: January freeze -> independent formal runs.

Each command writes its own directory; commands for different branches/clocks
may run in separate processes. A numerical-core fingerprint prevents stale
January evidence or old monthly checkpoints from being silently resumed.
"""
from __future__ import annotations
import argparse,hashlib,json,shutil,subprocess,sys
from dataclasses import asdict,replace
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from q4_data import load_q4_inputs
from q4_price import PriceConfig
from q4_sim import SimConfig,SimState,simulate_range,replay_summary
from q4_metrics import day_metrics,paired_block_bootstrap,relative_improvement
from q4_inheritance import VERSION,execute_inherited
from q4_sim_v2 import regular_fee
from q4_select import promotion_gate

ROOT=Path(__file__).resolve().parents[2];CODE=Path(__file__).resolve().parent
REV=CODE/'inheritance_revision';RUNS=REV/'runs'
CORE_FILES=('q4_inheritance.py','q4_sim_v2.py','q4_sim.py','q4_price.py','q4_metrics.py','q4_select.py')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def state_dict(s):
    return {'soc':float(s.soc),'prev_B':s.prev_B.tolist(),'prev_A':s.prev_A.tolist(),
            'prev_adjustment_prices':None if s.prev_adjustment_prices is None else s.prev_adjustment_prices.tolist()}
def state_load(v):
    p=v.get('prev_adjustment_prices');return SimState(float(v['soc']),np.asarray(v['prev_B'],float),np.asarray(v['prev_A'],float),None if p is None else np.asarray(p,float))
def cfg_load(v):
    d=dict(v);d['price_config']=PriceConfig(**d['price_config']);d['disabled_vintage_hours']=tuple(d.get('disabled_vintage_hours',()))
    return SimConfig(**d)
def fingerprint():
    source={str((CODE/n).relative_to(ROOT)).replace('\\','/'):sha(CODE/n) for n in CORE_FILES}
    source['protected_hashes']=sha(REV/'protected_hashes.json')
    return hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()
def provenance(cfg):
    protected=load(REV/'protected_hashes.json')
    return {'backend':VERSION,'base_q3_head':protected['base_q3_head'],
        'run_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'numerical_core_hash':fingerprint(),'core_file_sha256':{n:sha(CODE/n) for n in CORE_FILES},
        'official_input_sha256':{n:v for n,v in protected['files'].items() if n.startswith('26C题/')},
        'model_config_sha256':hashlib.sha256(json.dumps(asdict(cfg),sort_keys=True).encode()).hexdigest(),
        'created_utc':datetime.now(timezone.utc).isoformat(),'robustness':'NOT_RUN_BY_USER_REQUEST'}
def log_progress(tag):
    def log(d,row):
        if d%14==0 or d in (7,30,31,364):print(json.dumps({'run':tag,'day_index':d,'date':row['date'],'cash':row['cash_fee_yuan'],'soc':row['S24_kwh']},ensure_ascii=False),flush=True)
    return log

def save_result(path,res):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    for df,name in [(res.slots,'physical_10min.csv'),(res.contracts,'contract_ledger.csv'),(res.events,'event_ledger.csv'),(res.days,'daily_ledger.csv')]:
        df.to_csv(path/name,index=False,encoding='utf-8-sig')
    summary={'backend':VERSION,'config':asdict(res.config),'state_end':state_dict(res.state_end),
             'replay':replay_summary(res),'metrics':day_metrics(res.days),'provenance':provenance(res.config)}
    summary['result_sha256']={p.name:sha(p) for p in path.glob('*.csv')}
    dump(path/'summary.json',summary)
    return summary


def january(branch,clock='delivery'):
    base=SimConfig(branch,price_config=PriceConfig('lag7','global',1.0,42,48),settlement_clock=clock,label='JANUARY_B0')
    d=load_q4_inputs(ROOT);out=RUNS/f'{branch}_{clock}_january';out.mkdir(parents=True,exist_ok=True)
    # An unchanged, earlier JANUARY price-prediction comparison is reused, not
    # misrepresented as a rerun of the now-changed decision policy.
    pp=ROOT/'问题四/output/02_model_selection/price_freeze_decision.json';pf=load(pp)
    if pf['chosen']!=asdict(base.price_config):raise AssertionError('January predictive freeze conflicts with production model')
    warm=simulate_range(d,None,base,0,8,progress=log_progress(branch+' warm'))
    dump(out/'common_warmup_state.json',state_dict(warm.state_end))
    warm.days.to_csv(out/'common_warmup_days.csv',index=False,encoding='utf-8-sig')
    candidates={};rows={}
    for level in ('B0','B1'):
        cfg=replace(base,level=level,label='JANUARY_'+level)
        rr=simulate_range(d,None,cfg,8,31,state_load(state_dict(warm.state_end)),progress=log_progress(branch+' '+clock+' '+level))
        candidates[level]=rr;summary=save_result(out/level,rr);rows[level]={'name':level,**summary['metrics']}
    b0,b1=candidates['B0'],candidates['B1'];m0,m1=rows['B0'],rows['B1']
    row=rows['B1'];row['mean_cash_improvement_pct']=100*relative_improvement(m0['cash_mean_yuan_per_day'],m1['cash_mean_yuan_per_day'])
    row['cvar95_improvement_pct']=100*relative_improvement(m0['cash_cvar95_yuan_per_day'],m1['cash_cvar95_yuan_per_day'])
    row['paired']=paired_block_bootstrap(b0.days,b1.days,block_days=7,reps=2000)
    gate=promotion_gate(row);selected='B1' if gate['promote'] else 'B0';chosen=candidates[selected]
    decision={'backend':VERSION,'branch':branch,'settlement_clock':clock,'selection_period':'2025-01-09..2025-01-31',
        'formal_period_examined':False,'B0':rows['B0'],'B1':row,'B1_gate':gate,'B1_pass':gate['promote'],
        'B2_status':'NOT_RUN_BY_USER_REQUEST; no robust candidate is promoted without fresh evidence',
        'selected':selected,'config':asdict(chosen.config),'state_end':state_dict(chosen.state_end),
        'numerical_core_hash':fingerprint(),'provenance':provenance(chosen.config),
        'price_prediction_evidence':{'status':'REUSED_UNCHANGED_JANUARY_PRICE_TASK','path':str(pp.relative_to(ROOT)),'sha256':sha(pp),'chosen':pf['chosen']},
        'selection_reason':'统一成本主导晋级规则：现金改善至少1%且80%配对区块区间下界>0；或现金恶化不超1%、CVaR95改善至少2%且对应区间下界>0。否则保留简单B0。',
        'level_definition':'B0/B1均继承Q2的9个净负荷场景、48h视野和lambda=.02；B0电价为因果点预测，B1在相同历史源日加入配对价格残差。不是旧版单场景B0/3场景B1。'}
    dump(out/'decision.json',decision)
    print(json.dumps({'JANUARY_DONE':branch,'clock':clock,'selected':selected,'cash_B0':m0['cash_total_yuan'],'cash_B1':m1['cash_total_yuan'],'gate':gate},ensure_ascii=False),flush=True)
    return decision


def publish_freeze():
    frozen={};ladder={'backend':VERSION,'selection_period':'2025-01-09..2025-01-31','formal_period_examined':False,'branches':{}}
    for br in ('q4_2','q4_3'):
        dec=load(RUNS/f'{br}_delivery_january/decision.json')
        if dec['numerical_core_hash']!=fingerprint():raise AssertionError('stale January freeze')
        frozen[br]=dec;ladder['branches'][br]=dec
    # Preserve previous freeze evidence before replacing the public entrypoint.
    old=REV/'legacy_pre_inheritance';old.mkdir(parents=True,exist_ok=True)
    for name in ('january_frozen_selection.json','baseline_ladder.json'):
        p=CODE/name
        if p.exists() and not (old/name).exists():shutil.copy2(p,old/name)
    dump(CODE/'january_frozen_selection.json',frozen);dump(CODE/'baseline_ladder.json',ladder)
    return frozen


def make_tail(data,cfg,res):
    st=res.state_end;B=float(st.prev_B[143]);A=float(st.prev_A[143]);p=float(data.fixed_price_plan[-1] if cfg.fixed_price else data.price_plan[364,143])
    pa=float(st.prev_adjustment_prices[143]) if cfg.settlement_clock=='adjustment_time' else p
    L=float(data.load_plan_kwh[364,143]);G=float(data.pv_plan_kwh[364,143]);target=float(res.days.next_tail_target_soc_kwh.iloc[-1])
    f=execute_inherited(A,L,G,st.soc,target);rf=float(regular_fee(B,A,p,pa));ef=5*p*f.eL
    return dict(timestamp='2026-01-01T00:00:00',source_plan_date='2025-12-31',plan_j=143,backend=VERSION,
        B_kwh=B,A_kwh=A,price_yuan_per_kwh=p,adjustment_price_yuan_per_kwh=pa,load_kwh=L,pv_kwh=G,
        S0_kwh=f.S0,S1_kwh=f.S1,target_soc_kwh=target,qL=f.qL,qB=f.qB,u=f.u,gL=f.gL,gB=f.gB,kappa=f.kappa,
        eL=f.eL,eB=0.,x=f.x,y=f.y,I_kwh=f.I,regular_fee_yuan=rf,emergency_fee_yuan=ef,cash_fee_yuan=rf+ef,
        physics_residual=f.residual_max,schema_hash=f.schema_hash,
        scope='committed +1 tail: excluded from Feb-Dec natural-day KPI, included in plan-row bridge')


def formal(branch,clock='delivery',mode='selected'):
    dec=load(RUNS/f'{branch}_{clock}_january/decision.json')
    if dec['numerical_core_hash']!=fingerprint():raise AssertionError('numerical core changed after January freeze; rerun selection')
    cfg=replace(cfg_load(dec['config']),label='FORMAL_FROZEN');state=state_load(dec['state_end']);data=load_q4_inputs(ROOT)
    if mode=='backup':
        level='B1' if dec['selected']=='B0' else 'B0';sm=load(RUNS/f'{branch}_{clock}_january/{level}/summary.json')
        cfg=replace(cfg_load(sm['config']),label='FORMAL_BACKUP');state=state_load(sm['state_end'])
    elif mode=='matched_fixed_decision':
        cfg=replace(cfg,level='B0',decision_fixed_price=True,label='MATCHED_FIXED_DECISION')
    elif mode=='price_oracle':cfg=replace(cfg,level='B0',price_oracle=True,label='PRICE_ORACLE_DIAGNOSTIC')
    elif mode!='selected':raise ValueError(mode)
    before=state_dict(state);out=RUNS/f'{branch}_{clock}_{mode}'
    rr=simulate_range(data,None,cfg,31,365,state,progress=log_progress(f'{branch} {clock} {mode}'))
    if len(rr.days)!=334 or len(rr.slots)!=48096 or len(rr.contracts)!=48096:raise AssertionError('formal coverage mismatch')
    summary=save_result(out,rr);tail=make_tail(data,cfg,rr);dump(out/'tail_bridge.json',tail)
    final={'backend':VERSION,'branch':branch,'selected':cfg.level,'mode':mode,'config':asdict(cfg),'formal':day_metrics(rr.days),
        'tail_bridge':tail,'formal_state_start':before,'formal_state_end':state_dict(rr.state_end),
        'row_counts':{'slots':len(rr.slots),'events':len(rr.events),'contracts':len(rr.contracts),'days':len(rr.days)},
        'provenance':summary['provenance'],'january_decision_sha256':sha(RUNS/f'{branch}_{clock}_january/decision.json'),
        'result_sha256':summary['result_sha256']}
    dump(out/'formal_summary.json',final)
    print(json.dumps({'FORMAL_DONE':branch,'clock':clock,'mode':mode,'selected':cfg.level,'formal':final['formal']},ensure_ascii=False),flush=True)
    return final


def bridge(branch):
    data=load_q4_inputs(ROOT);cfg=SimConfig(branch,fixed_price=True,label='FIXED_PRICE_INHERITANCE_CHECK')
    rr=simulate_range(data,None,cfg,0,365,progress=log_progress('bridge '+branch))
    p=ROOT/'问题三/code'/('run_A.npz' if branch=='q4_2' else 'run_D.npz');z=np.load(p,allow_pickle=True)
    slots=rr.slots;idx=slots.day_index.to_numpy(int);slot=slots.slot.to_numpy(int)
    errors={'B':float(np.max(np.abs(rr.contracts.B_kwh.to_numpy().reshape(365,144)-z['B']))),
            'A':float(np.max(np.abs(rr.contracts.A_kwh.to_numpy().reshape(365,144)-z['A']))),
            'SOC':float(np.max(np.abs(slots.S1_kwh.to_numpy()-z['soc_path'][idx,slot+1])))}
    for actual,key in [('x','charge'),('y','discharge'),('e_kwh','emergency'),('kappa','curtail'),('I_kwh','grid_import'),('u','unused_contract')]:
        errors[key]=float(np.max(np.abs(slots[actual].to_numpy()-z[key][idx,slot])))
    errors['cash']=float(np.max(np.abs(rr.days.cash_fee_yuan.to_numpy()-(z['natural_regular_fee']+z['emergency_fee']))))
    if branch=='q4_2':
        q2=np.load(ROOT/'问题二/code/run_data.npz');errors['Q2_native_plan']=float(np.max(np.abs(rr.contracts.B_kwh.to_numpy().reshape(365,144)-q2['plan_q'])))
    result={'status':'PASS' if max(errors.values())<1e-6 else 'FAIL','backend':VERSION,'branch':branch,'days_recomputed':365,
        'actual_slots_checked':len(slots),'max_absolute_errors':errors,'formal_metrics':day_metrics(rr.days[rr.days.day_index>=31]),
        'comparison_file':str(p.relative_to(ROOT)),'comparison_sha256':sha(p),'numerical_core_hash':fingerprint(),
        'scope':'new independent full continuous simulation from Jan1 6000kWh; saved upstream arrays are comparison targets, never simulation inputs'}
    dump(REV/f'bridge_{branch}.json',result)
    print(json.dumps({'BRIDGE_DONE':result},ensure_ascii=False),flush=True)
    if result['status']!='PASS':raise AssertionError(errors)
    return result


def publish_results():
    old=REV/'legacy_pre_inheritance';old.mkdir(parents=True,exist_ok=True)
    for br in ('q4_2','q4_3'):
        src=RUNS/f'{br}_delivery_selected';m=load(src/'formal_summary.json')
        if m['provenance']['numerical_core_hash']!=fingerprint():raise AssertionError('stale formal result')
        dest=CODE/br
        if dest.exists():
            if (old/br).exists():
                # Never overwrite the preserved pre-revision evidence.
                for p in dest.glob('*'):
                    if p.is_file() and p.name in ('physical_10min.csv','contract_ledger.csv','event_ledger.csv','daily_ledger.csv','tail_bridge.json','formal_summary.json'):p.unlink()
            else:shutil.move(str(dest),str(old/br))
        dest.mkdir(parents=True,exist_ok=True)
        for name in ('physical_10min.csv','contract_ledger.csv','event_ledger.csv','daily_ledger.csv','tail_bridge.json','formal_summary.json'):
            shutil.copy2(src/name,dest/name)
    print('PUBLISHED_CURRENT_FORMAL_LEDGERS',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['january','formal','bridge','freeze','publish']);ap.add_argument('--branch',choices=['q4_2','q4_3'],default='q4_3');ap.add_argument('--clock',choices=['delivery','adjustment_time'],default='delivery');ap.add_argument('--mode',choices=['selected','backup','matched_fixed_decision','price_oracle'],default='selected');a=ap.parse_args()
    if a.stage=='january':january(a.branch,a.clock)
    elif a.stage=='formal':formal(a.branch,a.clock,a.mode)
    elif a.stage=='bridge':bridge(a.branch)
    elif a.stage=='freeze':publish_freeze()
    elif a.stage=='publish':publish_results()

if __name__=='__main__':main()
