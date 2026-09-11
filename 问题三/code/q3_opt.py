# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime,timedelta
from typing import Dict,List,Optional,Tuple
import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from q3_data import *

_PF_CACHE={}
_ACT_CACHE={}

def _pf(data,day_idx,event_hour,horizon,use_new_vintage):
    key=(id(data),day_idx,event_hour,horizon,bool(use_new_vintage))
    if key not in _PF_CACHE:
        _PF_CACHE[key]=event_point_forecast(data,day_idx,event_hour,horizon,use_new_vintage)
    return _PF_CACHE[key]

def _actual(data,day_idx,event_hour,horizon):
    key=(id(data),day_idx,event_hour,horizon)
    if key not in _ACT_CACHE:
        issue=datetime(data.dates[day_idx].year,data.dates[day_idx].month,data.dates[day_idx].day)+timedelta(hours=event_hour)
        _ACT_CACHE[key]=actual_horizon(data,issue,horizon)
    return _ACT_CACHE[key]

@dataclass
class ScenarioBundle:
    point_net: np.ndarray; scenarios: np.ndarray; source_days: List[int]; weights: np.ndarray
    slot_starts: List[datetime]; slot_prices: np.ndarray
    tree_nodes: Dict[int,Dict[int,int]]; node_members: Dict[Tuple[int,int],Tuple[int,...]]
    latest_history_day: int

@dataclass
class EventSolution:
    event_hour:int; current_contract:np.ndarray; expected_soc:np.ndarray
    scenario_soc:np.ndarray; scenario_emergency:np.ndarray; objective:float; solve_status:str
    solve_seconds:float; max_eq_residual:float; max_ub_violation:float; scenario_count:int; node_count:int
    continuation_first_time:Optional[datetime]

def _tree(source_days,data,current_day,event_hour,use_new_vintage):
    K=len(source_days); tree={event_hour:{k:0 for k in range(K)}}; members={(event_hour,0):tuple(range(K))}; groups=[tuple(range(K))]; nid=1
    for eh in [h for h in EVENT_HOURS if h>event_hour]:
        mp={}; ng=[]
        for g in groups:
            if len(g)<=1: parts=[g]
            else:
                scored=[]
                for k in g:
                    sig=prefix_signal(data,source_days[k],eh,use_new_vintage); scored.append((float(sig[0]+.0002*sig[1]),k))
                scored.sort(); cut=max(1,len(scored)//2); parts=[tuple(x[1] for x in scored[:cut]),tuple(x[1] for x in scored[cut:])]; parts=[p for p in parts if p]
            for p in parts:
                for k in p: mp[k]=nid
                members[(eh,nid)]=p; ng.append(p); nid+=1
        tree[eh]=mp; groups=ng
    return tree,members

def build_scenarios(data,day_idx,event_hour,horizon=145,use_new_vintage=True,max_scenarios=3,history_window=56):
    point,times,prices=_pf(data,day_idx,event_hour,horizon,use_new_vintage); latest=day_idx-2; rows=[]; cur=prefix_signal(data,day_idx,event_hour,use_new_vintage)
    for h in range(max(0,latest-history_window+1),latest+1):
        actual=_actual(data,h,event_hour,horizon)
        if actual is None: continue
        pred,_,_=_pf(data,h,event_hour,horizon,use_new_vintage); sig=prefix_signal(data,h,event_hour,use_new_vintage)
        dist=float(np.linalg.norm((sig-cur)/np.asarray([500.,1000.,1000.]))); resid=actual-pred; risk=float(np.dot(prices,np.maximum(resid,0))); rows.append((dist,risk,h,resid))
    if not rows:
        tree={}; members={}; nid=0
        for eh in [h for h in EVENT_HOURS if h>=event_hour]:
            tree[eh]={0:nid}; members[(eh,nid)]=(0,); nid+=1
        return ScenarioBundle(point,point[None,:],[],np.ones(1),times,prices,tree,members,latest)
    rows.sort(key=lambda z:(z[0],z[2])); pool=rows[:min(15,len(rows))]; pool.sort(key=lambda z:z[1]); K=min(max_scenarios,len(pool))
    ids=np.linspace(0,len(pool)-1,K).round().astype(int); pick=[]; seen=set()
    for i in ids:
        if int(i) not in seen: seen.add(int(i)); pick.append(pool[int(i)])
    scenarios=np.asarray([point+r[3] for r in pick]); scenarios=np.clip(scenarios,-2500,3000); src=[int(r[2]) for r in pick]; w=np.full(len(src),1/len(src)); tree,members=_tree(src,data,day_idx,event_hour,use_new_vintage)
    return ScenarioBundle(point,scenarios,src,w,times,prices,tree,members,latest)

def _stage(day,event_hour,t):
    base=datetime(day.year,day.month,day.day); valid=[h for h in EVENT_HOURS if h>=event_hour and base+timedelta(hours=h)<=t]
    return max(valid) if valid else event_hour

def execute_actual_interval(soc,contract,net_actual,target_after):
    margin=float(max(0,contract)-net_actual)
    if margin>=0:
        x=min(margin,XMAX,max(0,(SOC_MAX-soc)/ETA_C)); return soc+ETA_C*x,x,0.0,0.0,max(0,margin-x)
    deficit=-margin; floor=float(np.clip(target_after,SOC_MIN,SOC_MAX)); y=min(deficit,XMAX,max(0,(soc-floor)*ETA_D)); return soc-y/ETA_D,0.0,y,max(0,deficit-y),0.0

def solve_event_lp(data,day_idx,event_hour,soc0,B,A_before,lead_contract=0.0,*,use_new_vintage=True,allow_revision=True,scenario_count=3,horizon=145,down_settlement='cancel_settlement',revision_anchor='original_anchor',terminal_value=.45):
    bundle=build_scenarios(data,day_idx,event_hour,horizon,use_new_vintage,scenario_count); K,H=bundle.scenarios.shape; day=data.dates[day_idx]; startj=EVENT_PLAN_START[event_hour]
    if event_hour>0 and (B is None or A_before is None): raise ValueError('日内事件缺B/A')
    B0=None if B is None else np.asarray(B,float); A0=None if A_before is None else np.asarray(A_before,float)
    names=[]; lo=[]; hi=[]; cost=[]
    def add(name,l=0.0,u=None,c=0.0):
        i=len(names); names.append(name); lo.append(l); hi.append(u); cost.append(c); return i
    bvar={}
    if event_hour==0:
        for j in range(144): bvar[j]=add(('B',j))
    qnode={}
    if allow_revision:
        if event_hour>0:
            for j in range(startj,144): qnode[(event_hour,0,j)]=add(('A',event_hour,0,j))
        for sh in [h for h in EVENT_HOURS if h>event_hour]:
            for nid in sorted(set(bundle.tree_nodes[sh].values())):
                for j in range(EVENT_PLAN_START[sh],144): qnode[(sh,nid,j)]=add(('A',sh,nid,j))
    xv={};yv={};ev={};sv={};qcont={};rterm={}
    for k in range(K):
        for t in range(H):
            xv[k,t]=add(('x',k,t),0,XMAX); yv[k,t]=add(('y',k,t),0,XMAX); ev[k,t]=add(('e',k,t),0,None,float(bundle.weights[k]*5*bundle.slot_prices[t]))
            tt=bundle.slot_starts[t]
            if not (event_hour==0 and t==0) and plan_slot_index_for_time(day,tt) is None:
                qcont[k,t]=add(('qcont',k,t),0,None,float(bundle.weights[k]*bundle.slot_prices[t]))
        for t in range(H+1): sv[k,t]=add(('s',k,t),SOC_MIN,SOC_MAX)
        rterm[k]=add(('rterm',k),0,None,float(bundle.weights[k]*terminal_value))
    phiv={}
    for k in range(K):
        for t,tt in enumerate(bundle.slot_starts):
            j=plan_slot_index_for_time(day,tt)
            if j is not None and j>=startj: phiv[k,t]=add(('phi',k,t),0,None,float(bundle.weights[k]))
    eqr=[];eqb=[];ubr=[];ubb=[]
    def eq(row,b): eqr.append(row);eqb.append(float(b))
    def le(row,b): ubr.append(row);ubb.append(float(b))
    def b_expr(j): return ({bvar[j]:1.0},0.0) if event_hour==0 else ({},float(B0[j]))
    def q_expr(k,t):
        tt=bundle.slot_starts[t]
        if event_hour==0 and t==0: return {},float(lead_contract),None
        j=plan_slot_index_for_time(day,tt)
        if j is None: return ({qcont[k,t]:1.0},0.0,None)
        if not allow_revision:
            return (({bvar[j]:1.0},0.0,j) if event_hour==0 else ({},float(A0[j]),j))
        sh=_stage(day,event_hour,tt)
        if event_hour==0 and sh==0: return {bvar[j]:1.0},0.0,j
        if sh==event_hour:
            if event_hour==0: return {bvar[j]:1.0},0.0,j
            return {qnode[(event_hour,0,j)]:1.0},0.0,j
        nid=bundle.tree_nodes[sh][k]
        key=(sh,nid,j)
        if key in qnode: return {qnode[key]:1.0},0.0,j
        prev=max(h for h in EVENT_HOURS if event_hour<=h<sh)
        if prev==event_hour:
            if event_hour==0:return {bvar[j]:1.0},0.0,j
            return {qnode[(event_hour,0,j)]:1.0},0.0,j
        nid2=bundle.tree_nodes[prev][k]; return {qnode[(prev,nid2,j)]:1.0},0.0,j
    for k in range(K):
        eq({sv[k,0]:1.0},soc0)
        for t in range(H):
            eq({sv[k,t+1]:1,sv[k,t]:-1,xv[k,t]:-ETA_C,yv[k,t]:1/ETA_D},0)
            qc,qconst,_=q_expr(k,t); row={xv[k,t]:1,yv[k,t]:-1,ev[k,t]:-1}
            for ix,co in qc.items(): row[ix]=row.get(ix,0)-co
            le(row,-float(bundle.scenarios[k,t])+qconst)
        le({sv[k,H]:-1,rterm[k]:-1},-float(np.clip(soc0,SOC_MIN,SOC_MAX)))
    if revision_anchor not in ('original_anchor','stepwise_revision'):
        raise ValueError(revision_anchor)
    if revision_anchor=='original_anchor':
        for (k,t),pi in phiv.items():
            qc,qconst,j=q_expr(k,t); bc,bconst=b_expr(j); p=float(bundle.slot_prices[t])
            if down_settlement=='cancel_settlement': lines=((.5*p,.5*p),(1.5*p,-.5*p))
            elif down_settlement=='sunk_plan_plus_penalty': lines=((-0.5*p,1.5*p),(1.5*p,-.5*p))
            else: raise ValueError(down_settlement)
            for aq,ab in lines:
                row={pi:-1.0}
                for ix,co in qc.items():row[ix]=row.get(ix,0)+aq*co
                for ix,co in bc.items():row[ix]=row.get(ix,0)+ab*co
                le(row,-aq*qconst-ab*bconst)
    else:
        # Strict stepwise semantics: delivery-price purchase of final accepted
        # quantity plus 0.5*p for every legal contract revision along the
        # information-node path. Past revision costs are sunk at a later event.
        for (k,t),pi in phiv.items():
            qc,qconst,j=q_expr(k,t); p=float(bundle.slot_prices[t])
            row={pi:-1.0}
            for ix,co in qc.items():row[ix]=row.get(ix,0)+p*co
            le(row,-p*qconst)

            # Accepted schedule immediately before the first decision that is
            # still controllable at this event.
            if event_hour==0:
                prev_c,prev_v=b_expr(j)
                stages=[h for h in EVENT_HOURS if h>0]
            else:
                prev_c,prev_v={},float(A0[j])
                stages=[h for h in EVENT_HOURS if h>=event_hour]
            tt=bundle.slot_starts[t]
            for sh in stages:
                # Revision at sh may only affect slots whose delivery has not
                # started and whose plan index is within that stage's boundary.
                if j<EVENT_PLAN_START[sh] or datetime(day.year,day.month,day.day)+timedelta(hours=sh)>tt:
                    continue
                if sh==event_hour and event_hour>0:
                    curr_c,curr_v={qnode[(sh,0,j)]:1.0},0.0
                else:
                    nid=bundle.tree_nodes[sh][k]
                    key=(sh,nid,j)
                    if key not in qnode:
                        continue
                    curr_c,curr_v={qnode[key]:1.0},0.0
                dev=add(('stepdev',k,t,sh),0,None,float(bundle.weights[k]*0.5*p))
                row={dev:-1.0}
                for ix,co in curr_c.items():row[ix]=row.get(ix,0)+co
                for ix,co in prev_c.items():row[ix]=row.get(ix,0)-co
                le(row,prev_v-curr_v)
                row={dev:-1.0}
                for ix,co in prev_c.items():row[ix]=row.get(ix,0)+co
                for ix,co in curr_c.items():row[ix]=row.get(ix,0)-co
                le(row,curr_v-prev_v)
                prev_c,prev_v=curr_c,curr_v
    N=len(names)
    def sp(rows):
        rr=[];cc=[];vv=[]
        for r,row in enumerate(rows):
            for j,x in row.items(): rr.append(r);cc.append(j);vv.append(x)
        return coo_matrix((vv,(rr,cc)),shape=(len(rows),N)).tocsr()
    Aeq=sp(eqr); Aub=sp(ubr); eqb_a=np.asarray(eqb); ubb_a=np.asarray(ubb); t0=time.perf_counter()
    res=linprog(np.asarray(cost),A_ub=Aub,b_ub=ubb_a,A_eq=Aeq,b_eq=eqb_a,bounds=list(zip(lo,hi)),method='highs',options={'presolve':True})
    secs=time.perf_counter()-t0
    if not res.success: raise RuntimeError(f'event LP失败 day={day.date()} event={event_hour}: {res.status} {res.message}')
    v=res.x; eqres=float(np.max(np.abs(Aeq@v-eqb_a))) if len(eqr) else 0.; ubv=float(max(0,np.max(Aub@v-ubb_a))) if len(ubr) else 0.
    if event_hour==0:
        out=np.asarray([v[bvar[j]] for j in range(144)])
    else:
        out=A0.copy()
        if allow_revision:
            for j in range(startj,144): out[j]=v[qnode[(event_hour,0,j)]]
    ss=np.zeros((K,H+1)); ee=np.zeros((K,H))
    for k in range(K):
        for t in range(H+1):ss[k,t]=v[sv[k,t]]
        for t in range(H):ee[k,t]=v[ev[k,t]]
    expected=np.average(ss,axis=0,weights=bundle.weights); ctimes=[bundle.slot_starts[t] for k,t in qcont]
    return EventSolution(event_hour,out,expected,ss,ee,float(res.fun),str(res.message),secs,eqres,ubv,K,len(bundle.node_members),min(ctimes) if ctimes else None)

def natural_contract(plan_A,d,i):
    if i==0:return 0.0 if d==0 else float(plan_A[d-1,143])
    return float(plan_A[d,i-1])

def natural_regular_fee(plan_B,plan_A,data,d,down_settlement='cancel_settlement'):
    total=0.0
    for i in range(144):
        if i==0:
            if d==0:continue
            pd,j=d-1,143
        else:pd,j=d,i-1
        total+=float(settlement_components([plan_B[pd,j]],[plan_A[pd,j]],[data.price_plan[j]],down_settlement)['F_regular'][0])
    return total
