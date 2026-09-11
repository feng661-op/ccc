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

def _pf(data,day_idx,event_hour,horizon,use_new_vintage,disabled_vintage_hours=()):
    disabled=tuple(sorted(int(x) for x in disabled_vintage_hours)); key=(id(data),day_idx,event_hour,horizon,bool(use_new_vintage),disabled)
    if key not in _PF_CACHE:
        _PF_CACHE[key]=event_point_forecast(data,day_idx,event_hour,horizon,use_new_vintage,disabled)
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
    latest_history_day: int; clip_rate: float=0.0; point_only: bool=False

@dataclass
class EventSolution:
    event_hour:int; current_contract:np.ndarray; expected_soc:np.ndarray
    scenario_soc:np.ndarray; scenario_emergency:np.ndarray; objective:float; solve_status:str
    solve_seconds:float; max_eq_residual:float; max_ub_violation:float; scenario_count:int; node_count:int
    continuation_first_time:Optional[datetime]; terminal_time:datetime; point_net:np.ndarray; point_prices:np.ndarray
    scenario_weights:np.ndarray; clip_rate:float; terminal_shortfall_expected:float; terminal_shortfall_binding:int

def _tree(source_days,data,current_day,event_hour,use_new_vintage,disabled_vintage_hours=()):
    K=len(source_days); tree={event_hour:{k:0 for k in range(K)}}; members={(event_hour,0):tuple(range(K))}; groups=[tuple(range(K))]; nid=1
    for eh in [h for h in EVENT_HOURS if h>event_hour]:
        mp={}; ng=[]
        for g in groups:
            if len(g)<=1: parts=[g]
            else:
                scored=[]
                for k in g:
                    sig=prefix_signal(data,source_days[k],eh,use_new_vintage,disabled_vintage_hours); scored.append((float(sig[0]+.0002*sig[1]),k))
                scored.sort(); cut=max(1,len(scored)//2); parts=[tuple(x[1] for x in scored[:cut]),tuple(x[1] for x in scored[cut:])]; parts=[p for p in parts if p]
            for p in parts:
                for k in p: mp[k]=nid
                members[(eh,nid)]=p; ng.append(p); nid+=1
        tree[eh]=mp; groups=ng
    return tree,members

def build_scenarios(data,day_idx,event_hour,horizon=None,use_new_vintage=True,max_scenarios=3,history_window=56,disabled_vintage_hours=()):
    if horizon is None: horizon=EVENT_PLAN_HORIZON[event_hour]
    disabled=tuple(sorted(int(x) for x in disabled_vintage_hours))
    point,times,prices=_pf(data,day_idx,event_hour,horizon,use_new_vintage,disabled); latest=day_idx-2
    cur=prefix_signal(data,day_idx,event_hour,use_new_vintage,disabled); cand=[]
    for h in range(max(0,latest-history_window+1),latest+1):
        actual=_actual(data,h,event_hour,horizon)
        if actual is None: continue
        pred,_,_=_pf(data,h,event_hour,horizon,use_new_vintage,disabled); sig=prefix_signal(data,h,event_hour,use_new_vintage,disabled)
        resid=actual-pred; risk=float(np.dot(prices,np.maximum(resid,0))); cand.append((h,resid,risk,sig))
    # K=1 is deliberately the pure point forecast, never a tail representative.
    if max_scenarios<=1 or not cand:
        tree={}; members={}; nid=0
        for eh in [h for h in EVENT_HOURS if h>=event_hour]:
            tree[eh]={0:nid}; members[(eh,nid)]=(0,); nid+=1
        return ScenarioBundle(point,point[None,:],[],np.ones(1),times,prices,tree,members,latest,0.0,True)
    sigs=np.asarray([x[3] for x in cand],float)
    med=np.median(sigs,axis=0); scale=1.4826*np.median(np.abs(sigs-med),axis=0)
    std=np.std(sigs,axis=0); scale=np.where(scale>1e-9,scale,np.where(std>1e-9,std,1.0))
    rows=[]
    for h,resid,risk,sig in cand:
        dist=float(np.linalg.norm((sig-cur)/scale)); rows.append((dist,risk,h,resid))
    rows.sort(key=lambda z:(z[0],z[2])); pool=rows[:min(15,len(rows))]; pool.sort(key=lambda z:(z[1],z[2])); K=min(int(max_scenarios),len(pool))
    groups=[np.asarray(g,dtype=int) for g in np.array_split(np.arange(len(pool)),K) if len(g)]
    pick=[]; weights=[]
    for ids in groups:
        rr=np.asarray([pool[int(i)][3] for i in ids],float)
        if len(ids)==1: m=0
        else:
            # empirical risk-bin medoid; avoids equal-weighting one extreme day as a full third of probability mass
            dmat=np.linalg.norm(rr[:,None,:]-rr[None,:,:],axis=2); m=int(np.argmin(dmat.sum(axis=1)))
        pick.append(pool[int(ids[m])]); weights.append(len(ids)/len(pool))
    raw=np.asarray([point+r[3] for r in pick],float); scenarios=np.clip(raw,-2500,3000); clipped=int(np.count_nonzero(np.abs(raw-scenarios)>1e-12)); total=int(raw.size)
    src=[int(r[2]) for r in pick]; w=np.asarray(weights,float); w=w/w.sum(); tree,members=_tree(src,data,day_idx,event_hour,use_new_vintage,disabled)
    return ScenarioBundle(point,scenarios,src,w,times,prices,tree,members,latest,clipped/max(1,total),False)

def _stage(day,event_hour,t):
    base=datetime(day.year,day.month,day.day); valid=[h for h in EVENT_HOURS if h>=event_hour and base+timedelta(hours=h)<=t]
    return max(valid) if valid else event_hour

def execute_actual_interval(soc,contract,net_actual,target_after):
    """Legacy target-floor executor retained only for regression comparisons."""
    margin=float(max(0,contract)-net_actual)
    if margin>=0:
        x=min(margin,XMAX,max(0,(SOC_MAX-soc)/ETA_C)); return soc+ETA_C*x,x,0.0,0.0,max(0,margin-x)
    deficit=-margin; floor=float(np.clip(target_after,SOC_MIN,SOC_MAX)); y=min(deficit,XMAX,max(0,(soc-floor)*ETA_D)); return soc-y/ETA_D,0.0,y,max(0,deficit-y),0.0

def solve_dispatch_mpc(soc0,contracts,net_forecast,prices,terminal_value=.45):
    """Strictly causal deterministic 10-min receding-horizon storage dispatch.

    Contracts are fixed until the next 0/6/12/18 contract event. Future net load
    is the latest causal point forecast; only the first slot is replaced by the
    causal forecast available before the interval starts. The LP is re-solved every 10 minutes, so no
    current/future realized aggregate load or expected-SOC floor enters the current action.
    """
    q=np.maximum(0,np.asarray(contracts,float)); net=np.asarray(net_forecast,float).copy(); pr=np.asarray(prices,float); H=len(q)
    if H<1 or len(net)<H or len(pr)<H: raise ValueError('dispatch MPC horizon mismatch')
    net=net[:H]; pr=pr[:H]
    # x,y,e,s[0:H+1],terminal shortfall r
    nx=H; ny=H; ne=H; ns=H+1; ir=nx+ny+ne+ns; N=ir+1
    c=np.zeros(N); c[:H]=1e-8; c[H:2*H]=1e-8; c[2*H:3*H]=5*pr; c[ir]=float(terminal_value)
    eqr=[]; eqb=[]; ubr=[]; ubb=[]
    row={3*H:1.0}; eqr.append(row); eqb.append(float(soc0))
    for h in range(H):
        eqr.append({3*H+h+1:1.0,3*H+h:-1.0,h:-ETA_C,H+h:1/ETA_D}); eqb.append(0.0)
        # q+y+e >= net+x  -> x-y-e <= q-net
        ubr.append({h:1.0,H+h:-1.0,2*H+h:-1.0}); ubb.append(float(q[h]-net[h]))
    # s_H + r >= s_0: soft opportunity value, not a hard SOC floor.
    ubr.append({3*H+H:-1.0,ir:-1.0}); ubb.append(-float(np.clip(soc0,SOC_MIN,SOC_MAX)))
    def dense(rows):
        A=np.zeros((len(rows),N))
        for r,d in enumerate(rows):
            for j,v in d.items(): A[r,j]=v
        return A
    bounds=[(0,XMAX)]*(2*H)+[(0,None)]*H+[(SOC_MIN,SOC_MAX)]*(H+1)+[(0,None)]
    res=linprog(c,A_ub=dense(ubr),b_ub=np.asarray(ubb),A_eq=dense(eqr),b_eq=np.asarray(eqb),bounds=bounds,method='highs',options={'presolve':True})
    if not res.success: raise RuntimeError(f'dispatch MPC failed: {res.status} {res.message}')
    # Project any LP-degenerate simultaneous charge/discharge to the unique
    # unidirectional action with the SAME SOC increment. This never increases
    # grid demand (the removed round-trip loss becomes curtailment), so it is
    # physically dominant when export is forbidden and preserves the receding
    # horizon state exactly.
    xr=float(res.x[0]); yr=float(res.x[H]); delta=ETA_C*xr-yr/ETA_D
    if delta>=-1e-10:
        x=max(0.0,delta/ETA_C); y=0.0
    else:
        x=0.0; y=max(0.0,-delta*ETA_D)
    soc1=float(soc0+ETA_C*x-y/ETA_D)
    e_pred=max(0.0,float(net[0]+x-q[0]-y))
    return soc1,x,y,e_pred,float(res.x[ir])

def solve_event_lp(data,day_idx,event_hour,soc0,B,A_before,lead_contract=0.0,*,use_new_vintage=True,allow_revision=True,scenario_count=3,horizon=None,down_settlement='cancel_settlement',revision_anchor='original_anchor',terminal_value=.45,disabled_vintage_hours=()):
    if horizon is None: horizon=EVENT_PLAN_HORIZON[event_hour]
    bundle=build_scenarios(data,day_idx,event_hour,horizon,use_new_vintage,scenario_count,disabled_vintage_hours=disabled_vintage_hours); K,H=bundle.scenarios.shape; day=data.dates[day_idx]; startj=EVENT_PLAN_START[event_hour]
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
    xv={};yv={};ev={};sv={};rterm={}
    for k in range(K):
        for t in range(H):
            xv[k,t]=add(('x',k,t),0,XMAX); yv[k,t]=add(('y',k,t),0,XMAX); ev[k,t]=add(('e',k,t),0,None,float(bundle.weights[k]*5*bundle.slot_prices[t]))
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
        if j is None: raise AssertionError(f'contract horizon crossed next-day 00:10: {tt}')
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
    expected=np.average(ss,axis=0,weights=bundle.weights); rvals=np.asarray([v[rterm[k]] for k in range(K)],float)
    terminal_time=bundle.slot_starts[-1]+timedelta(minutes=10)
    return EventSolution(event_hour,out,expected,ss,ee,float(res.fun),str(res.message),secs,eqres,ubv,K,len(bundle.node_members),None,terminal_time,bundle.point_net.copy(),bundle.slot_prices.copy(),bundle.weights.copy(),float(bundle.clip_rate),float(np.dot(bundle.weights,rvals)),int(np.count_nonzero(rvals>1e-8)))

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
