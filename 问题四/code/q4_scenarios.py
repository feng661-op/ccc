# -*- coding: utf-8 -*-
"""Leakage-safe joint (load, PV, price) residual paths and transport utilities."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Sequence, Tuple
import numpy as np
from scipy.optimize import linprog

from q4_data import Q4Data, actual_at
from q4_forecast import event_forecast
from q4_information import legal_residual_days
from q4_price import PriceForecaster, PriceConfig


@dataclass
class ScenarioSet:
    load: np.ndarray       # (K,H), kWh
    pv: np.ndarray         # (K,H), kWh
    price: np.ndarray      # (K,H), yuan/kWh
    weights: np.ndarray    # (K,)
    source_days: List[int]
    medoid_source_days: List[int]
    available_at: List[str]
    distance: np.ndarray   # KxK standardized path distance
    point_load: np.ndarray
    point_pv: np.ndarray
    point_price: np.ndarray
    branch: str
    event_hour: int


def _actual_path(data: Q4Data, times: Sequence[datetime]):
    L=[]; G=[]; C=[]
    for t in times:
        l=actual_at(data,t,'load'); g=actual_at(data,t,'pv'); c=actual_at(data,t,'price')
        if l is None or g is None or c is None: return None
        L.append(l);G.append(g);C.append(c)
    return np.asarray(L),np.asarray(G),np.asarray(C)


def _path_residual_for_source(data:Q4Data,pf:PriceForecaster,src_day:int,event_hour:int,horizon:int,branch:str,cfg:PriceConfig,
                              *,use_new_vintage:bool=True,disabled_vintage_hours:Sequence[int]=()):
    cache=getattr(pf,'_joint_residual_cache',None)
    if cache is None:
        cache={}; setattr(pf,'_joint_residual_cache',cache)
    disabled=tuple(int(x) for x in disabled_vintage_hours)
    key=(int(src_day),int(event_hour),int(horizon),str(branch),cfg.version,bool(use_new_vintage),disabled)
    if key in cache:
        return cache[key]
    Lp,Gp,Np,times,_=event_forecast(data,src_day,event_hour,horizon,branch,
                                     use_new_vintage=use_new_vintage,disabled_vintage_hours=disabled)
    issue=data.dates[src_day]+timedelta(hours=event_hour)
    Cp,_=pf.predict(issue,times,Np,cfg)
    act=_actual_path(data,times)
    if act is None:
        cache[key]=None; return None
    La,Ga,Ca=act
    ans=(La-Lp,Ga-Gp,Ca-Cp)
    cache[key]=ans
    return ans


def _robust_scale(x:np.ndarray)->float:
    z0=np.asarray(x,float).ravel()
    # Solar residuals contain structurally zero night slots.  Letting those
    # zeros dominate MAD/IQR can collapse s_G to a near-zero number and make
    # the transport metric numerically meaningless.  When a channel is
    # genuinely zero-inflated, estimate its nonzero-state scale on the same
    # pre-selection training sample; this is still a frozen robust training
    # scale, not a future-data floor or formal-period retuning.
    nz=z0[np.abs(z0)>1e-8]
    z=nz if (len(nz)>=max(10,int(0.25*len(z0))) and len(nz)<0.80*len(z0)) else z0
    med=float(np.median(z))
    mad=float(np.median(np.abs(z-med))*1.4826)
    if mad>1e-8: return mad
    q=float(np.quantile(z,.75)-np.quantile(z,.25))
    if q>1e-8: return q/1.349
    s=float(np.std(z))
    return s if s>1e-8 else 1.0


def frozen_distance_scales(data:Q4Data,pf:PriceForecaster,cfg:PriceConfig,branch:str,event_hour:int,horizon:int)->Tuple[float,float,float]:
    """Freeze s_N,s_G,s_c before the January model-selection window.

    Selection starts at 2025-01-09 00:00 (day index 8).  At that instant only
    complete residual paths through 2025-01-07 are legal for every event-node
    horizon, so source day indices 1..6 are the common causal training pool.
    The resulting numerical scales are cached and reused unchanged in January
    walk-forward selection, formal Feb-Dec replay, and every sensitivity.  This
    prevents future formal residuals from redefining the transport geometry.
    """
    cache=getattr(pf,'_frozen_distance_scale_cache',None)
    if cache is None:
        cache={}; setattr(pf,'_frozen_distance_scale_cache',cache)
    key=(str(branch),int(event_hour),int(horizon),cfg.version)
    if key in cache:
        return cache[key]
    residuals=[]
    for d in range(1,7):
        r=_path_residual_for_source(data,pf,d,event_hour,horizon,branch,cfg,
                                    use_new_vintage=True,disabled_vintage_hours=())
        if r is not None:
            residuals.append(r)
    if len(residuals)<3:
        raise RuntimeError(f'insufficient pre-selection residual paths for frozen distance scale: {key}, n={len(residuals)}')
    rL=np.stack([x[0] for x in residuals]); rG=np.stack([x[1] for x in residuals]); rC=np.stack([x[2] for x in residuals])
    rN=rL-rG
    scales=(float(_robust_scale(rN)),float(_robust_scale(rG)),float(_robust_scale(rC)))
    if not all(np.isfinite(x) and x>0 for x in scales):
        raise RuntimeError(f'invalid frozen distance scales {scales}')
    cache[key]=scales
    return scales


def _joint_distance_matrix(rL:np.ndarray,rG:np.ndarray,rC:np.ndarray,scales:Tuple[float,float,float])->np.ndarray:
    """Frozen V1.2 standardized path distance on (net-load, PV, price).

    All supports at one event share the same information-node type, hence the
    d_nu term is identically zero here. Positive channel weights are 1/3 each.
    """
    rN=np.asarray(rL,float)-np.asarray(rG,float)
    rG=np.asarray(rG,float); rC=np.asarray(rC,float)
    sN,sG,sC=(float(x) for x in scales)
    dn=np.mean(np.abs(rN[:,None,:]-rN[None,:,:]),axis=2)/sN
    dg=np.mean(np.abs(rG[:,None,:]-rG[None,:,:]),axis=2)/sG
    dc=np.mean(np.abs(rC[:,None,:]-rC[None,:,:]),axis=2)/sC
    return (dn+dg+dc)/3.0


def _joint_cross_distance(a:Tuple[np.ndarray,np.ndarray,np.ndarray], b:Tuple[np.ndarray,np.ndarray,np.ndarray],
                          scales:Tuple[float,float,float])->np.ndarray:
    aL,aG,aC=(np.asarray(x,float) for x in a); bL,bG,bC=(np.asarray(x,float) for x in b)
    aN=aL-aG; bN=bL-bG
    sN,sG,sC=(float(x) for x in scales)
    dn=np.mean(np.abs(aN[:,None,:]-bN[None,:,:]),axis=2)/sN
    dg=np.mean(np.abs(aG[:,None,:]-bG[None,:,:]),axis=2)/sG
    dc=np.mean(np.abs(aC[:,None,:]-bC[None,:,:]),axis=2)/sC
    return (dn+dg+dc)/3.0


def _pam_medoids(D:np.ndarray,k:int):
    D=np.asarray(D,float); n=D.shape[0]
    if D.shape!=(n,n): raise ValueError('PAM distance must be square')
    if k>=n: return list(range(n)),np.arange(n)
    # deterministic farthest-first seed from point nearest global centroid
    first=int(np.argmin(D.sum(axis=1)))
    meds=[first]
    while len(meds)<k:
        dmin=np.min(D[:,meds],axis=1); dmin[meds]=-1
        meds.append(int(np.argmax(dmin)))
    improved=True
    while improved:
        improved=False; base=np.min(D[:,meds],axis=1).sum()
        best=(base,None,None)
        non=[i for i in range(n) if i not in meds]
        for mi,m in enumerate(meds):
            for cand in non:
                mm=meds.copy(); mm[mi]=cand
                val=np.min(D[:,mm],axis=1).sum()
                if val < best[0]-1e-10: best=(val,mi,cand)
        if best[1] is not None:
            meds[best[1]]=int(best[2]); improved=True
    labels=np.argmin(D[:,meds],axis=1)
    return meds,labels


def build_joint_scenarios(data:Q4Data,pf:PriceForecaster,day_idx:int,event_hour:int,horizon:int,branch:str,
                          cfg:PriceConfig,k:int=3,history_days:int=42,mismatch:bool=False,
                          *,use_new_vintage:bool=True,disabled_vintage_hours:Sequence[int]=())->ScenarioSet:
    disabled=tuple(int(x) for x in disabled_vintage_hours)
    L0,G0,N0,times,_=event_forecast(data,day_idx,event_hour,horizon,branch,
                                     use_new_vintage=use_new_vintage,disabled_vintage_hours=disabled)
    issue=data.dates[day_idx]+timedelta(hours=event_hour)
    C0,_=pf.predict(issue,times,N0,cfg)
    legal=legal_residual_days(day_idx,event_hour,cap=history_days)
    residuals=[]; used=[]
    for d in legal:
        r=_path_residual_for_source(data,pf,d,event_hour,horizon,branch,cfg,
                                    use_new_vintage=use_new_vintage,disabled_vintage_hours=disabled)
        if r is None: continue
        residuals.append(r); used.append(d)
    if not residuals:
        load=L0[None,:]; pv=G0[None,:]; price=C0[None,:]; w=np.ones(1)
        return ScenarioSet(load,pv,price,w,[],[],[],np.zeros((1,1)),L0,G0,C0,branch,event_hour)
    rL=np.stack([x[0] for x in residuals]); rG=np.stack([x[1] for x in residuals]); rC=np.stack([x[2] for x in residuals])
    if mismatch and len(rC)>1:
        rC=np.roll(rC,1,axis=0)
    scales=frozen_distance_scales(data,pf,cfg,branch,event_hour,horizon)
    Dfull=_joint_distance_matrix(rL,rG,rC,scales)
    kk=max(1,min(int(k),len(used)))
    meds,labels=_pam_medoids(Dfull,kk)
    weights=np.asarray([(labels==j).mean() for j in range(kk)],float)
    medL=rL[meds]; medG=rG[meds]; medC=rC[meds]
    load=np.maximum(0.0,L0[None,:]+medL)
    pv=np.maximum(0.0,G0[None,:]+medG)
    price=np.maximum(0.0,C0[None,:]+medC)
    D=Dfull[np.ix_(meds,meds)]
    available=[(data.dates[d]+timedelta(days=1,minutes=10)).isoformat() if event_hour==0 else (data.dates[d]+timedelta(days=1)).isoformat() for d in used]
    return ScenarioSet(load,pv,price,weights,[int(x) for x in used],[int(used[m]) for m in meds],available,D,L0,G0,C0,branch,event_hour)


def transport_worst_expectation_primal(costs:Sequence[float],p:Sequence[float],D:np.ndarray,epsilon:float):
    """Finite-support Wasserstein transport primal: max_T sum_ab T_ab cost_b."""
    c=np.asarray(costs,float); p=np.asarray(p,float); D=np.asarray(D,float); K=len(c)
    if p.shape!=(K,) or D.shape!=(K,K): raise ValueError('shape')
    n=K*K
    obj=-np.tile(c,K)
    Aeq=np.zeros((K,n)); beq=p.copy()
    for a in range(K): Aeq[a,a*K:(a+1)*K]=1.0
    Aub=D.reshape(1,-1); bub=np.asarray([float(epsilon)])
    res=linprog(obj,A_ub=Aub,b_ub=bub,A_eq=Aeq,b_eq=beq,bounds=[(0,None)]*n,method='highs')
    if not res.success: raise RuntimeError(res.message)
    T=res.x.reshape(K,K); q=T.sum(axis=0)
    return float(-res.fun),q,T


def transport_worst_expectation_dual(costs:Sequence[float],p:Sequence[float],D:np.ndarray,epsilon:float):
    """Dual: min beta*eps + sum_a p_a v_a s.t. v_a+beta*d_ab >= cost_b."""
    c=np.asarray(costs,float); p=np.asarray(p,float); D=np.asarray(D,float); K=len(c)
    # variables [beta, v_0..v_K-1], v free
    obj=np.r_[float(epsilon),p]
    Aub=[]; bub=[]
    for a in range(K):
        for b in range(K):
            row=np.zeros(K+1); row[0]=-D[a,b]; row[1+a]=-1.0
            Aub.append(row); bub.append(-c[b])
    bounds=[(0,None)]+[(None,None)]*K
    res=linprog(obj,A_ub=np.asarray(Aub),b_ub=np.asarray(bub),bounds=bounds,method='highs')
    if not res.success: raise RuntimeError(res.message)
    return float(res.fun),float(res.x[0]),res.x[1:]


def _wasserstein_between_supports(pa:np.ndarray,pb:np.ndarray,D:np.ndarray)->float:
    pa=np.asarray(pa,float); pb=np.asarray(pb,float); D=np.asarray(D,float)
    A=len(pa); B=len(pb)
    if D.shape!=(A,B) or not np.isclose(pa.sum(),1.0) or not np.isclose(pb.sum(),1.0): raise ValueError('W1 support shape')
    n=A*B; c=D.reshape(-1); rows=[]; rhs=[]
    for a in range(A):
        row=np.zeros(n); row[a*B:(a+1)*B]=1; rows.append(row); rhs.append(pa[a])
    for b in range(B):
        row=np.zeros(n); row[b::B]=1; rows.append(row); rhs.append(pb[b])
    # Drop one redundant marginal equality for numerical rank stability.
    res=linprog(c,A_eq=np.asarray(rows[:-1]),b_eq=np.asarray(rhs[:-1]),bounds=[(0,None)]*n,method='highs')
    if not res.success: raise RuntimeError(res.message)
    return float(res.fun)


def calibrate_epsilon_january(data:Q4Data,pf:PriceForecaster,cfg:PriceConfig,branch:str='q4_2')->dict:
    """Successive January W1 drift of legal reduced empirical residual laws.

    Distances use the exact frozen V1.2 standardized (net-load, PV, price) path
    metric.  This calibrates an empirical drift *scale* only, never confidence.
    """
    distances=[]; pairs=[]
    scales=frozen_distance_scales(data,pf,cfg,branch,0,145)
    prev=None
    for d in range(9,31):
        s=build_joint_scenarios(data,pf,d,0,145,branch,cfg,k=3,history_days=42)
        cur=(s, (s.load-s.point_load, s.pv-s.point_pv, s.price-s.point_price))
        if prev is not None:
            ps,pr=prev; cs,cr=cur
            D=_joint_cross_distance(pr,cr,scales)
            val=_wasserstein_between_supports(ps.weights,cs.weights,D)
            distances.append(val); pairs.append({'from_day':int(d-1),'to_day':int(d),'W1':float(val),'Ka':len(ps.weights),'Kb':len(cs.weights)})
        prev=cur
    a=np.asarray(distances,float)
    q50=float(np.quantile(a,.50)) if len(a) else 0.0
    q75=float(np.quantile(a,.75)) if len(a) else 0.0
    return {'window':'January walk-forward only','branch':branch,'distance_definition':'mean L1 standardized residual path on (net-load, PV, price), equal positive channel weights; same-node d_nu=0',
            'frozen_scale_training_window':'fully released source paths 2025-01-02..2025-01-07, frozen before 2025-01-09 selection begins',
            'frozen_scales':{'s_net_kwh':scales[0],'s_pv_kwh':scales[1],'s_price_yuan_per_kwh':scales[2]},
            'distances':[float(x) for x in a],'pairs':pairs,
            'n':int(len(a)),'q50':q50,'q75':q75,'candidates':[0.0,q50,q75],
            'interpretation':'successive empirical W1 distribution-drift scale; NOT a confidence radius or coverage guarantee'}


def validate_transport_duality(seed:int=2026)->dict:
    rng=np.random.default_rng(seed); rows=[]
    for K in (2,3):
        for rep in range(8):
            c=rng.normal(size=K); p=rng.random(K); p/=p.sum()
            X=rng.normal(size=(K,4)); D=np.sqrt(((X[:,None,:]-X[None,:,:])**2).mean(axis=2))
            eps=float(rng.uniform(0,max(1e-6,np.max(D))))
            vp,q,T=transport_worst_expectation_primal(c,p,D,eps)
            vd,beta,v=transport_worst_expectation_dual(c,p,D,eps)
            rows.append({'K':K,'rep':rep,'primal':vp,'dual':vd,'gap':abs(vp-vd),'epsilon':eps})
    return {'rows':rows,'max_gap':max(r['gap'] for r in rows)}


def build_point_scenario(data:Q4Data,pf:PriceForecaster,day_idx:int,event_hour:int,horizon:int,branch:str,cfg:PriceConfig,
                         *,price_oracle:bool=False,use_new_vintage:bool=True,disabled_vintage_hours:Sequence[int]=())->ScenarioSet:
    """B0 causal point path (or E14 price-only Oracle) with the same schema as B1/B2."""
    L,G,N,times,_=event_forecast(data,day_idx,event_hour,horizon,branch,use_new_vintage=use_new_vintage,
                                 disabled_vintage_hours=disabled_vintage_hours)
    issue=data.dates[day_idx]+timedelta(hours=event_hour)
    C,_=pf.predict(issue,times,N,cfg,oracle_price=price_oracle)
    return ScenarioSet(L[None,:],G[None,:],C[None,:],np.ones(1),[],[],[],np.zeros((1,1)),L,G,C,branch,event_hour)
