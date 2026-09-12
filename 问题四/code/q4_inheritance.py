"""Q4 inheritance-v2: extend, never replace, the current Q2/Q3 policy.

Q2/Q3 files are imported read-only. The 289-slot valuation window, historical
net-error scenarios, emergency CVaR and quarter-quantile reserve feedback are
inherited. Only price information/price scenario paths change. Future-day
purchases are valuation recourse, NOT contracts issued with future information.
The planning LP is the inherited net-balance relaxation; actual execution is
checked against the complete source-flow equations. They are not claimed to be
identical optimization models.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter
from typing import List, Tuple, Optional, Sequence
import sys
import numpy as np
from scipy.optimize import linprog, milp, Bounds, LinearConstraint
from scipy.sparse import lil_matrix, coo_matrix, vstack

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'问题三'/'code'))
import q3_inheritance as upstream
import q2_core as q2
from q3_data import Q3Data
from q3_physical import physical_flows
from q4_data import DT, ETA_C, ETA_D, SOC_MIN, SOC_MAX, SOC0, XMAX, EVENT_PLAN_START, actual_at
from q4_flow import structure_signature

VERSION='q4-inheritance-v2'
CVAR_ALPHA=q2.CVAR_ALPHA
TERMINAL_SHORTAGE_VALUE=q2.TERMINAL_SHORTAGE_VALUE
TERMINAL_RESERVE=q2.TERMINAL_RESERVE
DayPlanSolution=q2.DayPlanSolution
empirical_cvar_equal_prob=q2.empirical_cvar_equal_prob
_ADAPTERS={}

def adapter(data):
    hit=_ADAPTERS.get(id(data))
    if hit is not None and hit[0] is data: return hit[1]
    p=np.asarray(data.fixed_price_plan,float)
    a=Q3Data(data.dates,p,np.r_[p[-1],p[:143]],data.load_plan_kwh,data.pv_plan_kwh,
        data.net_plan_kwh,data.load_cal_kwh,data.pv_cal_kwh,data.net_cal_kwh,
        data.plan_headers,data.forecasts,data.forecast_records)
    _ADAPTERS[id(data)]=(data,a)
    return a

def price_point(data,day_idx,event_hour,times,cfg):
    """Lag7 is a causal weekly price forecast, including the second day.

    The current price is observable under our stated real-time-price assumption;
    no future current-day price/actual net load is read. An unavailable weekly
    source falls back to the known periodic Attachment-1 tariff.
    """
    decision=data.dates[day_idx]+timedelta(hours=event_hour)
    out=[]
    for t in times:
        j=((t.hour*60+t.minute)//10-1)%144
        base=float(data.fixed_price_plan[j])
        if cfg.fixed_price or cfg.decision_fixed_price:
            out.append(base); continue
        if cfg.price_config.model!='lag7':
            raise ValueError('inheritance-v2 production requires the January-frozen lag7 price model')
        if cfg.price_oracle:
            v=actual_at(data,t,'price')
        elif t<=decision:
            v=actual_at(data,t,'price')
        else:
            src=t-timedelta(days=7)
            if src>=decision: raise AssertionError('future price source')
            v=actual_at(data,src,'price')
        out.append(base if v is None else float(v))
    return np.maximum(np.asarray(out,float),0.0)

def make_bundle(data,day_idx,event_hour,cfg):
    a=adapter(data)
    use_official=(cfg.branch=='q4_3' and cfg.use_pv_updates)
    b=upstream.build_scenarios(a,day_idx,event_hour,use_new_vintage=use_official,
        max_scenarios=9,disabled_vintage_hours=cfg.disabled_vintage_hours)
    point=price_point(data,day_idx,event_hour,b.slot_starts,cfg)
    prices=np.tile(point,(len(b.weights),1))
    if cfg.level=='B1' and not (cfg.fixed_price or cfg.decision_fixed_price or cfg.price_oracle):
        # Couple price and net-load errors through the SAME historical triplet.
        # The last source row h+1 is complete before today's midnight.
        for k,h in enumerate(b.source_days):
            if h+1>day_idx-2: raise AssertionError('unreleased joint residual source')
            issue=data.dates[h]+timedelta(hours=event_hour)
            ht=[issue+timedelta(minutes=10*j) for j in range(len(point))]
            old_pred=price_point(data,h,event_hour,ht,cfg)
            realized=np.asarray([actual_at(data,t,'price') for t in ht],float)
            if not np.isfinite(realized).all(): raise AssertionError('missing historical price residual')
            prices[k]=np.maximum(0.0,point+realized-old_pred)
        # The current revealed price has no scenario uncertainty.
        prices[:,0]=point[0]
    elif cfg.level not in ('B0','B1'):
        raise ValueError('B2 robust search is not part of this no-robustness revision')
    return b,prices

@dataclass
class InheritedEvent:
    contract_A:np.ndarray
    original_B:np.ndarray
    scenario_soc:np.ndarray
    scenario_emergency:np.ndarray
    reserve_floor:np.ndarray
    objective:float
    solve_seconds:float
    max_eq_residual:float
    scenario_count:int
    source_days:list
    prices:np.ndarray
    binaries:int=0
    mip_gap:float=0.0


def solve_intraday(bundle,prices,soc0,B,A0,lead,eh,cfg,event_price=None,fixed_contract=None):
    K,H=bundle.scenarios.shape; start=eh*6; startj=EVENT_PLAN_START[eh]
    cost=[]; bounds=[]; eqs=[]; rhs=[]; ubs=[]; urhs=[]; integers=[]
    def add(c=0.0,bound=(0,None),integer=False):
        i=len(cost);cost.append(float(c));bounds.append(bound);integers.append(int(integer));return i
    q={j:add() for j in range(startj,144)}
    if not cfg.allow_contract_revision or fixed_contract is not None:
        fixed=A0 if fixed_contract is None else np.asarray(fixed_contract,float)
        for j,ix in q.items(): bounds[ix]=(float(fixed[j]),float(fixed[j]))
    future={};x={};y={};e={};s={};r={}
    for k in range(K):
        w=bundle.weights[k]
        for t in range(H):
            if start+t>=145: future[k,t]=add(w*prices[k,t])
            x[k,t]=add(bound=(0,XMAX));y[k,t]=add(bound=(0,XMAX));e[k,t]=add(w*5*prices[k,t])
        for t in range(H+1): s[k,t]=add(bound=(SOC_MIN,SOC_MAX))
        r[k]=add(w*cfg.terminal_value)
    zeta=add(cfg.risk_lambda)
    excess={k:add(cfg.risk_lambda*bundle.weights[k]/(1-cfg.alpha)) for k in range(K)}
    for k in range(K):
        eqs.append({s[k,0]:1});rhs.append(float(soc0));risk={zeta:-1,excess[k]:-1}
        for t in range(H):
            eqs.append({s[k,t+1]:1,s[k,t]:-1,x[k,t]:-ETA_C,y[k,t]:1/ETA_D});rhs.append(0)
            row={x[k,t]:1,y[k,t]:-1,e[k,t]:-1};g=start+t;v=-bundle.scenarios[k,t]
            if g==0:v+=lead
            elif g<=144:row[q[g-1]]=-1
            else:row[future[k,t]]=-1
            ubs.append(row);urhs.append(v)
            if g<=144:risk[e[k,t]]=5*prices[k,t]
        ubs.append({s[k,H]:-1,r[k]:-1});urhs.append(-cfg.terminal_reserve)
        ubs.append(risk);urhs.append(0)
    common=np.array_equal(prices,np.tile(prices[0],(K,1)))
    for j,ix in q.items():
        t=j+1-start;b=float(B[j]); groups=[(0,1.0)] if common else list(enumerate(bundle.weights))
        event_clock=getattr(cfg,'settlement_clock','delivery')=='adjustment_time'
        U=max(b,float(np.max(bundle.scenarios[:,t])+XMAX),1.0)
        # A bound above every scenario's useful import cannot remove a
        # positive-price optimum. It also supplies a finite, derived Big-M.
        nonconvex=event_clock and np.any(prices[:,t]>2*float(event_price)+1e-12)
        binary=None
        if nonconvex:
            binary=add(bound=(0,1),integer=True)
            if bounds[ix][0]!=bounds[ix][1]:bounds[ix]=(0,U)
            M=max(U,b)
            ubs.extend(({ix:1,binary:-M},{ix:-1,binary:M}));urhs.extend((b,M-b))
        for k,w in groups:
            p=float(prices[k,t]);pe=float(event_price) if event_clock else p
            phi=add(w)
            lines=((p-.5*pe,.5*pe*b),(1.5*pe,(p-1.5*pe)*b)) if event_clock else ((.5*p,.5*p*b),(1.5*p,-.5*p*b))
            if nonconvex:
                m0,b0=lines[0];m1,b1=lines[1];Mphi=abs(2*pe-p)*max(U,b)+1e-7
                ubs.extend(({ix:m0,phi:-1,binary:-Mphi},{ix:m1,phi:-1,binary:Mphi}))
                urhs.extend((-b0,Mphi-b1))
            else:
                for m0,b0 in lines:ubs.append({ix:m0,phi:-1});urhs.append(-b0)
    N=len(cost)
    def sparse(rows):
        rr=[];cc=[];vv=[]
        for i,row in enumerate(rows):
            for j,v in row.items():rr.append(i);cc.append(j);vv.append(v)
        return coo_matrix((vv,(rr,cc)),shape=(len(rows),N)).tocsr()
    eq=sparse(eqs);ub=sparse(ubs);t0=perf_counter()
    if any(integers):
        mat=vstack([eq,ub],format='csr');lo=np.r_[rhs,np.full(len(urhs),-np.inf)];hi=np.r_[rhs,urhs]
        lb=[-np.inf if v[0] is None else v[0] for v in bounds];upper=[np.inf if v[1] is None else v[1] for v in bounds]
        sol=milp(cost,integrality=np.asarray(integers),bounds=Bounds(lb,upper),constraints=LinearConstraint(mat,lo,hi),options={'mip_rel_gap':1e-7,'time_limit':60})
    else:
        sol=linprog(cost,A_eq=eq,b_eq=rhs,A_ub=ub,b_ub=urhs,bounds=bounds,method='highs',options={'presolve':True})
    elapsed=perf_counter()-t0
    if not sol.success:raise RuntimeError(f'inherited event LP/MILP failed: {sol.message}')
    v=sol.x;out=A0.copy()
    for j,ix in q.items():out[j]=max(0,float(v[ix]))
    ss=np.array([[v[s[k,t]] for t in range(H+1)] for k in range(K)])
    ee=np.array([[v[e[k,t]] for t in range(H)] for k in range(K)])
    resid=max(float(np.max(np.abs(eq@v-rhs))),float(max(0,np.max(ub@v-urhs))))
    return InheritedEvent(out,B.copy(),ss,ee,np.quantile(ss,.25,axis=0),float(sol.fun),elapsed,resid,K,
        list(bundle.source_days),prices,int(sum(integers)),float(getattr(sol,'mip_gap',0) or 0))


def solve_event(data,day_idx,eh,soc,B,A,lead,cfg,fixed_contract=None):
    bundle,prices=make_bundle(data,day_idx,eh,cfg)
    if eh==0:
        t0=perf_counter()
        sol=solve_midnight(bundle.scenarios,prices,soc,lead,cfg.risk_lambda,cfg.alpha,cfg.terminal_value)
        out=np.maximum(sol.q_current,0);out[np.abs(out)<1e-9]=0
        return InheritedEvent(out,out.copy(),sol.scenario_soc,sol.scenario_emergency,
            np.quantile(sol.scenario_soc,.25,axis=0),sol.objective,perf_counter()-t0,sol.max_residual,
            sol.scenario_count,list(bundle.source_days),prices)
    p=actual_at(data,data.dates[day_idx]+timedelta(hours=eh),'price')
    if cfg.fixed_price:p=float(data.fixed_price_plan[(eh*6-1)%144])
    return solve_intraday(bundle,prices,soc,B,A,lead,eh,cfg,p,fixed_contract)


def execute_inherited(Q,load_kwh,pv_kwh,S0,target_soc):
    s1,x,y,e,w=q2.execute_interval(S0,Q,load_kwh-pv_kwh,target_soc)
    f=physical_flows(Q,load_kwh,pv_kwh,x,y)
    gL=min(pv_kwh,max(0.0,load_kwh-y));gB=min(x,max(0.0,pv_kwh-gL))
    qB=max(0.0,x-gB);qL=max(0.0,f.grid_import-qB)
    kappa=max(0.0,pv_kwh-gL-gB);u=max(0.0,Q-qL-qB)
    residues=[qL+qB+u-Q,gL+gB+kappa-pv_kwh,qL+gL+y+e-load_kwh,x-qB-gB,
              s1-S0-ETA_C*x+y/ETA_D,e-f.emergency,w-f.supply_surplus]
    resid=max(abs(z) for z in residues)
    if resid>1e-6 or min(qL,qB,u,gL,gB,kappa,e,x,y)<-1e-8:raise AssertionError('inherited source-flow violation')
    return SimpleNamespace(qL=qL,qB=qB,u=u,gL=gL,gB=gB,kappa=kappa,eL=e,eB=0.0,x=x,y=y,
        S0=S0,S1=s1,I=f.grid_import,residual_max=resid,schema_hash=structure_signature())


# Native Q2 matrix/variable/constraint ordering retained; only the price
# vector is generalized to causal K x 289 paths. Protected upstream SHA256:
# 1724ae2aed8ca4f8336f8ee6083db0d268852d83baf449b8ef7f5ff02521aef8
def solve_midnight(
    scenario_net: np.ndarray,
    prices: np.ndarray,
    soc0: float,
    q_lead: float,
    risk_lambda: float,
    alpha: float = CVAR_ALPHA,
    terminal_value: float = TERMINAL_SHORTAGE_VALUE,
) -> DayPlanSolution:
    """两阶段随机LP：当天144段q为共同一阶段决策，下一日q为场景自适应的MPC辅助决策。

    H=289 = 1个已承诺lead interval + 当天144段 + 下一天144段。
    lead段购电量已由昨天承诺，今天不可修改且不重复计计划购电费。
    """
    scenario_net = np.asarray(scenario_net, dtype=float)
    if scenario_net.ndim != 2 or scenario_net.shape[1] != 289:
        raise ValueError(f'scenario_net应为(K,289)，实际{scenario_net.shape}')
    K, H = scenario_net.shape
    if K < 1:
        raise ValueError('至少需要1个场景')
    prices = np.asarray(prices, dtype=float)
    if prices.shape != (K,H) or not np.isfinite(prices).all() or np.min(prices)<0:
        raise ValueError('prices must be finite nonnegative K by 289 paths')
    common = np.array_equal(prices,np.tile(prices[0],(K,1)))
    price = prices[0,1:145] if common else np.mean(prices[:,1:145],axis=0)

    nq = 144
    nqf = 144
    block = nqf + 3 * H + (H + 1) + 1  # qfuture,x,y,e,s,r_terminal
    q0 = 0
    scen0 = nq
    zeta_idx = scen0 + K * block
    u0 = zeta_idx + 1
    N = u0 + K

    def offs(k: int):
        b = scen0 + k * block
        qf = b
        x = qf + nqf
        y = x + H
        e = y + H
        s = e + H
        r = s + H + 1
        return qf, x, y, e, s, r

    c = np.zeros(N, dtype=float)
    c[q0:q0 + nq] = price
    p_h = prices
    prob = 1.0 / K
    for k in range(K):
        qf, x, y, e, s, r = offs(k)
        c[qf:qf + nqf] = prob * prices[k,145:]
        c[e:e + H] = prob * 5.0 * prices[k]
        c[r] = prob * terminal_value
    if risk_lambda > 0:
        c[zeta_idx] = risk_lambda
        c[u0:u0 + K] = risk_lambda / ((1.0 - alpha) * K)

    # 等式：每场景固定起始SOC + H段SOC递推
    n_eq = K * (H + 1)
    Aeq = lil_matrix((n_eq, N), dtype=float)
    beq = np.zeros(n_eq, dtype=float)
    rr = 0
    for k in range(K):
        qf, x, y, e, s, rterm = offs(k)
        Aeq[rr, s] = 1.0
        beq[rr] = soc0
        rr += 1
        for t in range(H):
            Aeq[rr, s + t + 1] = 1.0
            Aeq[rr, s + t] = -1.0
            Aeq[rr, x + t] = -ETA_C
            Aeq[rr, y + t] = 1.0 / ETA_D
            rr += 1

    # 不等式：平衡 + 软终端储备 + CVaR
    n_ub = K * H + K + K
    Aub = lil_matrix((n_ub, N), dtype=float)
    bub = np.zeros(n_ub, dtype=float)
    rr = 0
    for k in range(K):
        qf, x, y, e, s, rterm = offs(k)
        for t in range(H):
            # q + y + emergency >= net + x => x-y-e-q <= -net
            Aub[rr, x + t] = 1.0
            Aub[rr, y + t] = -1.0
            Aub[rr, e + t] = -1.0
            if t == 0:
                bub[rr] = float(q_lead - scenario_net[k, t])
            elif 1 <= t <= 144:
                Aub[rr, q0 + (t - 1)] = -1.0
                bub[rr] = -float(scenario_net[k, t])
            else:
                Aub[rr, qf + (t - 145)] = -1.0
                bub[rr] = -float(scenario_net[k, t])
            rr += 1
        # r_terminal >= TERMINAL_RESERVE - S_H
        Aub[rr, s + H] = -1.0
        Aub[rr, rterm] = -1.0
        bub[rr] = -TERMINAL_RESERVE
        rr += 1

    # CVaR：Z_k - zeta - u_k <= 0，Z只针对lead+当前承诺日(t=0..144)紧急购电风险。
    for k in range(K):
        qf, x, y, e, s, rterm = offs(k)
        for t in range(145):
            Aub[rr, e + t] = 5.0 * prices[k,t]
        Aub[rr, zeta_idx] = -1.0
        Aub[rr, u0 + k] = -1.0
        bub[rr] = 0.0
        rr += 1

    bounds: List[Tuple[float, Optional[float]]] = [(0.0, None)] * N
    # 精确覆盖各变量边界
    for k in range(K):
        qf, x, y, e, s, rterm = offs(k)
        for i in range(x, x + H):
            bounds[i] = (0.0, XMAX)
        for i in range(y, y + H):
            bounds[i] = (0.0, XMAX)
        for i in range(e, e + H):
            bounds[i] = (0.0, None)
        for i in range(s, s + H + 1):
            bounds[i] = (SOC_MIN, SOC_MAX)
        bounds[rterm] = (0.0, None)
    bounds[zeta_idx] = (0.0, None)
    for i in range(u0, u0 + K):
        bounds[i] = (0.0, None)

    res = linprog(
        c,
        A_ub=Aub.tocsr(), b_ub=bub,
        A_eq=Aeq.tocsr(), b_eq=beq,
        bounds=bounds,
        method='highs',
        options={'presolve': True},
    )
    if not res.success:
        raise RuntimeError(f'随机MPC求解失败: {res.status} {res.message}')
    v = res.x
    q_current = v[q0:q0 + nq].copy()
    ss = np.zeros((K, H + 1), dtype=float)
    ee = np.zeros((K, H), dtype=float)
    scenario_costs = np.zeros(K, dtype=float)
    for k in range(K):
        qf, x, y, e, s, rterm = offs(k)
        ss[k] = v[s:s + H + 1]
        ee[k] = v[e:e + H]
        scenario_costs[k] = float(np.dot(5.0 * prices[k,:145], ee[k, :145]))
    expected_emergency = float(np.mean(scenario_costs))
    cvar = empirical_cvar_equal_prob(scenario_costs, alpha)
    out = DayPlanSolution(
        q_current=q_current,
        scenario_soc=ss,
        scenario_emergency=ee,
        objective=float(res.fun),
        expected_emergency_cost=expected_emergency,
        cvar_emergency_cost=cvar,
        scenario_count=K,
        solve_status=str(res.message),
    )
    out.max_residual=max(float(np.max(np.abs(Aeq.tocsr() @ v-beq))),float(max(0,np.max(Aub.tocsr() @ v-bub))))
    return out
