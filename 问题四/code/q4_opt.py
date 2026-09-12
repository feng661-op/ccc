# -*- coding: utf-8 -*-
"""Event-level source-flow LP/SAA/transport-DRO optimizer for Q4 V1.2.

The optimizer and the real-time executor share q4_flow.flow_constraint_rows.
There is no net-balance solve followed by a physical projection.  Contracts are
scenario-shared variables (nonanticipativity); physical source flows and SOC are
scenario recourse variables.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

from q4_data import SOC_MIN, SOC_MAX
from q4_flow import (
    Expr, FlowIndex, canonical_bounds, const_expr, flow_constraint_rows,
    structure_signature, var_expr,
)
from q4_scenarios import ScenarioSet
from q4_settlement import convex_fee_lines
from q4_tree import event_horizon

CVAR_ALPHA_DEFAULT = 0.80
TERMINAL_RESERVE_DEFAULT = 6000.0
TERMINAL_VALUE_DEFAULT = 0.80


class _LP:
    def __init__(self):
        self.n=0
        self.names: List[str]=[]
        self.bounds: List[Tuple[Optional[float],Optional[float]]]=[]
        self.eq: List[Tuple[Dict[int,float],float,str]]=[]
        self.ub: List[Tuple[Dict[int,float],float,str]]=[]

    def var(self,name:str,lb:Optional[float]=0.0,ub:Optional[float]=None)->int:
        j=self.n; self.n+=1; self.names.append(name); self.bounds.append((lb,ub)); return j

    def add_eq(self,coeff:Mapping[int,float],rhs:float,name:str=''):
        self.eq.append(({int(k):float(v) for k,v in coeff.items() if abs(float(v))>0},float(rhs),name))

    def add_ub(self,coeff:Mapping[int,float],rhs:float,name:str=''):
        self.ub.append(({int(k):float(v) for k,v in coeff.items() if abs(float(v))>0},float(rhs),name))

    def matrix(self,extra_ub:Sequence[Tuple[np.ndarray,float]]=()):
        def sparse(rows):
            rr=[];cc=[];vv=[];b=[]
            for i,(co,rhs,_) in enumerate(rows):
                for j,a in co.items(): rr.append(i);cc.append(j);vv.append(a)
                b.append(rhs)
            return csr_matrix((vv,(rr,cc)),shape=(len(rows),self.n)),np.asarray(b,float)
        Aeq,beq=sparse(self.eq)
        ub=list(self.ub)
        for ix,(vec,rhs) in enumerate(extra_ub):
            co={int(j):float(v) for j,v in enumerate(np.asarray(vec,float)) if abs(float(v))>0}
            ub.append((co,float(rhs),f'lex_extra_{ix}'))
        Aub,bub=sparse(ub)
        return Aeq,beq,Aub,bub

    def solve(self,c:np.ndarray,extra_ub:Sequence[Tuple[np.ndarray,float]]=()):
        Aeq,beq,Aub,bub=self.matrix(extra_ub)
        return linprog(np.asarray(c,float),A_ub=Aub,b_ub=bub,A_eq=Aeq,b_eq=beq,
                       bounds=self.bounds,method='highs')


def _expr_add(dst:Dict[int,float],src:Mapping[int,float],scale:float=1.0):
    for j,a in src.items(): dst[int(j)]=dst.get(int(j),0.0)+float(scale)*float(a)


def _eval_expr(co:Mapping[int,float],x:np.ndarray)->float:
    return float(sum(float(a)*float(x[int(j)]) for j,a in co.items()))


@dataclass
class EventSolution:
    success: bool
    message: str
    branch: str
    event_hour: int
    model_level: str
    contract_A: np.ndarray
    original_B: np.ndarray
    expected_soc_end: np.ndarray
    scenario_cash: np.ndarray
    scenario_augmented_cost: np.ndarray
    scenario_weights: np.ndarray
    expected_cash: float
    weighted_cvar_cash: float
    level1_objective: float
    level2_throughput: float
    level3_curtailment: float
    solve_seconds: float
    max_eq_residual: float
    lex_tolerance1: float
    lex_tolerance2: float
    physical_schema_hash: str
    predicted_emergency_kwh: float
    predicted_unused_contract_kwh: float
    predicted_curtailment_kwh: float
    predicted_terminal_shortage_kwh: float
    dro_epsilon: float
    dro_expected_value: Optional[float]
    dro_cvar_value: Optional[float]
    final_level1_objective: float = float('nan')
    final_level2_objective: float = float('nan')


def _discrete_cvar(values:Sequence[float],weights:Sequence[float],alpha:float)->float:
    z=np.asarray(values,float); w=np.asarray(weights,float)
    order=np.argsort(z)
    zs=z[order]; ws=w[order]
    # integrate upper (1-alpha) mass exactly, splitting atoms at VaR
    remaining=1.0-float(alpha); acc=0.0; taken=0.0
    for zi,wi in zip(zs[::-1],ws[::-1]):
        take=min(float(wi),remaining-taken)
        if take>0: acc += take*float(zi); taken += take
        if taken>=remaining-1e-14: break
    return acc/remaining


def solve_event(
    *, scenarios:ScenarioSet, soc0:float, day, event_hour:int, branch:str,
    original_B:Sequence[float], current_A:Sequence[float], lead_q:float,
    model_level:str='B1', risk_lambda:float=0.0, alpha:float=CVAR_ALPHA_DEFAULT,
    epsilon:float=0.0, terminal_reserve:float=TERMINAL_RESERVE_DEFAULT,
    terminal_value:float=TERMINAL_VALUE_DEFAULT, allow_emergency_charging:bool=False,
    lock_contracts:bool=False, lex_abs_tol:float=1e-7, lex_rel_tol:float=1e-9,
)->EventSolution:
    if model_level not in ('B0','B1','B2'):
        raise ValueError(model_level)
    B=np.asarray(original_B,float).copy(); A0=np.asarray(current_A,float).copy()
    if B.shape!=(144,) or A0.shape!=(144,): raise ValueError('B/A must be 144 slots')
    slots=event_horizon(day,event_hour,branch)
    H=len(slots); K=len(scenarios.weights)
    if scenarios.load.shape!=(K,H) or scenarios.pv.shape!=(K,H) or scenarios.price.shape!=(K,H):
        raise ValueError(f'scenario shape does not match event horizon: K={K}, H={H}')
    w=np.asarray(scenarios.weights,float)
    if np.any(w<0) or not np.isclose(w.sum(),1.0,atol=1e-10): raise ValueError('bad scenario weights')
    if not SOC_MIN-1e-8 <= float(soc0) <= SOC_MAX+1e-8: raise ValueError('soc0')
    lp=_LP()

    # Shared contract variables implement nonanticipativity.
    qidx: Dict[int,int]={}
    for s in slots:
        if s.mutable and not lock_contracts and s.plan_j not in qidx:
            qidx[s.plan_j]=lp.var(f'A[{s.plan_j}]',0.0,None)

    flows: List[List[FlowIndex]]=[]
    scenario_cash: List[Dict[int,float]]=[]
    scenario_aug: List[Dict[int,float]]=[]
    zterm=[]
    phi: Dict[Tuple[int,int],int]={}

    for k in range(K):
        fk=[]; cash:Dict[int,float]={}
        prevS:Expr=const_expr(float(soc0))
        for h,s in enumerate(slots):
            ids=[lp.var(f'{nm}[{k},{h}]') for nm in ('qL','qB','u','gL','gB','kappa','eL','eB','x','y','S1')]
            f=FlowIndex.from_sequence(ids); fk.append(f)
            # apply canonical per-flow bounds to global bound vector
            bnds=canonical_bounds(f,lp.n,allow_emergency_charging=allow_emergency_charging,base=lp.bounds)
            lp.bounds=bnds
            if s.lead_from_previous_contract:
                Q=const_expr(float(max(0.0,lead_q)))
            elif s.mutable and not lock_contracts:
                Q=var_expr(qidx[s.plan_j])
            else:
                Q=const_expr(float(max(0.0,A0[s.plan_j])))
            rows=flow_constraint_rows(f,Q=Q,load_kwh=float(scenarios.load[k,h]),pv_kwh=float(scenarios.pv[k,h]),S0=prevS)
            for r in rows:
                if r.kind=='eq': lp.add_eq(r.coeff,r.rhs,f'flow:{k}:{h}:{r.name}')
                else: lp.add_ub(r.coeff,r.rhs,f'flow:{k}:{h}:{r.name}')
            prevS=var_expr(f.S1)
            c=float(scenarios.price[k,h])
            # Contract cash for this delivery interval. The event-0 lead was
            # contracted yesterday and is sunk at the current event.
            if not s.lead_from_previous_contract and not lock_contracts:
                if event_hour==0:
                    _expr_add(cash,{qidx[s.plan_j]:c})
                else:
                    pvar=lp.var(f'phi[{k},{h}]',0.0,None); phi[(k,h)]=pvar
                    for line_id,(m,b) in enumerate(convex_fee_lines(float(B[s.plan_j]),c)):
                        # phi >= m*A+b -> m*A-phi <= -b
                        lp.add_ub({qidx[s.plan_j]:m,pvar:-1.0},-b,f'settle_epi:{k}:{h}:{line_id}')
                    _expr_add(cash,{pvar:1.0})
            _expr_add(cash,{f.eL:5.0*c,f.eB:5.0*c})
        flows.append(fk)
        z=lp.var(f'term_short[{k}]',0.0,None); zterm.append(z)
        # z >= reserve - S_H -> -S_H-z <= -reserve
        lp.add_ub({fk[-1].S1:-1.0,z:-1.0},-float(terminal_reserve),f'terminal:{k}')
        aug=dict(cash); _expr_add(aug,{z:float(terminal_value)})
        scenario_cash.append(cash); scenario_aug.append(aug)

    n_before_risk=lp.n
    c1=np.zeros(lp.n,float)
    dro_expected_vars=None; dro_cvar_vars=None
    if model_level in ('B0','B1'):
        # Empirical expectation of cash plus continuation value.  Tail risk is
        # defined on cash C only, exactly as frozen in V1.2.
        for k in range(K):
            for j,a in scenario_aug[k].items(): c1[j]+=w[k]*a
        if risk_lambda>0 and K>1:
            zeta=lp.var('cvar_zeta',None,None)
            excess=[lp.var(f'cvar_excess[{k}]',0.0,None) for k in range(K)]
            c1=np.pad(c1,(0,lp.n-len(c1)))
            for k in range(K):
                row=dict(scenario_cash[k]); _expr_add(row,{zeta:-1.0,excess[k]:-1.0})
                lp.add_ub(row,0.0,f'cvar:{k}')
            c1[zeta]+=float(risk_lambda)
            for k in range(K): c1[excess[k]]+=float(risk_lambda)*w[k]/(1.0-float(alpha))
    else:
        if scenarios.distance.shape!=(K,K): raise ValueError('DRO distance shape')
        D=np.asarray(scenarios.distance,float)
        # One transport inner problem over the augmented scenario expression:
        # C_b + V_b + lambda/(1-alpha) u_b, with u_b >= C_b-zeta.
        zeta=lp.var('dro_zeta',None,None) if (risk_lambda>0 and K>1) else None
        u=[lp.var(f'dro_tail_u[{b}]',0.0,None) for b in range(K)] if zeta is not None else []
        beta=lp.var('dro_beta',0.0,None)
        v=[lp.var(f'dro_v[{a}]',None,None) for a in range(K)]
        c1=np.pad(c1,(0,lp.n-len(c1)))
        if zeta is not None:
            for b in range(K):
                row=dict(scenario_cash[b]); _expr_add(row,{zeta:-1.0,u[b]:-1.0})
                lp.add_ub(row,0.0,f'dro_tail_loss:{b}')
        for a in range(K):
            for b in range(K):
                row=dict(scenario_aug[b])
                if zeta is not None:
                    _expr_add(row,{u[b]:float(risk_lambda)/(1.0-float(alpha))})
                _expr_add(row,{v[a]:-1.0,beta:-float(D[a,b])})
                lp.add_ub(row,0.0,f'dro:{a}:{b}')
        c1[beta]+=float(epsilon)
        for a in range(K): c1[v[a]]+=w[a]
        if zeta is not None: c1[zeta]+=float(risk_lambda)
        dro_expected_vars=(beta,v)
        if zeta is not None: dro_cvar_vars=(zeta,u)

    # Resize objective if risk vars were not used but LP grew by terminal/phi after c1 initialized.
    if len(c1)<lp.n: c1=np.pad(c1,(0,lp.n-len(c1)))
    t0=perf_counter(); r1=lp.solve(c1)
    if not r1.success:
        return EventSolution(False,r1.message,branch,event_hour,model_level,A0,B,np.full(H,np.nan),np.full(K,np.nan),np.full(K,np.nan),w,
                             np.nan,np.nan,np.nan,np.nan,np.nan,perf_counter()-t0,np.inf,0,0,structure_signature(),
                             np.nan,np.nan,np.nan,np.nan,float(epsilon),None,None)
    f1=float(c1@r1.x); tol1=max(float(lex_abs_tol),float(lex_rel_tol)*max(1.0,abs(f1)))

    c2=np.zeros(lp.n,float)
    for k in range(K):
        for h in range(H):
            c2[flows[k][h].x]+=w[k]; c2[flows[k][h].y]+=w[k]
    r2=lp.solve(c2,[(c1,f1+tol1)])
    if not r2.success: raise RuntimeError(f'lex level2 failed: {r2.message}')
    f2=float(c2@r2.x); tol2=max(float(lex_abs_tol),float(lex_rel_tol)*max(1.0,abs(f2)))

    c3=np.zeros(lp.n,float)
    for k in range(K):
        for h in range(H): c3[flows[k][h].kappa]+=w[k]
    r3=lp.solve(c3,[(c1,f1+tol1),(c2,f2+tol2)])
    if not r3.success: raise RuntimeError(f'lex level3 failed: {r3.message}')
    xsol=r3.x; elapsed=perf_counter()-t0

    A=A0.copy()
    for j,ix in qidx.items(): A[j]=max(0.0,float(xsol[ix]))
    # At 00:00 the original anchor is exactly the newly placed contract.
    Bout=B.copy()
    if event_hour==0:
        for j in qidx: Bout[j]=A[j]

    s_cash=np.asarray([_eval_expr(co,xsol) for co in scenario_cash],float)
    s_aug=np.asarray([_eval_expr(co,xsol) for co in scenario_aug],float)
    expected_soc=np.asarray([sum(w[k]*xsol[flows[k][h].S1] for k in range(K)) for h in range(H)],float)
    emerg=sum(w[k]*sum(xsol[flows[k][h].eL]+xsol[flows[k][h].eB] for h in range(H)) for k in range(K))
    unused=sum(w[k]*sum(xsol[flows[k][h].u] for h in range(H)) for k in range(K))
    curt=sum(w[k]*sum(xsol[flows[k][h].kappa] for h in range(H)) for k in range(K))
    tsh=sum(w[k]*xsol[zterm[k]] for k in range(K))
    Aeq,beq,_,_=lp.matrix([(c1,f1+tol1),(c2,f2+tol2)])
    maxres=float(np.max(np.abs(Aeq@xsol-beq))) if len(beq) else 0.0
    droE=None; droCV=None
    if dro_expected_vars is not None:
        beta,v=dro_expected_vars; droE=float(epsilon*xsol[beta]+sum(w[a]*xsol[v[a]] for a in range(K)))
    if dro_cvar_vars is not None:
        zeta,u=dro_cvar_vars
        # Diagnostic empirical CVaR at the selected decision; the optimized
        # robust tail term is already inside droE via the single transport dual.
        droCV=float(xsol[zeta]+sum(w[b]*xsol[u[b]] for b in range(K))/(1.0-alpha))
    return EventSolution(
        True,str(r3.message),branch,event_hour,model_level,A,Bout,expected_soc,s_cash,s_aug,w,
        float(w@s_cash),_discrete_cvar(s_cash,w,alpha),f1,f2,float(c3@xsol),elapsed,maxres,tol1,tol2,
        structure_signature(),float(emerg),float(unused),float(curt),float(tsh),float(epsilon),droE,droCV,
        float(c1@xsol),float(c2@xsol),
    )


def epsilon_zero_equivalence(**kwargs)->dict:
    """Numerical B1 vs B2(eps=0, lambda=0) equivalence diagnostic."""
    a=solve_event(model_level='B1',risk_lambda=0.0,epsilon=0.0,**kwargs)
    b=solve_event(model_level='B2',risk_lambda=0.0,epsilon=0.0,**kwargs)
    return {'B1_success':a.success,'B2_success':b.success,
            'contract_max_abs_diff':float(np.max(np.abs(a.contract_A-b.contract_A))),
            'level1_abs_diff':abs(a.level1_objective-b.level1_objective),
            'expected_cash_abs_diff':abs(a.expected_cash-b.expected_cash)}
