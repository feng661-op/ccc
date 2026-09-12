# -*- coding: utf-8 -*-
"""Price Oracle diagnostics and strict Full-information Oracle cash lower bound.

PO only removes future-price uncertainty and is NEVER called a theoretical
lower bound. FI relaxes information/nonanticipativity over the formal horizon,
keeps the identical q4_flow physics, already-locked Feb-01 lead contract,
settlement semantics, initial SOC, and an explicitly matched final SOC. Only
that matched FI cash optimum is reported as a strict cash lower bound.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, Mapping, Optional, Sequence, Tuple
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

from q4_data import Q4Data, SOC_MIN, SOC_MAX, FORMAL_START_DAY, FORMAL_END_DAY
from q4_flow import FlowIndex, canonical_bounds, const_expr, flow_constraint_rows, var_expr, structure_signature
from q4_settlement import settlement_components


class SparseLP:
    def __init__(self):
        self.n=0; self.bounds=[]; self.eq=[]; self.ub=[]
    def var(self,lb=0.0,ub=None):
        j=self.n; self.n+=1; self.bounds.append((lb,ub)); return j
    def add_eq(self,co,rhs): self.eq.append((dict(co),float(rhs)))
    def add_ub(self,co,rhs): self.ub.append((dict(co),float(rhs)))
    def matrix(self,rows):
        rr=[];cc=[];vv=[];bb=[]
        for i,(co,b) in enumerate(rows):
            for j,a in co.items():
                if a: rr.append(i);cc.append(int(j));vv.append(float(a))
            bb.append(float(b))
        return csr_matrix((vv,(rr,cc)),shape=(len(rows),self.n)),np.asarray(bb,float)


@dataclass
class FIResult:
    success:bool
    message:str
    strict_cash_lower_bound_yuan:float
    variable_lp_cash_yuan:float
    locked_lead_regular_fee_yuan:float
    final_soc_target_kwh:float
    initial_soc_kwh:float
    locked_lead_B_kwh:float
    locked_lead_A_kwh:float
    emergency_kwh:float
    contract_kwh:float
    solve_seconds:float
    max_eq_residual:float
    schema_hash:str
    scope:str


def full_information_cash_lower_bound(data:Q4Data, *, initial_soc:float, final_soc_target:float,
                                      locked_lead_B:float, locked_lead_A:float,
                                      start_day:int=FORMAL_START_DAY,end_day:int=FORMAL_END_DAY)->FIResult:
    """Solve one global formal-period LP under perfect future information.

    Natural slots are [Feb-01 00:00, Jan-01 00:00) for the 334 formal days.
    The very first 00:00-00:10 entitlement is Jan-31 plan-row j=143 and is
    already locked, so its original-B settlement is a fixed cash constant. All
    later entitlements may be chosen with full information (B=A=Q); allowing
    later revisions cannot improve on this direct full-information choice.
    """
    if not (SOC_MIN<=initial_soc<=SOC_MAX and SOC_MIN<=final_soc_target<=SOC_MAX): raise ValueError('SOC boundary')
    lp=SparseLP(); flows=[]; qvars=[]; c_contract={}; c_emg={}; c_thru={}; c_kappa={}
    prevS=const_expr(float(initial_soc)); total_slots=(int(end_day)-int(start_day))*144
    hglobal=0
    for d in range(int(start_day),int(end_day)):
        for i in range(144):
            L=float(data.load_cal_kwh[d,i]); G=float(data.pv_cal_kwh[d,i]); c=float(data.price_cal[d,i])
            if not (np.isfinite(L) and np.isfinite(G) and np.isfinite(c)): raise ValueError(f'missing formal actual at {d},{i}')
            q=None if hglobal==0 else lp.var(0.0,None)
            ids=[lp.var() for _ in range(11)]
            f=FlowIndex.from_sequence(ids); flows.append(f); qvars.append(q)
            lp.bounds=canonical_bounds(f,lp.n,allow_emergency_charging=False,base=lp.bounds)
            if hglobal==0:
                Q=const_expr(max(0.0,float(locked_lead_A)))
            else:
                Q=var_expr(q)
                c_contract[q]=c_contract.get(q,0.0)+c
            rows=flow_constraint_rows(f,Q=Q,load_kwh=L,pv_kwh=G,S0=prevS)
            for r in rows:
                if r.kind=='eq': lp.add_eq(r.coeff,r.rhs)
                else: lp.add_ub(r.coeff,r.rhs)
            c_emg[f.eL]=c_emg.get(f.eL,0.0)+5.0*c
            c_emg[f.eB]=c_emg.get(f.eB,0.0)+5.0*c
            c_thru[f.x]=1.0; c_thru[f.y]=1.0; c_kappa[f.kappa]=1.0
            prevS=var_expr(f.S1); hglobal+=1
    # Matched terminal state is an equality, not a penalty.
    lp.add_eq({flows[-1].S1:1.0},float(final_soc_target))
    c1=np.zeros(lp.n,float)
    for j,a in c_contract.items(): c1[j]+=a
    for j,a in c_emg.items(): c1[j]+=a
    Aeq,beq=lp.matrix(lp.eq); Aub,bub=lp.matrix(lp.ub)
    t0=perf_counter(); res=linprog(c1,A_ub=Aub,b_ub=bub,A_eq=Aeq,b_eq=beq,bounds=lp.bounds,method='highs'); sec=perf_counter()-t0
    lead_regular=settlement_components(float(locked_lead_B),float(locked_lead_A),float(data.price_cal[start_day,0]),0.0).regular_fee
    if not res.success:
        return FIResult(False,res.message,float('nan'),float('nan'),lead_regular,float(final_soc_target),float(initial_soc),
                        float(locked_lead_B),float(locked_lead_A),float('nan'),float('nan'),sec,float('inf'),structure_signature(),
                        'matched-physics/contracts/settlement/initial-final-SOC formal cash objective')
    x=res.x; residual=float(np.max(np.abs(Aeq@x-beq))) if len(beq) else 0.0
    emg=float(sum(x[f.eL]+x[f.eB] for f in flows))
    contract=float(max(0.0,locked_lead_A)+sum(x[q] for q in qvars[1:] if q is not None))
    variable=float(c1@x); total=variable+lead_regular
    return FIResult(True,res.message,total,variable,lead_regular,float(final_soc_target),float(initial_soc),float(locked_lead_B),
                    float(locked_lead_A),emg,contract,sec,residual,structure_signature(),
                    'STRICT cash lower bound: same q4_flow physics; same original-B settlement for locked first lead; identical initial/final SOC; same formal natural-day cash objective; future information/nonanticipativity relaxed')


def audit_oracle_claim(po_cash:float,fi:FIResult,main_cash:float,tol:float=1e-5)->dict:
    return {
        'Price_Oracle':{'cash_yuan':float(po_cash),'claim':'diagnostic information-value comparator only; NOT a theoretical lower bound'},
        'Full_information_Oracle':{'cash_yuan':float(fi.strict_cash_lower_bound_yuan),'claim':'strict cash lower bound only under the matched conditions recorded in FIResult',
                                   'success':bool(fi.success),'schema_hash':fi.schema_hash,'scope':fi.scope},
        'main_cash_yuan':float(main_cash),
        'strict_lower_bound_check_pass':bool(fi.success and fi.strict_cash_lower_bound_yuan<=float(main_cash)+tol*max(1.0,abs(main_cash))),
        'main_minus_FI_yuan':float(main_cash-fi.strict_cash_lower_bound_yuan) if fi.success else None,
        'PO_minus_FI_yuan':float(po_cash-fi.strict_cash_lower_bound_yuan) if fi.success else None,
    }
