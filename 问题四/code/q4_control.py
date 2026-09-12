# -*- coding: utf-8 -*-
"""10-minute causal source-flow executor.

The event LP supplies a causal SOC reference path.  At execution the current
load/PV/price are observed at the slot start.  The controller first minimizes
current emergency load purchase, then tracks the event SOC reference with the
minimum battery throughput required, and finally minimizes PV curtailment.
Every returned action is checked against q4_flow's canonical source-flow rows;
there is no post-optimization projection.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np

from q4_data import ETA_C, ETA_D, SOC_MIN, SOC_MAX, XMAX
from q4_flow import FlowIndex, const_expr, flow_constraint_rows, structure_signature


@dataclass
class ControlAction:
    qL:float; qB:float; u:float; gL:float; gB:float; kappa:float
    eL:float; eB:float; x:float; y:float; S0:float; S1:float; I:float
    target_soc:float; residual_max:float; schema_hash:str


def execute_slot(*,Q:float,load_kwh:float,pv_kwh:float,S0:float,target_soc:float,
                 allow_emergency_charging:bool=False)->ControlAction:
    Q=max(0.0,float(Q)); L=max(0.0,float(load_kwh)); G=max(0.0,float(pv_kwh))
    S0=float(np.clip(S0,SOC_MIN,SOC_MAX)); target=float(np.clip(target_soc,SOC_MIN,SOC_MAX))
    # Level 1: use physical non-emergency sources before emergency.  PV receives
    # load priority over contract so that equivalent solutions consume renewable
    # energy first (the third lexicographic objective).
    gL=min(G,L); rem=L-gL
    qL=min(Q,rem); rem-=qL
    # Battery discharge only if needed to avoid current emergency purchase.
    y=min(rem,XMAX,max(0.0,(S0-SOC_MIN)*ETA_D))
    rem-=y
    eL=max(0.0,rem)
    # Remaining source entitlement/solar can charge only toward the causal event
    # SOC reference.  This is the compact continuation-value tracking rule.
    Qr=Q-qL; Gr=G-gL
    charge_need=max(0.0,(target-(S0-y/ETA_D))/ETA_C)
    charge_cap=min(XMAX,max(0.0,(SOC_MAX-(S0-y/ETA_D))/ETA_C))
    x=min(charge_need,charge_cap,Qr+Gr)
    # Renewable-first charging minimizes curtailment among same tracked state.
    gB=min(Gr,x); qB=min(Qr,x-gB)
    eB=0.0
    if allow_emergency_charging and x < min(charge_need,charge_cap):
        # Explicit sensitivity only.  Main model never enters this branch.
        eB=min(min(charge_need,charge_cap)-x,XMAX-x); x+=eB
    u=max(0.0,Qr-qB); kappa=max(0.0,Gr-gB)
    S1=S0+ETA_C*x-y/ETA_D
    S1=float(np.clip(S1,SOC_MIN,SOC_MAX))

    # Canonical-constraint audit using the exact shared q4_flow definition.
    f=FlowIndex.from_sequence(range(11))
    rows=flow_constraint_rows(f,Q=const_expr(Q),load_kwh=L,pv_kwh=G,S0=const_expr(S0))
    v=np.asarray([qL,qB,u,gL,gB,kappa,eL,eB,x,y,S1],float)
    residual=[]
    for r in rows:
        lhs=sum(a*v[j] for j,a in r.coeff.items()); residual.append(abs(lhs-r.rhs))
    rmax=float(max(residual) if residual else 0.0)
    if rmax>2e-7 or not (SOC_MIN-2e-7<=S1<=SOC_MAX+2e-7) or x>XMAX+2e-7 or y>XMAX+2e-7:
        raise AssertionError(f'executor violates canonical physics: residual={rmax}')
    return ControlAction(qL,qB,u,gL,gB,kappa,eL,eB,x,y,S0,S1,qL+qB,target,rmax,structure_signature())


def isomorphism_audit()->dict:
    cases=[
        dict(Q=1000,L=500,G=0,S=10800,T=10800),
        dict(Q=0,L=300,G=1200,S=10800,T=10800),
        dict(Q=0,L=100,G=0,S=1200,T=1200),
        dict(Q=800,L=1000,G=300,S=6000,T=6200),
        dict(Q=1200,L=500,G=700,S=5000,T=5600),
    ]
    rows=[]
    for c in cases:
        a=execute_slot(Q=c['Q'],load_kwh=c['L'],pv_kwh=c['G'],S0=c['S'],target_soc=c['T'])
        rows.append({'case':c,'residual':a.residual_max,'eB':a.eB,'S1':a.S1,'unused':a.u,'curtail':a.kappa})
    return {'schema_hash':structure_signature(),'rows':rows,'max_residual':max(r['residual'] for r in rows)}
