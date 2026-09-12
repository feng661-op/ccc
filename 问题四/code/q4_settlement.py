# -*- coding: utf-8 -*-
"""Authoritative Q4 contract and emergency settlement formulas."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Sequence
import numpy as np


@dataclass(frozen=True)
class Settlement:
    keep_kwh: float
    down_kwh: float
    up_kwh: float
    keep_fee: float
    down_fee: float
    up_fee: float
    regular_fee: float
    emergency_kwh: float
    emergency_fee: float
    total_cash: float


def settlement_components(B: float, A: float, price: float, emergency_kwh: float = 0.0) -> Settlement:
    """Original-B anchor; cancellation=.5c, upward increment=1.5c, emergency=5c.

    B is the 00:00 original contract, A the final contract after legal revisions.
    Unused physical entitlement is *not* refunded here; settlement depends on the
    contractual A/B pair, not realized physical import I.
    """
    B=max(0.0,float(B)); A=max(0.0,float(A)); c=max(0.0,float(price)); e=max(0.0,float(emergency_kwh))
    keep=min(A,B); down=max(B-A,0.0); up=max(A-B,0.0)
    fkeep=c*keep; fdown=0.5*c*down; fup=1.5*c*up
    regular=fkeep+fdown+fup; emg=5.0*c*e
    return Settlement(keep,down,up,fkeep,fdown,fup,regular,e,emg,regular+emg)


def regular_fee(B: float, A: float, price: float) -> float:
    return settlement_components(B,A,price,0.0).regular_fee


def settlement_vector(B: Sequence[float], A: Sequence[float], price: Sequence[float], emergency: Sequence[float] | None = None):
    B=np.asarray(B,float); A=np.asarray(A,float); c=np.asarray(price,float)
    if B.shape!=A.shape or B.shape!=c.shape: raise ValueError('shape mismatch')
    e=np.zeros_like(B) if emergency is None else np.asarray(emergency,float)
    if e.shape!=B.shape: raise ValueError('emergency shape mismatch')
    keep=np.minimum(B,A); down=np.maximum(B-A,0.0); up=np.maximum(A-B,0.0)
    return {
        'keep_kwh':keep,'down_kwh':down,'up_kwh':up,
        'keep_fee':c*keep,'down_fee':0.5*c*down,'up_fee':1.5*c*up,
        'regular_fee':c*keep+0.5*c*down+1.5*c*up,
        'emergency_fee':5.0*c*e,
        'total_cash':c*keep+0.5*c*down+1.5*c*up+5.0*c*e,
    }


def convex_fee_lines(B: float, price: float):
    """Return (slope, intercept) lines whose max equals regular fee in A."""
    B=max(0.0,float(B)); c=max(0.0,float(price))
    return ((0.5*c,0.5*c*B),(1.5*c,-0.5*c*B))


def hand_check() -> dict:
    rows=[]
    for B,A,c in [(100,60,0.8),(100,100,0.8),(100,140,0.8),(0,50,1.2)]:
        s=settlement_components(B,A,c,7)
        lines=convex_fee_lines(B,c)
        phi=max(m*A+b for m,b in lines)
        rows.append({'B':B,'A':A,'c':c,'formula':s.regular_fee,'epigraph':phi,'diff':abs(phi-s.regular_fee)})
    return {'rows':rows,'max_abs_diff':max(x['diff'] for x in rows),'semantics':'original_B_anchor_delivery_price'}
