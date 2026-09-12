# -*- coding: utf-8 -*-
"""Frozen Q4 evaluation metrics and paired moving-block comparisons."""
from __future__ import annotations
from typing import Dict, Sequence
import numpy as np
import pandas as pd


def empirical_cvar(x:Sequence[float],alpha:float=.95)->float:
    z=np.asarray(x,float)
    if z.size==0: return float('nan')
    z=np.sort(z)[::-1]; remaining=(1.0-float(alpha))*len(z); acc=0.; mass=0.
    for v in z:
        take=min(1.0,remaining-mass)
        if take>0: acc+=take*float(v); mass+=take
        if mass>=remaining-1e-12: break
    return float(acc/remaining)


def day_metrics(days:pd.DataFrame)->dict:
    c=days.cash_fee_yuan.to_numpy(float); e=days.emergency_kwh.to_numpy(float)
    return {
        'n_days':int(len(days)),'cash_total_yuan':float(c.sum()),'cash_mean_yuan_per_day':float(c.mean()),
        'cash_median_yuan_per_day':float(np.median(c)),'cash_p95_yuan_per_day':float(np.quantile(c,.95)),
        'cash_cvar90_yuan_per_day':empirical_cvar(c,.90),'cash_cvar95_yuan_per_day':empirical_cvar(c,.95),
        'cash_worst_day_yuan':float(c.max()),'emergency_total_kwh':float(e.sum()),
        'emergency_days':int(np.sum(e>1e-8)),'emergency_worst_day_kwh':float(e.max()),
        'curtailment_total_kwh':float(days.curtailment_kwh.sum()),'unused_contract_total_kwh':float(days.unused_contract_kwh.sum()),
        'soc_end_kwh':float(days.S24_kwh.iloc[-1]),'max_physics_residual':float(days.max_physics_residual.max()),
    }


def _moving_block_indices(n:int,block:int,rng:np.random.Generator)->np.ndarray:
    out=[]
    while len(out)<n:
        start=int(rng.integers(0,max(1,n-block+1)))
        out.extend(range(start,min(n,start+block)))
    return np.asarray(out[:n],int)


def paired_block_bootstrap(simple:pd.DataFrame,complex_:pd.DataFrame, *,block_days:int=7,reps:int=2000,seed:int=20260912)->dict:
    a=simple[['date','cash_fee_yuan']].rename(columns={'cash_fee_yuan':'a'})
    b=complex_[['date','cash_fee_yuan']].rename(columns={'cash_fee_yuan':'b'})
    m=a.merge(b,on='date',how='inner').sort_values('date')
    if len(m)<5: raise ValueError('too few paired days')
    A=m.a.to_numpy(float); B=m.b.to_numpy(float); n=len(m); rng=np.random.default_rng(seed)
    mean_gain=[]; tail_gain=[]
    for _ in range(int(reps)):
        ix=_moving_block_indices(n,min(int(block_days),n),rng)
        aa=A[ix]; bb=B[ix]
        mean_gain.append(float(np.mean(aa-bb)))
        tail_gain.append(float(empirical_cvar(aa,.95)-empirical_cvar(bb,.95)))
    mg=np.asarray(mean_gain); tg=np.asarray(tail_gain)
    return {
        'n_days':n,'block_days':min(int(block_days),n),'reps':int(reps),
        'mean_cash_gain_yuan':float(np.mean(A-B)),
        'mean_cash_gain_80pct_interval':[float(np.quantile(mg,.1)),float(np.quantile(mg,.9))],
        'mean_cash_improvement_probability':float(np.mean(mg>0)),
        'cvar95_gain_yuan':float(empirical_cvar(A,.95)-empirical_cvar(B,.95)),
        'cvar95_gain_80pct_interval':[float(np.quantile(tg,.1)),float(np.quantile(tg,.9))],
        'cvar95_improvement_probability':float(np.mean(tg>0)),
    }


def relative_improvement(simple_value:float,complex_value:float)->float:
    return float((float(simple_value)-float(complex_value))/max(1e-12,abs(float(simple_value))))
