# -*- coding: utf-8 -*-
"""R0 repricing and fixed-price bridge diagnostics for Q4 V1.2.

R0 freezes the already-validated Q2/Q3 trajectories and changes no decision.
It merely replaces Attachment-1 tariff settlement by Attachment-4 realized
price settlement.  It is a price-exposure diagnostic, never an executable Q4
strategy and never included in the B0/B1/B2 ranking.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple
import json
import numpy as np

from q4_data import Q4Data, FORMAL_START_DAY, FORMAL_END_DAY
from q4_settlement import settlement_vector


def _natural_from_plan(a:np.ndarray)->np.ndarray:
    a=np.asarray(a,float)
    if a.shape!=(365,144): raise ValueError(a.shape)
    out=np.full_like(a,np.nan)
    out[1:,0]=a[:-1,143]
    out[:,1:]=a[:,:143]
    return out


def _summary(cash:np.ndarray,regular:np.ndarray,emergency_fee:np.ndarray,emergency_kwh:np.ndarray)->dict:
    s=slice(FORMAL_START_DAY,FORMAL_END_DAY)
    daily=np.nansum(cash[s],axis=1)
    return {
        'period':'2025-02-01..2025-12-31','n_days':334,
        'cash_total_yuan':float(np.nansum(cash[s])),
        'regular_total_yuan':float(np.nansum(regular[s])),
        'emergency_total_yuan':float(np.nansum(emergency_fee[s])),
        'emergency_total_kwh':float(np.nansum(emergency_kwh[s])),
        'daily_mean_yuan':float(np.mean(daily)),'daily_p95_yuan':float(np.quantile(daily,.95)),
        'daily_worst_yuan':float(np.max(daily)),
    }


def reprice_q2(repo:Path,data:Q4Data)->dict:
    z=np.load(Path(repo)/'问题二'/'code'/'run_data.npz',allow_pickle=True)
    Qp=np.asarray(z['plan_q'],float); e=np.asarray(z['emergency'],float)
    Qn=_natural_from_plan(Qp)
    fixed=data.fixed_price_cal; var=data.price_cal
    reg_fixed=fixed*Qn; emg_fixed=5.0*fixed*e; cash_fixed=reg_fixed+emg_fixed
    reg_var=var*Qn; emg_var=5.0*var*e; cash_var=reg_var+emg_var
    stored=float(np.sum(np.asarray(z['plan_cost'])[31:])+np.sum(np.asarray(z['emergency_cost'])[31:]))
    # Stored Q2 plan-row accounting includes each current row's +1 tail whereas
    # this natural-day ledger includes the previous row lead instead.  The two
    # differ only by Jan31/Dec31 boundary entitlement; compare also exact natural
    # fixed-price emergency semantics without pretending equality of coordinates.
    sf=_summary(cash_fixed,reg_fixed,emg_fixed,e); sv=_summary(cash_var,reg_var,emg_var,e)
    return {'branch':'q4_2','source':'问题二/code/run_data.npz','trajectory_frozen':True,
            'semantics':'existing Q2 plan+storage+emergency frozen; natural-day contracts mapped from plan rows; only tariff repriced',
            'stored_q2_planrow_total_yuan':stored,'fixed_price_natural_reconstruction':sf,'R0_variable_price':sv,
            'variable_minus_fixed_natural_yuan':float(sv['cash_total_yuan']-sf['cash_total_yuan']),
            'variable_vs_fixed_natural_pct':float((sv['cash_total_yuan']-sf['cash_total_yuan'])/sf['cash_total_yuan'])}


def reprice_q3(repo:Path,data:Q4Data)->dict:
    z=np.load(Path(repo)/'问题三'/'code'/'run_D.npz',allow_pickle=True)
    Bp=np.asarray(z['B'],float); Ap=np.asarray(z['A'],float); e=np.asarray(z['emergency'],float)
    Bn=_natural_from_plan(Bp); An=_natural_from_plan(Ap)
    out={}
    for tag,price in [('fixed',data.fixed_price_cal),('variable',data.price_cal)]:
        sv=settlement_vector(Bn,An,price,e)
        out[tag]=_summary(sv['total_cash'],sv['regular_fee'],sv['emergency_fee'],e)
    stored=float(np.sum(np.asarray(z['natural_regular_fee'])[31:])+np.sum(np.asarray(z['emergency_fee'])[31:]))
    # Exact fixed-price natural ledger should reconstruct the repaired Q3 run.
    diff=float(out['fixed']['cash_total_yuan']-stored)
    return {'branch':'q4_3','source':'问题三/code/run_D.npz','trajectory_frozen':True,
            'semantics':'repaired Q3 original-B/final-A/storage/emergency frozen; only tariff repriced under identical natural-day settlement',
            'stored_q3_natural_total_yuan':stored,'fixed_price_natural_reconstruction':out['fixed'],
            'fixed_reconstruction_diff_yuan':diff,'fixed_reconstruction_pass':bool(abs(diff)<=1e-5*max(1.0,abs(stored))),
            'R0_variable_price':out['variable'],
            'variable_minus_fixed_natural_yuan':float(out['variable']['cash_total_yuan']-out['fixed']['cash_total_yuan']),
            'variable_vs_fixed_natural_pct':float((out['variable']['cash_total_yuan']-out['fixed']['cash_total_yuan'])/out['fixed']['cash_total_yuan'])}


def build_r0(repo:Path,data:Q4Data)->dict:
    q2=reprice_q2(repo,data); q3=reprice_q3(repo,data)
    return {'name':'R0 Repricing Exposure','executable_strategy':False,
            'ranking_role':'diagnostic only; excluded from B0/B1/B2 algorithm ranking',
            'q4_2':q2,'q4_3':q3,
            'claim_boundary':'R0 isolates historical price exposure. It does not measure savings of any Q4 decision rule.'}
