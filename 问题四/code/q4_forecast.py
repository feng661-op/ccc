# -*- coding: utf-8 -*-
"""Strictly causal load/PV forecasts inherited from Q2/Q3 information rights."""
from __future__ import annotations
from datetime import datetime, timedelta
from functools import lru_cache
from typing import List, Sequence, Tuple
import numpy as np

from q4_data import Q4Data, DT, actual_at


def _masked_previous_plan_profile(actual_plan: np.ndarray, plan_day: int) -> np.ndarray | None:
    """Exact Q2 visibility: previous plan row 0..142 seen; col143 is still future at 00:00."""
    if plan_day <= 0:
        return None
    p = np.asarray(actual_plan[plan_day - 1], float).copy()
    if plan_day >= 2:
        lo=max(0, plan_day-15); hi=plan_day-1
        hist=np.asarray(actual_plan[lo:hi,143], float)
        fill=float(np.median(hist)) if hist.size else float(p[142])
    else:
        fill=float(p[142])
    p[143]=fill
    return p


def q2_profile_forecast(data: Q4Data, plan_day: int, target_plan_day: int, field: str) -> np.ndarray:
    """Port of the frozen Q2 causal profile forecaster; permissions are identical."""
    if field not in ('load','pv'):
        raise ValueError(field)
    actual=data.load_plan_kwh if field=='load' else data.pv_plan_kwh
    n_days=actual.shape[0]; full_end=int(plan_day)-2
    if not 0 <= int(plan_day) < n_days:
        if field=='load': return np.full(144,5000.0*DT,float)
        return np.zeros(144,float)
    target_date=data.dates[0]+timedelta(days=int(target_plan_day))
    prev=_masked_previous_plan_profile(actual,int(plan_day))
    candidates=[]; weights=[]
    is_current=(int(target_plan_day)==int(plan_day))
    if prev is not None:
        candidates.append(prev)
        weights.append(0.35 if (field=='load' and is_current) else 0.30 if (field=='pv' and is_current) else 0.20)
    lag7=int(target_plan_day)-7
    if 0 <= lag7 <= full_end and lag7 < n_days:
        candidates.append(actual[lag7]); weights.append(0.35 if field=='load' else 0.25)
    if full_end >= 0:
        full_idx=list(range(0,min(full_end,n_days-1)+1))
        same=[j for j in full_idx if data.dates[j].weekday()==target_date.weekday()][-4:]
        if same:
            candidates.append(np.mean(actual[same],axis=0)); weights.append(0.20)
        recent=full_idx[-7:]
        if recent:
            candidates.append(np.median(actual[recent],axis=0)); weights.append(0.10 if field=='load' else 0.25)
    if not candidates:
        return np.full(144,5000.0*DT,float) if field=='load' else np.zeros(144,float)
    w=np.asarray(weights,float); w/=w.sum()
    pred=np.zeros(144,float)
    for wi,ci in zip(w,candidates): pred += wi*np.asarray(ci,float)
    return np.maximum(pred,0.0)


def historical_forecast_kwh(data: Q4Data, issue: datetime, times: Sequence[datetime], field: str) -> np.ndarray:
    """Q2-exact causal forecast mapped from plan coordinates to absolute target times.

    A 06/12/18 Q4-3 event may update PV via Attachment 3, but its historical load
    component is still the forecast that was legal at that day's 00:00.  This
    function therefore never learns from same-day actual load/PV.
    """
    if field not in ('load','pv'): raise ValueError(field)
    issue_day=(issue.date()-data.dates[0].date()).days
    profiles={}
    def prof(target_plan_day:int):
        key=int(target_plan_day)
        if key not in profiles:
            profiles[key]=q2_profile_forecast(data,issue_day,key,field)
        return profiles[key]
    out=np.zeros(len(times),float)
    for k,t in enumerate(times):
        td=(t.date()-data.dates[0].date()).days
        minute=t.hour*60+t.minute
        if minute==0:
            # Natural 00:00-00:10 is previous plan row's final slot.  If that row
            # was issued yesterday, reproduce the forecast that was legal then.
            pd=td-1
            if pd < 0:
                out[k]=5000.0*DT if field=='load' else 0.0
            elif pd < issue_day:
                out[k]=q2_profile_forecast(data,pd,pd,field)[143]
            else:
                out[k]=prof(pd)[143]
        else:
            pd=td; j=minute//10-1
            out[k]=prof(pd)[j]
    return np.maximum(out,0.0)


def _official_pv_kw(data: Q4Data, issue: datetime, times: Sequence[datetime], vintage: datetime) -> np.ndarray:
    fallback = historical_forecast_kwh(data, issue, times, 'pv') / DT
    mp = data.forecasts.get(vintage)
    if not mp:
        return fallback
    ts = sorted(mp)
    xs = np.asarray([(x-vintage).total_seconds()/3600.0 for x in ts], dtype=float)
    ys = np.asarray([mp[x] for x in ts], dtype=float)
    # Contract decisions must not read the still-unfinished current 10-minute
    # actual PV.  Anchor interpolation with the causal historical forecast; the
    # current actual measurement is reserved for the lower execution layer.
    anchor_kw = float(fallback[0]) if len(fallback) else 0.0
    xs = np.r_[0.0, xs]; ys = np.r_[anchor_kw, ys]
    out = np.zeros(len(times), dtype=float)
    for k, t in enumerate(times):
        rel = (t-vintage).total_seconds()/3600.0
        out[k] = np.interp(rel, xs, ys) if 0.0 <= rel <= 24.0 else fallback[k]
    return np.maximum(out, 0.0)


def event_forecast(
    data: Q4Data, day_idx: int, event_hour: int, horizon: int, branch: str,
    *, use_new_vintage: bool = True, disabled_vintage_hours: Sequence[int] = (),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[datetime], str]:
    """Return causal (load,pv,net) kWh forecasts from an event time.

    Q4-2 always uses historical PV regardless of event/vintage flags. Q4-3 uses
    only a vintage issued no later than the event; disabling a vintage falls back
    to the latest prior legal issue.
    """
    day = data.dates[int(day_idx)]
    issue = datetime(day.year, day.month, day.day) + timedelta(hours=int(event_hour))
    times = [issue + timedelta(minutes=10*k) for k in range(int(horizon))]
    load = historical_forecast_kwh(data, issue, times, 'load')
    if branch == 'q4_2':
        pv = historical_forecast_kwh(data, issue, times, 'pv')
        version = 'Q2_HISTORICAL_PV'
    elif branch == 'q4_3':
        disabled = set(int(x) for x in disabled_vintage_hours)
        legal = [h for h in (0,6,12,18) if h <= int(event_hour) and h not in disabled]
        vh = max(legal) if (use_new_vintage and legal) else 0
        vintage = datetime(day.year, day.month, day.day) + timedelta(hours=vh)
        pv = _official_pv_kw(data, issue, times, vintage) * DT
        version = f'ATT3_VINTAGE_{vh:02d}'
    else:
        raise ValueError(branch)
    return load, pv, load-pv, times, version


def actual_horizon(data: Q4Data, issue: datetime, horizon: int):
    load = np.zeros(horizon); pv = np.zeros(horizon); price = np.zeros(horizon)
    for k in range(horizon):
        t = issue + timedelta(minutes=10*k)
        lv = actual_at(data, t, 'load'); gv = actual_at(data, t, 'pv'); cv = actual_at(data, t, 'price')
        if lv is None or gv is None or cv is None:
            return None
        load[k] = lv; pv[k] = gv; price[k] = cv
    return load, pv, load-pv, price


def previous_completed_residual(data: Q4Data, now: datetime, branch: str) -> float | None:
    """One-step feedback signal based only on the previous completed 10-min slot."""
    prev = now - timedelta(minutes=10)
    a = actual_at(data, prev, 'net')
    if a is None: return None
    d = (now.date() - data.dates[0].date()).days
    issue = prev
    p = historical_forecast_kwh(data, issue, [prev], 'load')[0] - historical_forecast_kwh(data, issue, [prev], 'pv')[0]
    return float(a-p)
