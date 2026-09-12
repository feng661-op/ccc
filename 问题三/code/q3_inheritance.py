"""Q3 extends the frozen Q2 planner and executor; no Q2 files are modified.

At midnight call Q2's solver itself. At later events solve its remaining
48-hour approximation with common accepted contracts and piecewise fees.
Future-day purchases remain valuation recourse, never an issued contract.
The all-information-off, revision-off path retains the midnight SOC floors.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
import sys
import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from q3_data import (DT, SOC_MIN, SOC_MAX, XMAX, ETA_C, ETA_D,
                     EVENT_PLAN_START, completed_actual_kw, _selected_vintage_hour)

Q2_CODE = Path(__file__).resolve().parents[2] / '问题二' / 'code'
sys.path.insert(0, str(Q2_CODE))
import q2_core as q2

RISK_LAMBDA = 0.02
SCENARIO_COUNT = 9
TERMINAL_VALUE = q2.TERMINAL_SHORTAGE_VALUE
_CONTEXTS = {}


def released_pv_kw(data, vintage, times):
    """Exact existing interpolation on released support, without calculating
    an unused historical fallback for every supported target time."""
    mp=data.forecasts[vintage]; targets=sorted(mp)
    xs=np.r_[0.0,[(t-vintage).total_seconds()/3600 for t in targets]]
    ys=np.r_[completed_actual_kw(data,vintage,'pv') or 0.0,[mp[t] for t in targets]]
    rel=np.array([(t-vintage).total_seconds()/3600 for t in times])
    if np.any((rel<0)|(rel>24)):
        raise ValueError('interpolation outside released forecast support')
    return np.maximum(np.interp(rel,xs,ys),0.0)


def context(data):
    cached = _CONTEXTS.get(id(data))
    if cached is not None and cached[0] is data:
        return cached[1:]
    inp = q2.InputData(data.dates, data.price_plan, data.load_plan_kwh/DT,
                       data.pv_plan_kwh/DT, data.load_plan_kwh,
                       data.pv_plan_kwh, data.net_plan_kwh, [], data.plan_headers)
    fc = q2.precompute_causal_forecasts(inp)
    _CONTEXTS[id(data)] = (data, inp, fc, {})
    return inp, fc, _CONTEXTS[id(data)][3]


def forecast_bundle(data, use_official):
    inp, base, cache = context(data)
    if not use_official:
        return base
    if 'official' not in cache:
        pv = base.pv_current.copy()
        # Only the midnight release is available at the daily planning time.
        for d, day in enumerate(data.dates):
            times = [day+timedelta(minutes=10*(j+1)) for j in range(144)]
            pv[d] = released_pv_kw(data, day, times)*DT
        cache['official'] = replace(base, pv_current=pv,
                                    net_current=base.load_current-pv)
    return cache['official']


def point_48h(data, d, event_hour, use_new_vintage, disabled=()):
    _, _, cache = context(data)
    key = ('point', d, event_hour, bool(use_new_vintage), tuple(disabled))
    if key in cache:
        return cache[key].copy()
    fc = forecast_bundle(data, use_new_vintage)
    point = np.r_[0.0 if d == 0 else fc.net_current[d-1, 143],
                  fc.net_current[d], fc.net_future[d]]
    if use_new_vintage and event_hour > 0:
        vh = _selected_vintage_hour(event_hour, True, disabled)
        if vh > 0:
            day = data.dates[d]; issue = day+timedelta(hours=event_hour)
            vintage = day+timedelta(hours=vh)
            times = [day+timedelta(minutes=10*k) for k in range(289)]
            pv0 = np.r_[0.0 if d == 0 else fc.pv_current[d-1, 143],
                         fc.pv_current[d], fc.pv_future[d]]
            valid = np.array([vintage <= t <= vintage+timedelta(hours=24) for t in times])
            # Existing midnight/future history forecast outside released support.
            pvnew = released_pv_kw(data, vintage, [t for t,ok in zip(times,valid) if ok])*DT
            point[valid] += pv0[valid]-pvnew
    cache[key] = point
    return point.copy()


def source_days(day, inp, fc, quantiles, window):
    # Same ranking, tie-breaking, deduplication and warmup as Q2.
    centers = list(range(1, day-2))[-window:]
    if not centers:
        return []
    residuals = inp.net-fc.net_current
    scores = [float(np.dot(inp.price, np.maximum(residuals[j], 0))) for j in centers]
    ranked = [centers[int(i)] for i in np.argsort(scores)]
    picked = []
    for q in quantiles:
        j = ranked[int(round(q*(len(ranked)-1)))]
        if j not in picked:
            picked.append(j)
    for j in reversed(centers):
        if len(picked) >= min(len(quantiles), len(centers)):
            break
        if j not in picked:
            picked.append(j)
    return picked


def build_scenarios(data, day_idx, event_hour, horizon=None,
                    use_new_vintage=True, max_scenarios=SCENARIO_COUNT,
                    history_window=q2.SCENARIO_WINDOW_DAYS,
                    disabled_vintage_hours=()):
    from q3_opt import ScenarioBundle
    inp, _, _ = context(data)
    fc = forecast_bundle(data, use_new_vintage)
    start = event_hour*6
    H = 289-start if horizon is None else int(horizon)
    if not 1 <= H <= 289-start:
        raise ValueError('event horizon outside inherited 48-hour window')
    quantiles = q2.SCENARIO_QUANTILES if max_scenarios == 9 else tuple(np.linspace(.05, .95, max_scenarios))
    point0 = point_48h(data, day_idx, 0, use_new_vintage)
    point = point_48h(data, day_idx, event_hour, use_new_vintage, disabled_vintage_hours)
    src = source_days(day_idx, inp, fc, quantiles, history_window)
    if max_scenarios == 1:
        raw = point[None, :]; src = []
    else:
        raw = q2.build_net_scenarios(day_idx, inp, fc, quantiles, history_window)
        if event_hour > 0 and use_new_vintage:
            # Update both forecast and its historical errors at the same release
            # time; no formal-period future realization enters this calculation.
            raw = raw + (point-point0)[None, :]
            for k, h in enumerate(src):
                old = point_48h(data, h, 0, True)
                new = point_48h(data, h, event_hour, True, disabled_vintage_hours)
                raw[k] += old-new
    raw = raw[:, start:start+H]
    scenarios = np.clip(raw, -2000, 2500) if src else raw.copy()
    K = len(scenarios); weights = np.full(K, 1/K)
    times = [data.dates[day_idx]+timedelta(minutes=10*(start+t)) for t in range(H)]
    prices = np.r_[data.price_plan[-1], data.price_plan, data.price_plan][start:start+H]
    # Current contracts share one information set. Future-day recourse is only
    # the same valuation relaxation as in Q2, not a clairvoyant emitted policy.
    tree = {event_hour: {k: 0 for k in range(K)}}
    members = {(event_hour, 0): tuple(range(K))}
    return ScenarioBundle(point[start:start+H], scenarios, src, weights, times,
                          prices, tree, members, day_idx-2,
                          float(np.mean(np.abs(raw-scenarios) > 1e-12)), max_scenarios == 1)


def solve_event_lp(data, day_idx, event_hour, soc0, B, A_before,
                   lead_contract=0.0, *, use_new_vintage=True,
                   allow_revision=True, scenario_count=SCENARIO_COUNT,
                   horizon=None, down_settlement='cancel_settlement',
                   revision_anchor='original_anchor', terminal_value=TERMINAL_VALUE,
                   disabled_vintage_hours=(), risk_lambda=RISK_LAMBDA,
                   fixed_contract=None):
    from q3_opt import EventSolution
    # Resolve through q3_opt so finite-difference scenario perturbations audit
    # the exact same production solver, rather than a detached diagnostic copy.
    import q3_opt
    bundle = q3_opt.build_scenarios(data, day_idx, event_hour, horizon,
        use_new_vintage, scenario_count, disabled_vintage_hours=disabled_vintage_hours)
    K, H = bundle.scenarios.shape; start = event_hour*6
    startj = EVENT_PLAN_START[event_hour]
    if event_hour == 0 and H == 289 and fixed_contract is None:
        t0 = time.perf_counter()
        # Q2's public result does not expose matrix residuals. Observe its
        # unchanged native solve rather than inventing a zero diagnostic.
        native=q2.linprog; diagnostic={}
        def observed(c, **kwargs):
            result=native(c, **kwargs)
            if result.success:
                diagnostic['eq']=float(np.max(np.abs(kwargs['A_eq']@result.x-kwargs['b_eq'])))
                diagnostic['ub']=float(max(0,np.max(kwargs['A_ub']@result.x-kwargs['b_ub'])))
            return result
        q2.linprog=observed
        try:
            sol = q2.solve_stochastic_mpc(bundle.scenarios, data.price_plan,
                soc0, lead_contract, risk_lambda, q2.CVAR_ALPHA, terminal_value)
        finally:
            q2.linprog=native
        out = np.maximum(sol.q_current, 0); out[np.abs(out) < 1e-9] = 0
        rvals = np.maximum(q2.TERMINAL_RESERVE-sol.scenario_soc[:, -1], 0)
        answer = EventSolution(event_hour, out, sol.scenario_soc.mean(axis=0),
            sol.scenario_soc, sol.scenario_emergency, sol.objective, sol.solve_status,
            time.perf_counter()-t0, diagnostic['eq'], diagnostic['ub'], K, 1,
            data.dates[day_idx]+timedelta(days=1, minutes=10),
            bundle.slot_starts[-1]+timedelta(minutes=10), bundle.point_net,
            bundle.slot_prices, bundle.weights, bundle.clip_rate,
            float(rvals.mean()), int(np.count_nonzero(rvals > 1e-8)))
        answer.reserve_floor = np.quantile(sol.scenario_soc, .25, axis=0)
        return answer
    if event_hour > 0 and (B is None or A_before is None):
        raise ValueError('intraday event requires original and accepted contract')
    B0 = np.zeros(144) if B is None else np.asarray(B, float)
    A0 = B0.copy() if A_before is None else np.asarray(A_before, float)
    cost, bounds, eqs, rhs, ubs, ub_rhs = [], [], [], [], [], []
    def add(c=0.0, bound=(0, None)):
        i=len(cost); cost.append(c); bounds.append(bound); return i
    q = {j:add() for j in range(startj, 144)}
    if not allow_revision or fixed_contract is not None:
        fixed = A0 if fixed_contract is None else np.asarray(fixed_contract, float)
        for j, ix in q.items(): bounds[ix] = (float(fixed[j]), float(fixed[j]))
    future, x, y, e, s, r = {}, {}, {}, {}, {}, {}
    for k in range(K):
        w=bundle.weights[k]
        for t in range(H):
            global_t=start+t
            if global_t >= 145: future[k,t]=add(w*bundle.slot_prices[t])
            x[k,t]=add(bound=(0,XMAX)); y[k,t]=add(bound=(0,XMAX))
            e[k,t]=add(w*5*bundle.slot_prices[t])
        for t in range(H+1): s[k,t]=add(bound=(SOC_MIN,SOC_MAX))
        r[k]=add(w*terminal_value)
    zeta=add(risk_lambda)
    excess={k:add(risk_lambda*bundle.weights[k]/(1-q2.CVAR_ALPHA)) for k in range(K)}
    for k in range(K):
        eqs.append({s[k,0]:1}); rhs.append(soc0)
        risk={zeta:-1, excess[k]:-1}
        for t in range(H):
            eqs.append({s[k,t+1]:1,s[k,t]:-1,x[k,t]:-ETA_C,y[k,t]:1/ETA_D}); rhs.append(0)
            row={x[k,t]:1,y[k,t]:-1,e[k,t]:-1}; global_t=start+t
            b=-bundle.scenarios[k,t]
            if global_t==0: b += lead_contract
            elif global_t<=144: row[q[global_t-1]]=-1
            else: row[future[k,t]]=-1
            ubs.append(row); ub_rhs.append(b)
            if global_t<=144: risk[e[k,t]]=5*bundle.slot_prices[t]
        ubs.append({s[k,H]:-1,r[k]:-1}); ub_rhs.append(-q2.TERMINAL_RESERVE)
        ubs.append(risk); ub_rhs.append(0)
    for j, ix in q.items():
        p=float(data.price_plan[j]); b=float(B0[j]); a=float(A0[j])
        if event_hour==0:
            cost[ix]=p
            continue
        phi=add(1)
        if revision_anchor=='stepwise_revision':
            lines=((.5*p,.5*p*a),(1.5*p,-.5*p*a))
        elif revision_anchor=='original_anchor' and down_settlement=='cancel_settlement':
            lines=((.5*p,.5*p*b),(1.5*p,-.5*p*b))
        elif revision_anchor=='original_anchor' and down_settlement=='sunk_plan_plus_penalty':
            lines=((-0.5*p,1.5*p*b),(1.5*p,-.5*p*b))
        else: raise ValueError('unsupported settlement')
        for slope, intercept in lines:
            ubs.append({ix:slope,phi:-1}); ub_rhs.append(-intercept)
    N=len(cost)
    def sparse(rows):
        rr,cc,vv=[],[],[]
        for i,row in enumerate(rows):
            for j,v in row.items(): rr.append(i);cc.append(j);vv.append(v)
        return coo_matrix((vv,(rr,cc)),shape=(len(rows),N)).tocsr()
    eq=sparse(eqs); ub=sparse(ubs); t0=time.perf_counter()
    sol=linprog(cost,A_eq=eq,b_eq=rhs,A_ub=ub,b_ub=ub_rhs,bounds=bounds,
                method='highs',options={'presolve':True})
    seconds=time.perf_counter()-t0
    if not sol.success: raise RuntimeError(f'Q2-extension LP failed: {sol.message}')
    v=sol.x; out=A0.copy()
    for j,ix in q.items(): out[j]=max(0,float(v[ix]))
    ss=np.array([[v[s[k,t]] for t in range(H+1)] for k in range(K)])
    ee=np.array([[v[e[k,t]] for t in range(H)] for k in range(K)])
    rv=np.array([v[r[k]] for k in range(K)])
    answer=EventSolution(event_hour,out,ss.mean(axis=0),ss,ee,float(sol.fun),str(sol.message),
        seconds,float(np.max(np.abs(eq@v-rhs))),float(max(0,np.max(ub@v-ub_rhs))),K,1,
        data.dates[day_idx]+timedelta(days=1,minutes=10) if start+H>145 else None,
        bundle.slot_starts[-1]+timedelta(minutes=10),bundle.point_net,bundle.slot_prices,
        bundle.weights,bundle.clip_rate,float(np.dot(bundle.weights,rv)),int(np.count_nonzero(rv>1e-8)))
    answer.reserve_floor=np.quantile(ss,.25,axis=0)
    return answer
