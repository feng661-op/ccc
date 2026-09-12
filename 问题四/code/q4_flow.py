# -*- coding: utf-8 -*-
"""Single source of truth for Q4 source-flow physics.

Both event optimization and 10-minute execution call ``flow_constraint_rows``.
There is no net-balance optimizer followed by a physical projection.  The main
semantics forbid emergency-funded charging (eB=0); the alternative is exposed
only through an explicit sensitivity flag.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, vstack

from q4_data import ETA_C, ETA_D, SOC_MIN, SOC_MAX, XMAX

FLOW_VARIABLES = ('qL','qB','u','gL','gB','kappa','eL','eB','x','y','S1')
FLOW_EQUATIONS = (
    'qL+qB+u=Q',
    'gL+gB+kappa=G',
    'qL+gL+y+eL=L',
    'x=qB+gB+eB',
    'S1=S0+eta_c*x-y/eta_d',
)
FLOW_SCHEMA_VERSION = 'q4-source-flow-v1.2'


def structure_signature() -> str:
    payload = json.dumps({
        'version': FLOW_SCHEMA_VERSION,
        'variables': FLOW_VARIABLES,
        'equations': FLOW_EQUATIONS,
        'eta_c': ETA_C,
        'eta_d': ETA_D,
        'xmax': XMAX,
        'soc_min': SOC_MIN,
        'soc_max': SOC_MAX,
    }, sort_keys=True, ensure_ascii=True).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class Expr:
    coeff: Mapping[int, float]
    const: float = 0.0


def const_expr(v: float) -> Expr:
    return Expr({}, float(v))


def var_expr(i: int, coef: float = 1.0, const: float = 0.0) -> Expr:
    return Expr({int(i): float(coef)}, float(const))


@dataclass(frozen=True)
class FlowIndex:
    qL: int; qB: int; u: int; gL: int; gB: int; kappa: int
    eL: int; eB: int; x: int; y: int; S1: int

    @classmethod
    def from_sequence(cls, ids: Sequence[int]) -> 'FlowIndex':
        if len(ids) != 11:
            raise ValueError('FlowIndex requires 11 ids')
        return cls(*[int(x) for x in ids])

    def as_dict(self) -> Dict[str, int]:
        return {k: int(getattr(self, k)) for k in FLOW_VARIABLES}


@dataclass(frozen=True)
class ConstraintRow:
    coeff: Dict[int, float]
    rhs: float
    kind: str  # 'eq' or 'ub'
    name: str


def _merge(row: MutableMapping[int, float], expr: Expr, scale: float) -> float:
    for j, a in expr.coeff.items():
        row[int(j)] = row.get(int(j), 0.0) + float(scale) * float(a)
    return float(scale) * float(expr.const)


def flow_constraint_rows(
    f: FlowIndex,
    *,
    Q: Expr,
    load_kwh: float,
    pv_kwh: float,
    S0: Expr,
) -> List[ConstraintRow]:
    """Return the canonical physical equalities for one 10-minute slot.

    Bounds (nonnegative flows, x/y<=XMAX, SOC range, eB=0 for main semantics)
    are deliberately kept outside these rows so the same equations are shared
    by event and executor layers.
    """
    L = float(load_kwh); G = float(pv_kwh)
    if not (np.isfinite(L) and np.isfinite(G) and L >= -1e-9 and G >= -1e-9):
        raise ValueError('load/PV must be finite nonnegative energies')
    rows: List[ConstraintRow] = []

    # qL + qB + u = Q
    r = {f.qL:1.0, f.qB:1.0, f.u:1.0}; c = _merge(r, Q, -1.0)
    rows.append(ConstraintRow(r, -c, 'eq', 'contract_source'))
    # gL + gB + kappa = G
    rows.append(ConstraintRow({f.gL:1.0, f.gB:1.0, f.kappa:1.0}, G, 'eq', 'pv_source'))
    # qL + gL + y + eL = L
    rows.append(ConstraintRow({f.qL:1.0, f.gL:1.0, f.y:1.0, f.eL:1.0}, L, 'eq', 'load_sink'))
    # x = qB + gB + eB
    rows.append(ConstraintRow({f.x:1.0, f.qB:-1.0, f.gB:-1.0, f.eB:-1.0}, 0.0, 'eq', 'charge_sink'))
    # S1 = S0 + eta_c*x - y/eta_d
    r = {f.S1:1.0, f.x:-ETA_C, f.y:1.0/ETA_D}; c = _merge(r, S0, -1.0)
    rows.append(ConstraintRow(r, -c, 'eq', 'soc_transition'))
    return rows


def canonical_bounds(f: FlowIndex, nvars: int, *, allow_emergency_charging: bool = False,
                     base: Optional[List[Tuple[Optional[float], Optional[float]]]] = None):
    bounds = list(base) if base is not None else [(0.0, None)] * int(nvars)
    if len(bounds) != nvars:
        raise ValueError('bounds length mismatch')
    for name in ('qL','qB','u','gL','gB','kappa','eL'):
        bounds[getattr(f,name)] = (0.0, None)
    bounds[f.eB] = (0.0, None) if allow_emergency_charging else (0.0, 0.0)
    bounds[f.x] = (0.0, XMAX)
    bounds[f.y] = (0.0, XMAX)
    bounds[f.S1] = (SOC_MIN, SOC_MAX)
    return bounds


def rows_to_dense(rows: Sequence[ConstraintRow], nvars: int):
    eq = [r for r in rows if r.kind == 'eq']; ub = [r for r in rows if r.kind == 'ub']
    def mat(rr):
        A = np.zeros((len(rr), nvars), dtype=float); b = np.zeros(len(rr), dtype=float)
        for i, x in enumerate(rr):
            for j, v in x.coeff.items(): A[i, j] = v
            b[i] = x.rhs
        return A, b
    return (*mat(eq), *mat(ub))


def weighted_cvar(costs: Sequence[float], weights: Sequence[float], alpha: float) -> float:
    """Exact upper-tail CVaR for arbitrary discrete weights."""
    z = np.asarray(costs, dtype=float); w = np.asarray(weights, dtype=float)
    if z.ndim != 1 or w.shape != z.shape or z.size == 0:
        raise ValueError('cost/weight shape mismatch')
    if np.any(w < -1e-15) or not np.isclose(w.sum(), 1.0, atol=1e-10):
        raise ValueError('weights must be nonnegative and sum to one')
    if not 0.0 <= alpha < 1.0:
        raise ValueError('alpha out of range')
    order = np.argsort(z)[::-1]
    remaining = 1.0 - alpha; acc = 0.0; mass = 0.0
    for idx in order:
        take = min(float(w[idx]), remaining - mass)
        if take > 0:
            acc += take * float(z[idx]); mass += take
        if mass >= remaining - 1e-14:
            break
    return acc / remaining


@dataclass
class OneSlotFlow:
    qL: float; qB: float; u: float; gL: float; gB: float; kappa: float
    eL: float; eB: float; x: float; y: float; S0: float; S1: float
    I: float; main_objective: float; throughput_objective: float
    lex_stage1_value: float; lex_stage2_value: float; lex_stage3_value: float
    residual_max: float; status: str


def solve_one_slot(
    *, Q: float, load_kwh: float, pv_kwh: float, S0: float, price: float,
    soc_value: float = 0.0, allow_emergency_charging: bool = False,
    lex_tol_abs: float = 1e-8, lex_tol_rel: float = 1e-9,
) -> OneSlotFlow:
    """Exact current-slot feedback LP under the same canonical source-flow rows.

    Q is already contracted and therefore sunk at execution.  Level 1 minimizes
    emergency cash minus a causal continuation value of ending SOC.  Level 2
    minimizes battery throughput at the same level-1 optimum; level 3 minimizes
    PV curtailment at the same first two optima.
    """
    vals = [Q, load_kwh, pv_kwh, S0, price, soc_value]
    if not all(np.isfinite(float(x)) for x in vals):
        raise ValueError('nonfinite one-slot input')
    if Q < -1e-9 or load_kwh < -1e-9 or pv_kwh < -1e-9 or price < -1e-12:
        raise ValueError('negative physical input')
    if not SOC_MIN - 1e-7 <= S0 <= SOC_MAX + 1e-7:
        raise ValueError('S0 outside bounds')

    n = 11; f = FlowIndex.from_sequence(range(n))
    rows = flow_constraint_rows(f, Q=const_expr(max(0.0,Q)), load_kwh=max(0.0,load_kwh),
                                pv_kwh=max(0.0,pv_kwh), S0=const_expr(float(S0)))
    Aeq, beq, Aub0, bub0 = rows_to_dense(rows, n)
    bounds = canonical_bounds(f, n, allow_emergency_charging=allow_emergency_charging)

    c1 = np.zeros(n); c1[f.eL] = 5.0 * float(price); c1[f.eB] = 5.0 * float(price); c1[f.S1] = -float(soc_value)
    res1 = linprog(c1, A_eq=Aeq, b_eq=beq, bounds=bounds, method='highs')
    if not res1.success:
        raise RuntimeError(f'one-slot level1 infeasible: {res1.status} {res1.message}')
    f1 = float(c1 @ res1.x); tol1 = max(float(lex_tol_abs), float(lex_tol_rel) * max(1.0, abs(f1)))

    c2 = np.zeros(n); c2[f.x] = 1.0; c2[f.y] = 1.0
    A1 = np.atleast_2d(c1); b1 = np.asarray([f1 + tol1])
    res2 = linprog(c2, A_ub=A1, b_ub=b1, A_eq=Aeq, b_eq=beq, bounds=bounds, method='highs')
    if not res2.success:
        raise RuntimeError(f'one-slot level2 infeasible: {res2.status} {res2.message}')
    f2 = float(c2 @ res2.x); tol2 = max(float(lex_tol_abs), float(lex_tol_rel) * max(1.0, abs(f2)))

    c3 = np.zeros(n); c3[f.kappa] = 1.0
    A2 = np.vstack([c1, c2]); b2 = np.asarray([f1 + tol1, f2 + tol2])
    res3 = linprog(c3, A_ub=A2, b_ub=b2, A_eq=Aeq, b_eq=beq, bounds=bounds, method='highs')
    if not res3.success:
        raise RuntimeError(f'one-slot level3 infeasible: {res3.status} {res3.message}')
    v = res3.x
    eq_resid = float(np.max(np.abs(Aeq @ v - beq))) if len(beq) else 0.0
    return OneSlotFlow(
        qL=float(v[f.qL]), qB=float(v[f.qB]), u=float(v[f.u]),
        gL=float(v[f.gL]), gB=float(v[f.gB]), kappa=float(v[f.kappa]),
        eL=float(v[f.eL]), eB=float(v[f.eB]), x=float(v[f.x]), y=float(v[f.y]),
        S0=float(S0), S1=float(v[f.S1]), I=float(v[f.qL] + v[f.qB]),
        main_objective=float(c1 @ v), throughput_objective=float(c2 @ v),
        lex_stage1_value=f1, lex_stage2_value=f2, lex_stage3_value=float(c3 @ v),
        residual_max=eq_resid, status=str(res3.message),
    )


def manual_counterexamples() -> Dict[str, Dict[str, float]]:
    """Return deterministic sanity cases for T19-T25, using the real solver."""
    cases = {}
    # Night: paid entitlement exceeds load. It must become unused contract, not PV curtailment.
    a = solve_one_slot(Q=1000.0, load_kwh=500.0, pv_kwh=0.0, S0=SOC_MAX, price=0.5, soc_value=0.2)
    cases['night_unused_contract'] = {'unused_contract':a.u,'kappa':a.kappa,'battery_dump':0.0,'residual':a.residual_max}
    # Midday PV surplus: curtailment is PV-source curtailment; contract is not relabeled as curtailment.
    b = solve_one_slot(Q=0.0, load_kwh=300.0, pv_kwh=1200.0, S0=SOC_MAX, price=0.5, soc_value=0.2)
    cases['pv_surplus'] = {'unused_contract':b.u,'kappa':b.kappa,'residual':b.residual_max}
    # Main semantics cannot emergency-charge.
    c = solve_one_slot(Q=0.0, load_kwh=100.0, pv_kwh=0.0, S0=SOC_MIN, price=0.2, soc_value=10.0, allow_emergency_charging=False)
    cases['no_emergency_charge'] = {'eL':c.eL,'eB':c.eB,'x':c.x,'residual':c.residual_max}
    return cases
