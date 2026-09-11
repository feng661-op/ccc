"""Physical flow accounting, separate from paid contract quantities.

Contract quantity is a paid entitlement, not forced physical import. PV is
used first; rejecting unused entitlement does not reduce its contractual fee.
Current-slot measurement experiments assume an ideal piecewise-constant
within-slot signal. They do not turn an end-of-slot average into an observed
start-of-slot measurement without that explicit approximation.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from q3_data import ETA_C, ETA_D, SOC_MIN, SOC_MAX, XMAX


@dataclass(frozen=True)
class PhysicalFlows:
    grid_import: float
    unused_contract: float
    pv_curtail: float
    emergency: float
    supply_surplus: float
    battery_dump: float = 0.0  # Diagnostic only; forbidden for the main policy.


def physical_flows(contract: float, load: float, pv: float,
                   charge: float, discharge: float, *,
                   allow_battery_dump: bool = False) -> PhysicalFlows:
    """Reconcile physical energy; a positive battery dump raises by default."""
    vals = (contract, load, pv, charge, discharge)
    if not all(math.isfinite(float(v)) and float(v) >= -1e-7 for v in vals):
        raise ValueError('physical inputs must be finite and nonnegative')
    a, load, pv, x, y = (max(0.0, float(v)) for v in vals)
    demand = load + x - y
    dump = max(0.0, -demand)
    if dump > 1e-6 and not allow_battery_dump:
        raise ValueError(f'battery output exceeds all useful demand: {dump} kWh')
    useful = max(0.0, demand)
    pv_used = min(pv, useful)
    remaining = max(0.0, useful - pv_used)
    imp = min(a, remaining)
    unused = max(0.0, a - imp)
    curt = max(0.0, pv - pv_used)
    emerg = max(0.0, remaining - imp)
    residual = imp + pv - curt + y + emerg - load - x - dump
    if abs(residual) > 1e-6:
        raise AssertionError(f'physical balance residual {residual}')
    return PhysicalFlows(imp, unused, curt, emerg, unused + curt + dump, dump)


def project_current_action(soc: float, contract: float, net_measured: float,
                           charge_command: float, discharge_command: float):
    """No wasted discharge and no emergency-funded charging in the main policy.

    A current-measurement safety projection, not a claim that historical
    ten-minute averages resolve sub-slot dynamics. It never changes B or A.
    """
    vals = (soc, contract, net_measured, charge_command, discharge_command)
    if not all(math.isfinite(float(v)) for v in vals):
        raise ValueError('nonfinite control input')
    if not SOC_MIN - 1e-6 <= soc <= SOC_MAX + 1e-6:
        raise ValueError('initial SOC outside operating bounds')
    if contract < -1e-7 or min(charge_command, discharge_command) < -1e-7:
        raise ValueError('negative contract or action')
    delta = ETA_C * max(0.0, charge_command) - max(0.0, discharge_command) / ETA_D
    xc = max(0.0, delta / ETA_C)
    yc = max(0.0, -delta * ETA_D)
    surplus = max(0.0, contract - net_measured)
    deficit = max(0.0, net_measured - contract)
    x = min(xc, XMAX, surplus, max(0.0, (SOC_MAX - soc) / ETA_C))
    y = min(yc, XMAX, deficit, max(0.0, (soc - SOC_MIN) * ETA_D))
    return float(soc + ETA_C * x - y / ETA_D), float(x), float(y)
