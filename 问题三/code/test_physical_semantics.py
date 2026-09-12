"""Regression tests for contract/physical separation and current-measurement control."""
from pathlib import Path
import copy
import numpy as np
import pytest
from q3_data import load_q3_inputs, ETA_C, ETA_D, SOC_MIN, SOC_MAX, XMAX
from q3_opt import solve_dispatch_mpc
from q3_physical import physical_flows, project_current_action
from q3_sim import simulate_strategy


def test_reported_night_counterexample_is_rejected_then_repaired():
    with pytest.raises(ValueError, match='battery output'):
        physical_flows(1610.75, 523.55, 0, 0, 833.33)
    s, x, y = project_current_action(6000, 1610.75, 523.55, 0, 833.33)
    f = physical_flows(1610.75, 523.55, 0, x, y)
    assert y == 0 and x == 0 and s == 6000
    assert f.pv_curtail == 0 and f.grid_import == pytest.approx(523.55)
    assert f.unused_contract == pytest.approx(1087.20)
    assert f.emergency == 0 and f.battery_dump == 0


def test_paid_contract_is_not_forced_import():
    f = physical_flows(100, 100, 50, 0, 0)
    assert (f.grid_import, f.unused_contract, f.pv_curtail) == (50, 50, 0)
    # The accounting function never changes the original paid contract.
    assert f.grid_import + f.unused_contract == 100


def test_true_pv_curtailment():
    f = physical_flows(100, 20, 150, 30, 0)
    assert f.pv_curtail == 100 and f.unused_contract == 100
    assert f.supply_surplus == 200 and f.grid_import == 0


def test_no_emergency_funded_charge():
    s, x, y = project_current_action(5000, 100, 500, 500, 0)
    f = physical_flows(100, 500, 0, x, y)
    assert x == 0 and f.emergency == 400 and s == 5000


@pytest.mark.parametrize('soc', [1200, 6000, 10800])
def test_randomized_physical_invariants(soc):
    rng = np.random.default_rng(20260911)
    for _ in range(1200):
        a, l, pv = rng.uniform(0, 2500, 3)
        xc, yc = rng.uniform(0, 1800, 2)
        s1, x, y = project_current_action(soc, a, l-pv, xc, yc)
        f = physical_flows(a, l, pv, x, y)
        assert SOC_MIN-1e-6 <= s1 <= SOC_MAX+1e-6
        assert min(x, y) < 1e-7 and max(x, y) <= XMAX+1e-6
        assert 0 <= f.pv_curtail <= pv+1e-6
        assert abs(f.grid_import + pv-f.pv_curtail + y+f.emergency-l-x) < 1e-6
        assert abs(f.grid_import+f.unused_contract-a) < 1e-6
        assert abs(s1-soc-ETA_C*x+y/ETA_D) < 1e-6
        assert f.battery_dump == 0
        assert not (y>1e-6 and f.supply_surplus>1e-6)
        assert not (x>1e-6 and f.emergency>1e-6)


def test_mpc_current_balance_limits():
    s, x, y, e, r = solve_dispatch_mpc(6000, [1610.75, 0], [523.55, 1000], [1, 1], current_balance_limits=True)
    assert y < 1e-7
    assert x <= 1087.2+1e-6
    f = physical_flows(1610.75, 523.55, 0, x, y)
    assert f.battery_dump == 0 and f.pv_curtail == 0


def test_future_observations_do_not_change_earlier_decisions():
    root = Path(__file__).resolve().parents[2]
    data = load_q3_inputs(root)
    pert = copy.deepcopy(data)
    # At current slot 72 (12:00), only 12:10 and later realized observations change.
    pert.load_cal_kwh[0,73:] += 177
    pert.net_cal_kwh[0,73:] += 177
    pert.load_plan_kwh[0,72:143] += 177
    pert.net_plan_kwh[0,72:143] += 177
    a = simulate_strategy(data, end_day=1)
    b = simulate_strategy(pert, end_day=1)
    np.testing.assert_allclose(a.B[0], b.B[0], atol=1e-7, rtol=0)
    np.testing.assert_allclose(a.A_stage[0,:3], b.A_stage[0,:3], atol=1e-7, rtol=0)
    np.testing.assert_allclose(a.charge[0,:73], b.charge[0,:73], atol=1e-7, rtol=0)
    np.testing.assert_allclose(a.discharge[0,:73], b.discharge[0,:73], atol=1e-7, rtol=0)
    assert np.max(a.battery_dump) == 0
    assert a.execution_mode == 'q2_reserve_feedback'


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -10])
def test_bad_physical_inputs_fail_closed(bad):
    with pytest.raises(ValueError):
        physical_flows(bad, 100, 0, 0, 0)
