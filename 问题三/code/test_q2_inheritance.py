"""Independent regression checks for the cross-problem policy embedding."""
from pathlib import Path
import copy
import numpy as np
import pytest
from q3_data import load_q3_inputs, settlement_components
from q3_inheritance import context, q2, point_48h
from q3_opt import build_scenarios, solve_event_lp
from q3_sim import simulate_strategy


def test_forecast_interpolation_matches_original_exactly(data):
    from datetime import timedelta
    from q3_data import official_pv_kw
    from q3_inheritance import released_pv_kw
    for day in (0,31,181,364):
        for hour in (0,6,12,18):
            issue=data.dates[day]+timedelta(hours=hour)
            times=[issue+timedelta(minutes=10*k) for k in range(145)]
            np.testing.assert_array_equal(released_pv_kw(data,issue,times),official_pv_kw(data,issue,issue,times))


def test_zero_forecast_only_never_replans_intraday(data,monkeypatch):
    import q3_sim
    original=q3_sim.solve_event_lp; calls=[]
    def observed(*args,**kwargs):
        calls.append(args[2]); return original(*args,**kwargs)
    monkeypatch.setattr(q3_sim,'solve_event_lp',observed)
    sim=q3_sim.simulate_strategy(data,end_day=1,use_new_vintage=True,
        allow_revision=False,disabled_vintage_hours=(6,12,18))
    assert calls==[0]
    np.testing.assert_array_equal(sim.A_stage[0],np.repeat(sim.B[0,None,:],4,axis=0))


@pytest.fixture(scope='module')
def data():
    return load_q3_inputs(Path(__file__).resolve().parents[2])


@pytest.mark.parametrize('day', [0, 3, 31, 181, 364])
def test_disabled_forecast_and_scenarios_are_q2(data, day):
    inp, fc, _ = context(data)
    b = build_scenarios(data, day, 0, use_new_vintage=False)
    np.testing.assert_array_equal(b.scenarios, q2.build_net_scenarios(day, inp, fc))
    np.testing.assert_array_equal(b.point_net, np.r_[0 if day == 0 else fc.net_current[day-1,143],fc.net_current[day],fc.net_future[day]])
    assert b.scenarios.shape[1] == 289


def test_native_q2_solver_residuals_are_checked(data, monkeypatch):
    native = q2.linprog; observed = []
    def audited(c, **kw):
        result = native(c, **kw)
        observed.append((float(np.max(np.abs(kw['A_eq']@result.x-kw['b_eq']))),
                         float(np.max(kw['A_ub']@result.x-kw['b_ub']))))
        return result
    monkeypatch.setattr(q2, 'linprog', audited)
    for d in (31, 181, 364):
        sol = solve_event_lp(data,d,0,6000,None,None)
        assert sol.scenario_count == 9
    assert len(observed) == 3
    assert max(max(x) for x in observed) < 1e-7


@pytest.mark.parametrize('hour', [6,12,18])
def test_no_revision_candidate_remains_feasible_and_no_worse_in_model(data, hour):
    daily = solve_event_lp(data,31,0,6000,None,None)
    b = daily.current_contract
    frozen = solve_event_lp(data,31,hour,6000,b,b,allow_revision=False)
    flexible = solve_event_lp(data,31,hour,6000,b,b,allow_revision=True)
    np.testing.assert_allclose(frozen.current_contract,b,atol=1e-7,rtol=0)
    assert flexible.objective <= frozen.objective+1e-6
    assert flexible.terminal_time == data.dates[31]+__import__('datetime').timedelta(days=2,minutes=10)


def test_future_day_valuation_never_becomes_a_current_contract(data):
    sol = solve_event_lp(data,364,0,6000,None,None)
    assert sol.current_contract.shape == (144,)
    assert sol.scenario_soc.shape == (9,290)
    assert np.all(np.isfinite(sol.current_contract))


@pytest.mark.parametrize('hour', [0,6,12,18])
def test_event_contract_never_reads_unrealized_data_or_unreleased_forecasts(data, hour):
    from datetime import timedelta
    d=80; day=data.dates[d]; altered=copy.deepcopy(data); i=hour*6
    # Change all representations of all unavailable actual data, including the
    # otherwise easy-to-miss previous plan row's midnight lead slot.
    altered.load_cal_kwh[d,i:]+=1000; altered.net_cal_kwh[d,i:]+=1000
    altered.load_cal_kwh[d+1:]+=1000; altered.net_cal_kwh[d+1:]+=1000
    if i==0:
        altered.load_plan_kwh[d-1,143]+=1000; altered.net_plan_kwh[d-1,143]+=1000
    altered.load_plan_kwh[d,max(0,i-1):]+=1000; altered.net_plan_kwh[d,max(0,i-1):]+=1000
    altered.load_plan_kwh[d+1:]+=1000; altered.net_plan_kwh[d+1:]+=1000
    for issue in altered.forecasts:
        if issue > day+timedelta(hours=hour):
            altered.forecasts[issue]={t:v+5000 for t,v in altered.forecasts[issue].items()}
    b=np.full(144,500.0)
    a=solve_event_lp(data,d,hour,6000,None if hour==0 else b,None if hour==0 else b)
    p=solve_event_lp(altered,d,hour,6000,None if hour==0 else b,None if hour==0 else b)
    np.testing.assert_allclose(a.current_contract,p.current_contract,rtol=0,atol=1e-7)
    np.testing.assert_allclose(a.reserve_floor,p.reserve_floor,rtol=0,atol=1e-7)


def test_equal_contract_has_no_adjustment_fee_under_both_settlements():
    b=np.array([0.,17.,900.]);price=np.array([.2,.6,1.3])
    for mode in ('cancel_settlement','sunk_plan_plus_penalty'):
        c=settlement_components(b,b,price,mode)
        np.testing.assert_array_equal(c['F_regular'],b*price)
        assert np.sum(c['F_cancel']+c['F_add'])==0
