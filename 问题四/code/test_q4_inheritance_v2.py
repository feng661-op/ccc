"""Fast correctness tests, NOT robustness/parameter sweeps."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from datetime import timedelta
from pathlib import Path
import numpy as np
import pytest
from q4_data import load_q4_inputs, _plan_to_calendar
from q4_sim import SimConfig, SimState, simulate_range
from q4_inheritance import make_bundle, solve_event, solve_intraday, execute_inherited, price_point
from q4_sim_v2 import regular_fee
from q4_select import promotion_gate

@pytest.fixture(scope='module')
def data():return load_q4_inputs(Path(__file__).resolve().parents[2])

@pytest.mark.parametrize('gain,tail,ml,tl,want',[(2,0,1,-1,True),(-.5,3,-10,1,True),(-6,20,-10,5,False),(-1.01,20,-10,5,False),(1,0,0,0,False),(0,2,-1,0,False),(0,2,-1,1,True)])
def test_uniform_selection(gain,tail,ml,tl,want):
    d={'mean_cash_improvement_pct':gain,'cvar95_improvement_pct':tail,'paired':{'mean_cash_gain_80pct_interval':[ml,ml+10],'cvar95_gain_80pct_interval':[tl,tl+10]}}
    assert promotion_gate(d)['promote'] is want

@pytest.mark.parametrize('level',['B0','B1'])
def test_future_cannot_change_decision(data,level):
    cfg=SimConfig('q4_3',level=level);D=20;h=6;a=deepcopy(data)
    # Alter every future observation after 06:10, preserving revealed history.
    for field in ('load_plan_kwh','pv_plan_kwh','price_plan'):
        v=getattr(a,field);v[D,36:]+=100 if field!='price_plan' else 2;v[D+1:]+=50 if field!='price_plan' else 1
    a.net_plan_kwh=a.load_plan_kwh-a.pv_plan_kwh
    for field in ('load','pv','net','price'):
        f=field+'_plan'+('' if field=='price' else '_kwh');g=field+'_cal'+('' if field=='price' else '_kwh');setattr(a,g,_plan_to_calendar(getattr(a,f)))
    # Unreleased PV forecast vintages are also changed.
    issue=data.dates[D]+timedelta(hours=h)
    assert any(k>issue for k in a.forecasts)
    a.forecasts={k:({target:value+777 for target,value in v.items()} if k>issue else v) for k,v in a.forecasts.items()}
    b,p=make_bundle(data,D,h,cfg);bb,pp=make_bundle(a,D,h,cfg)
    np.testing.assert_allclose(b.scenarios,bb.scenarios,atol=0,rtol=0);np.testing.assert_allclose(p,pp,atol=0,rtol=0)
    B=np.full(144,700.)
    x=solve_event(data,D,h,6000.,B,B,700.,cfg);y=solve_event(a,D,h,6000.,B,B,700.,cfg)
    np.testing.assert_allclose(x.contract_A,y.contract_A,atol=1e-7,rtol=0)
    np.testing.assert_allclose(x.reserve_floor,y.reserve_floor,atol=1e-7,rtol=0)

@pytest.mark.parametrize('level',['B0','B1'])
def test_revision_superset_and_prefix(data,level):
    cfg=SimConfig('q4_3',level=level);B=np.full(144,700.);b,p=make_bundle(data,20,6,cfg)
    free=solve_intraday(b,p,6500.,B,B,700.,6,cfg)
    locked=solve_intraday(b,p,6500.,B,B,700.,6,cfg,fixed_contract=B)
    assert free.objective<=locked.objective+1e-5
    np.testing.assert_array_equal(free.contract_A[:35],B[:35]);np.testing.assert_array_equal(locked.contract_A,B)


def test_disable_new_rights_is_q2(data):
    B=np.full(144,600.);s=SimState(6000.,B.copy(),B.copy())
    a=simulate_range(data,None,SimConfig('q4_2'),20,21,deepcopy(s))
    b=simulate_range(data,None,SimConfig('q4_3',use_pv_updates=False,allow_contract_revision=False),20,21,deepcopy(s))
    for x in ['B_kwh','A_kwh','S1_kwh','x','y','e_kwh','cash_fee_yuan']:np.testing.assert_allclose(a.slots[x],b.slots[x],atol=1e-8,rtol=0)
    assert len(a.events)==len(b.events)==1


def test_q42_never_uses_attachment3(data):
    cfg=SimConfig('q4_2');d=deepcopy(data);d.forecasts={}
    a,pa=make_bundle(data,20,0,cfg);b,pb=make_bundle(d,20,0,cfg)
    np.testing.assert_array_equal(a.scenarios,b.scenarios);np.testing.assert_array_equal(pa,pb)

@pytest.mark.parametrize('Q,L,G,S,target',[(1000,100,0,10800,6000),(0,1000,0,1200,6000),(800,900,300,6000,6000),(0,100,2000,6000,6000)])
def test_actual_source_flow(Q,L,G,S,target):
    a=execute_inherited(Q,L,G,S,target)
    assert abs(a.qL+a.qB+a.u-Q)<1e-8 and abs(a.gL+a.gB+a.kappa-G)<1e-8
    assert abs(a.qL+a.gL+a.y+a.eL-L)<1e-8 and abs(a.x-a.qB-a.gB)<1e-8
    assert a.eB==0 and a.x*a.y<1e-8 and 1200-1e-8<=a.S1<=10800+1e-8
    if G==0:assert a.kappa==0


def test_settlement_original_anchor():
    np.testing.assert_allclose(regular_fee(100,np.array([80,100,120]),1),[90,100,130])
    np.testing.assert_allclose(regular_fee(100,np.array([80,100,120]),1,.2),[82,100,106])


def test_event_clock_nonconvex_milp_and_locked_objective():
    # One-variable concave-kink fee: fixed-price LP convexification is wrong.
    # A true binary solution is checked against the same model with its
    # optimized contract locked; this is not a brute-force global proof.
    cfg=SimConfig('q4_3',settlement_clock='adjustment_time',terminal_reserve=1200.)
    net=np.array([[0.,500.]]);bundle=SimpleNamespace(scenarios=net,weights=np.array([1.]),source_days=[])
    # Actual event hours must map to a supported boundary, so use 18h and a
    # 37-slot bundle; all but the last useful import are zero-price/zero-load.
    net=np.zeros((1,37));net[0,-1]=500.;bundle.scenarios=net
    prices=np.full_like(net,.1);prices[0,-1]=1.;B=np.zeros(144);B[-1]=100.
    out=solve_intraday(bundle,prices,1200.,B,B,0.,18,cfg,event_price=.2)
    assert out.binaries>=1 and out.mip_gap<=1e-7 and out.max_eq_residual<1e-6
    # Independently lock to the optimized contract, which removes all
    # decision ambiguity and must preserve the mathematical objective.
    fixed=solve_intraday(bundle,prices,1200.,B,B,0.,18,cfg,event_price=.2,fixed_contract=out.contract_A)
    assert abs(fixed.objective-out.objective)<1e-5


@pytest.mark.parametrize('event_price',[.2,.7])
def test_exact_semantic_aggregation(event_price):
    from q4_semantics_fast import solve_intraday as fast
    K=2;H=37;net=np.zeros((K,H));net[:,-1]=[450.,600.]
    bundle=SimpleNamespace(scenarios=net,weights=np.array([.5,.5]),source_days=[1,2])
    prices=np.full((K,H),.1);prices[:,-1]=[.8,1.2]
    B=np.zeros(144);B[-1]=100.
    cfg=SimConfig('q4_3',settlement_clock='adjustment_time',terminal_reserve=1200.)
    original=solve_intraday(bundle,prices,1200.,B,B,0.,18,cfg,event_price=event_price)
    reduced=fast(bundle,prices,1200.,B,B,0.,18,cfg,event_price=event_price)
    assert abs(original.objective-reduced.objective)<1e-5
    assert reduced.max_eq_residual<1e-7 and reduced.mip_gap<=1e-7
    for q in [0.,20.,100.,200.,500.]:
        actual=np.dot(bundle.weights,regular_fee(100.,q,prices[:,-1],event_price))
        expected=regular_fee(100.,q,np.dot(bundle.weights,prices[:,-1]),event_price)
        assert abs(actual-expected)<1e-10


def test_q1_shared_physics(data):
    import ast
    from q4_data import ETA_C,ETA_D,SOC_MIN,SOC_MAX,XMAX
    src=Path(__file__).resolve().parents[2]/'问题一/code/问题1_模型求解.py'
    tree=ast.parse(src.read_text(encoding='utf-8-sig'));v={}
    for node in tree.body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            name=node.targets[0].id
            if name in ('CAP','PMAX','SOC0','SOC_MIN','SOC_MAX','ETA_C','ETA_D'):
                v[name]=eval(compile(ast.Expression(node.value),str(src),'eval'),{'np':np})
    assert v['SOC_MIN']==SOC_MIN and v['SOC_MAX']==SOC_MAX
    assert v['ETA_C']==ETA_C and v['ETA_D']==ETA_D and np.isclose(v['PMAX']/6,XMAX,atol=1e-10,rtol=0) and v['SOC0']==6000
