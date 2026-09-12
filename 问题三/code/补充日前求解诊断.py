"""Audit native Q2 midnight matrix residuals without changing any trajectories."""
from pathlib import Path
import csv
import json
import numpy as np
from q3_data import load_q3_inputs
from q3_opt import solve_event_lp

HERE=Path(__file__).resolve().parent
data=load_q3_inputs(HERE.parents[1]); records=[]
events=list(csv.DictReader((HERE/'event_audit.csv').open(encoding='utf-8-sig')))
for name in ('A','B','C','D'):
    z=np.load(HERE/f'run_{name}.npz');eq=[];ub=[];err=[]
    for d in range(365):
        sol=solve_event_lp(data,d,0,float(z['soc00'][d]),None,None,
             lead_contract=0 if d==0 else float(z['A'][d-1,143]),
             use_new_vintage=name in ('B','D'),allow_revision=name in ('C','D'))
        eq.append(sol.max_eq_residual);ub.append(sol.max_ub_violation)
        err.append(float(np.max(np.abs(sol.current_contract-z['B'][d]))))
        if name=='D':
            row=events[d*4];assert row['event_hour']=='0'
            row['max_eq_residual']=sol.max_eq_residual;row['max_ub_violation']=sol.max_ub_violation
    row={'strategy':name,'midnight_solves':365,'max_contract_reproduction_error':max(err),
         'max_eq_residual':max(eq),'max_ub_violation':max(ub)}
    assert max(err)<1e-6 and max(eq)<1e-7 and max(ub)<1e-7,row
    mpath=HERE/f'metric_{name}.json';m=json.loads(mpath.read_text(encoding='utf-8'))
    m['solver_max_eq_residual']=max(m['solver_max_eq_residual'],max(eq))
    m['solver_max_ub_violation']=max(m['solver_max_ub_violation'],max(ub))
    mpath.write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
    records.append(row);print(row,flush=True)
with (HERE/'event_audit.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(events[0]));writer.writeheader();writer.writerows(events)
(HERE/'cross_problem_evidence'/'midnight_solver_validation.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
