# -*- coding: utf-8 -*-
from pathlib import Path
from dataclasses import asdict
import json, numpy as np, pandas as pd
from q4_data import load_q4_inputs
from q4_price import PriceForecaster, PriceConfig
from q4_sim import SimConfig, SimState, simulate_range
from q4_metrics import day_metrics
from q4_oracle import full_information_cash_lower_bound, audit_oracle_claim
ROOT=Path(__file__).resolve().parents[2]; CODE=Path(__file__).resolve().parent
data=load_q4_inputs(ROOT); freeze=json.loads((CODE/'january_frozen_selection.json').read_text(encoding='utf-8'))
out={}
for br in ('q4_2','q4_3'):
 f=freeze[br]; c=f['config']; pc=PriceConfig(**c['price_config'])
 st=SimState(float(f['state_end']['soc']),np.asarray(f['state_end']['prev_B'],float),np.asarray(f['state_end']['prev_A'],float))
 # Price Oracle must relax only future-price information.  Preserve the exact
 # January-frozen causal model level and every other configuration knob;
 # otherwise a B1 formal policy compared with a B0 price-oracle would confound
 # information value with model-class changes.
 po_args=dict(c); po_args['price_config']=pc; po_args['disabled_vintage_hours']=tuple(po_args.get('disabled_vintage_hours') or ())
 po_args['price_oracle']=True; po_args['decision_fixed_price']=False; po_args['fixed_price']=False; po_args['label']='E14_PRICE_ORACLE'
 po_cfg=SimConfig(**po_args)
 po=simulate_range(data,PriceForecaster(data),po_cfg,31,365,st)
 pdir=CODE/br; po.days.to_csv(pdir/'price_oracle_daily.csv',index=False,encoding='utf-8-sig')
 po_metrics=day_metrics(po.days)
 main_days=pd.read_csv(pdir/'daily_ledger.csv'); main=day_metrics(main_days)
 final_soc=float(main_days.S24_kwh.iloc[-1])
 fi=full_information_cash_lower_bound(data,initial_soc=float(f['state_end']['soc']),final_soc_target=final_soc,
                                     locked_lead_B=float(f['state_end']['prev_B'][143]),locked_lead_A=float(f['state_end']['prev_A'][143]))
 audit=audit_oracle_claim(po_metrics['cash_total_yuan'],fi,main['cash_total_yuan'])
 out[br]={'causal':main,'price_oracle':po_metrics,'full_information':asdict(fi),'audit':audit,
          'price_oracle_boundary':'Actual Attachment-4 prices replace causal price forecasts on every scored/available slot. The Dec-31 contractual +1 tail lies outside Attachment 4 and outside scored execution, so that one unobserved terminal slot retains the causal point forecast rather than fabricated 2026 information.'}
 print(br,json.dumps({'causal':main['cash_total_yuan'],'PO':po_metrics['cash_total_yuan'],'FI':fi.strict_cash_lower_bound_yuan,'FI_ok':audit['strict_lower_bound_check_pass']},ensure_ascii=False),flush=True)
(CODE/'oracle_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
