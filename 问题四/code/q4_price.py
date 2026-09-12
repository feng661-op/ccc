# -*- coding: utf-8 -*-
"""Causal real-time price forecasting and January walk-forward model freeze."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from functools import lru_cache
import json
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from q4_data import Q4Data, DT, actual_at
from q4_forecast import event_forecast, q2_profile_forecast

PRICE_MODELS = ('lag1','lag7','lag7_R','lag7_dN','full_ridge')
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0)
BLOCK_SCHEMES = ('global','6block')


@dataclass(frozen=True)
class PriceConfig:
    model: str = 'full_ridge'
    block_scheme: str = 'global'
    ridge_alpha: float = 1.0
    history_days: int = 42
    min_samples: int = 48

    @property
    def version(self) -> str:
        return f'{self.model}|{self.block_scheme}|a={self.ridge_alpha:g}|H={self.history_days}'


def _ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    X = np.asarray(X,float); y = np.asarray(y,float)
    if X.ndim != 2 or y.ndim != 1 or len(y) != len(X): raise ValueError('ridge shape')
    xa = np.c_[np.ones(len(X)), X]
    pen = np.eye(xa.shape[1]); pen[0,0] = 0.0
    return np.linalg.solve(xa.T@xa + float(alpha)*pen, xa.T@y)


def _pred(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    X=np.asarray(X,float)
    if X.ndim==1: X=X[None,:]
    return np.c_[np.ones(len(X)),X]@np.asarray(beta,float)


def _cvar95(x: Sequence[float]) -> float:
    z=np.sort(np.asarray(x,float))
    if not len(z): return float('nan')
    a=.95; n=len(z); q=a*n; k=int(np.floor(q)); frac=q-k
    # exact empirical CVaR with equal masses, including fractional quantile atom
    mass=(1-a)*n; acc=0.0
    if k < n:
        first_mass=1.0-frac if frac>1e-12 else 1.0
        acc += first_mass*z[k]
        if k+1<n: acc += z[k+1:].sum()
    return float(acc/mass)


class PriceForecaster:
    def __init__(self, data: Q4Data):
        self.data=data
        self._feature_table=self._build_feature_table()
        self._fit_cache: Dict[Tuple[str,str], Dict[int,np.ndarray]]={}

    def _fixed(self,t:datetime)->float:
        v=actual_at(self.data,t,'fixed_price')
        return 0.8 if v is None or not np.isfinite(v) else float(v)

    def _lag(self,t:datetime,days:int,field='price')->float|None:
        return actual_at(self.data,t-timedelta(days=int(days)),field)

    def _historical_pred_net(self,d:int,i:int)->float:
        """Q2-exact net-load forecast that was legal at the event preceding target slot."""
        if i == 0:
            issue_day=d-1; target_plan_day=d-1; j=143
        else:
            issue_day=d; target_plan_day=d; j=i-1
        if issue_day < 0:
            return 5000.0*DT
        L=q2_profile_forecast(self.data,issue_day,target_plan_day,'load')[j]
        G=q2_profile_forecast(self.data,issue_day,target_plan_day,'pv')[j]
        return float(L-G)

    def _revealed_residual_mean(self, decision: datetime) -> float:
        """Mean same-day weekly-baseline price error revealed strictly before decision."""
        d=(decision.date()-self.data.dates[0].date()).days
        if d < 7:
            return 0.0
        end=(decision.hour*60+decision.minute)//10
        vals=[]
        for i in range(end):
            a=self.data.price_cal[d,i]; b=self.data.price_cal[d-7,i]
            if np.isfinite(a) and np.isfinite(b):
                vals.append(float(a-b))
        return float(np.mean(vals)) if vals else 0.0

    def _build_feature_table(self)->pd.DataFrame:
        rows=[]
        for d in range(365):
            for i in range(144):
                y=self.data.price_cal[d,i]
                if not np.isfinite(y): continue
                t=self.data.dates[d] + timedelta(minutes=10*i)
                lag1=self._lag(t,1) or self._fixed(t)
                lag7=self._lag(t,7) or self._fixed(t)
                n7=self._lag(t,7,'net')
                pn=self._historical_pred_net(d,i)
                dn=((pn-(n7 if n7 is not None else pn))/DT)/1000.0
                if i == 0:
                    issue=self.data.dates[d]-timedelta(hours=6)  # previous day 18:00 forecast of this lead slot
                else:
                    issue_i=((i-1)//36)*36
                    issue=self.data.dates[d]+timedelta(minutes=10*issue_i)
                R=self._revealed_residual_mean(issue)
                rows.append((d,i,t,issue,float(y),float(lag1),float(lag7),float(dn),float(R),i//24))
        return pd.DataFrame(rows,columns=['day','slot','target_time','canonical_issue','actual','lag1','lag7','dN_MW','R_price','block'])

    @staticmethod
    def _features(model:str, lag1:float, lag7:float, dN:float, R:float)->np.ndarray:
        # Ridge is a correction to the fixed-coefficient weekly baseline,
        # not a free regression of the absolute price.  This implements
        # c_hat = c_{t-7d} + beta0 + betaN*dN + betaR*R.
        if model=='lag7_dN': return np.asarray([dN],float)
        if model=='full_ridge': return np.asarray([dN,R],float)
        raise ValueError(model)

    def fit(self,decision:datetime,cfg:PriceConfig)->Dict[int,np.ndarray]:
        if cfg.model not in ('lag7_dN','full_ridge'): return {}
        key=(decision.isoformat(),cfg.version)
        if key in self._fit_cache: return self._fit_cache[key]
        dd=(decision.date()-self.data.dates[0].date()).days
        lo=max(0,dd-int(cfg.history_days))
        tab=self._feature_table
        sub=tab[(tab.day>=lo)&(tab.target_time<decision)]
        groups=[-1] if cfg.block_scheme=='global' else list(range(6))
        out={}
        for b in groups:
            s=sub if b==-1 else sub[sub.block==b]
            cols=['dN_MW'] if cfg.model=='lag7_dN' else ['dN_MW','R_price']
            if len(s)>=cfg.min_samples:
                out[b]=_ridge(s[cols].to_numpy(),(s.actual-s.lag7).to_numpy(),cfg.ridge_alpha)
        if cfg.block_scheme=='6block' and -1 not in out and len(sub)>=cfg.min_samples:
            cols=['dN_MW'] if cfg.model=='lag7_dN' else ['dN_MW','R_price']
            out[-1]=_ridge(sub[cols].to_numpy(),(sub.actual-sub.lag7).to_numpy(),cfg.ridge_alpha)
        self._fit_cache[key]=out
        return out

    def predict(self,decision:datetime,times:Sequence[datetime],pred_net_kwh:Sequence[float],cfg:PriceConfig,
                *,oracle_price:bool=False)->Tuple[np.ndarray,List[dict]]:
        if len(times)!=len(pred_net_kwh): raise ValueError('price target length')
        betas=self.fit(decision,cfg)
        R=self._revealed_residual_mean(decision)
        out=np.zeros(len(times)); audit=[]
        max_released=decision
        for k,(t,pn) in enumerate(zip(times,pred_net_kwh)):
            actual=actual_at(self.data,t,'price')
            if oracle_price or t<=decision:
                if actual is None: actual=self._fixed(t)
                y=float(actual); source='revealed' if not oracle_price else 'price_oracle'
                lag1=self._lag(t,1) or self._fixed(t); lag7=self._lag(t,7) or self._fixed(t); dn=0.; beta=None
            else:
                lag1=self._lag(t,1) or self._fixed(t); lag7=self._lag(t,7) or self._fixed(t)
                n7=self._lag(t,7,'net'); dn=((float(pn)-(n7 if n7 is not None else float(pn)))/DT)/1000.0
                if cfg.model=='lag1': y=float(lag1); beta=None
                elif cfg.model=='lag7': y=float(lag7); beta=None
                elif cfg.model=='lag7_R': y=float(lag7+R); beta=None
                else:
                    b=(t.hour//4) if cfg.block_scheme=='6block' else -1
                    beta=betas.get(b,betas.get(-1))
                    if beta is None: y=float(lag7)
                    else: y=float(lag7 + _pred(beta,self._features(cfg.model,lag1,lag7,dn,R))[0])
                y=max(0.0,y); source=cfg.version
            out[k]=y
            audit.append({'decision_time':decision.isoformat(),'target_time':t.isoformat(),'block':int(t.hour//4),
                          'lag1':float(lag1),'lag7':float(lag7),'dN_MW':float(dn),'R_price':float(R),
                          'ridge_beta':None if beta is None else [float(x) for x in beta],
                          'pred_price':float(y),'actual_price':None if actual is None else float(actual),
                          'model_version':source,'max_training_released_at':max_released.isoformat()})
        return out,audit

    def january_walkforward(self)->Tuple[pd.DataFrame,pd.DataFrame,PriceConfig,dict]:
        rows=[]
        configs=[]
        for m in PRICE_MODELS:
            if m in ('lag7_dN','full_ridge'):
                for scheme in BLOCK_SCHEMES:
                    for a in RIDGE_ALPHAS: configs.append(PriceConfig(m,scheme,a))
            else:
                configs.append(PriceConfig(m,'global',1.0))
        for day in range(8,31):
            for eh in (0,6,12,18):
                issue=self.data.dates[day]+timedelta(hours=eh)
                # Evaluate strictly future prices until the next event; current price is already revealed.
                _,_,net,times,_=event_forecast(self.data,day,eh,37,'q4_2')
                times=times[1:37]; net=net[1:37]
                actual=np.asarray([actual_at(self.data,t,'price') for t in times],dtype=object)
                for cfg in configs:
                    pred,aud=self.predict(issue,times,net,cfg)
                    for j,(p,a) in enumerate(zip(pred,actual)):
                        if a is None: continue
                        rows.append({'decision_time':issue,'target_time':times[j],'model':cfg.model,'block_scheme':cfg.block_scheme,
                                     'alpha':cfg.ridge_alpha,'pred':float(p),'actual':float(a),'error':float(p-float(a)),
                                     'target_block':times[j].hour//4,'model_version':cfg.version})
        df=pd.DataFrame(rows)
        metrics=[]
        for key,g in df.groupby(['model','block_scheme','alpha']):
            err=g.error.to_numpy(); ae=np.abs(err)
            block_mae={str(int(b)):float(np.mean(np.abs(s.error))) for b,s in g.groupby('target_block')}
            metrics.append({'model':key[0],'block_scheme':key[1],'alpha':float(key[2]),'n':len(g),
                            'mae':float(ae.mean()),'rmse':float(np.sqrt(np.mean(err**2))),
                            'q95_abs_error':float(np.quantile(ae,.95)),'cvar95_abs_error':_cvar95(ae),'block_mae':json.dumps(block_mae,sort_keys=True)})
        met=pd.DataFrame(metrics).sort_values(['mae','rmse']).reset_index(drop=True)
        # Fair predictive freeze across every prespecified candidate.  Structural
        # mechanism evidence does not grant Ridge a privileged slot in the
        # forecasting competition: January causal walk-forward MAE is primary,
        # then RMSE/q95, with complexity used only as an exact-near-tie break.
        full=met[met.model=='full_ridge'].copy()
        best_global=full[full.block_scheme=='global'].sort_values(['mae','rmse']).iloc[0]
        best_six=full[full.block_scheme=='6block'].sort_values(['mae','rmse']).iloc[0]
        bg=json.loads(best_global.block_mae); bs=json.loads(best_six.block_mae)
        improved_blocks=sum(float(bs.get(str(i),1e9)) < float(bg.get(str(i),-1)) for i in range(6))
        six_stable=(best_six.mae <= 0.99*best_global.mae and best_six.rmse <= 0.995*best_global.rmse and improved_blocks>=4)
        rank=met.copy()
        complexity={'lag7':0,'lag1':1,'lag7_R':2,'lag7_dN':3,'full_ridge':4}
        rank['complexity_rank']=rank.model.map(complexity).fillna(99)
        rank=rank.sort_values(['mae','rmse','q95_abs_error','complexity_rank','alpha']).reset_index(drop=True)
        chosen_row=rank.iloc[0]
        if chosen_row.model in ('lag1','lag7','lag7_R'):
            chosen=PriceConfig(str(chosen_row.model),'global',1.0)
        else:
            chosen=PriceConfig(str(chosen_row.model),str(chosen_row.block_scheme),float(chosen_row.alpha))
        lag7=met[(met.model=='lag7')].iloc[0]
        ridge_ref=best_six if six_stable else best_global
        ridge_predictive_gain=float((lag7.mae-ridge_ref.mae)/lag7.mae)
        winner_gain_vs_ridge=float((ridge_ref.mae-chosen_row.mae)/ridge_ref.mae)
        runner=rank.iloc[1] if len(rank)>1 else chosen_row
        decision={'chosen':asdict(chosen),
                  'selection_rule':'minimum January causal walk-forward MAE; RMSE then q95 tie-break; model complexity only after predictive ties',
                  'selected_metrics':{k:float(chosen_row[k]) for k in ('mae','rmse','q95_abs_error','cvar95_abs_error')},
                  'runner_up':{'model':str(runner.model),'block_scheme':str(runner.block_scheme),'alpha':float(runner.alpha),'mae':float(runner.mae),'rmse':float(runner.rmse)},
                  'six_block_stable':bool(six_stable),'six_block_improved_blocks':int(improved_blocks),
                  'ridge_vs_lag7_mae_relative_gain':ridge_predictive_gain,
                  'winner_vs_best_ridge_mae_relative_gain':winner_gain_vs_ridge,
                  'ridge_prespecified_unique_main':False,'predictive_ablation_warning':False,
                  'selection_window':'2025-01-09..2025-01-31 walk-forward; formal Feb-Dec unseen','history_rule':'expanding then cap42'}
        return df,met,chosen,decision


def structure_audit(data:Q4Data)->dict:
    c=data.price_plan[:31]; n=data.net_plan_kwh[:31]/DT
    cd=c-c.mean(axis=0,keepdims=True); nd=n-n.mean(axis=0,keepdims=True)
    corr=float(np.corrcoef(cd.ravel(),nd.ravel())[0,1])
    return {'window':'2025-01-01..2025-01-31','statistic':'corr(price-slotmean, netload_power-slotmean)',
            'value':corr,'role':'offline structural mechanism evidence only; not predictive accuracy and not dispatch benefit',
            'lag7_price_corr_all_year':float(np.corrcoef(data.price_plan[7:].ravel(),data.price_plan[:-7].ravel())[0,1])}
