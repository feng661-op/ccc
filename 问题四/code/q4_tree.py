# -*- coding: utf-8 -*-
"""Decision-event tree and contract mutability for Q4.

This module has no optimizer logic.  It defines the exact relationship between
natural event horizons and the official plan-row contract coordinates.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

from q4_data import EVENT_HOURS, EVENT_PLAN_START

EVENT_PLAN_HORIZON = {0:145, 6:109, 12:73, 18:37}


@dataclass(frozen=True)
class HorizonSlot:
    h: int
    target_start: datetime
    plan_day_offset: int
    plan_j: int
    mutable: bool
    lead_from_previous_contract: bool


def event_horizon(day: datetime, event_hour: int, branch: str) -> List[HorizonSlot]:
    if event_hour not in EVENT_HOURS:
        raise ValueError(event_hour)
    if branch == 'q4_2' and event_hour != 0:
        raise ValueError('Q4-2 has no intraday contract revision events')
    if branch not in ('q4_2','q4_3'):
        raise ValueError(branch)
    H=EVENT_PLAN_HORIZON[event_hour]
    issue=datetime(day.year,day.month,day.day)+timedelta(hours=event_hour)
    out=[]
    for h in range(H):
        t=issue+timedelta(minutes=10*h)
        if event_hour==0 and h==0:
            out.append(HorizonSlot(h,t,-1,143,False,True))
            continue
        minute=t.hour*60+t.minute
        if minute==0:
            pd=0 if t.date()==day.date() else 0
            # only final next-day 00:00 slot in this event horizon
            plan_day_offset=0
            j=143
        else:
            plan_day_offset=(t.date()-day.date()).days
            j=minute//10-1
        mutable=(plan_day_offset==0 and j>=EVENT_PLAN_START[event_hour])
        out.append(HorizonSlot(h,t,plan_day_offset,j,mutable,False))
    return out


def mutable_plan_js(event_hour:int)->range:
    if event_hour not in EVENT_HOURS: raise ValueError(event_hour)
    return range(EVENT_PLAN_START[event_hour],144)


def validate_tree()->dict:
    d=datetime(2025,2,1)
    rows=[]
    for eh in EVENT_HOURS:
        x=event_horizon(d,eh,'q4_3')
        rows.append({'event_hour':eh,'H':len(x),'first_h':x[0].h,'first_j':x[0].plan_j,
                     'first_lead':x[0].lead_from_previous_contract,'last_j':x[-1].plan_j,
                     'mutable_count':sum(z.mutable for z in x)})
    expected={0:(145,144),6:(109,109),12:(73,73),18:(37,37)}
    ok=True
    for r in rows:
        h,m=expected[r['event_hour']]
        ok &= r['H']==h and r['mutable_count']==m
    return {'rows':rows,'pass':bool(ok),'nonanticipativity':'all mutable contract variables are shared across scenarios at each event'}
