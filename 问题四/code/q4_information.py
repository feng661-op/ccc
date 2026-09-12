# -*- coding: utf-8 -*-
"""Information-boundary rules for Q4-2 and Q4-3."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Sequence

from q4_data import EVENT_HOURS, EVENT_PLAN_START

HISTORY_CAP_DAYS = 42


@dataclass(frozen=True)
class InformationState:
    branch: str
    decision_time: datetime
    event_hour: int
    price_revealed_through: datetime
    pv_vintage_time: datetime | None
    contract_start_j: int
    can_revise_contract: bool


def make_information_state(branch: str, day: datetime, event_hour: int) -> InformationState:
    if branch not in ('q4_2', 'q4_3'):
        raise ValueError(branch)
    if event_hour not in EVENT_HOURS:
        raise ValueError(event_hour)
    decision = datetime(day.year, day.month, day.day) + timedelta(hours=event_hour)
    if branch == 'q4_2':
        if event_hour != 0:
            raise ValueError('Q4-2 contract decision exists only at 00:00')
        return InformationState(branch, decision, 0, decision, None, 0, False)
    return InformationState(branch, decision, event_hour, decision, decision, EVENT_PLAN_START[event_hour], event_hour > 0)


def legal_history_days(day_idx: int, *, cap: int = HISTORY_CAP_DAYS) -> List[int]:
    """Completed natural days available at 00:00: at most day_idx-1.

    In prediction residual pools, callers may impose an additional one-day lag
    when a target includes the next-day tail.  This function enforces only the
    base expanding->cap rule.
    """
    if day_idx <= 0:
        return []
    lo = max(0, int(day_idx) - int(cap))
    return list(range(lo, int(day_idx)))


def legal_residual_days(day_idx: int, event_hour: int, *, cap: int = HISTORY_CAP_DAYS) -> List[int]:
    """Days whose same-event forecast horizon is fully realized.

    Event horizons end at next-day 00:10, so day h is only fully released after
    h+1 00:10.  At 00:00 we therefore stop at day_idx-2; at 06/12/18, day_idx-1
    is fully completed and may enter the pool.
    """
    latest = day_idx - 2 if int(event_hour) == 0 else day_idx - 1
    if latest < 0:
        return []
    lo = max(0, latest - int(cap) + 1)
    return list(range(lo, latest + 1))


def assert_target_released(target_end: datetime, decision_time: datetime) -> None:
    if target_end > decision_time:
        raise AssertionError(f'target not released: {target_end} > {decision_time}')


def contract_mutable(branch: str, event_hour: int, plan_j: int) -> bool:
    if branch == 'q4_2':
        return event_hour == 0 and 0 <= plan_j < 144
    if branch != 'q4_3':
        raise ValueError(branch)
    return int(plan_j) >= EVENT_PLAN_START[int(event_hour)]


def price_is_revealed(target_start: datetime, decision_time: datetime) -> bool:
    """Real-time price for the current delivery interval is known at its start."""
    return target_start <= decision_time
