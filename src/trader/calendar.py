"""東証（XTKS）の営業日判定。"""

from __future__ import annotations

from datetime import date
from functools import lru_cache

import exchange_calendars as xcals
import pandas as pd


@lru_cache(maxsize=1)
def _calendar() -> xcals.ExchangeCalendar:
    return xcals.get_calendar("XTKS", start="2000-01-04")


def to_timestamp(value: date | str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def is_session(value: date | str | pd.Timestamp) -> bool:
    return bool(_calendar().is_session(to_timestamp(value)))


def sessions_between(start: date | str | pd.Timestamp, end: date | str | pd.Timestamp) -> pd.DatetimeIndex:
    return _calendar().sessions_in_range(to_timestamp(start), to_timestamp(end))


def next_session(value: date | str | pd.Timestamp) -> pd.Timestamp:
    return _calendar().next_session(to_timestamp(value))


def previous_session(value: date | str | pd.Timestamp) -> pd.Timestamp:
    return _calendar().previous_session(to_timestamp(value))


def session_on_or_before(value: date | str | pd.Timestamp) -> pd.Timestamp:
    ts = to_timestamp(value)
    return ts if is_session(ts) else _calendar().date_to_session(ts, direction="previous")


def is_month_end_session(value: date | str | pd.Timestamp) -> bool:
    """その月の最終営業日なら True（月1回の入れ替え判断日）。"""
    ts = to_timestamp(value)
    return next_session(ts).month != ts.month


def is_week_end_session(value: date | str | pd.Timestamp) -> bool:
    """その週の最終営業日なら True（週次レポートの送信日）。"""
    ts = to_timestamp(value)
    return next_session(ts).isocalendar()[:2] != ts.isocalendar()[:2]
