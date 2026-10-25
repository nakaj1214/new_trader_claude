"""成績指標。入金の影響を除くため、基準価額（NAV、時間加重）で計算する。"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable

import pandas as pd

from trader.portfolio import Trade

SESSIONS_PER_YEAR = 245


def nav_metrics(nav: pd.Series) -> dict[str, float]:
    nav = nav.dropna()
    if len(nav) < 2 or nav.iloc[0] <= 0:
        return {}
    returns = nav.pct_change().dropna()
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    total = nav.iloc[-1] / nav.iloc[0] - 1
    std = returns.std()
    return {
        "start": nav.index[0].date().isoformat(),
        "end": nav.index[-1].date().isoformat(),
        "years": years,
        "total_return": total,
        "cagr": (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else float("nan"),
        "volatility": std * math.sqrt(SESSIONS_PER_YEAR),
        "sharpe": returns.mean() / std * math.sqrt(SESSIONS_PER_YEAR) if std > 0 else float("nan"),
        "max_drawdown": float((nav / nav.cummax() - 1).min()),
    }


def yearly_returns(nav: pd.Series) -> pd.Series:
    nav = nav.dropna()
    year_end = nav.groupby(nav.index.year).last()
    previous = year_end.shift(1)
    previous.iloc[0] = nav.iloc[0]
    return year_end / previous - 1


def trade_stats(trades: Iterable[Trade]) -> dict[str, float]:
    trades = list(trades)
    if not trades:
        return {"count": 0}
    returns = pd.Series([t.return_pct for t in trades])
    gains = sum(t.pnl for t in trades if t.pnl > 0)
    losses = -sum(t.pnl for t in trades if t.pnl < 0)
    reasons = Counter(t.exit_reason for t in trades)
    return {
        "count": len(trades),
        "win_rate": float((returns > 0).mean()),
        "avg_return": float(returns.mean()),
        "median_return": float(returns.median()),
        "best": float(returns.max()),
        "worst": float(returns.min()),
        "avg_holding_sessions": float(pd.Series([t.holding_sessions for t in trades]).mean()),
        "profit_factor": gains / losses if losses > 0 else float("inf"),
        **{f"exit_{k}": float(v) for k, v in reasons.items()},
    }


def money_weighted_benchmark(bench_index: pd.Series, flows: pd.Series, initial: float) -> pd.Series:
    """同じ日に同じ金額をベンチマークへ入れた場合の評価額。"""
    ratio = (bench_index / bench_index.shift(1)).fillna(1.0)
    values = []
    value = initial
    for day in bench_index.index:
        value = value * ratio.loc[day] + flows.get(day, 0.0)
        values.append(value)
    return pd.Series(values, index=bench_index.index)
