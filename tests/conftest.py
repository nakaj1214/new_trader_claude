"""合成データ（外部通信なし）でテストするための共通部品。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import requests

from trader import calendar as tse
from trader.market import MarketData, from_frames

BENCH = "1306.T"
PROXY = "2559.T"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """テスト中の外部通信を禁止する。"""

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("テストから外部通信しようとしました")

    monkeypatch.setattr(requests.Session, "request", refuse)
    monkeypatch.setattr(requests, "get", refuse)
    monkeypatch.setattr(requests, "post", refuse)


def sessions(start: str, count: int) -> pd.DatetimeIndex:
    days = tse.sessions_between(start, pd.Timestamp(start) + pd.Timedelta(days=int(count * 1.6) + 30))
    assert len(days) >= count
    return days[:count]


def market_from_arrays(
    index: pd.DatetimeIndex,
    close: dict[str, list[float] | np.ndarray],
    open_: dict[str, list[float] | np.ndarray] | None = None,
    volume: float = 1_000_000,
    splits: dict[tuple[str, str], float] | None = None,
    dividends: dict[tuple[str, str], float] | None = None,
) -> MarketData:
    close_df = pd.DataFrame({k: np.asarray(v, dtype=float) for k, v in close.items()}, index=index)
    open_df = close_df.copy() if open_ is None else pd.DataFrame(
        {k: np.asarray(open_.get(k, close[k]), dtype=float) for k in close}, index=index
    )
    split_df = pd.DataFrame(0.0, index=index, columns=close_df.columns)
    for (ticker, day), ratio in (splits or {}).items():
        split_df.loc[pd.Timestamp(day), ticker] = ratio
    div_df = pd.DataFrame(0.0, index=index, columns=close_df.columns)
    for (ticker, day), amount in (dividends or {}).items():
        div_df.loc[pd.Timestamp(day), ticker] = amount
    return from_frames(
        {
            "open": open_df,
            "high": np.maximum(open_df, close_df),
            "low": np.minimum(open_df, close_df),
            "close": close_df,
            "adj_close": close_df,
            "volume": pd.DataFrame(volume, index=index, columns=close_df.columns),
            "dividends": div_df,
            "splits": split_df,
        }
    )


def random_market(
    index: pd.DatetimeIndex, drifts: dict[str, float], seed: int = 0, vol: float = 0.01, price: float = 1000.0
) -> MarketData:
    rng = np.random.default_rng(seed)
    close = {}
    opens = {}
    for ticker, drift in drifts.items():
        steps = drift + vol * rng.standard_normal(len(index))
        path = price * np.exp(np.cumsum(steps))
        close[ticker] = path
        gap = 1 + 0.002 * rng.standard_normal(len(index))
        opens[ticker] = np.concatenate([[path[0]], path[:-1] * gap[1:]])
    return market_from_arrays(index, close, opens)


def make_universe(tickers: list[str], sectors: dict[str, str] | None = None) -> pd.DataFrame:
    sectors = sectors or {}
    return pd.DataFrame(
        {
            "code": [t.removesuffix(".T") for t in tickers],
            "ticker": tickers,
            "name": [f"銘柄{t.removesuffix('.T')}" for t in tickers],
            "sector": [sectors.get(t, f"業種{i}") for i, t in enumerate(tickers)],
            "size": "TOPIX Core30",
            "as_of": "2026-10-01",
        }
    )
