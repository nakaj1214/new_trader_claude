"""株価データ（営業日 × 銘柄の表）と、そこから計算する指標。"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

from trader import calendar as tse
from trader.config import StrategyConfig
from trader.quality import QualityReport, repair_market

log = logging.getLogger(__name__)

FIELDS = {
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "adj_close": "Adj Close",
    "volume": "Volume",
    "dividends": "Dividends",
    "splits": "Stock Splits",
}


class DataNotReadyError(RuntimeError):
    """当日の終値がまだ取得できないとき（後で再実行すれば解消する可能性がある）。"""


class DataError(RuntimeError):
    """取得したデータが判断に使えないとき。"""


@dataclass
class MarketData:
    """営業日を行、ティッカーを列にした価格表。

    close は分割調整済み・配当未調整、adj_close は分割と配当を調整した値（Yahoo Finance の定義）。
    """

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    adj_close: pd.DataFrame
    volume: pd.DataFrame
    dividends: pd.DataFrame
    splits: pd.DataFrame
    quality: QualityReport = field(default_factory=QualityReport)
    _signals: dict[tuple, Signals] = field(default_factory=dict, repr=False)

    @property
    def sessions(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.close.index)

    @property
    def tickers(self) -> list[str]:
        return list(self.close.columns)

    def value(self, frame: str, session: pd.Timestamp, ticker: str) -> float:
        table: pd.DataFrame = getattr(self, frame)
        if ticker not in table.columns or session not in table.index:
            return float("nan")
        return float(table.at[session, ticker])

    def signals(self, cfg: StrategyConfig) -> Signals:
        key = (cfg.momentum_lookback, cfg.momentum_skip, cfg.trend_window)
        if key not in self._signals:
            self._signals[key] = Signals.build(self, cfg)
        return self._signals[key]


@dataclass(frozen=True)
class Signals:
    """売買判断に使う指標。行方向の shift / rolling だけを使うため、各日付の値は過去データのみに依存する。"""

    momentum: pd.DataFrame
    trend: pd.DataFrame
    turnover: pd.DataFrame

    @staticmethod
    def build(md: MarketData, cfg: StrategyConfig) -> Signals:
        adj = md.adj_close
        momentum = adj.shift(cfg.momentum_skip) / adj.shift(cfg.momentum_lookback) - 1.0
        trend = adj.rolling(cfg.trend_window, min_periods=cfg.trend_window).mean()
        turnover = (md.close * md.volume).rolling(20, min_periods=15).median()
        return Signals(momentum=momentum, trend=trend, turnover=turnover)


def _normalize_index(index: pd.Index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def _dedupe(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.index = _normalize_index(frame.index)
    return frame[~frame.index.duplicated(keep="last")].sort_index()


def from_frames(frames: dict[str, pd.DataFrame]) -> MarketData:
    """列名を揃え、営業日だけに絞った MarketData を作る（テストとダウンロード結果の共通入口）。"""
    if missing := [n for n in ("open", "close", "adj_close") if n not in frames]:
        raise DataError(f"必要な価格列がありません: {missing}")
    close = _dedupe(frames["close"])
    index = pd.DatetimeIndex([ts for ts in close.index if tse.is_session(ts)])
    columns = list(close.columns)
    built: dict[str, pd.DataFrame] = {}
    for name in FIELDS:
        neutral = name in {"dividends", "splits"}
        frame = frames.get(name)
        if frame is None:
            frame = pd.DataFrame(0.0 if neutral else float("nan"), index=index, columns=columns)
        else:
            frame = _dedupe(frame).reindex(index=index, columns=columns)
        frame = frame.astype(float)
        built[name] = frame.fillna(0.0) if neutral else frame
    return MarketData(**built)


def _download(tickers: list[str], start: date, end: date | None) -> pd.DataFrame:
    import yfinance as yf

    return yf.download(
        tickers,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat() if end else None,
        auto_adjust=False,
        actions=True,
        group_by="column",
        threads=True,
        progress=False,
        multi_level_index=True,
        timeout=30,
    )


def _frames_from_download(raw: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for name, column in FIELDS.items():
        if column in raw.columns.get_level_values(0):
            frames[name] = raw[column].reindex(columns=tickers)
    return frames


def download_market_data(tickers: Iterable[str], start: date, end: date | None = None) -> MarketData:
    """Yahoo Finance から日足を取得する。全く取れなかった銘柄は1回だけ取り直す。"""
    tickers = sorted(set(tickers))
    raw = _download(tickers, start, end)
    if raw is None or raw.empty:
        raise DataError("株価データを1件も取得できませんでした")
    frames = _frames_from_download(raw, tickers)
    empty = [t for t in tickers if frames["close"][t].dropna().empty]
    if empty:
        log.warning("取得できなかった %d 銘柄を取り直します", len(empty))
        retry = _download(empty, start, end)
        if retry is not None and not retry.empty:
            retry_frames = _frames_from_download(retry, empty)
            for name, frame in retry_frames.items():
                base = frames.setdefault(name, pd.DataFrame(index=frame.index, columns=tickers))
                base = base.reindex(base.index.union(frame.index))
                base.update(frame)
                frames[name] = base
    md = from_frames(frames)
    md.quality = repair_market(md)
    for repair in md.quality.repairs:
        log.info("分割の未調整を補正: %s %s 倍率 %g", repair.ticker, repair.date.date(), repair.factor)
    for ticker, days in md.quality.unresolved.items():
        log.warning("不自然な値動きのため判断から外す: %s %s", ticker, [d.date().isoformat() for d in days[:3]])
    return md


def require_session(md: MarketData, session: pd.Timestamp, required: Iterable[str]) -> list[str]:
    """当日の終値が揃っているか確認する。必須銘柄が欠けていれば DataNotReadyError。

    戻り値は当日の終値が無い（必須以外の）銘柄。
    """
    if len(md.sessions) == 0 or md.sessions[-1] < session:
        last = md.sessions[-1].date() if len(md.sessions) else "なし"
        raise DataNotReadyError(f"{session.date()} の日足がまだありません（最新: {last}）")
    row = md.close.loc[session]
    missing_required = [t for t in required if pd.isna(row.get(t, float("nan")))]
    if missing_required:
        raise DataNotReadyError(f"{session.date()} の終値が未取得: {missing_required}")
    return [t for t in md.tickers if pd.isna(row[t])]
