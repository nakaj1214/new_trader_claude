"""売買ルール。状態を変更しない純粋関数だけを置く（バックテストと本番で共通）。"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import pandas as pd

from trader.config import SatelliteConfig, StrategyConfig
from trader.market import MarketData, Signals
from trader.portfolio import Order, PortfolioState

RISK_ON = "risk_on"
RISK_OFF = "risk_off"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class Candidate:
    ticker: str
    rank: int
    momentum: float
    close: float
    sector: str


@dataclass(frozen=True)
class RebalancePlan:
    orders: list[Order]
    skipped: list[tuple[Candidate, str]]  # 買えなかった上位候補と理由
    kept: list[str]
    notes: list[str]


def _at(frame: pd.DataFrame, session: pd.Timestamp, ticker: str) -> float:
    if ticker not in frame.columns or session not in frame.index:
        return float("nan")
    return float(frame.at[session, ticker])


def market_regime(md: MarketData, signals: Signals, session: pd.Timestamp, cfg: StrategyConfig) -> str:
    """TOPIX 連動ETFが200日平均より上なら risk_on。データが無ければ unknown（新規買いしない）。"""
    if not cfg.regime_filter:
        return RISK_ON
    price = _at(md.adj_close, session, cfg.benchmark_ticker)
    trend = _at(signals.trend, session, cfg.benchmark_ticker)
    if math.isnan(price) or math.isnan(trend):
        return UNKNOWN
    return RISK_ON if price > trend else RISK_OFF


def rank_universe(
    md: MarketData,
    signals: Signals,
    universe: pd.DataFrame,
    session: pd.Timestamp,
    cfg: StrategyConfig,
) -> list[Candidate]:
    """長期トレンドが上向きで流動性のある銘柄を、過去12か月（直近1か月除く）の上昇率順に並べる。"""
    rows = []
    unreliable = md.quality.unreliable
    for ticker, sector in zip(universe["ticker"], universe["sector"], strict=True):
        if ticker in unreliable:
            continue
        close = _at(md.close, session, ticker)
        adj = _at(md.adj_close, session, ticker)
        trend = _at(signals.trend, session, ticker)
        momentum = _at(signals.momentum, session, ticker)
        turnover = _at(signals.turnover, session, ticker)
        if any(math.isnan(v) for v in (close, adj, trend, momentum, turnover)):
            continue
        if adj <= trend or turnover < cfg.min_turnover or close <= 0:
            continue
        rows.append((ticker, momentum, close, sector))
    rows.sort(key=lambda r: (-r[1], r[0]))
    return [Candidate(t, i + 1, m, c, s) for i, (t, m, c, s) in enumerate(rows)]


def stop_triggered(close: float, high_water: float, cfg: StrategyConfig) -> bool:
    if cfg.trailing_stop is None or math.isnan(close) or high_water <= 0:
        return False
    return close <= high_water * (1.0 - cfg.trailing_stop)


def held_exit_reason(
    md: MarketData, signals: Signals, session: pd.Timestamp, ticker: str, rank: int | None, cfg: StrategyConfig
) -> str | None:
    """月末の入れ替えで保有銘柄を売る理由。売らない場合は None。"""
    if rank is not None:
        return "rank_dropped" if rank > cfg.keep_rank else None
    adj = _at(md.adj_close, session, ticker)
    trend = _at(signals.trend, session, ticker)
    if math.isnan(adj) or math.isnan(trend):
        return None  # データ欠損だけでは売らない（警告は呼び出し側で出す）
    return "below_trend" if adj <= trend else None


def plan_rebalance(
    state: PortfolioState,
    md: MarketData,
    signals: Signals,
    session: pd.Timestamp,
    candidates: list[Candidate],
    sectors: dict[str, str],
    regime: str,
    excluded: set[str],
    incoming_cash: float,
    cfg: StrategyConfig,
    sat: SatelliteConfig,
    frozen: frozenset[str] = frozenset(),
) -> RebalancePlan:
    """月末の入れ替え。excluded は既に売り注文がある銘柄、incoming_cash はその売却見込み額。

    frozen は株価の急落で判断を保留中の銘柄で、売らずに保有枠を1つ使ったままにする。
    """
    day = session.date().isoformat()
    held = [t for t in state.positions if t not in excluded]
    notes: list[str] = []
    if regime == UNKNOWN:
        notes.append("相場全体（TOPIX）のデータが無いため、今回の入れ替えは見送ります")
        return RebalancePlan([], [], held, notes)
    if regime == RISK_OFF:
        sells = [
            Order("sell", t, state.positions[t].shares, "market_downtrend", day, state.positions[t].last_price)
            for t in held
            if t not in frozen
        ]
        return RebalancePlan(sells, [], [t for t in held if t in frozen], notes)

    by_ticker = {c.ticker: c for c in candidates}
    sells: list[Order] = []
    kept: list[str] = []
    for ticker in held:
        cand = by_ticker.get(ticker)
        pos = state.positions[ticker]
        if ticker in frozen:
            kept.append(ticker)
            continue
        if ticker not in sectors:
            sells.append(Order("sell", ticker, pos.shares, "universe_exit", day, pos.last_price))
            continue
        reason = held_exit_reason(md, signals, session, ticker, cand.rank if cand else None, cfg)
        if reason is None:
            kept.append(ticker)
            if cand is None and math.isnan(_at(md.close, session, ticker)):
                notes.append(f"{ticker} の当日データが無いため、売買判断せず保有を続けます")
            continue
        sells.append(Order("sell", ticker, pos.shares, reason, day, pos.last_price, cand.rank if cand else None))

    cash = state.cash + incoming_cash + sum(o.shares * o.ref_price * (1 - sat.cost_rate) for o in sells)
    kept_value = sum(state.positions[t].shares * state.positions[t].last_price for t in kept)
    target = (cash + kept_value) / cfg.max_positions
    slots = cfg.max_positions - len(kept)
    sector_count = Counter(sectors.get(t, "") for t in kept)
    buys: list[Order] = []
    skipped: list[tuple[Candidate, str]] = []
    for cand in candidates:
        if slots <= 0 or cand.rank > cfg.keep_rank:
            break
        if cand.ticker in state.positions or cand.ticker in excluded:
            continue
        if sector_count[cand.sector] >= cfg.max_per_sector:
            skipped.append((cand, "sector_limit"))
            continue
        unit_cost = cand.close * (1 + sat.cost_rate)
        alloc = min(target, cash / slots)
        shares = int(alloc // (unit_cost * sat.lot_size)) * sat.lot_size
        if shares <= 0:
            skipped.append((cand, "too_expensive"))
            continue
        buys.append(Order("buy", cand.ticker, shares, "momentum", day, cand.close, cand.rank, cand.momentum))
        cash -= shares * unit_cost
        slots -= 1
        sector_count[cand.sector] += 1
    return RebalancePlan(sells + buys, skipped, kept, notes)
