from __future__ import annotations

import numpy as np
import pandas as pd
from conftest import BENCH, make_universe, market_from_arrays, sessions

from trader import strategy
from trader.config import Settings
from trader.portfolio import PortfolioState, Position

N = 300


def _trend(start: float, end: float) -> np.ndarray:
    return np.geomspace(start, end, N)


def _setup() -> tuple:
    idx = sessions("2024-01-04", N)
    close = {
        "A.T": _trend(1000, 3000),  # 最も強い
        "B.T": _trend(1000, 2000),
        "C.T": _trend(1000, 1500),
        "D.T": _trend(3000, 1000),  # 下落トレンド
        "E.T": _trend(1000, 1800),
        BENCH: _trend(1000, 1300),
    }
    md = market_from_arrays(idx, close)
    return md, idx[-1]


def test_rank_universe_orders_by_momentum_and_excludes_downtrend() -> None:
    md, session = _setup()
    settings = Settings()
    uni = make_universe(["A.T", "B.T", "C.T", "D.T", "E.T"])
    ranked = strategy.rank_universe(md, md.signals(settings.strategy), uni, session, settings.strategy)
    assert [c.ticker for c in ranked] == ["A.T", "B.T", "E.T", "C.T"]
    assert [c.rank for c in ranked] == [1, 2, 3, 4]


def test_illiquid_stocks_are_excluded() -> None:
    md, session = _setup()
    md.volume.loc[:, "A.T"] = 10.0  # 売買代金がほぼ無い
    settings = Settings()
    uni = make_universe(["A.T", "B.T"])
    ranked = strategy.rank_universe(md, md.signals(settings.strategy), uni, session, settings.strategy)
    assert [c.ticker for c in ranked] == ["B.T"]


def test_market_regime() -> None:
    md, session = _setup()
    settings = Settings()
    sig = md.signals(settings.strategy)
    assert strategy.market_regime(md, sig, session, settings.strategy) == strategy.RISK_ON
    down = market_from_arrays(md.sessions, {BENCH: _trend(1300, 900)})
    assert strategy.market_regime(down, down.signals(settings.strategy), session, settings.strategy) == "risk_off"
    empty = market_from_arrays(md.sessions, {"A.T": _trend(1, 2)})
    assert strategy.market_regime(empty, empty.signals(settings.strategy), session, settings.strategy) == "unknown"


def _plan(state: PortfolioState, settings: Settings, uni: pd.DataFrame, regime: str = strategy.RISK_ON):
    md, session = _setup()
    sig = md.signals(settings.strategy)
    candidates = strategy.rank_universe(md, sig, uni, session, settings.strategy)
    sectors = dict(zip(uni["ticker"], uni["sector"], strict=True))
    return strategy.plan_rebalance(
        state, md, sig, session, candidates, sectors, regime, set(), 0.0, settings.strategy, settings.satellite
    )


def test_plan_rebalance_respects_positions_sector_and_budget() -> None:
    settings = Settings().with_strategy(max_positions=2, max_per_sector=1)
    uni = make_universe(["A.T", "B.T", "C.T", "E.T"], sectors={"A.T": "機械", "B.T": "機械"})
    state = PortfolioState.new("2024-01-04", 100_000)
    plan = _plan(state, settings, uni)
    buys = [o for o in plan.orders if o.side == "buy"]
    assert [o.ticker for o in buys] == ["A.T", "E.T"]  # B は同業種上限で見送り
    assert ("B.T", "sector_limit") in [(c.ticker, r) for c, r in plan.skipped]
    spent = sum(o.shares * o.ref_price * (1 + settings.satellite.cost_rate) for o in buys)
    assert spent <= 100_000
    assert all(o.shares * o.ref_price <= 50_000 for o in buys)  # 1銘柄は資金の1/2まで


def test_plan_rebalance_skips_too_expensive_stock() -> None:
    settings = Settings().with_strategy(max_positions=1)
    uni = make_universe(["A.T", "B.T"])
    state = PortfolioState.new("2024-01-04", 2_500)  # A(約3000円)は1株も買えない、B(約2000円)は1株買える
    plan = _plan(state, settings, uni)
    assert ("A.T", "too_expensive") in [(c.ticker, r) for c, r in plan.skipped]
    assert [o.ticker for o in plan.orders] == ["B.T"]


def test_plan_rebalance_keeps_within_rank_buffer_and_sells_downtrend() -> None:
    settings = Settings().with_strategy(max_positions=1, keep_rank=3)
    uni = make_universe(["A.T", "B.T", "C.T", "D.T", "E.T"])
    state = PortfolioState.new("2024-01-04", 0.0)
    state.positions["C.T"] = Position("C.T", 10, "2024-06-03", 1200, 12000, 1500, 1500)  # 4位: 売る
    state.positions["D.T"] = Position("D.T", 10, "2024-06-03", 2000, 20000, 2000, 1000)  # 下落: 売る
    plan = _plan(state, settings, uni)
    sells = {o.ticker: o.reason for o in plan.orders if o.side == "sell"}
    assert sells == {"C.T": "rank_dropped", "D.T": "below_trend"}
    state.positions.pop("D.T")
    state.positions["E.T"] = state.positions.pop("C.T")  # 3位なら保有継続
    plan = _plan(state, settings, uni)
    assert plan.kept == ["E.T"] and plan.orders == []


def test_risk_off_sells_everything_and_buys_nothing() -> None:
    settings = Settings()
    uni = make_universe(["A.T", "B.T"])
    state = PortfolioState.new("2024-01-04", 100_000)
    state.positions["A.T"] = Position("A.T", 5, "2024-06-03", 2000, 10000, 3000, 3000)
    plan = _plan(state, settings, uni, regime=strategy.RISK_OFF)
    assert [(o.side, o.ticker, o.reason) for o in plan.orders] == [("sell", "A.T", "market_downtrend")]


def test_held_stock_outside_universe_is_sold() -> None:
    settings = Settings()
    uni = make_universe(["A.T", "B.T"])
    state = PortfolioState.new("2024-01-04", 0.0)
    state.positions["C.T"] = Position("C.T", 5, "2024-06-03", 1200, 6000, 1500, 1500)
    plan = _plan(state, settings, uni)
    assert [(o.ticker, o.reason) for o in plan.orders if o.side == "sell"] == [("C.T", "universe_exit")]


def test_stop_triggered() -> None:
    cfg = Settings().strategy
    assert strategy.stop_triggered(850, 1000, cfg)
    assert not strategy.stop_triggered(851, 1000, cfg)
    assert not strategy.stop_triggered(100, 1000, Settings().with_strategy(trailing_stop=None).strategy)


def test_frozen_position_keeps_its_slot_and_is_not_sold() -> None:
    md, session = _setup()
    settings = Settings().with_strategy(max_positions=1)
    uni = make_universe(["A.T", "B.T", "D.T"])
    state = PortfolioState.new("2024-01-04", 100_000)
    state.positions["D.T"] = Position("D.T", 10, "2024-06-03", 2000, 20000, 2000, 2000)  # 下落中だが判断保留
    sig = md.signals(settings.strategy)
    candidates = strategy.rank_universe(md, sig, uni, session, settings.strategy)
    sectors = dict(zip(uni["ticker"], uni["sector"], strict=True))
    plan = strategy.plan_rebalance(
        state, md, sig, session, candidates, sectors, strategy.RISK_ON, set(), 0.0,
        settings.strategy, settings.satellite, frozen=frozenset({"D.T"}),
    )
    assert plan.orders == [] and plan.kept == ["D.T"]
