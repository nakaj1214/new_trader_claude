from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from conftest import BENCH, PROXY, make_universe, market_from_arrays, sessions

from trader import engine
from trader.config import Settings
from trader.portfolio import Order, PortfolioState, Position

SETTINGS = Settings().with_strategy(max_positions=2)
UNI = make_universe(["A.T", "B.T"])
# 3月前半: 月末を含まない15営業日（入れ替え判断が起きない）
MID_MONTH = sessions("2024-03-04", 15)


def _state_holding(last: int, shares: int = 10, price: float = 1000.0, cash: float = 0.0) -> PortfolioState:
    state = PortfolioState.new(MID_MONTH[0].date().isoformat(), cash + shares * price)
    state.cash = cash
    state.positions["A.T"] = Position("A.T", shares, "2024-03-04", price, shares * price * 1.001, price, price)
    state.last_session = MID_MONTH[last].date().isoformat()
    state.last_rebalance = state.last_session
    return state


def _flat(value: float, n: int = 15) -> np.ndarray:
    return np.full(n, value, dtype=float)


def _step(state, md, i, settings=SETTINGS, **kw):
    return engine.step(state, md, md.sessions[i], settings, UNI, apply_splits=kw.pop("apply_splits", True), **kw)


def test_first_step_orders_and_next_step_fills_at_open_with_cost() -> None:
    idx = sessions("2023-01-04", 300)
    md = market_from_arrays(
        idx,
        {
            "A.T": np.geomspace(1000, 3000, 300),
            "B.T": np.geomspace(1000, 2000, 300),
            BENCH: np.geomspace(1000, 1300, 300),
            PROXY: np.geomspace(1000, 1100, 300),
        },
    )
    state = PortfolioState.new(idx[280].date().isoformat(), 100_000, 200_000)
    res = engine.step(state, md, idx[280], SETTINGS, UNI, apply_splits=False)
    assert res.is_rebalance and res.regime == "risk_on"
    assert [o.ticker for o in res.orders] == ["A.T", "B.T"]
    with pytest.raises(engine.AlreadyProcessedError):
        engine.step(state, md, idx[280], SETTINGS, UNI, apply_splits=False)

    res = engine.step(state, md, idx[281], SETTINGS, UNI, apply_splits=False)
    assert {f.ticker for f in res.fills} == {"A.T", "B.T"}
    for fill in res.fills:
        price = md.value("open", idx[281], fill.ticker)
        assert fill.amount == pytest.approx(fill.shares * price * 1.001)
    assert state.cash == pytest.approx(100_000 - sum(f.amount for f in res.fills))
    assert state.cash >= 0
    assert state.core.value > 200_000  # 参考ETFの上昇で積立の推定額が増える


def test_trailing_stop_sells_next_open_and_records_trade() -> None:
    close = _flat(1000)
    close[3], close[4] = 1200, 1000  # 最高値1200から16.7%下落
    md = market_from_arrays(MID_MONTH, {"A.T": close, BENCH: _flat(1000)})
    state = _state_holding(last=2)
    assert _step(state, md, 3).orders == []
    res = _step(state, md, 4)
    assert [(o.ticker, o.reason) for o in res.orders] == [("A.T", "trailing_stop")]
    res = _step(state, md, 5)
    assert [f.side for f in res.fills] == ["sell"] and "A.T" not in state.positions
    trade = state.trades[-1]
    assert trade.exit_reason == "trailing_stop" and trade.holding_sessions == 5
    assert trade.pnl == pytest.approx(10 * 1000 * 0.999 - 10 * 1000 * 1.001)


def test_sell_without_open_price_is_carried_over() -> None:
    opens = _flat(1000)
    opens[3] = np.nan
    md = market_from_arrays(MID_MONTH, {"A.T": _flat(1000), BENCH: _flat(1000)}, {"A.T": opens})
    state = _state_holding(last=1)
    state.pending = [Order("sell", "A.T", 10, "trailing_stop", MID_MONTH[1].date().isoformat(), 1000)]
    _step(state, md, 2)  # 2日目に約定する
    assert "A.T" not in state.positions

    state = _state_holding(last=2)
    state.pending = [Order("sell", "A.T", 10, "trailing_stop", MID_MONTH[2].date().isoformat(), 1000)]
    res = _step(state, md, 3)
    assert "A.T" in state.positions and [o.ticker for o in res.orders] == ["A.T"]
    res = _step(state, md, 4)
    assert "A.T" not in state.positions and len(res.fills) == 1


def test_buy_is_reduced_to_available_cash() -> None:
    md = market_from_arrays(MID_MONTH, {"A.T": _flat(1000), BENCH: _flat(1000)})
    state = PortfolioState.new("2024-03-04", 10_000)
    state.last_session = state.last_rebalance = MID_MONTH[1].date().isoformat()
    state.pending = [Order("buy", "A.T", 10, "momentum", MID_MONTH[1].date().isoformat(), 1000, 1, 0.3)]
    res = _step(state, md, 2)
    assert state.positions["A.T"].shares == 9
    assert any("9 株に減らして" in w for w in res.warnings)


def test_split_is_applied_without_false_stop() -> None:
    close = _flat(1000)
    close[3:] = 500
    md = market_from_arrays(MID_MONTH, {"A.T": close, BENCH: _flat(1000)}, splits={("A.T", "2024-03-07"): 2.0})
    state = _state_holding(last=2)
    res = _step(state, md, 3)
    pos = state.positions["A.T"]
    assert pos.shares == 20 and pos.high_water == pytest.approx(500) and res.orders == []
    assert state.equity() == pytest.approx(10_000)


def test_delayed_split_record_is_held_one_day_then_applied() -> None:
    close = _flat(1000)
    close[3:] = 500
    without = market_from_arrays(MID_MONTH, {"A.T": close, BENCH: _flat(1000)})
    with_split = market_from_arrays(
        MID_MONTH, {"A.T": close, BENCH: _flat(1000)}, splits={("A.T", "2024-03-07"): 2.0}
    )
    state = _state_holding(last=2)
    res = _step(state, without, 3)
    assert res.orders == [] and any("分割" in w for w in res.warnings)
    assert state.equity() == pytest.approx(10_000)  # 判断保留中は前日の価格で評価
    res = _step(state, with_split, 4)
    assert state.positions["A.T"].shares == 20 and res.orders == []


def test_real_crash_is_accepted_on_second_day() -> None:
    close = _flat(1000)
    close[3:] = 400
    md = market_from_arrays(MID_MONTH, {"A.T": close, BENCH: _flat(1000)})
    state = _state_holding(last=2, cash=100_000)  # 資金全体の下落は20%未満
    assert _step(state, md, 3).orders == []
    res = _step(state, md, 4)
    assert [(o.ticker, o.reason) for o in res.orders] == [("A.T", "trailing_stop")]


def test_dividends_are_credited_after_tax() -> None:
    md = market_from_arrays(
        MID_MONTH, {"A.T": _flat(1000), BENCH: _flat(1000)}, dividends={("A.T", "2024-03-07"): 30.0}
    )
    state = _state_holding(last=2)
    res = _step(state, md, 3, settings=SETTINGS.with_satellite(tax_rate=0.2))
    assert res.dividends == pytest.approx(10 * 30 * 0.8)
    assert state.cash == pytest.approx(240)


def test_month_end_contribution_keeps_nav_and_is_not_double_counted() -> None:
    idx = sessions("2024-03-25", 10)  # 3/29 が月末
    md = market_from_arrays(idx, {"A.T": _flat(1000, 10), BENCH: _flat(1000, 10)})
    settings = replace(SETTINGS, core=replace(SETTINGS.core, proxy_ticker=None))
    state = PortfolioState.new("2024-03-25", 50_000, 200_000)
    state.last_session = state.last_rebalance = "2024-03-25"
    month_end = list(idx).index(next(d for d in idx if d.day == 29))
    for i in range(1, month_end + 1):
        res = engine.step(state, md, idx[i], settings, UNI, apply_splits=True)
    assert res.is_rebalance and res.contribution == 10_000
    assert state.cash == 60_000 and state.units == pytest.approx(60_000) and state.nav() == pytest.approx(1.0)
    assert state.core.value == 240_000 and state.contributed_months == ["2024-03"]
    assert state.total_contributed == 60_000


def test_drawdown_limit_sells_all_pauses_and_resumes() -> None:
    idx = sessions("2024-01-04", 90)
    close = _flat(1000, 90)
    close[2:] = 750  # 25% 下落
    md = market_from_arrays(idx, {"A.T": close, BENCH: _flat(1000, 90)})
    settings = replace(SETTINGS, core=replace(SETTINGS.core, proxy_ticker=None))
    state = PortfolioState.new("2024-01-04", 100_000)
    state.cash = 0
    state.positions["A.T"] = Position("A.T", 100, "2024-01-04", 1000, 100_100, 1000, 1000)
    state.last_session = state.last_rebalance = idx[1].date().isoformat()
    res = engine.step(state, md, idx[2], settings, UNI, apply_splits=True)
    assert [(o.ticker, o.reason) for o in res.orders] == [("A.T", "drawdown_limit")]
    assert state.pause_rebalances_left == 3
    resumed = []
    for i in range(3, 90):
        res = engine.step(state, md, idx[i], settings, UNI, apply_splits=True)
        assert not [o for o in res.orders if o.side == "buy"] or state.pause_rebalances_left == 0
        if any("再開" in e for e in res.events):
            resumed.append(res.session)
    assert len(resumed) == 1 and resumed[0].startswith("2024-03")
    assert state.pause_rebalances_left == 0 and state.peak_nav == pytest.approx(state.nav())


def test_tax_on_previous_year_gains_is_deducted_once() -> None:
    idx = sessions("2024-12-25", 6)
    md = market_from_arrays(idx, {"A.T": _flat(1000, 6), BENCH: _flat(1000, 6)})
    settings = replace(SETTINGS.with_satellite(tax_rate=0.2), core=replace(SETTINGS.core, proxy_ticker=None))
    state = PortfolioState.new("2024-12-25", 50_000)
    state.realized_by_year = {"2024": 10_000}
    state.last_session = state.last_rebalance = "2024-12-30"
    first_2025 = next(i for i, d in enumerate(idx) if d.year == 2025)
    res = engine.step(state, md, idx[first_2025], settings, UNI, apply_splits=True)
    assert res.tax_paid == pytest.approx(2_000) and state.cash == pytest.approx(48_000)
    res = engine.step(state, md, idx[first_2025 + 1], settings, UNI, apply_splits=True)
    assert res.tax_paid == 0


def test_split_records_on_different_days_are_applied_once() -> None:
    close = _flat(1000)
    close[3:] = 500
    md = market_from_arrays(
        MID_MONTH,
        {"A.T": close, BENCH: _flat(1000)},
        splits={("A.T", "2024-03-07"): 2.0, ("A.T", "2024-03-08"): 2.0},  # 同じ分割が2日に記録された
    )
    state = _state_holding(last=2)
    _step(state, md, 3)
    _step(state, md, 4)
    assert state.positions["A.T"].shares == 20
