from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import BENCH, PROXY, make_universe, random_market, sessions

from trader.backtest import (
    Scenario,
    default_scenarios,
    render_report,
    run_backtest,
    sensitivity_scenarios,
)
from trader.config import Settings
from trader.market import from_frames
from trader.metrics import money_weighted_benchmark, nav_metrics, trade_stats, yearly_returns

TICKERS = [f"{1000 + i}.T" for i in range(12)]


def _market(seed: int = 1):
    idx = sessions("2019-01-04", 900)
    drifts = {t: 0.0002 * (i - 4) for i, t in enumerate(TICKERS)}
    drifts[BENCH] = 0.0003
    drifts[PROXY] = 0.0003
    return random_market(idx, drifts, seed=seed, vol=0.015)


def test_backtest_runs_and_report_renders() -> None:
    md = _market()
    uni = make_universe(TICKERS)
    settings = Settings()
    results = [run_backtest(md, uni, s) for s in default_scenarios(settings)]
    ref = results[0]
    assert len(ref.history) == len(md.sessions) - 257
    assert (ref.history["equity"] > 0).all()
    assert ref.state.trades, "合成データでも売買が発生するはず"
    assert ref.history["cash"].min() >= -1e-6
    plan = results[1]
    months = plan.history["flow"].gt(0).sum()
    assert plan.state.total_contributed == pytest.approx(50_000 + 10_000 * months)
    report = render_report(results, [], md, uni, settings, pd.Timestamp("2026-10-25").date())
    for heading in ["## 前提", "生存者バイアス", "## 2. 期間を前半・後半", "## 4.", "## 7. まとめ"]:
        assert heading in report


def test_no_lookahead_future_prices_do_not_change_past_decisions() -> None:
    md = _market(seed=2)
    uni = make_universe(TICKERS)
    scenario = default_scenarios(Settings())[0]
    base = run_backtest(md, uni, scenario)
    cut = md.sessions[600]
    frames = {
        name: getattr(md, name).copy()
        for name in ["open", "high", "low", "close", "adj_close", "volume", "dividends", "splits"]
    }
    rng = np.random.default_rng(99)
    for name in ["open", "high", "low", "close", "adj_close"]:
        future = frames[name].index > cut
        frames[name].loc[future] *= rng.uniform(0.5, 1.5, size=frames[name].loc[future].shape)
    changed = run_backtest(from_frames(frames), uni, scenario)
    pd.testing.assert_frame_equal(base.history.loc[:cut], changed.history.loc[:cut])


def test_sensitivity_scenarios_cover_grid() -> None:
    scenarios = sensitivity_scenarios(Settings())
    assert len(scenarios) == 18
    assert all(isinstance(s, Scenario) for s in scenarios)
    assert all(s.settings.strategy.keep_rank >= s.settings.strategy.max_positions for s in scenarios)


def test_nav_metrics_and_yearly_returns() -> None:
    idx = pd.DatetimeIndex(pd.date_range("2020-01-01", "2021-01-01", freq="D"))
    nav = pd.Series(np.linspace(1.0, 2.0, len(idx)), index=idx)
    nav.iloc[100] = 0.5  # 一時的な急落
    m = nav_metrics(nav)
    assert m["total_return"] == pytest.approx(1.0)
    assert m["cagr"] == pytest.approx(1.0, rel=0.01)
    assert m["max_drawdown"] < -0.6
    yearly = yearly_returns(nav)
    assert list(yearly.index) == [2020, 2021]


def test_money_weighted_benchmark_and_trade_stats() -> None:
    idx = pd.DatetimeIndex(["2024-01-04", "2024-01-05", "2024-01-09"])
    bench = pd.Series([1.0, 1.1, 1.21], index=idx)
    flows = pd.Series([0.0, 100.0, 0.0], index=idx)
    value = money_weighted_benchmark(bench, flows, 1000.0)
    assert value.iloc[-1] == pytest.approx((1000 * 1.1 + 100) * 1.1)
    assert trade_stats([]) == {"count": 0}
