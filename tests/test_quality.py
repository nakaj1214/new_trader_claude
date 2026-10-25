from __future__ import annotations

import numpy as np
import pytest
from conftest import BENCH, make_universe, market_from_arrays, sessions

from trader import strategy
from trader.config import Settings
from trader.quality import repair_market, split_factor


def test_split_factor_matching() -> None:
    assert split_factor(0.5) == 2.0
    assert split_factor(0.105) == 10.0
    assert split_factor(2.02) == 0.5
    assert split_factor(0.4) is None


def test_unadjusted_split_is_repaired_and_recorded() -> None:
    idx = sessions("2024-03-04", 10)
    close = np.array([1000, 1010, 1020, 1030, 103, 104, 105, 106, 107, 108], dtype=float)
    md = market_from_arrays(idx, {"A.T": close, BENCH: np.full(10, 1000.0)})
    report = repair_market(md)
    assert [(r.ticker, r.factor) for r in report.repairs] == [("A.T", 10.0)]
    assert md.close["A.T"].iloc[3] == pytest.approx(103.0)
    assert md.splits.at[idx[4], "A.T"] == 10.0
    assert md.volume["A.T"].iloc[0] == pytest.approx(10_000_000)
    assert report.unresolved == {}


def test_single_day_spike_is_reverted_and_unexplained_jump_excludes_ticker() -> None:
    idx = sessions("2024-01-04", 300)
    spike = np.geomspace(1000, 3000, 300)
    spike[150] /= 100  # 1日だけ100分の1になる誤り
    bad = np.geomspace(1000, 2500, 300)
    bad[200:] *= 0.4  # 分割比率で説明できない急落
    md = market_from_arrays(idx, {"A.T": spike, "B.T": bad, BENCH: np.geomspace(1000, 1300, 300)})
    report = repair_market(md)
    assert md.close["A.T"].iloc[150] == pytest.approx(np.geomspace(1000, 3000, 300)[150], rel=1e-9)
    assert report.unreliable == {"B.T"}
    settings = Settings()
    uni = make_universe(["A.T", "B.T"])
    ranked = strategy.rank_universe(md, md.signals(settings.strategy), uni, idx[-1], settings.strategy)
    assert [c.ticker for c in ranked] == ["A.T"]
