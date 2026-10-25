from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from trader import calendar as tse
from trader.config import DEFAULT_CONFIG_PATH, ConfigError, load_settings
from trader.daily import target_session
from trader.universe import UniverseError, ensure_universe, parse_jpx_listing, to_ticker

JST = ZoneInfo("Asia/Tokyo")


def test_repository_settings_load() -> None:
    settings = load_settings(DEFAULT_CONFIG_PATH)
    assert settings.mode == "paper"
    assert settings.satellite.initial_capital == 50_000
    assert settings.core.monthly_amount + settings.satellite.monthly_contribution == 50_000


@pytest.mark.parametrize(
    "body",
    [
        "mode: real\n",
        "strategy:\n  max_positions: 5\n  keep_rank: 3\n",
        "strategy:\n  unknown_key: 1\n",
        "satellite:\n  lot_size: 0\n",
        "typo_section: {}\n",
    ],
)
def test_invalid_settings_are_rejected(tmp_path, body: str) -> None:
    path = tmp_path / "settings.yaml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(path)


def test_month_and_week_end_sessions() -> None:
    assert tse.is_month_end_session("2026-10-30")
    assert not tse.is_month_end_session("2026-10-29")
    assert tse.is_week_end_session("2026-10-30")
    # 2026-12-31 は休場のため 12/30 が月末・年末の最終営業日
    assert tse.is_month_end_session("2026-12-30")


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        (datetime(2026, 10, 30, 18, 0, tzinfo=JST), "2026-10-30"),
        (datetime(2026, 10, 30, 10, 0, tzinfo=JST), "2026-10-29"),
        (datetime(2026, 10, 31, 12, 0, tzinfo=JST), "2026-10-30"),  # 土曜
        (datetime(2026, 11, 3, 8, 0, tzinfo=JST), "2026-11-02"),  # 文化の日
    ],
)
def test_target_session(now: datetime, expected: str) -> None:
    assert target_session(now) == pd.Timestamp(expected)


def _jpx_raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "日付": ["20261001"] * 4,
            "コード": ["7203", "130A", "1306", "9999"],
            "銘柄名": ["トヨタ自動車", "新コード銘柄", "TOPIX連動型上場投資信託", "小型株"],
            "33業種区分": ["輸送用機器", "情報・通信業", "-", "小売業"],
            "規模区分": ["TOPIX Core30", "TOPIX Large70", "-", "TOPIX Small 1"],
        }
    )


def test_parse_jpx_listing_filters_tiers() -> None:
    df = parse_jpx_listing(_jpx_raw(), ["TOPIX Core30", "TOPIX Large70"], date(2026, 10, 1))
    assert list(df["ticker"]) == ["130A.T", "7203.T"]
    assert to_ticker("7203.0") == "7203.T"
    with pytest.raises(UniverseError):
        parse_jpx_listing(_jpx_raw().drop(columns=["規模区分"]), ["TOPIX Core30"], date(2026, 10, 1))


def test_ensure_universe_refresh_and_fallback(tmp_path) -> None:
    path = tmp_path / "universe.csv"
    tiers = ["TOPIX Core30"]

    def broken() -> pd.DataFrame:
        raise OSError("network down")

    with pytest.raises(UniverseError):
        ensure_universe(path, tiers, 40, date(2026, 10, 1), broken)

    df, warnings = ensure_universe(path, tiers, 40, date(2026, 10, 1), _jpx_raw)
    assert path.exists() and list(df["ticker"]) == ["7203.T"] and warnings == []

    calls = []
    df, warnings = ensure_universe(path, tiers, 40, date(2026, 10, 20), lambda: calls.append(1) or _jpx_raw())
    assert calls == []  # まだ新しいので取り直さない

    df, warnings = ensure_universe(path, tiers, 40, date(2026, 12, 31), broken)
    assert list(df["ticker"]) == ["7203.T"]
    assert warnings and "前回の一覧" in warnings[0]
