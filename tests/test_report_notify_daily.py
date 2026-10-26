from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from conftest import BENCH, PROXY, market_from_arrays

from trader import calendar as tse
from trader.config import Settings
from trader.daily import run
from trader.engine import StepResult
from trader.market import DataNotReadyError
from trader.notify import NotifyError, send_discord, split_message
from trader.portfolio import Order, PortfolioState
from trader.report import jp_date, render_notification

JST = ZoneInfo("Asia/Tokyo")
NOW = datetime(2026, 10, 30, 18, 0, tzinfo=JST)  # 金曜・月末


def test_jp_date() -> None:
    assert jp_date("2026-11-02") == "11/2(月)"


def test_render_notification_none_on_quiet_day() -> None:
    state = PortfolioState.new("2026-10-01", 50_000)
    state.session_count = 5
    res = StepResult(session="2026-10-28", equity=50_000, cash=50_000)
    assert render_notification([res], state, Settings(), {}) is None


def test_render_notification_orders_are_explicit() -> None:
    state = PortfolioState.new("2026-10-01", 50_000)
    state.session_count = 5
    order = Order("buy", "7203.T", 4, "momentum", "2026-10-28", 3_100, 1, 0.45)
    sell = Order("sell", "6758.T", 3, "trailing_stop", "2026-10-28", 2_000)
    res = StepResult(session="2026-10-28", orders=[sell, order], equity=50_000, cash=50_000)
    text = render_notification([res], state, Settings(), {"7203.T": "トヨタ自動車"})
    assert text is not None
    assert "【模擬運用】" in text
    assert "10/29(木) の朝9:00" in text
    assert "7203 トヨタ自動車 **4株**" in text
    assert "S株" in text
    assert "最高値から15%下がった" in text
    assert "投資の助言ではありません" in text


def test_split_message_respects_limit() -> None:
    text = "\n".join(f"行{i} " + "あ" * 50 for i in range(200)) + "\n" + "い" * 5000
    chunks = split_message(text, limit=500)
    assert all(len(c) <= 500 for c in chunks)
    assert "".join(c.replace("\n", "") for c in chunks) == text.replace("\n", "")


class FakeResponse:
    def __init__(self, status: int, headers: dict[str, str] | None = None) -> None:
        self.status_code = status
        self.headers = headers or {}


class FakeSession:
    def __init__(self, statuses: list[int]) -> None:
        self.statuses = statuses
        self.posts: list[dict] = []

    def post(self, url: str, json: dict, timeout: float) -> FakeResponse:
        self.posts.append(json)
        return FakeResponse(self.statuses.pop(0), {"Retry-After": "0"})


def test_send_discord_retries_rate_limit_and_hides_url() -> None:
    session = FakeSession([429, 204])
    send_discord("https://discord.example/secret-token", "hello", session=session)  # type: ignore[arg-type]
    assert len(session.posts) == 2
    with pytest.raises(NotifyError) as err:
        send_discord("https://discord.example/secret-token", "x", session=FakeSession([400]))  # type: ignore[arg-type]
    assert "secret-token" not in str(err.value)


# ---------------------------------------------------------------- daily.run


def _jpx() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "コード": ["1001", "1002", "1003"],
            "銘柄名": ["銘柄A", "銘柄B", "銘柄C"],
            "33業種区分": ["機械", "小売業", "銀行業"],
            "規模区分": ["TOPIX Core30", "TOPIX Large70", "TOPIX Core30"],
        }
    )


def _downloader(end: str = "2026-10-30"):
    idx = tse.sessions_between("2024-09-02", end)
    n = len(idx)
    close = {
        "1001.T": np.geomspace(1000, 3000, n),
        "1002.T": np.geomspace(1000, 2000, n),
        "1003.T": np.geomspace(2000, 1000, n),
        BENCH: np.geomspace(1000, 1300, n),
        PROXY: np.geomspace(1000, 1200, n),
    }
    md = market_from_arrays(idx, close)
    calls: list[list[str]] = []

    def download(tickers, start, stop):
        calls.append(list(tickers))
        return md

    return download, calls


def test_daily_run_without_webhook_keeps_paper_running_but_refuses_live(tmp_path) -> None:
    from dataclasses import replace

    download, _ = _downloader()
    kwargs = dict(now=NOW, universe_path=tmp_path / "u.csv", downloader=download, fetch_listing=_jpx)
    assert run(settings=Settings(), state_path=tmp_path / "paper.json", **kwargs) == 0
    assert (tmp_path / "paper.json").exists()  # 通知なしでも模擬運用の記録は進む
    live = replace(Settings(), mode="live")
    assert run(settings=live, state_path=tmp_path / "live.json", **kwargs) == 1
    assert not (tmp_path / "live.json").exists()


def test_daily_run_first_run_then_idempotent(tmp_path) -> None:
    sent: list[str] = []
    download, calls = _downloader()
    kwargs = dict(
        settings=Settings(),
        now=NOW,
        state_path=tmp_path / "state.json",
        universe_path=tmp_path / "universe.csv",
        webhook="https://discord.example/hook",
        downloader=download,
        fetch_listing=_jpx,
        sender=lambda url, text: sent.append(text),
    )
    assert run(**kwargs) == 0
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["last_session"] == "2026-10-30"
    assert {o["ticker"] for o in state["pending"]} == {"1001.T", "1002.T"}
    assert len(sent) == 1 and "運用を開始しました" in sent[0] and "11/2(月)" in sent[0]
    assert run(**kwargs) == 0  # 同じ日の再実行は何もしない
    assert len(sent) == 1 and len(calls) == 1


def test_daily_run_data_not_ready_retries_then_fails_on_final_attempt(tmp_path) -> None:
    sent: list[str] = []
    download, _ = _downloader(end="2026-10-29")
    kwargs = dict(
        settings=Settings(),
        now=NOW,
        state_path=tmp_path / "state.json",
        universe_path=tmp_path / "universe.csv",
        webhook="https://discord.example/hook",
        downloader=download,
        fetch_listing=_jpx,
        sender=lambda url, text: sent.append(text),
    )
    assert run(**kwargs) == 0 and sent == [] and not (tmp_path / "state.json").exists()
    assert run(**kwargs, final_attempt=True) == 3  # 通知済みの失敗
    assert len(sent) == 1 and "エラー" in sent[0]


def test_daily_run_notify_failure_returns_error(tmp_path) -> None:
    download, _ = _downloader()

    def broken(url: str, text: str) -> None:
        raise NotifyError("Discord が送信を拒否しました（HTTP 401）")

    code = run(
        settings=Settings(),
        now=NOW,
        state_path=tmp_path / "state.json",
        universe_path=tmp_path / "universe.csv",
        webhook="https://discord.example/hook",
        downloader=download,
        fetch_listing=_jpx,
        sender=broken,
    )
    assert code == 1


def test_data_not_ready_error_is_distinct() -> None:
    assert not issubclass(DataNotReadyError, NotifyError)


def test_switching_paper_to_live_archives_and_starts_fresh(tmp_path) -> None:
    from dataclasses import replace

    sent: list[str] = []
    download, _ = _downloader("2026-11-02")
    kwargs = dict(
        state_path=tmp_path / "state.json",
        universe_path=tmp_path / "universe.csv",
        webhook="https://discord.example/hook",
        downloader=download,
        fetch_listing=_jpx,
        sender=lambda url, text: sent.append(text),
    )
    assert run(settings=Settings(), now=NOW, **kwargs) == 0
    live = replace(Settings(), mode="live")
    later = datetime(2026, 11, 2, 18, 0, tzinfo=JST)
    assert run(settings=live, now=later, **kwargs) == 0
    archived = json.loads((tmp_path / "paper_until_2026-10-30.json").read_text(encoding="utf-8"))
    current = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert archived["mode"] == "paper" and current["mode"] == "live"
    assert current["start_date"] == "2026-11-02" and current["positions"] == {}
    assert "実運用を新しく始めました" in sent[-1] and "【模擬運用】" not in sent[-1]
    # 実運用から paper へは戻さない
    assert run(settings=Settings(), now=datetime(2026, 11, 2, 19, 0, tzinfo=JST), **kwargs) == 3


def test_dashboard_renders_with_and_without_state() -> None:
    from trader.dashboard import _nice_ticks, build_html

    now = datetime(2026, 10, 30, 18, 0, tzinfo=JST)
    empty = build_html(None, Settings(), None, None, now, "owner/repo", "main")
    assert "まだ運用が始まっていません" in empty
    assert "https://github.com/owner/repo/edit/main/config/settings.yaml" in empty

    state = PortfolioState.new("2026-10-01", 50_000, 200_000)
    state.last_session = "2026-10-30"
    state.history = [
        {"date": "2026-10-29", "nav": 1.0, "bench": 1.0, "equity": 50_000, "cash": 50_000, "core": 200_000, "flow": 0},
        {"date": "2026-10-30", "nav": 1.05, "bench": 1.02, "equity": 52_500, "cash": 0, "core": 201_000, "flow": 0},
    ]
    state.pending = [Order("buy", "7203.T", 4, "momentum", "2026-10-30", 3_100, 1, 0.45)]
    names = pd.DataFrame({"ticker": ["7203.T"], "name": ["<script>alert(1)</script>"]})
    html = build_html(state, Settings(), names, None, now)
    assert "<script>alert(1)</script>" not in html  # 銘柄名はエスケープする
    assert "11/2(月)" in html and "4株" in html and "<svg" in html
    assert _nice_ticks(-0.12, 0.54) == pytest.approx([-0.2, 0.0, 0.2, 0.4, 0.6])
