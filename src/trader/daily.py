"""毎営業日の夕方に実行する本番処理。"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Callable
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from trader import calendar as tse
from trader import engine
from trader.config import DEFAULT_CONFIG_PATH, Settings, load_settings
from trader.market import DataNotReadyError, MarketData, download_market_data, require_session
from trader.notify import NotifyError, send_discord
from trader.portfolio import PortfolioState, load_state, save_state
from trader.report import render_error, render_notification
from trader.universe import UniverseError, ensure_universe, fetch_jpx_listing

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE = ROOT / "state" / "portfolio.json"
DEFAULT_UNIVERSE = ROOT / "data" / "universe.csv"
JST = ZoneInfo("Asia/Tokyo")
CLOSE_READY = time(16, 30)  # 東証の終値（15:30）が無料データに反映されるまでの余裕を見た時刻
HISTORY_DAYS = 800  # 12か月の上昇率と200日平均に必要な期間 + 余裕
MAX_CATCH_UP_SESSIONS = 10
MAX_MISSING_RATIO = 0.2
EXIT_NOTIFIED = 3  # 失敗を Discord へ通知済み（workflow 側で重ねて通知しない）

log = logging.getLogger("trader.daily")

Downloader = Callable[[list[str], date, date], MarketData]
Sender = Callable[[str, str], None]


def target_session(now: datetime) -> pd.Timestamp:
    """終値が確定している最新の営業日。"""
    now = now.astimezone(JST)
    today = pd.Timestamp(now.date())
    if tse.is_session(today):
        return today if now.time() >= CLOSE_READY else tse.previous_session(today)
    return tse.session_on_or_before(today)


def _fail(title: str, detail: str, webhook: str | None, settings: Settings | None, sender: Sender,
          dry_run: bool) -> int:
    print(f"ERROR: {title}: {detail}", file=sys.stderr)
    if webhook and not dry_run:
        try:
            sender(webhook, render_error(title, detail, settings))
            return EXIT_NOTIFIED
        except NotifyError as exc:
            print(f"ERROR: エラー通知も送れませんでした: {exc}", file=sys.stderr)
    return 1


def run(
    *,
    settings: Settings,
    now: datetime,
    state_path: Path = DEFAULT_STATE,
    universe_path: Path = DEFAULT_UNIVERSE,
    webhook: str | None = None,
    dry_run: bool = False,
    final_attempt: bool = False,
    downloader: Downloader = download_market_data,
    fetch_listing: Callable[[], pd.DataFrame] = fetch_jpx_listing,
    sender: Sender = send_discord,
) -> int:
    if not webhook and not dry_run:
        print("DISCORD_WEBHOOK_URL が未設定のため、何もせず終了します（設定すると次回から動き始めます）")
        return 0
    target = target_session(now)
    state = load_state(state_path)

    def fail(title: str, detail: str) -> int:
        return _fail(title, detail, webhook, settings, sender, dry_run)

    archive: tuple[Path, PortfolioState] | None = None
    if state is not None and state.mode != settings.mode:
        if state.mode == "live":
            return fail(
                "実運用の記録があるため paper に戻せません",
                "戻す場合は state/portfolio.json を別名に移してから実行してください",
            )
        archive = (state_path.with_name(f"paper_until_{state.last_session}.json"), state)
        state = None  # 模擬運用の記録は保存し、実運用を新しく始める
    if state is not None and state.last_session and pd.Timestamp(state.last_session) >= target:
        print(f"{target.date()} は処理済みです")
        return 0

    try:
        universe, warnings = ensure_universe(
            universe_path, settings.universe.tiers, settings.universe.max_age_days, target.date(), fetch_listing
        )
    except UniverseError as exc:
        return fail("売買対象の銘柄一覧を用意できません", str(exc))

    held = sorted(set(state.positions) | {o.ticker for o in state.pending}) if state else []
    extra = [settings.strategy.benchmark_ticker, settings.core.proxy_ticker, *held]
    tickers = sorted(set(universe["ticker"]) | {t for t in extra if t})
    try:
        md = downloader(tickers, (target - timedelta(days=HISTORY_DAYS)).date(), target.date())
        missing = require_session(md, target, [settings.strategy.benchmark_ticker])
    except DataNotReadyError as exc:
        if final_attempt:
            return fail("株価データがまだ揃っていません", str(exc))
        print(f"{exc}。次の実行で再試行します")
        return 0
    except Exception as exc:  # noqa: BLE001 - 原因の種類を残して通知する
        return fail("株価データの取得に失敗しました", f"{type(exc).__name__}: {exc}")

    missing_universe = [t for t in universe["ticker"] if t in set(missing)]
    if len(missing_universe) > MAX_MISSING_RATIO * len(universe):
        return fail(
            "当日の株価が取れない銘柄が多すぎます",
            f"{len(missing_universe)}/{len(universe)} 銘柄が欠損。データ提供元の不調の可能性があります",
        )
    if missing_universe:
        warnings.append(f"当日の株価が取れなかった {len(missing_universe)} 銘柄は今回の判断から外しました")

    if archive is not None:
        warnings.append(f"模擬運用の記録を {archive[0].name} に保存し、実運用を新しく始めました")
    if state is None:
        state = PortfolioState.new(
            target.date().isoformat(),
            settings.satellite.initial_capital,
            settings.core.initial_amount,
            settings.mode,
        )
        sessions = pd.DatetimeIndex([target])
    else:
        last = pd.Timestamp(state.last_session)
        sessions = md.sessions[(md.sessions > last) & (md.sessions <= target)]
        if len(sessions) > MAX_CATCH_UP_SESSIONS:
            return fail(
                "前回の実行から日数が空きすぎています",
                f"前回 {last.date()}、未処理 {len(sessions)} 営業日。"
                "自動で取り戻すと実際の売買とずれるため停止しました",
            )
        if len(sessions) == 0:
            print("処理する営業日がありません")
            return 0

    results = [engine.step(state, md, s, settings, universe, apply_splits=True) for s in sessions]
    names = dict(zip(universe["ticker"], universe["name"], strict=True))
    message = render_notification(results, state, settings, names, warnings)
    if dry_run:
        print(message or "（今日は通知する内容がありません）")
        return 0
    if archive is not None:
        save_state(archive[1], archive[0])
    save_state(state, state_path)
    if message:
        try:
            sender(webhook or "", message)
        except NotifyError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    last = results[-1]
    print(f"{last.session}: 注文 {len(last.orders)} 件、約定 {sum(len(r.fills) for r in results)} 件、"
          f"評価額 {last.equity:,.0f}円、通知 {'あり' if message else 'なし'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="毎営業日の売買判断と Discord 通知")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--dry-run", action="store_true", help="保存も通知もせず、通知文を表示するだけ")
    parser.add_argument("--final-attempt", action="store_true", help="データ未確定でもエラー通知する最後の実行")
    parser.add_argument("--now", help="動作確認用の現在時刻（例: 2026-10-30T18:00:00+09:00）")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    now = datetime.fromisoformat(args.now) if args.now else datetime.now(JST)
    return run(
        settings=load_settings(args.config),
        now=now,
        state_path=args.state,
        universe_path=args.universe,
        webhook=os.environ.get("DISCORD_WEBHOOK_URL") or None,
        dry_run=args.dry_run,
        final_attempt=args.final_attempt,
    )


if __name__ == "__main__":
    raise SystemExit(main())
