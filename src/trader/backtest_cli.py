"""バックテストを実行してレポート（Markdown）を書き出す。"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

from trader.backtest import default_scenarios, render_report, run_backtest, sensitivity_scenarios, summarize
from trader.config import DEFAULT_CONFIG_PATH, load_settings
from trader.daily import DEFAULT_UNIVERSE, ROOT
from trader.market import download_market_data
from trader.universe import ensure_universe

log = logging.getLogger("trader.backtest")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="過去データでの検証")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2006, 1, 1),
                        help="株価データの取得開始日（検証はここから約1年後に始まる）")
    parser.add_argument("--end", type=date.fromisoformat, default=None)
    parser.add_argument("--tiers", nargs="+", default=None, help="対象の規模区分（例: 'TOPIX Mid400'）")
    parser.add_argument("--sensitivity", action="store_true", help="ルールを少し変えた場合も計算する")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "backtest_latest.md")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    settings = load_settings(args.config)
    today = date.today()
    if args.tiers:
        settings = replace(settings, universe=replace(settings.universe, tiers=tuple(args.tiers)))
        universe_path = ROOT / ".cache" / "universe_custom.csv"
        universe_path.unlink(missing_ok=True)
    else:
        universe_path = DEFAULT_UNIVERSE
    universe, warnings = ensure_universe(
        universe_path, settings.universe.tiers, settings.universe.max_age_days, today
    )
    for warning in warnings:
        log.warning(warning)
    tickers = set(universe["ticker"]) | {settings.strategy.benchmark_ticker}
    if settings.core.proxy_ticker:
        tickers.add(settings.core.proxy_ticker)
    log.info("%d 銘柄の株価を取得します（%s〜）", len(tickers), args.start)
    md = download_market_data(sorted(tickers), args.start, args.end)
    usable = [t for t in universe["ticker"] if t in md.tickers and md.close[t].notna().any()]
    log.info("株価を取得できた銘柄: %d/%d", len(usable), len(universe))
    bench = settings.strategy.benchmark_ticker
    log.info("%s のデータ開始日: %s", bench, md.adj_close[bench].first_valid_index())
    universe = universe[universe["ticker"].isin(usable)].reset_index(drop=True)

    start = args.start + timedelta(days=1)
    results = [run_backtest(md, universe, s, start, args.end) for s in default_scenarios(settings)]
    sensitivity = []
    if args.sensitivity:
        sensitivity = [run_backtest(md, universe, s, start, args.end) for s in sensitivity_scenarios(settings)]
    report = render_report(results, sensitivity, md, universe, settings, today)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report + "\n", encoding="utf-8")
    summary_path = args.out.with_name("backtest_summary.json")
    summary_path.write_text(
        json.dumps(summarize(results, md, universe, today), ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(report)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(report + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
