"""過去データでの検証。毎日の本番と同じ engine.step を営業日ごとに呼ぶ。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from trader import engine
from trader.config import Settings
from trader.market import MarketData
from trader.metrics import money_weighted_benchmark, nav_metrics, trade_stats, yearly_returns
from trader.portfolio import PortfolioState

REFERENCE_CAPITAL = 10_000_000


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    settings: Settings


@dataclass
class BacktestResult:
    scenario: Scenario
    history: pd.DataFrame
    state: PortfolioState
    skipped: Counter = field(default_factory=Counter)

    @property
    def nav(self) -> pd.Series:
        return self.history["nav"]

    @property
    def bench(self) -> pd.Series:
        return self.history["bench"]

    @property
    def metrics(self) -> dict[str, float]:
        return nav_metrics(self.nav)

    @property
    def bench_metrics(self) -> dict[str, float]:
        return nav_metrics(self.bench)

    @property
    def exposure(self) -> float:
        h = self.history
        return float((1 - h["cash"] / h["equity"]).mean())


def warmup_sessions(settings: Settings) -> int:
    s = settings.strategy
    return max(s.momentum_lookback, s.trend_window) + 5


def run_backtest(
    md: MarketData,
    universe: pd.DataFrame,
    scenario: Scenario,
    start: date | None = None,
    end: date | None = None,
) -> BacktestResult:
    settings = scenario.settings
    sessions = md.sessions
    earliest = sessions[min(warmup_sessions(settings), len(sessions) - 1)]
    bench_trend = md.signals(settings.strategy).trend.get(settings.strategy.benchmark_ticker)
    if bench_trend is not None and bench_trend.notna().any():
        earliest = max(earliest, bench_trend.first_valid_index())  # TOPIX のデータが揃ってから始める
    first = max(pd.Timestamp(start), earliest) if start else earliest
    mask = sessions >= first
    if end:
        mask &= sessions <= pd.Timestamp(end)
    sessions = sessions[mask]
    if len(sessions) < 2:
        raise ValueError("検証期間が短すぎます")
    state = PortfolioState.new(
        sessions[0].date().isoformat(), settings.satellite.initial_capital, settings.core.initial_amount, settings.mode
    )
    skipped: Counter = Counter()
    for session in sessions:
        result = engine.step(state, md, session, settings, universe, apply_splits=False)
        skipped.update(reason for _, reason in result.skipped)
    history = pd.DataFrame(state.history)
    history.index = pd.DatetimeIndex(history.pop("date"))
    return BacktestResult(scenario, history, state, skipped)


def equal_weight_index(md: MarketData, tickers: list[str], index: pd.DatetimeIndex) -> pd.Series:
    """同じ銘柄群を毎日均等に持った場合の指数（コスト無し、参考値）。"""
    cols = [t for t in tickers if t in md.adj_close.columns and t not in md.quality.unreliable]
    returns = md.adj_close[cols].pct_change(fill_method=None).loc[index].mean(axis=1, skipna=True)
    returns.iloc[0] = 0.0
    return (1 + returns.fillna(0.0)).cumprod()


def default_scenarios(settings: Settings) -> list[Scenario]:
    reference = settings.with_satellite(initial_capital=REFERENCE_CAPITAL, monthly_contribution=0)
    return [
        Scenario("reference", "資金の影響を除いた参考（1,000万円で開始・追加入金なし）", reference),
        Scenario("plan", "あなたの資金計画（設定ファイルの開始額・毎月の入金）", settings),
    ]


def sensitivity_scenarios(settings: Settings) -> list[Scenario]:
    base = settings.with_satellite(initial_capital=REFERENCE_CAPITAL, monthly_contribution=0)
    out = []
    for positions in (3, 4, 6, 8):
        for stop in (0.10, 0.15, 0.20, None):
            keep = max(base.strategy.keep_rank, positions * 2)
            variant = base.with_strategy(max_positions=positions, trailing_stop=stop, keep_rank=keep)
            label = f"保有{positions}銘柄・損切り{'なし' if stop is None else f'{stop:.0%}'}"
            out.append(Scenario(f"p{positions}_s{stop}", label, variant))
    out.append(Scenario("no_regime", "TOPIXによる相場判断なし", base.with_strategy(regime_filter=False)))
    out.append(Scenario("no_dd", "最大下落ルールなし", base.with_strategy(max_drawdown=None)))
    return out


# ---------------------------------------------------------------- 表示


def _pct(value: float | None, signed: bool = True) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:+.1%}" if signed else f"{value:.1%}"


def _num(value: float | None) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:.2f}"


def _yen(value: float) -> str:
    return f"{value:,.0f}円"


def _metric_rows(columns: list[tuple[str, dict[str, float]]]) -> list[str]:
    header = "| | " + " | ".join(name for name, _ in columns) + " |"
    sep = "|---|" + "---|" * len(columns)
    rows = [
        ("年率リターン", "cagr", True),
        ("累積リターン", "total_return", True),
        ("最大下落（最高値から）", "max_drawdown", True),
        ("値動きの大きさ（年率）", "volatility", False),
    ]
    lines = [header, sep]
    for label, key, signed in rows:
        lines.append(f"| {label} | " + " | ".join(_pct(m.get(key), signed) for _, m in columns) + " |")
    lines.append("| シャープレシオ（高いほど効率的） | " + " | ".join(_num(m.get("sharpe")) for _, m in columns) + " |")
    return lines


def _split_halves(series: pd.Series) -> list[tuple[str, pd.Series]]:
    mid = series.index[len(series) // 2]
    first, second = series.loc[:mid], series.loc[mid:]
    return [
        (f"前半 {first.index[0].date()}〜{first.index[-1].date()}", first),
        (f"後半 {second.index[0].date()}〜{second.index[-1].date()}", second),
    ]


def _verdict(ref: BacktestResult) -> list[str]:
    m, b = ref.metrics, ref.bench_metrics
    diff = m["cagr"] - b["cagr"]
    halves = [
        nav_metrics(part)["cagr"] - nav_metrics(ref.bench.loc[part.index])["cagr"]
        for _, part in _split_halves(ref.nav)
    ]
    lines = [
        f"- 年率リターンは TOPIX より **{diff:+.1%}ポイント**"
        f"（前半 {halves[0]:+.1%} / 後半 {halves[1]:+.1%}）。",
        f"- 最大下落は戦略 {m['max_drawdown']:.1%}、TOPIX {b['max_drawdown']:.1%}。",
    ]
    if diff > 0 and min(halves) > 0:
        lines.append("- 前半・後半とも TOPIX を上回りました。ただし生存者バイアスの分だけ割り引いて見てください。")
    elif diff > 0:
        lines.append("- 全期間では上回りましたが、片方の期間では下回っており、安定しているとは言えません。")
    else:
        lines.append(
            "- TOPIX を下回りました。銘柄売買に回す額を増やさず、積立中心にすることを優先してください。"
        )
    return lines


def _quality_lines(md: MarketData, universe: pd.DataFrame, settings: Settings) -> list[str]:
    names = dict(zip(universe["ticker"], universe["name"], strict=True))
    bench = settings.strategy.benchmark_ticker
    first = md.adj_close[bench].first_valid_index() if bench in md.adj_close else None
    q = md.quality
    lines = [
        "## データ品質",
        "",
        f"- TOPIX（{bench}）のデータ開始日: {first.date() if first is not None else 'なし'}"
        "（検証はこの日から200営業日以上たってから始めています）",
        f"- 株式分割が未調整だった値動きを補正: {len(q.repairs)} 件",
    ]
    for r in q.repairs[:10]:
        label = names.get(r.ticker, r.ticker)
        lines.append(f"  - {label}（{r.ticker}）{r.date.date()} 株数 {r.factor:g} 倍")
    if q.unresolved:
        lines.append(f"- 分割で説明できない不自然な値動きがあり、判断から外した銘柄: {len(q.unresolved)} 件")
        for ticker, days in list(q.unresolved.items())[:10]:
            lines.append(f"  - {names.get(ticker, ticker)}（{ticker}）{', '.join(str(d.date()) for d in days[:3])}")
    else:
        lines.append("- 分割で説明できない不自然な値動き: なし")
    return lines + [""]


def render_report(
    results: list[BacktestResult],
    sensitivity: list[BacktestResult],
    md: MarketData,
    universe: pd.DataFrame,
    settings: Settings,
    created: date,
) -> str:
    ref = next(r for r in results if r.scenario.key == "reference")
    plan = next((r for r in results if r.scenario.key == "plan"), None)
    s = settings.strategy
    ew = equal_weight_index(md, list(universe["ticker"]), ref.nav.index)
    m = ref.metrics
    lines = [
        f"# バックテスト結果（{created.isoformat()} 作成）",
        "",
        "> 過去の株価でルールを機械的に再現した結果です。将来の利益を保証するものではありません。",
        "",
        "## 前提",
        "",
        f"- 期間: {m['start']} 〜 {m['end']}（約{m['years']:.1f}年）",
        f"- 対象: 現在の {' + '.join(settings.universe.tiers)} の {len(universe)} 銘柄",
        "- **生存者バイアスあり**: 現在まで大型株として残った銘柄だけで検証しているため、"
        "結果は実際より良く出やすい。下の「同じ銘柄群の均等保有」も同じバイアスを含むので、"
        "戦略の銘柄選びの効果はこの列との差で見るのが公平です。",
        f"- ルール: 月末に、長期トレンド（{s.trend_window}日平均）より上で、過去12か月（直近1か月除く）の"
        f"上昇率が高い順に最大 {s.max_positions} 銘柄（同業種 {s.max_per_sector} 銘柄まで）を保有。"
        f"順位が {s.keep_rank} 位より下がったら入れ替え。"
        f"保有中の最高値から {_pct(s.trailing_stop, False) if s.trailing_stop else 'なし'} 下落で売却。"
        f"TOPIX が {s.trend_window} 日平均より下なら全て売却。"
        f"資金が最高時から {_pct(s.max_drawdown, False) if s.max_drawdown else 'なし'} 減ったら全て売って"
        f" {s.drawdown_pause_months} か月休止（休止明けは基準をリセットするため、通算の最大下落はこの値を超え得る）。",
        f"- 約定: 判断の翌営業日の始値。片道コスト {settings.satellite.cost_rate:.2%}、"
        f"税率 {settings.satellite.tax_rate:.3%}。配当は受け取り扱い。",
        "- ルールの数値は検証前に決めたもので、この結果を見て調整していません。",
        "",
        *_quality_lines(md, universe, settings),
        f"## 1. 主な結果 — {ref.scenario.title}",
        "",
        *_metric_rows([("戦略", m), ("TOPIX（1306）", ref.bench_metrics), ("同じ銘柄群の均等保有", nav_metrics(ew))]),
        "",
        f"株を持っていた期間の割合（平均）: {ref.exposure:.0%}",
        "",
        "## 2. 期間を前半・後半に分けた結果",
        "",
        "| 期間 | 戦略 年率 | TOPIX 年率 | 戦略 最大下落 | TOPIX 最大下落 |",
        "|---|---|---|---|---|",
    ]
    for label, part in _split_halves(ref.nav):
        pm, bm = nav_metrics(part), nav_metrics(ref.bench.loc[part.index])
        lines.append(
            f"| {label} | {_pct(pm['cagr'])} | {_pct(bm['cagr'])} | "
            f"{_pct(pm['max_drawdown'])} | {_pct(bm['max_drawdown'])} |"
        )
    lines += ["", "## 3. 年ごとの成績", "", "| 年 | 戦略 | TOPIX | 差 |", "|---|---|---|---|"]
    strat_y, bench_y = yearly_returns(ref.nav), yearly_returns(ref.bench)
    for year in strat_y.index:
        lines.append(
            f"| {year} | {_pct(strat_y[year])} | {_pct(bench_y[year])} | {_pct(strat_y[year] - bench_y[year])} |"
        )
    if plan is not None:
        h = plan.history
        flows = h["flow"]
        mw = money_weighted_benchmark(plan.bench, flows, plan.scenario.settings.satellite.initial_capital)
        lines += [
            "",
            f"## 4. {plan.scenario.title}",
            "",
            f"- 入金累計 {_yen(plan.state.total_contributed)} → 最終評価額 **{_yen(h['equity'].iloc[-1])}**",
            f"- 同じ日に同じ額を TOPIX ETF に入れていた場合: {_yen(mw.iloc[-1])}",
            f"- 最大下落（入金の影響を除く）: {_pct(plan.metrics['max_drawdown'])}",
            f"- 1株の価格が予算を超えて買えなかった回数: {plan.skipped.get('too_expensive', 0)}"
            f"（同業種の上限で見送った回数: {plan.skipped.get('sector_limit', 0)}）",
            "- 少額のうちは1株単位の端数で現金が余りやすく、参考シナリオより成績がぶれます。",
        ]
    stats = trade_stats(ref.state.trades)
    lines += ["", "## 5. 売買の内訳（参考シナリオ）", ""]
    if stats["count"]:
        lines += [
            f"- 売買回数: {stats['count']} 回（年あたり約 {stats['count'] / m['years']:.0f} 回）",
            f"- 勝率: {stats['win_rate']:.0%}、1回あたり平均 {_pct(stats['avg_return'])}"
            f"（中央値 {_pct(stats['median_return'])}、最大 {_pct(stats['best'])}、最悪 {_pct(stats['worst'])}）",
            f"- 平均保有期間: {stats['avg_holding_sessions']:.0f} 営業日",
            f"- 利益合計 ÷ 損失合計: {_num(stats['profit_factor'])}",
            "- 売った理由: "
            + "、".join(
                f"{name} {int(stats.get(f'exit_{code}', 0))}回"
                for code, name in [
                    ("trailing_stop", "損切り/利益確定"),
                    ("rank_dropped", "順位低下"),
                    ("below_trend", "トレンド割れ"),
                    ("market_downtrend", "相場全体の下落"),
                    ("drawdown_limit", "最大下落ルール"),
                ]
            ),
        ]
    else:
        lines.append("- 売買はありませんでした。")
    if sensitivity:
        lines += [
            "",
            "## 6. ルールを少し変えた場合（感度分析・参考）",
            "",
            "既定値は事前に決めたものです。この表から一番良い組み合わせを選び直すと、"
            "過去にだけ合ったルールになりやすいので、**値が近い範囲で安定しているか**を見るために使います。",
            "",
            "| 条件 | 年率 | 最大下落 | シャープ | TOPIXとの差（年率） |",
            "|---|---|---|---|---|",
        ]
        for r in sensitivity:
            rm, rb = r.metrics, r.bench_metrics
            lines.append(
                f"| {r.scenario.title} | {_pct(rm['cagr'])} | {_pct(rm['max_drawdown'])} | "
                f"{_num(rm['sharpe'])} | {_pct(rm['cagr'] - rb['cagr'])} |"
            )
    lines += ["", "## 7. まとめ", "", *_verdict(ref), ""]
    return "\n".join(lines)
