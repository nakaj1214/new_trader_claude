"""運用状況と設定を確認する Web ページ（GitHub Pages 用の静的 HTML）を作る。

外部のライブラリや通信を使わない1ファイルの HTML にする（グラフも SVG で埋め込む）。
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from trader import calendar as tse
from trader.config import DEFAULT_CONFIG_PATH, Settings, load_settings
from trader.daily import DEFAULT_STATE, DEFAULT_UNIVERSE, ROOT
from trader.portfolio import PortfolioState, load_state
from trader.report import buy_reason, jp_date, months_elapsed, sell_reason, sell_reason_text, stock_label
from trader.universe import load_universe

JST = ZoneInfo("Asia/Tokyo")
DEFAULT_BACKTEST_SUMMARY = ROOT / "reports" / "backtest_summary.json"
RECENT_TRADES = 20


def _yen(value: float | None) -> str:
    return "—" if value is None else f"{value:,.0f}円"


def _pct(value: float | None, signed: bool = True) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:+.1%}" if signed else f"{value:.1%}"


def _tone(value: float | None) -> str:
    if value is None or pd.isna(value) or abs(value) < 1e-12:
        return ""
    return "pos" if value > 0 else "neg"


def _signed(text: str, value: float | None) -> str:
    return f'<span class="{_tone(value)}">{escape(text)}</span>'


def _table(
    headers: list[str], rows: list[list[str]], empty: str, cards: bool = True, wrap: tuple[int, ...] = ()
) -> str:
    """表を作る。cards=True ならスマホ幅では1行ずつのカード表示になる。wrap の列は折り返す。"""
    if not rows:
        return f'<p class="muted">{escape(empty)}</p>'
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    body = ""
    for row in rows:
        cells = ""
        for i, cell in enumerate(row):
            label = f' data-label="{escape(headers[i])}"' if headers[i] else ""
            cls = ' class="wrap"' if i in wrap else ""
            cells += f"<td{label}{cls}><span>{cell}</span></td>"
        body += f"<tr>{cells}</tr>"
    kind = "cards" if cards else "plain"
    table = f'<table class="{kind}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'
    return f'<div class="scroll">{table}</div>'


def _stat(label: str, value: str, note: str = "", small: bool = False) -> str:
    note_html = f'<div class="note">{note}</div>' if note else ""
    return (
        f'<div class="stat"><div class="label">{escape(label)}</div>'
        f'<div class="value{" small" if small else ""}">{value}</div>{note_html}</div>'
    )


def _nice_ticks(low: float, high: float) -> list[float]:
    for step in (0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0):
        if (high - low) / step <= 5:
            break
    start, end = math.floor(low / step) * step, math.ceil(high / step) * step
    if end - start < step / 2:
        end = start + step
    return [start + i * step for i in range(round((end - start) / step) + 1)]


def _chart(history: list[dict]) -> str:
    """銘柄売買（基準価額）と TOPIX の開始来騰落率の折れ線。軸の文字は HTML にして画面幅に関係なく読めるようにする。"""
    if len(history) < 2:
        return '<p class="muted">グラフは2営業日以上たつと表示されます。</p>'
    nav0, bench0 = history[0]["nav"], history[0]["bench"]
    strat = [h["nav"] / nav0 - 1 for h in history]
    bench = [h["bench"] / bench0 - 1 for h in history]
    ticks = _nice_ticks(min(strat + bench + [0.0]), max(strat + bench + [0.0]))
    low, high = ticks[0], ticks[-1]
    width, height = 1000, 300

    def y(v: float) -> float:
        return height * (high - v) / (high - low)

    def line(values: list[float], cls: str) -> str:
        n = len(values) - 1
        points = " ".join(f"{width * i / n:.1f},{y(v):.1f}" for i, v in enumerate(values))
        return f'<polyline class="{cls}" points="{points}" />'

    grid = "".join(f'<line class="grid" x1="0" x2="{width}" y1="{y(t):.1f}" y2="{y(t):.1f}" />' for t in ticks)
    labels = "".join(
        f'<span class="ylab" style="top:{(high - t) / (high - low) * 100:.2f}%">{t:+.0%}</span>' for t in ticks
    )
    return (
        '<div class="chart"><div class="plot">'
        f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none" role="img" '
        'aria-label="開始からの騰落率の推移">'
        + grid
        + f'<line class="zero" x1="0" x2="{width}" y1="{y(0):.1f}" y2="{y(0):.1f}" />'
        + line(bench, "bench")
        + line(strat, "strat")
        + "</svg>"
        + labels
        + "</div>"
        + f'<div class="xlabels"><span>{escape(history[0]["date"])}</span>'
        + f'<span>{escape(history[-1]["date"])}</span></div>'
        + '<div class="legend">'
        + f'<span class="key strat"></span>銘柄売買 {_signed(_pct(strat[-1]), strat[-1])}'
        + f'<span class="key bench"></span>TOPIX {_signed(_pct(bench[-1]), bench[-1])}'
        + "</div></div>"
    )


def _todo_section(state: PortfolioState | None, settings: Settings, names: dict[str, str]) -> str:
    if state is None or state.last_session is None:
        return (
            "<p>まだ運用が始まっていません。平日の夕方（17:30ごろ）の自動実行で模擬運用が始まり、"
            "このページに結果が表示されます。</p>"
        )
    next_day = jp_date(tse.next_session(state.last_session))
    if not state.pending:
        return f"<p>次の営業日（{escape(next_day)}）の注文は<strong>ありません</strong>。何もしなくて大丈夫です。</p>"
    rows = []
    for o in state.pending:
        if o.side == "sell":
            rows.append(['<span class="tag sell">売り</span>', escape(stock_label(o.ticker, names)),
                         f"{o.shares}株（全部）", escape(sell_reason(o, settings))])
        else:
            rows.append(['<span class="tag buy">買い</span>', escape(stock_label(o.ticker, names)),
                         f"<strong>{o.shares}株</strong>（目安 {_yen(o.shares * o.ref_price)}）",
                         escape(buy_reason(o))])
    label = "【模擬運用のため注文は不要】" if settings.mode == "paper" else ""
    return (
        f"<p>{escape(label)}次の営業日 <strong>{escape(next_day)}</strong> の朝9:00の取引開始前に、"
        "売り→買いの順に成行で注文します。</p>"
        + _table(["", "銘柄", "株数", "理由"], rows, "", wrap=(3,))
    )


def _overview_section(state: PortfolioState | None, settings: Settings) -> str:
    core = settings.core
    if state is None:
        stats = [
            _stat("積立（毎月）", _yen(core.monthly_amount), escape(core.fund_name)),
            _stat("銘柄売買（毎月）", _yen(settings.satellite.monthly_contribution)),
        ]
        return f'<div class="stats">{"".join(stats)}</div>'
    equity = state.equity()
    pnl = equity - state.total_contributed
    core_pnl = state.core.value - state.core.contributed
    total = state.core.value + equity
    nav, bench = state.nav() - 1, state.bench_index - 1
    drawdown = max(1 - state.nav() / state.peak_nav, 0.0) if state.peak_nav > 0 else 0.0
    core_note = "参考ETFでの推定" + ("（データ欠損のため更新停止中）" if state.core.stale else "")
    stats = [
        _stat("積立 推定評価額", _yen(state.core.value),
              f"入金 {_yen(state.core.contributed)}・損益 {_signed(f'{core_pnl:+,.0f}円', core_pnl)}・{core_note}"),
        _stat("銘柄売買 評価額", _yen(equity),
              f"入金 {_yen(state.total_contributed)}・損益 {_signed(f'{pnl:+,.0f}円', pnl)}"),
        _stat("銘柄売買の成績（開始来）", _signed(_pct(nav), nav),
              f"同じ期間のTOPIX {_signed(_pct(bench), bench)}"),
        _stat("最高時からの下落", _pct(-drawdown),
              f"上限 {_pct(settings.strategy.max_drawdown, False)}" if settings.strategy.max_drawdown else ""),
    ]
    if total > 0 and not state.core.stale:
        stats.append(_stat("配分（積立 : 銘柄売買）",
                           f"{state.core.value / total:.0%} : {equity / total:.0%}", "目安 8 : 2"))
    review = settings.review
    elapsed = months_elapsed(state, state.last_session or state.start_date)
    gap = nav - bench
    if elapsed < review.min_months:
        verdict = f"運用 {elapsed:.0f}/{review.min_months} か月（まだ判定しません）"
    elif gap < -review.underperform_threshold:
        verdict = '<span class="neg">見直しをおすすめします</span>'
    else:
        verdict = "問題なし"
    stats.append(_stat("成績の見直し判定", verdict, f"TOPIXとの差 {_signed(_pct(gap), gap)}", small=True))
    return f'<div class="stats">{"".join(stats)}</div>'


def _holdings_section(state: PortfolioState, settings: Settings, names: dict[str, str]) -> str:
    stop = settings.strategy.trailing_stop
    rows = []
    for p in state.positions.values():
        change = p.last_price / p.entry_price - 1 if p.entry_price else 0.0
        rows.append([
            escape(stock_label(p.ticker, names)),
            f"{p.shares}株",
            _yen(p.entry_price),
            _yen(p.last_price),
            _signed(_pct(change), change),
            _yen(p.high_water * (1 - stop)) if stop else "—",
            escape(p.entry_date),
        ])
    headers = ["銘柄", "株数", "買値", "現在値", "損益", "売却ライン", "買った日"]
    table = _table(headers, rows, "保有している株はありません。")
    return table + f'<p class="muted">現金 {_yen(state.cash)}（{escape(jp_date(state.last_session))} 終値時点）</p>'


def _trades_section(state: PortfolioState, settings: Settings, names: dict[str, str]) -> str:
    rows = []
    for t in reversed(state.trades[-RECENT_TRADES:]):
        rows.append([
            escape(t.exit_date),
            escape(stock_label(t.ticker, names)),
            f"{t.shares}株",
            _signed(f"{t.pnl:+,.0f}円", t.pnl),
            _signed(_pct(t.return_pct), t.return_pct),
            escape(sell_reason_text(t.exit_reason, settings)),
            f"{t.holding_sessions}日",
        ])
    headers = ["売った日", "銘柄", "株数", "損益", "損益率", "理由", "保有"]
    if not rows:
        return '<p class="muted">まだ売買の記録はありません。</p>'
    wins = sum(1 for t in state.trades if t.pnl > 0)
    total = sum(t.pnl for t in state.trades)
    summary = (
        f"これまで {len(state.trades)} 回売却・利益が出た回 {wins} 回・"
        f"損益合計 {_signed(f'{total:+,.0f}円', total)}（クリックで直近{RECENT_TRADES}件を表示）"
    )
    return f"<details><summary>{summary}</summary>{_table(headers, rows, '', wrap=(5,))}</details>"


def _settings_section(settings: Settings, repo: str | None, branch: str | None) -> str:
    s, sat, core = settings.strategy, settings.satellite, settings.core
    rows = [
        ["運用モード", "模擬運用（実際には注文しない）" if settings.mode == "paper" else "実運用"],
        ["証券会社", "SBI証券" if settings.broker == "sbi" else "楽天証券"],
        ["積立の投資信託", escape(core.fund_name)],
        ["積立 最初の額 / 毎月", f"{_yen(core.initial_amount)} / {_yen(core.monthly_amount)}"],
        ["銘柄売買 最初の額 / 毎月", f"{_yen(sat.initial_capital)} / {_yen(sat.monthly_contribution)}"],
        ["買い方", "1株から（単元未満株）" if sat.lot_size == 1 else f"{sat.lot_size}株単位"],
        ["税金", "かからない（NISA成長投資枠）" if sat.tax_rate == 0 else f"利益の {sat.tax_rate:.3%}"],
        ["取引コストの想定（片道）", _pct(sat.cost_rate, False)],
        ["対象の銘柄", escape(" + ".join(settings.universe.tiers))],
        ["同時に持つ銘柄数", f"最大 {s.max_positions} 銘柄（同じ業種は {s.max_per_sector} 銘柄まで）"],
        ["銘柄の選び方", f"過去12か月の上昇率が高い順（{s.trend_window}日平均より上の銘柄だけ）"],
        ["入れ替え", f"月末。順位が {s.keep_rank} 位以内なら持ち続ける"],
        ["損切り", f"保有中の最高値から {_pct(s.trailing_stop, False)} 下落で売る" if s.trailing_stop else "なし"],
        ["相場全体の判断", f"TOPIXが{s.trend_window}日平均より下なら全て売る" if s.regime_filter else "使わない"],
        ["損失の上限", f"最高時から {_pct(s.max_drawdown, False)} 減ったら全て売り {s.drawdown_pause_months} か月休む"
         if s.max_drawdown else "なし"],
        ["見直しの基準", f"{settings.review.min_months}か月後にTOPIXより"
         f"{settings.review.underperform_threshold:.0%}ポイント以上悪ければ"],
    ]
    table = _table(["項目", "設定値"], [[escape(k), v] for k, v in rows], "", cards=False, wrap=(1,))
    if repo and branch:
        url = f"https://github.com/{repo}/edit/{branch}/config/settings.yaml"
        table += f'<p><a class="button" href="{escape(url)}">設定を変更する（GitHub）</a></p>'
    return table


def _backtest_section(summary: dict | None, repo: str | None, branch: str | None) -> str:
    if not summary:
        return '<p class="muted">まだ実行されていません（GitHub の Actions →「バックテスト」で実行できます）。</p>'
    st, tp, ew = summary["strategy"], summary["topix"], summary["equal_weight"]
    rows = [
        ["年率リターン", _pct(st.get("cagr")), _pct(tp.get("cagr")), _pct(ew.get("cagr"))],
        ["最大下落", _pct(st.get("max_drawdown")), _pct(tp.get("max_drawdown")), _pct(ew.get("max_drawdown"))],
        ["シャープレシオ", *(f"{m['sharpe']:.2f}" if m.get("sharpe") is not None else "—" for m in (st, tp, ew))],
    ]
    html = (
        f'<p class="muted">期間 {escape(summary["start"])} 〜 {escape(summary["end"])}・'
        f'{escape(summary["created"])} 作成・過去の結果は将来を保証しません</p>'
        + _table(["", "銘柄売買のルール", "TOPIX", "同じ銘柄を均等に保有"], rows, "", cards=False)
        + "<ul>" + "".join(f"<li>{escape(line)}</li>" for line in summary.get("verdict", [])) + "</ul>"
    )
    plan = summary.get("plan")
    if plan:
        html += (
            f'<p>あなたの資金計画で続けた場合: 入金 {_yen(plan["contributed"])} → '
            f'<strong>{_yen(plan["final_equity"])}</strong>'
            f'（同じ額を TOPIX に入れた場合 {_yen(plan["topix_same_flows"])}）</p>'
        )
    if repo and branch:
        url = f"https://github.com/{repo}/blob/{branch}/reports/backtest_latest.md"
        html += f'<p><a href="{escape(url)}">詳しいレポートを見る</a></p>'
    return html


CSS = """
:root{--bg:#f6f7f9;--card:#fff;--text:#1d2330;--muted:#5f6b7a;--line:#e3e7ec;--accent:#2563eb;
--pos:#0f7b3f;--neg:#c0262d;--bench:#8a94a3;--tag-buy:#e7f0ff;--tag-sell:#fde8e8}
@media (prefers-color-scheme:dark){:root{--bg:#0f1318;--card:#181e26;--text:#e6eaf0;--muted:#9aa5b4;
--line:#2a323d;--accent:#6ea0ff;--pos:#4cc483;--neg:#ff7b7b;--bench:#7b8594;--tag-buy:#1c2b47;--tag-sell:#46201f}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);line-height:1.6;
font-family:-apple-system,BlinkMacSystemFont,"Hiragino Sans","Noto Sans JP","Yu Gothic UI",Meiryo,sans-serif}
main{max-width:960px;margin:0 auto;padding:16px}
header h1{font-size:1.4rem;margin:8px 0 2px}
.muted{color:var(--muted);font-size:.9rem}
.badge{display:inline-block;padding:2px 10px;border-radius:999px;font-size:.8rem;font-weight:600;
background:var(--tag-buy);color:var(--accent);margin-left:6px;vertical-align:middle}
.badge.live{background:var(--tag-sell);color:var(--neg)}
section{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin:16px 0}
section h2{font-size:1.1rem;margin:0 0 12px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.stat{border:1px solid var(--line);border-radius:10px;padding:12px}
.stat .label{color:var(--muted);font-size:.85rem}
.stat .value{font-size:1.35rem;font-weight:700;font-variant-numeric:tabular-nums}
.stat .value.small{font-size:1rem}
.stat .note{color:var(--muted);font-size:.8rem;margin-top:2px}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:.92rem;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
td.wrap{white-space:normal;min-width:14em}
th{color:var(--muted);font-weight:600;font-size:.82rem}
.pos{color:var(--pos)}.neg{color:var(--neg)}
.tag{padding:2px 8px;border-radius:6px;font-size:.8rem;font-weight:700}
.tag.buy{background:var(--tag-buy);color:var(--accent)}.tag.sell{background:var(--tag-sell);color:var(--neg)}
.chart .plot{position:relative;height:220px;margin:8px 0 0 52px}
.chart svg{position:absolute;inset:0;width:100%;height:100%;overflow:visible}
.chart line,.chart polyline{vector-effect:non-scaling-stroke}
svg .grid{stroke:var(--line);stroke-width:1}svg .zero{stroke:var(--muted);stroke-width:1;stroke-dasharray:4 4}
svg polyline{fill:none;stroke-width:2;stroke-linejoin:round}
.ylab{position:absolute;left:-52px;width:44px;text-align:right;transform:translateY(-50%);
font-size:.75rem;color:var(--muted);font-variant-numeric:tabular-nums}
.xlabels{display:flex;justify-content:space-between;margin-left:52px;font-size:.75rem;color:var(--muted)}
svg .strat{stroke:var(--accent)}svg .bench{stroke:var(--bench)}
.legend{display:flex;gap:6px;align-items:center;color:var(--muted);font-size:.85rem;margin-top:6px}
.key{display:inline-block;width:14px;height:3px;margin-left:10px}.key.strat{background:var(--accent)}
.key.bench{background:var(--bench)}
a{color:var(--accent)}
summary{cursor:pointer;color:var(--muted)}details[open] summary{margin-bottom:10px}
.button{display:inline-block;padding:8px 14px;border:1px solid var(--accent);border-radius:8px;text-decoration:none}
footer{color:var(--muted);font-size:.8rem;margin:24px 0}
@media (max-width:640px){
main{padding:12px}section{padding:14px}
table.cards thead{display:none}
table.cards,table.cards tbody,table.cards tr,table.cards td{display:block;width:100%}
table.cards tr{border:1px solid var(--line);border-radius:10px;padding:6px 12px;margin-bottom:10px}
table.cards td{display:flex;justify-content:space-between;gap:16px;border:none;padding:4px 0;
white-space:normal;text-align:right;min-width:0}
table.cards td::before{content:attr(data-label);color:var(--muted);font-size:.82rem;text-align:left;flex-shrink:0}
table.cards td:not([data-label]){justify-content:flex-start;text-align:left}
table.plain td{white-space:normal}
}
"""


def build_html(
    state: PortfolioState | None,
    settings: Settings,
    universe: pd.DataFrame | None,
    backtest: dict | None,
    generated_at: datetime,
    repo: str | None = None,
    branch: str | None = None,
) -> str:
    names = dict(zip(universe["ticker"], universe["name"], strict=True)) if universe is not None else {}
    paper = settings.mode == "paper"
    mode = '<span class="badge">模擬運用</span>' if paper else '<span class="badge live">実運用</span>'
    as_of = f"{jp_date(state.last_session)} 終値時点" if state and state.last_session else "運用開始前"
    sections = [
        ("次にやること", _todo_section(state, settings, names)),
        ("全体の状況", _overview_section(state, settings)),
    ]
    if state is not None and state.last_session:
        sections += [
            ("銘柄売買の成績の推移（開始からの騰落率）", _chart(state.history)),
            ("保有している株", _holdings_section(state, settings, names)),
            ("売買の記録", _trades_section(state, settings, names)),
        ]
    sections += [
        ("過去データでの検証（バックテスト）", _backtest_section(backtest, repo, branch)),
        ("今の設定", _settings_section(settings, repo, branch)),
    ]
    body = "".join(f"<section><h2>{escape(title)}</h2>{content}</section>" for title, content in sections)
    stamp = generated_at.astimezone(JST).strftime("%Y-%m-%d %H:%M")
    return (
        '<!doctype html><html lang="ja"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta name="robots" content="noindex">'
        "<title>株の運用ダッシュボード</title>"
        f"<style>{CSS}</style></head><body><main>"
        f"<header><h1>株の運用ダッシュボード{mode}</h1>"
        f'<div class="muted">{escape(as_of)}・ページ更新 {escape(stamp)}（平日の夕方に自動更新）</div></header>'
        f"{body}"
        "<footer>このページは自動で作られた記録です。投資の助言ではなく、利益を保証するものではありません。"
        "積立の評価額は値動きの近いETFでの推定で、実際の残高とは異なります。</footer>"
        "</main></body></html>"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="運用状況の Web ページを作る")
    parser.add_argument("--out", type=Path, default=ROOT / "site")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--backtest", type=Path, default=DEFAULT_BACKTEST_SUMMARY)
    parser.add_argument("--repo", help="owner/name（設定変更リンク用）")
    parser.add_argument("--branch", help="設定変更リンクのブランチ")
    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    state = load_state(args.state)
    universe = load_universe(args.universe) if args.universe.exists() else None
    backtest = json.loads(args.backtest.read_text(encoding="utf-8")) if args.backtest.exists() else None
    html = build_html(state, settings, universe, backtest, datetime.now(JST), args.repo, args.branch)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "index.html").write_text(html, encoding="utf-8")
    print(f"{args.out / 'index.html'} を作成しました")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
