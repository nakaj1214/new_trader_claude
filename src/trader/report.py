"""Discord に送る日本語のメッセージを組み立てる。"""

from __future__ import annotations

import pandas as pd

from trader import calendar as tse
from trader.config import Settings
from trader.engine import StepResult
from trader.portfolio import Fill, Order, PortfolioState

WEEKDAYS = "月火水木金土日"
SKIP_REASONS = {
    "too_expensive": "1株の価格が今の予算を超えるため",
    "sector_limit": "同じ業種をすでに上限まで持っているため",
}


def jp_date(value: str | pd.Timestamp) -> str:
    ts = pd.Timestamp(value)
    return f"{ts.month}/{ts.day}({WEEKDAYS[ts.weekday()]})"


def yen(value: float) -> str:
    return f"{value:,.0f}円"


def stock_label(ticker: str, names: dict[str, str]) -> str:
    code = ticker.removesuffix(".T")
    name = names.get(ticker)
    return f"{code} {name}" if name else code


def order_howto(settings: Settings) -> str:
    if settings.satellite.lot_size == 1:
        if settings.broker == "rakuten":
            return "楽天証券アプリの「かぶミニ®」で「寄付取引」を選び、成行で注文"
        return "SBI証券アプリの「S株（単元未満株）」で、翌営業日の寄付（始値）で約定する時間帯に成行で注文"
    return "通常の現物取引で、成行・寄付（始値）で約定する条件で注文"


def sell_reason(order: Order | Fill, settings: Settings) -> str:
    s = settings.strategy
    texts = {
        "trailing_stop": f"保有中の最高値から{s.trailing_stop or 0:.0%}下がった（損切り・利益確定のルール）",
        "rank_dropped": f"上昇率の順位が{s.keep_rank}位より下がった（月末の入れ替え）",
        "below_trend": f"株価が{s.trend_window}日平均線を下回った（月末の入れ替え）",
        "market_downtrend": f"TOPIXが{s.trend_window}日平均線を下回り、相場全体が下落傾向（一旦すべて売る）",
        "drawdown_limit": f"銘柄売買の資金が最高時から{s.max_drawdown or 0:.0%}減った（最大下落ルール）",
        "universe_exit": "売買対象の銘柄一覧（TOPIX100など）から外れた（月末の入れ替え）",
    }
    return texts.get(order.reason, order.reason)


def buy_reason(order: Order) -> str:
    mom = f"{order.momentum:+.0%}" if order.momentum is not None else "—"
    return f"過去12か月の上昇率 {mom}（{order.rank}位）・長期トレンドが上向き"


def _fills_section(results: list[StepResult], names: dict[str, str], settings: Settings) -> list[str]:
    lines: list[str] = []
    for res in results:
        for fill in res.fills:
            label = stock_label(fill.ticker, names)
            if fill.side == "buy":
                lines.append(
                    f"・{jp_date(fill.date)} 買い {label} {fill.shares}株 @{yen(fill.price)}（{yen(fill.amount)}）"
                )
            else:
                pnl = fill.pnl or 0.0
                cost = fill.amount - pnl
                rate = pnl / cost if cost else 0.0
                lines.append(
                    f"・{jp_date(fill.date)} 売り {label} {fill.shares}株 @{yen(fill.price)}"
                    f" 損益 {pnl:+,.0f}円（{rate:+.1%}）"
                )
    if not lines:
        return []
    return [
        "✅ **約定したとみなした取引**（始値で計算）",
        *lines,
        "※ 実際の約定が少し違っても問題ありません。システムはこの値で計算を続けます。",
        "",
    ]


def _orders_section(last: StepResult, next_session: pd.Timestamp, names: dict[str, str],
                    settings: Settings) -> list[str]:
    if not last.orders:
        return []
    sells = [o for o in last.orders if o.side == "sell"]
    buys = [o for o in last.orders if o.side == "buy"]
    lines = [
        f"📝 **次の営業日 {jp_date(next_session)} の朝9:00の取引開始前に注文してください**",
        f"注文方法: {order_howto(settings)}",
    ]
    if sells:
        lines.append("🔴 売り（持っている株をすべて売る）")
        for o in sells:
            lines.append(f"・{stock_label(o.ticker, names)} {o.shares}株 — 理由: {sell_reason(o, settings)}")
    if buys:
        lines.append("🟢 買い")
        for o in buys:
            lines.append(
                f"・{stock_label(o.ticker, names)} **{o.shares}株**（目安 {yen(o.ref_price)}/株・"
                f"約{yen(o.shares * o.ref_price)}）— 理由: {buy_reason(o)}"
            )
        lines.append("※ 売りの注文を先に出してください。翌朝の株価次第で、買える株数が1株ほど減ることがあります。")
    lines.append("")
    return lines


def _skipped_section(last: StepResult, names: dict[str, str]) -> list[str]:
    if not last.skipped:
        return []
    lines = ["ℹ️ 上位だが今回は買わない銘柄"]
    for cand, reason in last.skipped[:5]:
        lines.append(f"・{stock_label(cand.ticker, names)}（{cand.rank}位）: {SKIP_REASONS.get(reason, reason)}")
    return lines + [""]


def _holdings_section(state: PortfolioState, last: StepResult, names: dict[str, str],
                      settings: Settings) -> list[str]:
    s = settings.strategy
    lines = [f"📊 **保有状況**（{jp_date(last.session)} 終値）"]
    if state.positions:
        for pos in state.positions.values():
            change = pos.last_price / pos.entry_price - 1 if pos.entry_price else 0.0
            stop = f" 売却ライン {yen(pos.high_water * (1 - s.trailing_stop))}" if s.trailing_stop else ""
            lines.append(
                f"・{stock_label(pos.ticker, names)} {pos.shares}株 {yen(pos.last_price)}"
                f"（買値から {change:+.1%}）{stop}"
            )
    else:
        lines.append("・保有している株はありません")
    pnl = last.equity - state.total_contributed
    lines.append(
        f"現金 {yen(last.cash)} / 合計 {yen(last.equity)}"
        f"（入金累計 {yen(state.total_contributed)}、損益 {pnl:+,.0f}円）"
    )
    lines.append(
        f"開始来 {last.nav - 1:+.1%}（同じ期間のTOPIX {state.bench_index - 1:+.1%}）・"
        f"最高時からの下落 {max(last.drawdown, 0.0):.1%}"
        + (f"（上限 {s.max_drawdown:.0%}）" if s.max_drawdown else "")
    )
    return lines + [""]


def months_elapsed(state: PortfolioState, session: str) -> float:
    return (pd.Timestamp(session) - pd.Timestamp(state.start_date)).days / 30.44


def _monthly_section(state: PortfolioState, last: StepResult, settings: Settings) -> list[str]:
    core = settings.core
    session = pd.Timestamp(last.session)
    lines = [f"🗓 **月次レポート**（{session.year}年{session.month}月）"]
    if last.contribution:
        lines.append(
            f"・銘柄売買用の入金 {yen(last.contribution)} を今回の判断に含めました。"
            "月末までに証券口座へ入金しておいてください。"
        )
    if core.monthly_amount or core.initial_amount:
        estimate = "推定できません（参考ETFのデータ欠損）" if state.core.stale else f"約{yen(state.core.value)}"
        lines.append(
            f"・積立: {core.fund_name} 毎月 {yen(core.monthly_amount)}（証券会社の積立設定で自動購入）"
        )
        lines.append(f"  推定評価額 {estimate}（入金累計 {yen(state.core.contributed)}）")
        total = state.core.value + last.equity
        if total > 0 and not state.core.stale:
            lines.append(
                f"・配分: 積立 {state.core.value / total:.0%} / 銘柄売買 {last.equity / total:.0%}（目安 8:2）"
            )
    regime = {"risk_on": "上（買いOK）", "risk_off": "下（新しく買わない）", "unknown": "判断できず"}
    if last.regime:
        lines.append(f"・相場全体: TOPIXは{settings.strategy.trend_window}日平均線の{regime[last.regime]}")
    if last.paused:
        lines.append(f"・最大下落ルールにより休止中（残り {state.pause_rebalances_left} 回の月末）")
    lines += _review_lines(state, last, settings)
    if last.candidates:
        lines.append("・今月の上昇率ランキング上位: " + "、".join(
            f"{c.rank}位 {c.ticker.removesuffix('.T')}" for c in last.candidates[:5]
        ))
    return lines + [""]


def _review_lines(state: PortfolioState, last: StepResult, settings: Settings) -> list[str]:
    review = settings.review
    elapsed = months_elapsed(state, last.session)
    gap = (last.nav - 1) - (state.bench_index - 1)
    if elapsed < review.min_months:
        return [f"・成績の見直し判定: 運用 {elapsed:.0f}/{review.min_months} か月（期間が短く、まだ判定しません）"]
    if gap < -review.underperform_threshold:
        return [
            f"⚠️ **見直しをおすすめします**: 銘柄売買の成績がTOPIXを {gap:.1%} 下回っています。"
            "銘柄売買への入金を止め、その分を積立に回すことを検討してください。"
        ]
    return [f"・成績の見直し判定: TOPIXとの差 {gap:+.1%}（基準 -{review.underperform_threshold:.0%} 以内）"]


def render_notification(
    results: list[StepResult],
    state: PortfolioState,
    settings: Settings,
    names: dict[str, str],
    extra_warnings: list[str] | None = None,
) -> str | None:
    """通知が必要なときだけメッセージを返す。何も無い日は None。"""
    if not results:
        return None
    last = results[-1]
    next_session = tse.next_session(last.session)
    warnings = list(extra_warnings or []) + [w for r in results for w in r.warnings]
    events = [e for r in results for e in r.events]
    weekly = settings.notify.weekly_summary and tse.is_week_end_session(last.session)
    has_fills = any(r.fills for r in results)
    if not (has_fills or last.orders or warnings or events or weekly or last.is_rebalance):
        return None

    title = "【模擬運用】" if settings.mode == "paper" else ""
    lines = [f"**{title}株の売買通知 — {jp_date(last.session)} 終値で判断**", ""]
    if state.session_count == 1:
        lines += [
            "🎉 運用を開始しました。毎営業日の夕方に確認し、やることがある日だけ通知します。",
            "",
        ]
    if any(r.orders for r in results[:-1]):
        lines += [
            f"⚠️ 実行が遅れたため {len(results)} 営業日分をまとめて処理しました。"
            "途中の注文は通知が間に合わなかった可能性がありますが、システムは約定したものとして計算しています。",
            "",
        ]
    lines += _fills_section(results, names, settings)
    for event in events:
        lines.append(f"📌 {event}")
    for warning in warnings:
        lines.append(f"⚠️ {warning}")
    if events or warnings:
        lines.append("")
    lines += _orders_section(last, next_session, names, settings)
    if last.is_rebalance:
        if not last.orders and not last.paused:
            lines += ["📝 今月の入れ替え: 注文はありません（今の保有を続けます）", ""]
        lines += _skipped_section(last, names)
        lines += _monthly_section(state, last, settings)
    if weekly or last.is_rebalance or has_fills:
        lines += _holdings_section(state, last, names, settings)
    lines.append("※ このシステムの通知は投資の助言ではありません。最終判断はご自身で行ってください。")
    return "\n".join(lines).strip()


def render_error(title: str, detail: str, settings: Settings | None = None) -> str:
    label = "【模擬運用】" if settings is not None and settings.mode == "paper" else ""
    return (
        f"❌ **{label}株の売買システムでエラーが発生しました**: {title}\n{detail}\n"
        "この通知が続く場合は GitHub の Actions 画面でログを確認してください。"
        "今日の売買指示は出ていません（保有中の株はそのままで大丈夫です）。"
    )
