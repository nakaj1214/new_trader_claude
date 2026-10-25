"""1営業日分の処理。毎日の本番実行とバックテストで同じ関数を使い、結果の食い違いを防ぐ。

処理順（終値が確定した後に実行する想定）:
1. 株式分割・配当・年をまたいだ税金の反映
2. 前回の注文を当日の始値で約定（売り→買いの順）
3. 終値で評価額・最高値を更新
4. 最大下落ルール・損切りルールの判定
5. 月末なら入金と銘柄の入れ替え判断
6. 次の営業日の始値で実行する注文を保存
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from trader import calendar as tse
from trader import strategy
from trader.config import Settings
from trader.market import MarketData
from trader.portfolio import Fill, Order, PortfolioState, Position, Trade
from trader.quality import is_implausible

SPLIT_LOOKBACK_SESSIONS = 10
SPLIT_DEDUPE_DAYS = 7  # 同じ倍率の分割記録が日付違いで二重に付いても1回だけ反映する
ANOMALY_DROP_RATIO = 0.7  # 1:1.5 の分割（0.67倍）まで拾う
TOP_CANDIDATES_TO_REPORT = 10


class AlreadyProcessedError(RuntimeError):
    """同じ営業日を二重に処理しようとしたとき。"""


@dataclass
class StepResult:
    session: str
    is_rebalance: bool = False
    fills: list[Fill] = field(default_factory=list)
    orders: list[Order] = field(default_factory=list)
    skipped: list[tuple[strategy.Candidate, str]] = field(default_factory=list)
    candidates: list[strategy.Candidate] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    regime: str | None = None
    dividends: float = 0.0
    tax_paid: float = 0.0
    contribution: float = 0.0
    core_contribution: float = 0.0
    equity: float = 0.0
    cash: float = 0.0
    nav: float = 1.0
    drawdown: float = 0.0
    paused: bool = False


def _valid(value: float) -> bool:
    return not math.isnan(value) and value > 0


def _recently_applied(state: PortfolioState, ticker: str, day: pd.Timestamp, ratio: float) -> bool:
    for key in state.applied_splits:
        parts = key.split("@")
        if len(parts) != 3 or parts[0] != ticker or abs(float(parts[2]) - ratio) > 1e-9:
            continue
        if abs((pd.Timestamp(parts[1]) - day).days) <= SPLIT_DEDUPE_DAYS:
            return True
    return False


def _apply_splits(state: PortfolioState, md: MarketData, session: pd.Timestamp, res: StepResult) -> None:
    """Yahoo Finance の分割記録は遅れて付くことがあるため、直近数営業日を見て未適用の分を反映する。"""
    idx = md.sessions.get_loc(session)
    window = md.sessions[max(0, idx - SPLIT_LOOKBACK_SESSIONS + 1) : idx + 1]
    for ticker, pos in list(state.positions.items()):
        for day in window:
            ratio = md.value("splits", day, ticker)
            if not _valid(ratio) or ratio == 1.0:
                continue
            key = f"{ticker}@{day.date().isoformat()}@{ratio:g}"
            if key in state.applied_splits or _recently_applied(state, ticker, day, ratio):
                continue
            state.applied_splits.append(key)
            if pd.Timestamp(pos.entry_date) >= day:
                continue  # 分割後に買った株は調整不要
            new_shares = pos.shares * ratio
            whole = int(math.floor(new_shares + 1e-9))
            fraction = new_shares - whole
            pos.shares = whole
            pos.entry_price /= ratio
            pos.high_water /= ratio
            pos.last_price /= ratio
            if fraction > 1e-9:
                state.cash += fraction * pos.last_price
            res.events.append(f"{ticker} の株式分割（1:{ratio:g}）を反映しました")
            state.price_anomalies.pop(ticker, None)
    for order in state.pending:
        if order.side != "buy":
            continue
        ratio = md.value("splits", session, order.ticker)
        if _valid(ratio) and ratio != 1.0 and pd.Timestamp(order.created) < session:
            order.shares = max(1, int(order.shares * ratio))
            order.ref_price /= ratio
    del state.applied_splits[:-200]


def _apply_dividends(
    state: PortfolioState, md: MarketData, prev: pd.Timestamp, session: pd.Timestamp, tax: float
) -> float:
    days = md.sessions[(md.sessions > prev) & (md.sessions <= session)]
    total = 0.0
    for ticker, pos in state.positions.items():
        for day in days:
            amount = md.value("dividends", day, ticker)
            if _valid(amount):
                total += pos.shares * amount * (1 - tax)
    state.cash += total
    return total


def _fill_orders(state: PortfolioState, md: MarketData, session: pd.Timestamp, settings: Settings,
                 res: StepResult) -> list[Order]:
    """前回の注文を始値で約定させる。始値が無い売り注文は次の営業日へ持ち越す。"""
    sat = settings.satellite
    day = session.date().isoformat()
    carry: list[Order] = []
    sells = [o for o in state.pending if o.side == "sell"]
    buys = [o for o in state.pending if o.side == "buy"]
    for order in sells:
        pos = state.positions.get(order.ticker)
        if pos is None:
            continue
        price = md.value("open", session, order.ticker)
        if not _valid(price):
            carry.append(order)
            res.warnings.append(f"{order.ticker} の始値が無いため、売り注文を次の営業日へ持ち越します")
            continue
        proceeds = pos.shares * price * (1 - sat.cost_rate)
        pnl = proceeds - pos.cost_basis
        state.cash += proceeds
        year = str(session.year)
        state.realized_by_year[year] = state.realized_by_year.get(year, 0.0) + pnl
        held = max(len(tse.sessions_between(pos.entry_date, session)) - 1, 0)
        state.trades.append(
            Trade(order.ticker, pos.entry_date, day, pos.shares, pos.cost_basis, proceeds, pnl,
                  pnl / pos.cost_basis if pos.cost_basis else 0.0, order.reason, held)
        )
        res.fills.append(Fill("sell", order.ticker, pos.shares, price, proceeds, day, order.reason, pnl))
        del state.positions[order.ticker]
        state.price_anomalies.pop(order.ticker, None)
    for order in buys:
        price = md.value("open", session, order.ticker)
        if not _valid(price):
            res.warnings.append(f"{order.ticker} の始値が無いため、買い注文を取り消しました")
            continue
        if order.ticker in state.positions:
            continue
        unit = price * (1 + sat.cost_rate)
        affordable = int(state.cash // (unit * sat.lot_size)) * sat.lot_size
        shares = min(order.shares, affordable)
        if shares <= 0:
            res.warnings.append(f"{order.ticker} は始値が上がり現金が足りないため、買いを見送りました")
            continue
        if shares < order.shares:
            res.warnings.append(f"{order.ticker} は現金の範囲で {shares} 株に減らして買いました")
        cost = shares * unit
        state.cash -= cost
        state.positions[order.ticker] = Position(order.ticker, shares, day, price, cost, price, price)
        res.fills.append(Fill("buy", order.ticker, shares, price, cost, day, order.reason))
    return carry


def _mark_to_market(state: PortfolioState, md: MarketData, session: pd.Timestamp, res: StepResult) -> set[str]:
    """終値で評価する。前日比で3割以上の急落は分割記録の遅れの可能性があるため1日だけ判断を保留する。"""
    anomalies: set[str] = set()
    day = session.date().isoformat()
    for ticker, pos in state.positions.items():
        close = md.value("close", session, ticker)
        if not _valid(close):
            res.warnings.append(f"{ticker} の終値が無いため、前回の価格で評価しています")
            continue
        if pos.last_price > 0 and close / pos.last_price <= ANOMALY_DROP_RATIO:
            first = state.price_anomalies.get(ticker)
            if first is None:
                state.price_anomalies[ticker] = day
                anomalies.add(ticker)
                res.warnings.append(
                    f"{ticker} が前日から3割以上下がりました。株式分割の記録遅れの可能性があるため、"
                    "今日は売買判断を保留します。証券会社のアプリで株価を確認してください"
                )
                continue
            res.warnings.append(f"{ticker} の急落が2日続いたため、実際の値下がりとして扱います")
        state.price_anomalies.pop(ticker, None)
        pos.last_price = close
        pos.high_water = max(pos.high_water, close)
    return anomalies


def _track_index(md: MarketData, ticker: str | None, prev: pd.Timestamp, session: pd.Timestamp) -> float | None:
    if not ticker:
        return None
    now, before = md.value("adj_close", session, ticker), md.value("adj_close", prev, ticker)
    if not (_valid(now) and _valid(before)) or is_implausible(now / before):
        return None
    return now / before


def step(
    state: PortfolioState,
    md: MarketData,
    session: pd.Timestamp,
    settings: Settings,
    universe: pd.DataFrame,
    *,
    apply_splits: bool,
    force_rebalance: bool = False,
) -> StepResult:
    """1営業日を処理し、state を更新する。apply_splits は本番のみ True（バックテストの株価は分割調整済み）。"""
    cfg, sat = settings.strategy, settings.satellite
    session = tse.to_timestamp(session)
    if session not in md.sessions:
        raise ValueError(f"{session.date()} の株価データがありません")
    prev = pd.Timestamp(state.last_session) if state.last_session else None
    if prev is not None and session <= prev:
        raise AlreadyProcessedError(f"{session.date()} は処理済みです")
    res = StepResult(session=session.date().isoformat())

    # 1. 分割・配当・税金
    if apply_splits:
        _apply_splits(state, md, session, res)
    if prev is not None:
        res.dividends = _apply_dividends(state, md, prev, session, sat.tax_rate)
        if session.year > prev.year and sat.tax_rate > 0:
            gain = state.realized_by_year.get(str(prev.year), 0.0)
            if gain > 0:
                res.tax_paid = gain * sat.tax_rate
                state.cash -= res.tax_paid
                res.events.append(f"{prev.year}年の売却益にかかる税金 {res.tax_paid:,.0f}円 を差し引きました")

    # 2. 約定
    carry = _fill_orders(state, md, session, settings, res)

    # 3. 評価
    anomalies = _mark_to_market(state, md, session, res)
    if prev is not None:
        bench = _track_index(md, cfg.benchmark_ticker, prev, session)
        if bench is not None:
            state.bench_index *= bench
        core = _track_index(md, settings.core.proxy_ticker, prev, session)
        state.core.stale = core is None and bool(settings.core.proxy_ticker)
        if core is not None:
            state.core.value *= core
    nav = state.nav()
    if not anomalies and state.pause_rebalances_left == 0:
        state.peak_nav = max(state.peak_nav, nav)

    # 4. 最大下落・損切り
    excluded = {o.ticker for o in carry}
    new_orders: list[Order] = []
    day = res.session
    paused_before = state.pause_rebalances_left > 0
    drawdown = 1 - nav / state.peak_nav if state.peak_nav > 0 else 0.0
    if (
        not paused_before
        and not anomalies
        and cfg.max_drawdown is not None
        and drawdown >= cfg.max_drawdown
        and state.positions
    ):
        for ticker, pos in state.positions.items():
            if ticker not in excluded:
                new_orders.append(Order("sell", ticker, pos.shares, "drawdown_limit", day, pos.last_price))
        excluded |= set(state.positions)
        state.pause_rebalances_left = cfg.drawdown_pause_months
        res.events.append(
            f"銘柄売買の資金が最高時から {drawdown:.0%} 減ったため、全て売って "
            f"{cfg.drawdown_pause_months} か月分の入れ替えを休みます"
        )
    else:
        for ticker, pos in state.positions.items():
            if ticker in excluded or ticker in anomalies:
                continue
            if strategy.stop_triggered(pos.last_price, pos.high_water, cfg):
                new_orders.append(Order("sell", ticker, pos.shares, "trailing_stop", day, pos.last_price))
                excluded.add(ticker)

    # 5. 月末の入金と入れ替え
    res.is_rebalance = force_rebalance or state.last_rebalance is None or tse.is_month_end_session(session)
    if res.is_rebalance:
        month = session.strftime("%Y-%m")
        if state.last_rebalance is not None and month not in state.contributed_months:
            res.contribution = sat.monthly_contribution
            state.units += res.contribution / nav
            state.cash += res.contribution
            state.total_contributed += res.contribution
            res.core_contribution = settings.core.monthly_amount
            state.core.value += res.core_contribution
            state.core.contributed += res.core_contribution
            state.contributed_months.append(month)
        signals = md.signals(cfg)
        res.regime = strategy.market_regime(md, signals, session, cfg)
        if paused_before:
            state.pause_rebalances_left -= 1
            if state.pause_rebalances_left == 0:
                state.peak_nav = state.nav()
                res.events.append("休止期間が終わりました。次の月末から銘柄の入れ替えを再開します")
        elif state.pause_rebalances_left == 0:
            candidates = strategy.rank_universe(md, signals, universe, session, cfg)
            res.candidates = candidates[:TOP_CANDIDATES_TO_REPORT]
            incoming = sum(
                state.positions[o.ticker].shares * state.positions[o.ticker].last_price * (1 - sat.cost_rate)
                for o in carry + new_orders
                if o.side == "sell" and o.ticker in state.positions
            )
            sectors = dict(zip(universe["ticker"], universe["sector"], strict=True))
            plan = strategy.plan_rebalance(
                state, md, signals, session, candidates, sectors, res.regime, excluded, incoming, cfg, sat,
                frozen=frozenset(anomalies),
            )
            new_orders.extend(plan.orders)
            res.skipped = plan.skipped
            res.warnings.extend(plan.notes)
        state.last_rebalance = res.session

    # 6. 保存
    state.pending = carry + new_orders
    state.last_session = res.session
    state.session_count += 1
    res.orders = list(state.pending)
    res.equity, res.cash, res.nav = state.equity(), state.cash, state.nav()
    res.drawdown = 1 - res.nav / state.peak_nav if state.peak_nav > 0 else 0.0
    res.paused = state.pause_rebalances_left > 0
    state.history.append(
        {
            "date": res.session,
            "equity": round(res.equity, 2),
            "cash": round(res.cash, 2),
            "nav": round(res.nav, 6),
            "bench": round(state.bench_index, 6),
            "core": round(state.core.value, 2),
            "flow": res.contribution,
        }
    )
    return res
