"""モデル口座（システムが想定する保有・現金・注文）の状態と保存。"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

STATE_VERSION = 1


@dataclass
class Position:
    ticker: str
    shares: int
    entry_date: str
    entry_price: float  # 約定単価（コスト抜き）
    cost_basis: float  # 支払総額（コスト込み）
    high_water: float  # 保有開始後の最高終値
    last_price: float  # 直近の終値（欠損時の評価に使う）


@dataclass
class Order:
    side: str  # "buy" / "sell"
    ticker: str
    shares: int
    reason: str
    created: str  # 判断した営業日
    ref_price: float  # 判断時の終値（目安）
    rank: int | None = None
    momentum: float | None = None


@dataclass
class Fill:
    side: str
    ticker: str
    shares: int
    price: float
    amount: float  # 買いは支払額、売りは受取額（コスト込み）
    date: str
    reason: str
    pnl: float | None = None


@dataclass
class Trade:
    ticker: str
    entry_date: str
    exit_date: str
    shares: int
    cost_basis: float
    proceeds: float
    pnl: float
    return_pct: float
    exit_reason: str
    holding_sessions: int


@dataclass
class CoreState:
    """積立側の推定。値動きが近い ETF の日次騰落率で評価額を更新する。"""

    value: float = 0.0
    contributed: float = 0.0
    stale: bool = False


@dataclass
class PortfolioState:
    start_date: str
    cash: float
    units: float
    peak_nav: float
    total_contributed: float
    version: int = STATE_VERSION
    positions: dict[str, Position] = field(default_factory=dict)
    pending: list[Order] = field(default_factory=list)
    last_session: str | None = None
    last_rebalance: str | None = None
    contributed_months: list[str] = field(default_factory=list)
    pause_rebalances_left: int = 0
    realized_by_year: dict[str, float] = field(default_factory=dict)
    trades: list[Trade] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    bench_index: float = 1.0
    core: CoreState = field(default_factory=CoreState)
    session_count: int = 0
    mode: str = "paper"  # この記録が模擬運用か実運用か
    applied_splits: list[str] = field(default_factory=list)  # "TICKER@YYYY-MM-DD"
    price_anomalies: dict[str, str] = field(default_factory=dict)  # ticker -> 初めて急落を検知した日

    @classmethod
    def new(
        cls, start_date: str, initial_capital: float, core_initial: float = 0.0, mode: str = "paper"
    ) -> PortfolioState:
        return cls(
            start_date=start_date,
            mode=mode,
            cash=float(initial_capital),
            units=float(initial_capital),
            peak_nav=1.0,
            total_contributed=float(initial_capital),
            core=CoreState(value=float(core_initial), contributed=float(core_initial)),
        )

    def equity(self) -> float:
        return self.cash + sum(p.shares * p.last_price for p in self.positions.values())

    def nav(self) -> float:
        return self.equity() / self.units if self.units > 0 else 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PortfolioState:
        if raw.get("version") != STATE_VERSION:
            raise ValueError(f"未対応の state version です: {raw.get('version')}")
        data = dict(raw)
        data["positions"] = {k: Position(**v) for k, v in raw.get("positions", {}).items()}
        data["pending"] = [Order(**o) for o in raw.get("pending", [])]
        data["trades"] = [Trade(**t) for t in raw.get("trades", [])]
        data["core"] = CoreState(**raw.get("core", {}))
        return cls(**data)


def save_state(state: PortfolioState, path: Path) -> None:
    """途中で失敗しても壊れたファイルを残さないよう、一時ファイル経由で置き換える。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(state.to_dict(), ensure_ascii=False, indent=1, sort_keys=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".state-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload + "\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def load_state(path: Path) -> PortfolioState | None:
    if not path.exists():
        return None
    return PortfolioState.from_dict(json.loads(path.read_text(encoding="utf-8")))
