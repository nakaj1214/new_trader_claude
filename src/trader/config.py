"""settings.yaml を型付きの設定に変換する。不正な値は補正せずエラーにする。"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"


class ConfigError(ValueError):
    """設定ファイルの値が不正なときに送出する。"""


@dataclass(frozen=True)
class CoreConfig:
    initial_amount: float = 200_000
    monthly_amount: float = 40_000
    fund_name: str = "eMAXIS Slim 全世界株式（オール・カントリー）"
    proxy_ticker: str | None = "2559.T"


@dataclass(frozen=True)
class SatelliteConfig:
    initial_capital: float = 50_000
    monthly_contribution: float = 10_000
    lot_size: int = 1
    cost_rate: float = 0.001
    tax_rate: float = 0.0


@dataclass(frozen=True)
class StrategyConfig:
    max_positions: int = 4
    keep_rank: int = 10
    max_per_sector: int = 2
    momentum_lookback: int = 252
    momentum_skip: int = 21
    trend_window: int = 200
    trailing_stop: float | None = 0.15
    min_turnover: float = 100_000_000
    benchmark_ticker: str = "1306.T"
    regime_filter: bool = True
    max_drawdown: float | None = 0.20
    drawdown_pause_months: int = 3


@dataclass(frozen=True)
class UniverseConfig:
    tiers: tuple[str, ...] = ("TOPIX Core30", "TOPIX Large70")
    max_age_days: int = 40


@dataclass(frozen=True)
class ReviewConfig:
    min_months: int = 12
    underperform_threshold: float = 0.10


@dataclass(frozen=True)
class NotifyConfig:
    weekly_summary: bool = True


@dataclass(frozen=True)
class Settings:
    mode: str = "paper"
    broker: str = "sbi"
    core: CoreConfig = field(default_factory=CoreConfig)
    satellite: SatelliteConfig = field(default_factory=SatelliteConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    review: ReviewConfig = field(default_factory=ReviewConfig)
    notify: NotifyConfig = field(default_factory=NotifyConfig)

    def with_strategy(self, **changes: Any) -> Settings:
        return replace(self, strategy=replace(self.strategy, **changes))

    def with_satellite(self, **changes: Any) -> Settings:
        return replace(self, satellite=replace(self.satellite, **changes))


def _build(cls: type, raw: Any, section: str) -> Any:
    if raw is None:
        return cls()
    if not isinstance(raw, dict):
        raise ConfigError(f"{section} は key: value の形式で書いてください")
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"{section} に不明な項目があります: {sorted(unknown)}")
    values = dict(raw)
    if "tiers" in values:
        values["tiers"] = tuple(values["tiers"])
    return cls(**values)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def validate(settings: Settings) -> Settings:
    s, sat, core = settings.strategy, settings.satellite, settings.core
    _require(settings.mode in {"paper", "live"}, "mode は paper か live です")
    _require(settings.broker in {"sbi", "rakuten"}, "broker は sbi か rakuten です")
    _require(sat.initial_capital > 0, "satellite.initial_capital は正の値です")
    _require(sat.monthly_contribution >= 0, "satellite.monthly_contribution は0以上です")
    _require(sat.lot_size >= 1, "satellite.lot_size は1以上です")
    _require(0 <= sat.cost_rate < 0.05, "satellite.cost_rate は0以上0.05未満です")
    _require(0 <= sat.tax_rate < 1, "satellite.tax_rate は0以上1未満です")
    _require(core.initial_amount >= 0 and core.monthly_amount >= 0, "core の金額は0以上です")
    _require(s.max_positions >= 1, "strategy.max_positions は1以上です")
    _require(s.keep_rank >= s.max_positions, "strategy.keep_rank は max_positions 以上です")
    _require(s.max_per_sector >= 1, "strategy.max_per_sector は1以上です")
    _require(s.momentum_lookback > s.momentum_skip >= 0, "momentum_lookback > momentum_skip >= 0 です")
    _require(s.trend_window >= 20, "strategy.trend_window は20以上です")
    _require(s.trailing_stop is None or 0 < s.trailing_stop < 1, "strategy.trailing_stop は0〜1です")
    _require(s.max_drawdown is None or 0 < s.max_drawdown < 1, "strategy.max_drawdown は0〜1です")
    _require(s.drawdown_pause_months >= 0, "strategy.drawdown_pause_months は0以上です")
    _require(len(settings.universe.tiers) > 0, "universe.tiers を1つ以上指定してください")
    return settings


def load_settings(path: Path | str = DEFAULT_CONFIG_PATH) -> Settings:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError("settings.yaml の形式が不正です")
    sections = {
        "core": CoreConfig,
        "satellite": SatelliteConfig,
        "strategy": StrategyConfig,
        "universe": UniverseConfig,
        "review": ReviewConfig,
        "notify": NotifyConfig,
    }
    unknown = set(raw) - set(sections) - {"mode", "broker"}
    if unknown:
        raise ConfigError(f"settings.yaml に不明な項目があります: {sorted(unknown)}")
    built = {name: _build(cls, raw.get(name), name) for name, cls in sections.items()}
    settings = Settings(mode=raw.get("mode", "paper"), broker=raw.get("broker", "sbi"), **built)
    return validate(settings)
