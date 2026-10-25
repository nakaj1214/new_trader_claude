"""無料データにある明らかな誤り（株式分割の未調整など）の検出と補正。

東証には1日の値幅制限があり、TOPIX100 の銘柄や ETF が1日で 0.55 倍以下・1.8 倍以上に
なることは実質的に起こらない。そのような値動きは、分割比率に近ければ未調整の分割として
過去の価格を補正し、近くなければ「信頼できないデータ」として判断から外す。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

LOW_RATIO = 0.55
HIGH_RATIO = 1.8
SPLIT_FACTORS = (2, 3, 4, 5, 10, 20, 25, 50, 100)
TOLERANCE = 0.15
PRICE_FIELDS = ("open", "high", "low", "close", "adj_close")


@dataclass(frozen=True)
class Repair:
    ticker: str
    date: pd.Timestamp
    factor: float  # 株数の倍率（2 = 1株を2株に分割、0.5 = 2株を1株に併合）


@dataclass
class QualityReport:
    repairs: list[Repair] = field(default_factory=list)
    unresolved: dict[str, list[pd.Timestamp]] = field(default_factory=dict)

    @property
    def unreliable(self) -> set[str]:
        return set(self.unresolved)


def is_implausible(ratio: float) -> bool:
    return ratio <= LOW_RATIO or ratio >= HIGH_RATIO


def split_factor(ratio: float) -> float | None:
    """前日比 ratio が分割（または併合）で説明できるなら、その株数倍率を返す。"""
    for k in SPLIT_FACTORS:
        if abs(ratio * k - 1) <= TOLERANCE:
            return float(k)
        if abs(ratio / k - 1) <= TOLERANCE:
            return 1.0 / k
    return None


def repair_market(md) -> QualityReport:  # noqa: ANN001 - 循環 import を避けるため MarketData を型注釈しない
    """md を直接補正し、補正内容を返す。補正した分割は md.splits にも記録する（保有株数の調整に使う）。"""
    report = QualityReport()
    for ticker in md.tickers:
        close = md.close[ticker].dropna()
        ratios = close / close.shift(1)
        for day, ratio in ratios[ratios.map(is_implausible)].items():
            factor = split_factor(float(ratio))
            if factor is None:
                report.unresolved.setdefault(ticker, []).append(day)
                continue
            before = md.close.index < day
            for name in PRICE_FIELDS:
                frame = getattr(md, name)
                frame.loc[before, ticker] = frame.loc[before, ticker] / factor
            md.volume.loc[before, ticker] = md.volume.loc[before, ticker] * factor
            md.dividends.loc[before, ticker] = md.dividends.loc[before, ticker] / factor
            if md.splits.at[day, ticker] in (0.0, 1.0):
                md.splits.at[day, ticker] = factor
            report.repairs.append(Repair(ticker, day, factor))
    md._signals.clear()
    return report
