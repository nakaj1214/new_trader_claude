"""売買対象の銘柄一覧（JPX「東証上場銘柄一覧」から規模区分で抽出）。"""

from __future__ import annotations

import io
from collections.abc import Callable, Iterable
from datetime import date
from pathlib import Path

import pandas as pd
import requests

JPX_LISTING_URL = (
    "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
)
UNIVERSE_COLUMNS = ["code", "ticker", "name", "sector", "size", "as_of"]


class UniverseError(RuntimeError):
    """銘柄一覧を用意できないときに送出する（推測の一覧では動かさない）。"""


def to_ticker(code: str) -> str:
    """東証コード（例: 7203, 130A）を Yahoo Finance のティッカーへ変換する。"""
    code = str(code).strip().upper()
    if code.endswith(".0"):
        code = code[:-2]
    return f"{code}.T"


def parse_jpx_listing(raw: pd.DataFrame, tiers: Iterable[str], as_of: date) -> pd.DataFrame:
    """JPX の一覧表から指定した規模区分の銘柄を抜き出す。"""
    df = raw.rename(columns=lambda c: str(c).strip())
    required = {"コード", "銘柄名", "33業種区分", "規模区分"}
    missing = required - set(df.columns)
    if missing:
        raise UniverseError(f"JPXの一覧に必要な列がありません: {sorted(missing)}")
    wanted = {t.strip() for t in tiers}
    size = df["規模区分"].astype(str).str.strip()
    picked = df[size.isin(wanted)]
    if picked.empty:
        raise UniverseError(f"規模区分 {sorted(wanted)} に該当する銘柄がありません")
    out = pd.DataFrame(
        {
            "code": picked["コード"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True),
            "name": picked["銘柄名"].astype(str).str.strip(),
            "sector": picked["33業種区分"].astype(str).str.strip(),
            "size": picked["規模区分"].astype(str).str.strip(),
        }
    )
    out["ticker"] = out["code"].map(to_ticker)
    out["as_of"] = as_of.isoformat()
    return out[UNIVERSE_COLUMNS].sort_values("code").reset_index(drop=True)


def fetch_jpx_listing(timeout: float = 30.0) -> pd.DataFrame:
    response = requests.get(JPX_LISTING_URL, timeout=timeout)
    response.raise_for_status()
    return pd.read_excel(io.BytesIO(response.content), dtype=str)


def load_universe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    missing = set(UNIVERSE_COLUMNS) - set(df.columns)
    if missing or df.empty:
        raise UniverseError(f"{path} の形式が不正です（不足列: {sorted(missing)}）")
    return df


def ensure_universe(
    path: Path,
    tiers: Iterable[str],
    max_age_days: int,
    today: date,
    fetch: Callable[[], pd.DataFrame] = fetch_jpx_listing,
) -> tuple[pd.DataFrame, list[str]]:
    """保存済みの一覧が古ければ取り直す。取り直しに失敗しても既存の一覧があれば使い続ける。"""
    warnings: list[str] = []
    existing: pd.DataFrame | None = None
    if path.exists():
        existing = load_universe(path)
        age = (today - date.fromisoformat(existing["as_of"].iloc[0])).days
        if age <= max_age_days:
            return existing, warnings
    try:
        fresh = parse_jpx_listing(fetch(), tiers, today)
    except Exception as exc:  # noqa: BLE001 - 失敗の種類を残して既存一覧へ退避する
        if existing is None:
            raise UniverseError(f"銘柄一覧を取得できませんでした: {type(exc).__name__}: {exc}") from exc
        warnings.append(f"銘柄一覧の更新に失敗したため前回の一覧を使います（{type(exc).__name__}）")
        return existing, warnings
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh.to_csv(path, index=False)
    return fresh, warnings
