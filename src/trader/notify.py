"""Discord Webhook への送信。"""

from __future__ import annotations

import time

import requests

DISCORD_LIMIT = 1900  # Discord の上限は2000文字。余裕を持たせる


class NotifyError(RuntimeError):
    """通知を送れなかったとき（Webhook の URL はメッセージに含めない）。"""


def split_message(text: str, limit: int = DISCORD_LIMIT) -> list[str]:
    """行の区切りで分割する。1行が長すぎる場合だけ途中で切る。"""
    chunks: list[str] = []
    current = ""
    for line in text.splitlines():
        while len(line) > limit:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(line[:limit])
            line = line[limit:]
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > limit:
            chunks.append(current)
            current = line
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def send_discord(webhook_url: str, text: str, session: requests.Session | None = None) -> None:
    http = session or requests.Session()
    for chunk in split_message(text):
        for attempt in range(2):
            try:
                response = http.post(webhook_url, json={"content": chunk}, timeout=15)
            except requests.RequestException as exc:
                raise NotifyError(f"Discord へ接続できませんでした（{type(exc).__name__}）") from None
            if response.status_code == 429 and attempt == 0:
                time.sleep(min(float(response.headers.get("Retry-After", "2")), 10.0))
                continue
            if response.status_code >= 300:
                raise NotifyError(f"Discord が送信を拒否しました（HTTP {response.status_code}）")
            break
