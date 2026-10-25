"""GitHub Actions の想定外の失敗を Discord へ知らせる（Webhook 未設定なら何もしない）。"""

import os
import sys

from trader.notify import NotifyError, send_discord
from trader.report import render_error


def main() -> int:
    webhook = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook:
        return 0
    run_url = "{}/{}/actions/runs/{}".format(
        os.environ.get("GITHUB_SERVER_URL", "https://github.com"),
        os.environ.get("GITHUB_REPOSITORY", ""),
        os.environ.get("GITHUB_RUN_ID", ""),
    )
    title = sys.argv[1] if len(sys.argv) > 1 else "自動実行が失敗しました"
    try:
        send_discord(webhook, render_error(title, f"ログ: {run_url}"))
    except NotifyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
