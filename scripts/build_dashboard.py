"""運用状況の Web ページ（site/index.html）を作る（GitHub Actions から実行）。"""

from trader.dashboard import main

if __name__ == "__main__":
    raise SystemExit(main())
