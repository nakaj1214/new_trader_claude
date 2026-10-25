"""過去データでの検証（GitHub Actions の手動実行、またはローカルで実行）。"""

from trader.backtest_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
