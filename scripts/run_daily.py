"""毎営業日の売買判断と Discord 通知（GitHub Actions から実行）。"""

from trader.daily import main

if __name__ == "__main__":
    raise SystemExit(main())
