"""Compute today's target weights and rebalance the Alpaca account.

Intended to run on the first trading day of each month (see schedule/).
Uses prices through the prior close, so run `scripts/ingest.py` first (run_daily.sh does both).

  python scripts/rebalance.py            # respects settings: paper + dry_run
  python scripts/rebalance.py --send     # override dry_run (still paper unless --live)
  python scripts/rebalance.py --force    # run even if it's not the first trading day of the month
"""
import argparse
import sys

import _bootstrap  # noqa: F401
import pandas as pd

from deskfund.config import load_settings
from deskfund.data.store import Store
from deskfund.execution.alpaca_broker import AlpacaBroker
from deskfund.pipeline import sleeve_prices, target_weights


def first_trading_day_of_month(today: pd.Timestamp) -> bool:
    from pandas.tseries.holiday import USFederalHolidayCalendar
    from pandas.tseries.offsets import CustomBusinessDay

    bday = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    first = pd.date_range(today.replace(day=1), periods=1, freq=bday)[0]
    return today.normalize() == first.normalize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--send", action="store_true", help="send orders even if settings say dry_run")
    ap.add_argument("--live", action="store_true", help="required (with paper: false) to trade a live account")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    s = load_settings()
    ex = s.execution
    today = pd.Timestamp.today().normalize()
    if not a.force and not first_trading_day_of_month(today):
        print(f"{today.date()} is not the first trading day of the month — nothing to do (use --force).")
        return

    paper = bool(ex.get("paper", True))
    if not paper and not a.live:
        sys.exit("settings say paper: false but --live not passed. Refusing to trade a live account.")

    px, _, _ = sleeve_prices(s, Store(s.store_dir))
    targets, _ = target_weights(s, px)
    last_signal = targets.index[-1]
    if (today - last_signal).days > 10:
        sys.exit(f"Latest signal date {last_signal.date()} is stale — run scripts/ingest.py first.")

    w = targets.iloc[-1]
    ticker_of = dict(s.sleeves)
    if s.universe.get("cash"):
        ticker_of["CASH"] = s.universe["cash"]
    tw = {ticker_of[k]: float(v) for k, v in w.items() if v > 0}
    print(f"signal date {last_signal.date()} -> targets {tw}")

    broker = AlpacaBroker(s.env("ALPACA_API_KEY"), s.env("ALPACA_SECRET_KEY"), paper=paper)
    dry = bool(ex.get("dry_run", True)) and not a.send
    broker.rebalance(tw, min_trade_usd=float(ex.get("min_trade_usd", 50)), dry_run=dry)


if __name__ == "__main__":
    main()
