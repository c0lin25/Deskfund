"""Pull/refresh prices and macro data into data/.

  python scripts/ingest.py                    # provider from settings.yaml, incremental
  python scripts/ingest.py --provider yahoo   # override provider
  python scripts/ingest.py --full             # re-pull full history
"""
import argparse

import _bootstrap  # noqa: F401

from deskfund.config import load_settings
from deskfund.data.ingest import ingest_macro, ingest_prices
from deskfund.data.store import Store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--no-macro", action="store_true")
    a = ap.parse_args()

    s = load_settings()
    store = Store(s.store_dir)
    n = ingest_prices(s, store, a.provider, full=a.full)
    print(f"prices: +{n} rows")
    if not a.no_macro:
        try:
            print(f"macro:  {ingest_macro(s, store)} rows")
        except Exception as e:  # FRED is optional
            print(f"macro skipped: {e}")
    print(store.query("SELECT ticker, min(date) AS first, max(date) AS last, count(*) AS n FROM prices GROUP BY 1 ORDER BY 1"))


if __name__ == "__main__":
    main()
