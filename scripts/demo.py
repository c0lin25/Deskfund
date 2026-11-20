"""Offline smoke test: synthetic prices -> full pipeline -> data/demo/runs/demo.

No API keys or internet needed. Numbers are meaningless; it proves the plumbing works.
  python scripts/demo.py
  streamlit run app/dashboard.py   # then pick the 'demo' store in the sidebar
"""
import _bootstrap  # noqa: F401

from deskfund.backtest.metrics import format_summary
from deskfund.config import load_settings
from deskfund.data.ingest import ingest_prices
from deskfund.data.store import Store
from deskfund.pipeline import run_strategy


def main():
    s = load_settings()
    s.raw["data"]["price_provider"] = "synthetic"
    s.raw["data"]["store_dir"] = "data/demo"
    store = Store(s.store_dir)
    ingest_prices(s, store, "synthetic", full=True)
    out = run_strategy(s, store, "demo")
    print(format_summary(out.stats))
    print(out.attribution["linked"].round(4))
    print(f"\nsaved -> {out.run_dir}")


if __name__ == "__main__":
    main()
