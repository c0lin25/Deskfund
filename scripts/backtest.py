"""Run the strategy over stored history and save results to data/runs/<name>/.

  python scripts/backtest.py
  python scripts/backtest.py --name tilt_v2 --settings config/my_variant.yaml
"""
import argparse

import _bootstrap  # noqa: F401
import pandas as pd

from deskfund.backtest.metrics import format_summary
from deskfund.config import load_settings
from deskfund.data.store import Store
from deskfund.pipeline import run_strategy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name")
    ap.add_argument("--settings")
    a = ap.parse_args()

    s = load_settings(a.settings)
    out = run_strategy(s, Store(s.store_dir), a.name)
    pd.set_option("display.width", 140)
    r = out.result.returns
    print(f"\n=== {out.name}: {r.index[0].date()} -> {r.index[-1].date()} ===\n")
    print(format_summary(out.stats))
    print("\n--- signal IC (1-period forward, rank) ---")
    print(out.ic[["n", "mean_ic", "icir_annual", "t_stat", "hit_rate"]].round(3))
    print("\n--- Brinson-Fachler, Carino-linked (full period) ---")
    print(out.attribution["linked"].round(4))
    chk = out.attribution["check"]
    print(f"active {chk['active']:.4%} | effects {chk['sum_of_effects']:.4%} | residual {chk['residual']:.2e}")
    partial = out.signals_asof != (out.signals_asof + pd.offsets.BMonthEnd(0))
    note = " — month in progress, preview only" if partial else ""
    print(f"\nlatest target (signal {out.signals_asof.date()}{note}):")
    print(out.latest_target.round(4).to_string())
    print(f"\nsaved -> {out.run_dir}")


if __name__ == "__main__":
    main()
