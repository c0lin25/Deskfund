"""Information-coefficient analysis for any signal.

With a 3-sleeve universe, each period's rank IC is one of only a few values,
so judge the signal on the IC *series* (mean, ICIR, hit rate), not single months.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rebalance_dates(index: pd.DatetimeIndex, freq: str = "M") -> pd.DatetimeIndex:
    """Last trading day in each period that exists in `index`."""
    s = pd.Series(index, index=index)
    key = {"M": index.to_period("M"), "W": index.to_period("W"), "Q": index.to_period("Q")}[freq]
    return pd.DatetimeIndex(s.groupby(key).max().values)


def forward_returns(px: pd.DataFrame, dates: pd.DatetimeIndex, periods: int = 1) -> pd.DataFrame:
    """Return from date[i] close to date[i+periods] close."""
    p = px.reindex(dates)
    return p.shift(-periods) / p - 1


def ic_series(signal: pd.DataFrame, fwd: pd.DataFrame, method: str = "spearman") -> pd.Series:
    sig = signal.reindex(fwd.index)
    vals = {}
    for d in fwd.index:
        a, b = sig.loc[d], fwd.loc[d]
        m = a.notna() & b.notna()
        if m.sum() >= 3:
            vals[d] = a[m].corr(b[m], method=method)
    return pd.Series(vals, name="ic", dtype=float)


def ic_summary(ic: pd.Series, periods_per_year: int = 12) -> dict:
    ic = ic.dropna()
    if ic.empty:
        return {"n": 0}
    mean, sd = ic.mean(), ic.std(ddof=1)
    return {
        "n": int(len(ic)),
        "mean_ic": float(mean),
        "ic_std": float(sd),
        "icir": float(mean / sd) if sd > 0 else np.nan,
        "icir_annual": float(mean / sd * np.sqrt(periods_per_year)) if sd > 0 else np.nan,
        "t_stat": float(mean / sd * np.sqrt(len(ic))) if sd > 0 else np.nan,
        "hit_rate": float((ic > 0).mean()),
    }


def ic_decay(signal: pd.DataFrame, px: pd.DataFrame, dates: pd.DatetimeIndex, horizons=(1, 2, 3, 6, 12)) -> pd.DataFrame:
    """Mean IC and ICIR by forward horizon (in rebalance periods)."""
    rows = []
    for h in horizons:
        s = ic_summary(ic_series(signal, forward_returns(px, dates, h)))
        rows.append({"horizon": h, "mean_ic": s.get("mean_ic"), "icir": s.get("icir"), "n": s.get("n")})
    return pd.DataFrame(rows).set_index("horizon")


def signal_report(signals: dict[str, pd.DataFrame], px: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    """One row per signal: IC stats at a 1-period horizon."""
    dates = rebalance_dates(px.index, freq)
    fwd = forward_returns(px, dates, 1)
    ppy = {"M": 12, "W": 52, "Q": 4}[freq]
    return pd.DataFrame({name: ic_summary(ic_series(sig, fwd), ppy) for name, sig in signals.items()}).T
