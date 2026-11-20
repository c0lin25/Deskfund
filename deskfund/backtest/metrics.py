"""Performance and risk statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd

TD = 252


def _cagr(r: pd.Series) -> float:
    r = r.dropna()
    if r.empty:
        return np.nan
    years = len(r) / TD
    return float((1 + r).prod() ** (1 / years) - 1) if years > 0 else np.nan


def max_drawdown(r: pd.Series) -> float:
    eq = (1 + r.fillna(0)).cumprod()
    return float((eq / eq.cummax() - 1).min())


def drawdown_series(r: pd.Series) -> pd.Series:
    eq = (1 + r.fillna(0)).cumprod()
    return eq / eq.cummax() - 1


def summary(res, rf: pd.Series | None = None) -> pd.DataFrame:
    """Table of stats for port, bench, and active."""
    d = res.returns
    rf = (rf.reindex(d.index).fillna(0.0) if rf is not None else pd.Series(0.0, index=d.index))
    rows = {}
    for col in ["port", "bench"]:
        r = d[col]
        vol = r.std() * np.sqrt(TD)
        ex = r - rf
        rows[col] = {
            "CAGR": _cagr(r),
            "Vol": vol,
            "Sharpe": ex.mean() / ex.std() * np.sqrt(TD) if ex.std() > 0 else np.nan,
            "MaxDD": max_drawdown(r),
            "Calmar": _cagr(r) / abs(max_drawdown(r)) if max_drawdown(r) < 0 else np.nan,
        }
    a = d["active"]
    te = a.std() * np.sqrt(TD)
    m = (1 + d[["port", "bench"]]).resample("ME").prod() - 1
    years = len(d) / TD
    rows["active"] = {
        "CAGR": _cagr(d["port"]) - _cagr(d["bench"]),
        "Vol": te,
        "Sharpe": (a.mean() * TD) / te if te > 0 else np.nan,  # = information ratio
        "MaxDD": max_drawdown(a),
        "Calmar": np.nan,
    }
    out = pd.DataFrame(rows).T
    out.attrs["monthly_hit_rate"] = float((m["port"] > m["bench"]).mean())
    out.attrs["annual_turnover"] = float(res.turnover.sum() / years) if years > 0 else np.nan
    out.attrs["total_cost_bps"] = float(d["cost"].sum() * 1e4)
    return out


def format_summary(tbl: pd.DataFrame) -> str:
    t = tbl.copy()
    t = t.rename(index={"active": "active (IR in Sharpe col, TE in Vol col)"})
    lines = [t.to_string(float_format=lambda x: f"{x:,.3f}")]
    lines.append(f"monthly hit rate vs bench: {tbl.attrs['monthly_hit_rate']:.1%}")
    lines.append(f"annual one-way turnover:   {tbl.attrs['annual_turnover']:.1%}")
    lines.append(f"total costs paid:          {tbl.attrs['total_cost_bps']:.0f} bps")
    return "\n".join(lines)
