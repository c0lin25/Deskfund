"""Brinson-Fachler attribution with Carino multi-period linking.

Single period, per segment i:
  allocation  = (wp_i - wb_i) * (rb_i - Rb)
  selection   =  wb_i * (rp_i - rb_i)
  interaction = (wp_i - wb_i) * (rp_i - rb_i)
Sum over segments = Rp - Rb exactly (before costs).

When the portfolio holds the same instrument as the benchmark segment
(one ETF per region), rp_i == rb_i so selection and interaction are zero and
all active return is allocation. Plug in sleeve-level holdings later and the
same function splits selection out.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def brinson_fachler(wp: pd.Series, wb: pd.Series, rp: pd.Series, rb: pd.Series) -> pd.DataFrame:
    idx = wp.index.union(wb.index)
    wp, wb = wp.reindex(idx).fillna(0.0), wb.reindex(idx).fillna(0.0)
    rp, rb = rp.reindex(idx).fillna(0.0), rb.reindex(idx).fillna(0.0)
    Rb = float((wb * rb).sum())
    return pd.DataFrame({
        "allocation": (wp - wb) * (rb - Rb),
        "selection": wb * (rp - rb),
        "interaction": (wp - wb) * (rp - rb),
    })


def _carino_k(r: float, b: float) -> float:
    if abs(r - b) < 1e-12:
        return 1.0 / (1.0 + r)
    return (np.log1p(r) - np.log1p(b)) / (r - b)


def attribute(periods: list[dict], rp_by_period: list[pd.Series] | None = None) -> dict:
    """
    periods: from BacktestResult.periods (wp, wb, r, cost per holding period)
    rp_by_period: optional per-segment portfolio returns if they differ from benchmark segment returns

    Returns dict with:
      by_period   long table (end, segment, allocation, selection, interaction)
      linked      per-segment effects linked over the full horizon (Carino), plus cost row
      check       totals: port, bench, active, sum_of_effects, residual
    """
    rows, port_r, bench_r, costs = [], [], [], []
    for k, p in enumerate(periods):
        rb = p["r"]
        rp = rp_by_period[k] if rp_by_period is not None else rb
        bf = brinson_fachler(p["wp"], p["wb"], rp, rb)
        gross = float((p["wp"].reindex(bf.index).fillna(0) * rp.reindex(bf.index).fillna(0)).sum())
        R = (1 + gross) * (1 - p.get("cost", 0.0)) - 1  # cost charged on capital at period start
        B = float((p["wb"].reindex(bf.index).fillna(0) * rb.reindex(bf.index).fillna(0)).sum())
        port_r.append(R)
        bench_r.append(B)
        costs.append(gross - R)  # cost drag in return terms
        bf["end"] = p["end"]
        rows.append(bf.reset_index(names="segment"))

    if not rows:
        return {"by_period": pd.DataFrame(), "linked": pd.DataFrame(), "check": {}}

    by_period = pd.concat(rows, ignore_index=True)
    R_tot = float(np.prod([1 + x for x in port_r]) - 1)
    B_tot = float(np.prod([1 + x for x in bench_r]) - 1)
    K = _carino_k(R_tot, B_tot)
    ks = [_carino_k(r, b) / K for r, b in zip(port_r, bench_r)]

    ends = [p["end"] for p in periods]
    scale = pd.Series(ks, index=ends)
    eff = by_period.set_index("end")[["segment", "allocation", "selection", "interaction"]]
    scaled = eff[["allocation", "selection", "interaction"]].mul(scale.reindex(eff.index).values, axis=0)
    scaled["segment"] = eff["segment"].values
    linked = scaled.groupby("segment")[["allocation", "selection", "interaction"]].sum()
    linked["total"] = linked.sum(axis=1)
    cost_linked = -float(sum(c * k for c, k in zip(costs, ks)))
    linked.loc["(costs)"] = [0.0, 0.0, 0.0, cost_linked]

    effects_sum = float(linked["total"].sum())
    check = {
        "port": R_tot,
        "bench": B_tot,
        "active": R_tot - B_tot,
        "sum_of_effects": effects_sum,
        "residual": (R_tot - B_tot) - effects_sum,
    }
    return {"by_period": by_period, "linked": linked, "check": check}
