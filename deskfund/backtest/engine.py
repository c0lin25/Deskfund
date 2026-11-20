"""Daily drifting-weights backtest with periodic rebalances.

Timing (no look-ahead):
  signal computed at close of day d
  trade executes at close of d + `lag` trading days (default 1)
  new weights earn returns from the following day
Between trades weights drift with prices (buy-and-hold), so turnover and
attribution reflect what a real book would do.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    returns: pd.DataFrame            # daily: port, bench, active, cost
    targets: pd.DataFrame            # target weights, indexed by trade date
    turnover: pd.Series              # one-way turnover per trade date
    periods: list[dict] = field(default_factory=list)  # per holding period, for attribution
    bench_weights: pd.Series | None = None

    @property
    def equity(self) -> pd.DataFrame:
        return (1 + self.returns[["port", "bench"]]).cumprod()


def _simulate(rets: pd.DataFrame, targets: pd.DataFrame, cost_bps: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    cols = rets.columns
    w = None
    port = pd.Series(0.0, index=rets.index)
    cost = pd.Series(0.0, index=rets.index)
    turnover = {}
    trade_dates = set(targets.index)
    r_np = rets.fillna(0.0).values
    pending = 0.0  # cost of trading at yesterday's close, charged against today's capital
    for i, t in enumerate(rets.index):
        if w is not None:
            gross = float(w @ r_np[i])
            port.iloc[i] = (1 + gross) * (1 - pending) - 1
            cost.iloc[i] = pending
            pending = 0.0
            w = w * (1 + r_np[i]) / (1 + gross) if (1 + gross) != 0 else w
        if t in trade_dates:
            tgt = targets.loc[t].reindex(cols).fillna(0.0).values
            prev = np.zeros(len(cols)) if w is None else w
            to = float(np.abs(tgt - prev).sum()) / 2  # one-way
            turnover[t] = to
            pending = 2 * to * cost_bps / 1e4  # cost on both legs
            w = tgt
    return port, cost, pd.Series(turnover, name="turnover")


def run_backtest(
    px: pd.DataFrame,
    signal_weights: pd.DataFrame,
    bench_weights: pd.Series,
    cost_bps: float = 5.0,
    lag: int = 1,
) -> BacktestResult:
    """
    px              wide prices, columns = sleeve names (incl. cash if used)
    signal_weights  target weights indexed by *signal* date
    bench_weights   static policy weights (rebalanced on the same schedule, no costs)
    """
    px = px.sort_index()
    rets = px.pct_change()
    idx = px.index

    # map signal dates -> trade dates
    pos = idx.get_indexer(signal_weights.index)
    keep = (pos >= 0) & (pos + lag < len(idx))
    targets = signal_weights[keep].copy()
    targets.index = idx[pos[keep] + lag]
    targets = targets.reindex(columns=px.columns).fillna(0.0)

    bench_t = pd.DataFrame([bench_weights.reindex(px.columns).fillna(0.0)] * len(targets), index=targets.index)

    port, cost, turnover = _simulate(rets, targets, cost_bps)
    bench, _, _ = _simulate(rets, bench_t, 0.0)

    start = targets.index[0] if len(targets) else idx[0]
    out = pd.DataFrame({"port": port, "bench": bench, "cost": cost}).loc[start:]
    out.iloc[0] = 0.0  # first day: trade at close, no exposure yet
    out["active"] = out["port"] - out["bench"]

    # holding periods for attribution: trade close -> next trade close (or last date)
    periods = []
    tds = list(targets.index) + [idx[-1]]
    for a, b in zip(tds[:-1], tds[1:]):
        if b <= a:
            continue
        r = px.loc[b] / px.loc[a] - 1
        nxt = idx[idx.get_loc(a) + 1]  # trade cost lands on the first day of the period
        periods.append({
            "start": a, "end": b,
            "wp": targets.loc[a], "wb": bench_t.loc[a],
            "r": r.fillna(0.0),
            "cost": float(cost.loc[nxt]),
        })

    return BacktestResult(out, targets, turnover, periods, bench_weights)
