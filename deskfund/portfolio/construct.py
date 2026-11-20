"""Turn a composite score into target weights.

Methods
  active_tilt     benchmark + capped active tilt proportional to the score (default)
  inverse_vol     risk-parity-lite baseline, ignores the score
  black_litterman equilibrium prior from benchmark weights + score-driven views
                  (numpy closed form, long-only, no extra dependencies)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TD_YEAR = 252


def _project_active(active: pd.Series, cap: float, iters: int = 50) -> pd.Series:
    """Make actives sum to zero while each stays within +/- cap."""
    a = active.copy()
    for _ in range(iters):
        a = a.clip(-cap, cap)
        excess = a.sum()
        if abs(excess) < 1e-12:
            break
        # Subtracting (excess > 0): only names above the floor can move down; adding: only below the cap.
        free = a[a > -cap + 1e-12] if excess > 0 else a[a < cap - 1e-12]
        if free.empty:
            break
        a.loc[free.index] -= excess / len(free)
    return a.clip(-cap, cap)


def _finalize(w: pd.Series, min_weight: float) -> pd.Series:
    w = w.clip(lower=min_weight)
    return w / w.sum()


def active_tilt(score: pd.Series, bench: pd.Series, scale: float, cap: float, min_weight: float = 0.0) -> pd.Series:
    s = score.reindex(bench.index).fillna(0.0)
    active = _project_active(scale * (s - s.mean()), cap)
    return _finalize(bench + active, min_weight)


def inverse_vol(returns: pd.DataFrame, lookback: int = 63, min_weight: float = 0.0) -> pd.Series:
    vol = returns.tail(lookback).std()
    return _finalize(1.0 / vol.replace(0, np.nan).fillna(vol.max()), min_weight)


def black_litterman(
    score: pd.Series,
    bench: pd.Series,
    returns: pd.DataFrame,
    lookback: int = TD_YEAR,
    delta: float = 2.5,
    tau: float = 0.05,
    view_scale: float = 0.02,
    cap: float = 0.15,
    min_weight: float = 0.0,
) -> pd.Series:
    """Views: each sleeve's excess return = prior + view_scale * score * vol.

    view_scale=0.02 means a +1 z-score sleeve gets a +2% x (its vol / 1) annual view.
    Confidence follows He-Litterman (Omega = diag(P tau Sigma P')).
    """
    names = bench.index
    sigma = returns[names].tail(lookback).cov().values * TD_YEAR
    w_b = bench.values
    pi = delta * sigma @ w_b
    s = score.reindex(names).fillna(0.0).values
    vol = np.sqrt(np.diag(sigma))
    P = np.eye(len(names))
    Q = pi + view_scale * s * vol / max(vol.mean(), 1e-9)
    ts = tau * sigma
    omega = np.diag(np.diag(P @ ts @ P.T))
    mid = np.linalg.inv(np.linalg.inv(ts) + P.T @ np.linalg.inv(omega) @ P)
    mu_post = mid @ (np.linalg.inv(ts) @ pi + P.T @ np.linalg.inv(omega) @ Q)
    w = np.linalg.solve(delta * sigma, mu_post)
    w = pd.Series(np.clip(w, 0, None), index=names)
    w = w / w.sum() if w.sum() > 0 else bench.copy()
    active = _project_active(w - bench, cap)
    return _finalize(bench + active, min_weight)


def build_weights(
    method: str,
    score: pd.Series,
    bench: pd.Series,
    returns: pd.DataFrame,
    cfg: dict,
) -> pd.Series:
    cap = cfg.get("max_active_weight", 0.15)
    mw = cfg.get("min_weight", 0.0)
    if method == "active_tilt":
        return active_tilt(score, bench, cfg.get("tilt_scale", 0.10), cap, mw)
    if method == "inverse_vol":
        return inverse_vol(returns[bench.index], min_weight=mw)
    if method in ("black_litterman", "riskfolio_bl"):
        return black_litterman(score, bench, returns, cap=cap, min_weight=mw)
    raise ValueError(f"Unknown construction method: {method}")


def apply_risk_off(w: pd.Series, trend_row: pd.Series, cash_name: str, cash_weight: float) -> pd.Series:
    """If every sleeve is below its 200d MA, move `cash_weight` of the book to cash."""
    w = w.copy()
    if cash_name not in w.index:
        w[cash_name] = 0.0
    risky = w.index.drop(cash_name)
    if trend_row.reindex(risky).dropna().lt(0).all() and trend_row.reindex(risky).notna().any():
        w[risky] *= 1 - cash_weight
        w[cash_name] = cash_weight
    return w
