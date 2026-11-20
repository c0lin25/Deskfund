"""Signals. Input: wide price matrix (index=date, columns=sleeve). Output: same shape.

Every signal at date t uses data through the close of t only.
Add a signal: write a function, register it in SIGNALS, give it a weight in settings.yaml.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

TD_MONTH = 21
TD_YEAR = 252


def momentum_12_1(px: pd.DataFrame) -> pd.DataFrame:
    """12-month return skipping the most recent month (classic Jegadeesh-Titman)."""
    return px.shift(TD_MONTH) / px.shift(TD_YEAR) - 1


def momentum_3m(px: pd.DataFrame) -> pd.DataFrame:
    return px / px.shift(3 * TD_MONTH) - 1


def trend_200d(px: pd.DataFrame) -> pd.DataFrame:
    """Distance from 200-day moving average."""
    return px / px.rolling(200, min_periods=200).mean() - 1


def low_vol_63d(px: pd.DataFrame) -> pd.DataFrame:
    """Negative 3-month realized vol (higher = calmer)."""
    return -px.pct_change().rolling(63, min_periods=63).std() * np.sqrt(TD_YEAR)


SIGNALS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "momentum_12_1": momentum_12_1,
    "momentum_3m": momentum_3m,
    "trend_200d": trend_200d,
    "low_vol_63d": low_vol_63d,
}


def cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score per date. Rows with zero dispersion -> 0."""
    mu = df.mean(axis=1)
    sd = df.std(axis=1, ddof=0)
    z = df.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)
    z.loc[sd == 0] = 0.0  # no dispersion -> neutral
    return z.where(df.notna())


def compute_signals(px: pd.DataFrame, weights: dict[str, float]) -> dict[str, pd.DataFrame]:
    """Raw signals by name, plus 'composite' = weighted sum of z-scores."""
    out: dict[str, pd.DataFrame] = {}
    composite = None
    total_w = sum(abs(w) for w in weights.values()) or 1.0
    for name, w in weights.items():
        if name not in SIGNALS:
            raise KeyError(f"Unknown signal '{name}'. Available: {list(SIGNALS)}")
        raw = SIGNALS[name](px)
        out[name] = raw
        z = cs_zscore(raw) * (w / total_w)
        composite = z if composite is None else composite.add(z, fill_value=np.nan)
    out["composite"] = cs_zscore(composite)
    return out
