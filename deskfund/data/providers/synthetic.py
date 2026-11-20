"""Offline random-walk prices with regime-persistent drifts — for tests and demos.

Drifts persist for ~1 year then switch, so momentum has something real to find.
Results on this data say nothing about real markets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import PriceProvider


class SyntheticProvider(PriceProvider):
    name = "synthetic"

    def __init__(self, seed: int = 7):
        self.seed = seed

    def fetch(self, tickers, start, end=None):
        # Fixed calendar so tickers fetched in separate calls line up and share the common factor.
        full = pd.bdate_range("2000-01-03", end or pd.Timestamp.today().normalize())
        common = np.random.default_rng(self.seed).normal(0.0003, 0.009, len(full))
        keep = full >= pd.Timestamp(start)
        dates = full[keep]
        frames = []
        for t in sorted(tickers):
            tid = sum(ord(c) * 31 ** i for i, c in enumerate(t.upper())) % 10_000  # stable per ticker
            rng = np.random.default_rng(self.seed * 10_000 + tid)
            n_full = len(full)
            if t.upper() in {"BIL", "SHV", "SGOV"}:
                rets = rng.normal(0.00008, 0.0003, n_full)
            else:
                regime_len = 252
                drifts = np.repeat(rng.normal(0, 0.0006, n_full // regime_len + 1), regime_len)[:n_full]
                beta = rng.uniform(0.8, 1.2)
                rets = beta * common + drifts + rng.normal(0, 0.006, n_full)
            rets = rets[keep]
            n = len(rets)
            px = 100 * np.exp(np.cumsum(rets))
            frames.append(pd.DataFrame({
                "date": dates, "ticker": t.upper(), "open": px, "high": px * 1.005,
                "low": px * 0.995, "close": px, "adj_close": px, "volume": 1_000_000,
            }))
        return self._finalize(pd.concat(frames, ignore_index=True))
