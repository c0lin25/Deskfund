"""Yahoo via yfinance. No key, unofficial — prototyping only. `pip install yfinance`."""
from __future__ import annotations

import pandas as pd

from .base import PriceProvider


class YahooProvider(PriceProvider):
    name = "yahoo"

    def fetch(self, tickers, start, end=None):
        import yfinance as yf

        raw = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False, group_by="ticker")
        frames = []
        for t in tickers:
            sub = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            sub = sub.reset_index().rename(columns=str.lower).rename(columns={"adj close": "adj_close"})
            sub["ticker"] = t.upper()
            frames.append(sub)
        return self._finalize(pd.concat(frames, ignore_index=True))
