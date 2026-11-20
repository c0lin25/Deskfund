"""Tiingo EOD prices. Free key: https://www.tiingo.com (internal-use license)."""
from __future__ import annotations

import time

import pandas as pd
import requests

from .base import PriceProvider

URL = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"


class TiingoProvider(PriceProvider):
    name = "tiingo"

    def __init__(self, api_key: str | None):
        if not api_key:
            raise RuntimeError("TIINGO_API_KEY missing — add it to .env (free signup at tiingo.com)")
        self.api_key = api_key

    def fetch(self, tickers, start, end=None):
        frames = []
        for t in tickers:
            params = {"startDate": start, "token": self.api_key, "format": "json"}
            if end:
                params["endDate"] = end
            r = requests.get(URL.format(ticker=t.lower()), params=params, timeout=30)
            if r.status_code == 429:  # free tier is 50 req/hour
                time.sleep(60)
                r = requests.get(URL.format(ticker=t.lower()), params=params, timeout=30)
            r.raise_for_status()
            rows = r.json()
            if not rows:
                continue
            df = pd.DataFrame(rows).rename(columns={"adjClose": "adj_close"})
            df["ticker"] = t.upper()
            frames.append(df)
        if not frames:
            return self._finalize(pd.DataFrame(columns=["date", "ticker", "adj_close"]))
        return self._finalize(pd.concat(frames, ignore_index=True))
