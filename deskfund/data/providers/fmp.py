"""Financial Modeling Prep EOD prices (dividend-adjusted)."""
from __future__ import annotations

import pandas as pd
import requests

from .base import PriceProvider

URL = "https://financialmodelingprep.com/stable/historical-price-eod/dividend-adjusted"


class FMPProvider(PriceProvider):
    name = "fmp"

    def __init__(self, api_key: str | None):
        if not api_key:
            raise RuntimeError("FMP_API_KEY missing — add it to .env")
        self.api_key = api_key

    def fetch(self, tickers, start, end=None):
        frames = []
        for t in tickers:
            params = {"symbol": t.upper(), "from": start, "apikey": self.api_key}
            if end:
                params["to"] = end
            r = requests.get(URL, params=params, timeout=30)
            r.raise_for_status()
            rows = r.json()
            if isinstance(rows, dict):  # older endpoint shape
                rows = rows.get("historical", [])
            if not rows:
                continue
            df = pd.DataFrame(rows).rename(
                columns={"adjOpen": "open", "adjHigh": "high", "adjLow": "low", "adjClose": "adj_close"}
            )
            if "close" not in df:
                df["close"] = df["adj_close"]
            df["ticker"] = t.upper()
            frames.append(df)
        if not frames:
            return self._finalize(pd.DataFrame(columns=["date", "ticker", "adj_close"]))
        return self._finalize(pd.concat(frames, ignore_index=True))
