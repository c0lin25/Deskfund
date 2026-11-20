"""Provider interface. Swapping vendors is a config change, not a rewrite."""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

# Every provider returns this long format:
#   date (datetime64, tz-naive), ticker (str), open, high, low, close, adj_close, volume
PRICE_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]


class PriceProvider(ABC):
    name: str = "base"

    @abstractmethod
    def fetch(self, tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
        """Daily bars in PRICE_COLUMNS long format."""

    @staticmethod
    def _finalize(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        for c in PRICE_COLUMNS:
            if c not in df.columns:
                df[c] = pd.NA
        df = df[PRICE_COLUMNS].dropna(subset=["adj_close"])
        return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def get_provider(name: str, settings) -> PriceProvider:
    name = name.lower()
    if name == "tiingo":
        from .tiingo import TiingoProvider
        return TiingoProvider(settings.env("TIINGO_API_KEY"))
    if name == "fmp":
        from .fmp import FMPProvider
        return FMPProvider(settings.env("FMP_API_KEY"))
    if name == "yahoo":
        from .yahoo import YahooProvider
        return YahooProvider()
    if name == "synthetic":
        from .synthetic import SyntheticProvider
        return SyntheticProvider()
    raise ValueError(f"Unknown price provider: {name}")
