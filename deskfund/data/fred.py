"""FRED macro series via the public CSV endpoint (no API key needed)."""
from __future__ import annotations

import io

import pandas as pd
import requests

URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def fetch_fred(series: list[str], start: str) -> pd.DataFrame:
    """Long format: date, series, value."""
    frames = []
    for s in series:
        r = requests.get(URL, params={"id": s, "cosd": start}, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = ["date", "value"]
        df["value"] = pd.to_numeric(df["value"], errors="coerce")  # FRED uses "." for missing
        df["date"] = pd.to_datetime(df["date"])
        df["series"] = s
        frames.append(df.dropna())
    if not frames:
        return pd.DataFrame(columns=["date", "series", "value"])
    return pd.concat(frames, ignore_index=True)[["date", "series", "value"]]
