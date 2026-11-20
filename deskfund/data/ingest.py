"""Incremental ingest: only pulls dates after what's already stored."""
from __future__ import annotations

import logging

import pandas as pd

from ..config import Settings
from .fred import fetch_fred
from .providers import get_provider
from .store import Store

log = logging.getLogger(__name__)


def ingest_prices(settings: Settings, store: Store, provider_name: str | None = None, full: bool = False) -> int:
    provider = get_provider(provider_name or settings.data["price_provider"], settings)
    start_default = settings.data["start_date"]
    total = 0
    for t in settings.all_tickers:
        last = None if full else store.last_date(t)
        start = start_default if last is None else (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if last is not None and last >= pd.Timestamp.today().normalize() - pd.Timedelta(days=1):
            log.info("%s up to date", t)
            continue
        df = provider.fetch([t], start)
        store.write_prices(df)
        total += len(df)
        log.info("%s: +%d rows from %s (%s)", t, len(df), start, provider.name)
    return total


def ingest_macro(settings: Settings, store: Store) -> int:
    series = settings.data.get("fred_series") or []
    if not series:
        return 0
    df = fetch_fred(series, settings.data["start_date"])
    store.write_macro(df)
    return len(df)
