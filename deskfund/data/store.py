"""Parquet files on disk, queried through DuckDB.

data/
  prices.parquet   long daily bars (date, ticker, ..., adj_close)
  macro.parquet    long macro series (date, series, value)
  runs/<name>/     backtest outputs (weights, returns, attribution)
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


class Store:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # --- paths -------------------------------------------------------------
    @property
    def prices_path(self) -> Path:
        return self.root / "prices.parquet"

    @property
    def macro_path(self) -> Path:
        return self.root / "macro.parquet"

    def run_dir(self, name: str) -> Path:
        p = self.root / "runs" / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    # --- writes (upsert on key) -------------------------------------------
    def _upsert(self, path: Path, new: pd.DataFrame, keys: list[str]) -> None:
        if path.exists() and not new.empty:
            old = pd.read_parquet(path)
            new = pd.concat([old, new], ignore_index=True).drop_duplicates(keys, keep="last")
        new.sort_values(keys).to_parquet(path, index=False)

    def write_prices(self, df: pd.DataFrame) -> None:
        self._upsert(self.prices_path, df, ["ticker", "date"])

    def write_macro(self, df: pd.DataFrame) -> None:
        self._upsert(self.macro_path, df, ["series", "date"])

    # --- reads ---------------------------------------------------------------
    def query(self, sql: str) -> pd.DataFrame:
        """Ad-hoc SQL. Tables: prices, macro."""
        con = duckdb.connect()
        if self.prices_path.exists():
            con.execute(f"CREATE VIEW prices AS SELECT * FROM read_parquet('{self.prices_path.as_posix()}')")
        if self.macro_path.exists():
            con.execute(f"CREATE VIEW macro AS SELECT * FROM read_parquet('{self.macro_path.as_posix()}')")
        return con.execute(sql).df()

    def last_date(self, ticker: str) -> pd.Timestamp | None:
        if not self.prices_path.exists():
            return None
        d = self.query(f"SELECT max(date) AS d FROM prices WHERE ticker = '{ticker}'")["d"].iloc[0]
        return None if pd.isna(d) else pd.Timestamp(d)

    def adj_close(self, tickers: list[str] | None = None) -> pd.DataFrame:
        """Wide matrix: index=date, columns=ticker, values=adj_close."""
        where = ""
        if tickers:
            where = "WHERE ticker IN (" + ",".join(f"'{t}'" for t in tickers) + ")"
        df = self.query(f"SELECT date, ticker, adj_close FROM prices {where}")
        wide = df.pivot(index="date", columns="ticker", values="adj_close").sort_index()
        wide.index = pd.to_datetime(wide.index)
        return wide

    def macro_wide(self) -> pd.DataFrame:
        if not self.macro_path.exists():
            return pd.DataFrame()
        df = pd.read_parquet(self.macro_path)
        return df.pivot(index="date", columns="series", values="value").sort_index()
