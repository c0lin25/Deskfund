"""Load settings.yaml + .env into one object."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SETTINGS = ROOT / "config" / "settings.yaml"


@dataclass
class Settings:
    raw: dict[str, Any]
    root: Path = ROOT

    # --- convenience accessors -------------------------------------------
    @property
    def data(self) -> dict:
        return self.raw["data"]

    @property
    def universe(self) -> dict:
        return self.raw["universe"]

    @property
    def strategy(self) -> dict:
        return self.raw["strategy"]

    @property
    def execution(self) -> dict:
        return self.raw["execution"]

    @property
    def sleeves(self) -> dict[str, str]:
        """{'US': 'SPY', ...}"""
        return dict(self.universe["sleeves"])

    @property
    def benchmark_weights(self) -> dict[str, float]:
        w = dict(self.universe["benchmark"]["weights"])
        total = sum(w.values())
        return {k: v / total for k, v in w.items()}

    @property
    def all_tickers(self) -> list[str]:
        t = list(self.sleeves.values())
        t.append(self.universe["benchmark"]["ticker"])
        if self.universe.get("cash"):
            t.append(self.universe["cash"])
        return sorted(set(t))

    @property
    def store_dir(self) -> Path:
        p = Path(self.data["store_dir"])
        return p if p.is_absolute() else self.root / p

    @staticmethod
    def env(key: str) -> str | None:
        v = os.getenv(key)
        return v or None


def load_settings(path: str | Path | None = None) -> Settings:
    if load_dotenv is not None:
        load_dotenv(ROOT / ".env")
    path = Path(path) if path else DEFAULT_SETTINGS
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Settings(raw=raw)
