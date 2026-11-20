"""End-to-end strategy run: store -> signals -> weights -> backtest -> IC -> attribution -> saved run."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .attribution.brinson import attribute
from .backtest.engine import BacktestResult, run_backtest
from .backtest.metrics import summary
from .config import Settings
from .data.store import Store
from .portfolio.construct import apply_risk_off, build_weights
from .research.ic import ic_decay, rebalance_dates, signal_report
from .research.signals import compute_signals


@dataclass
class RunOutput:
    name: str
    result: BacktestResult
    stats: pd.DataFrame
    ic: pd.DataFrame
    ic_decay: pd.DataFrame
    attribution: dict
    latest_target: pd.Series
    signals_asof: pd.Timestamp
    run_dir: Path


def sleeve_prices(settings: Settings, store: Store) -> tuple[pd.DataFrame, pd.Series | None, pd.Series | None]:
    """Prices with sleeve names as columns (+ CASH if configured), benchmark series, cash returns."""
    sleeves = settings.sleeves
    cash = settings.universe.get("cash")
    bench_t = settings.universe["benchmark"]["ticker"]
    raw = store.adj_close(settings.all_tickers)
    missing = [t for t in sleeves.values() if t not in raw.columns]
    if missing:
        raise RuntimeError(f"No stored prices for {missing}. Run scripts/ingest.py first.")
    px = raw[list(sleeves.values())].rename(columns={v: k for k, v in sleeves.items()})
    cash_ret = None
    if cash and cash in raw.columns:
        px["CASH"] = raw[cash]
        cash_ret = raw[cash].pct_change()
    bench_px = raw[bench_t] if bench_t in raw.columns else None
    # Start where every sleeve has data; forward-fill only small holiday gaps
    px = px.dropna(how="all").ffill(limit=5)
    px = px.loc[px[list(sleeves)].dropna().index.min():]
    return px, bench_px, cash_ret


def target_weights(settings: Settings, px: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Target weights at every rebalance (signal) date."""
    strat = settings.strategy
    cons = strat["construction"]
    risk_off = strat.get("risk_off", {})
    sleeves = list(settings.sleeves)
    sig = compute_signals(px[sleeves], strat["signals"])
    if "trend_200d" not in sig:
        from .research.signals import trend_200d
        trend = trend_200d(px[sleeves])
    else:
        trend = sig["trend_200d"]
    bench = pd.Series(settings.benchmark_weights)[sleeves]
    rets = px[sleeves].pct_change()
    dates = rebalance_dates(px.index, strat.get("rebalance", "M"))
    rows = {}
    for d in dates:
        score = sig["composite"].loc[d]
        if score.isna().any():
            continue  # warm-up: not enough history for every signal
        w = build_weights(cons["method"], score, bench, rets.loc[:d], cons)
        if risk_off.get("enabled") and "CASH" in px.columns:
            w = apply_risk_off(w, trend.loc[d], "CASH", risk_off.get("cash_weight", 0.5))
        rows[d] = w
    return pd.DataFrame(rows).T.fillna(0.0), sig


def run_strategy(settings: Settings, store: Store, name: str | None = None, save: bool = True) -> RunOutput:
    strat = settings.strategy
    name = name or strat.get("name", "run")
    px, _bench_px, cash_ret = sleeve_prices(settings, store)
    targets, sig = target_weights(settings, px)
    if targets.empty:
        raise RuntimeError("No rebalance dates with full signal history — need ~13 months of prices.")

    bench_w = pd.Series(settings.benchmark_weights)
    res = run_backtest(px[targets.columns.union(bench_w.index)], targets, bench_w,
                       cost_bps=strat.get("costs_bps", 5), lag=1)
    stats = summary(res, rf=cash_ret)
    sleeves = list(settings.sleeves)
    ic = signal_report(sig, px[sleeves], strat.get("rebalance", "M"))
    decay = ic_decay(sig["composite"], px[sleeves], rebalance_dates(px.index, strat.get("rebalance", "M")))
    attr = attribute(res.periods)

    run_dir = store.run_dir(name)
    if save:
        res.returns.to_parquet(run_dir / "returns.parquet")
        res.targets.to_parquet(run_dir / "targets.parquet")
        res.turnover.to_frame().to_parquet(run_dir / "turnover.parquet")
        stats.to_parquet(run_dir / "stats.parquet")
        ic.to_parquet(run_dir / "ic.parquet")
        decay.to_parquet(run_dir / "ic_decay.parquet")
        attr["linked"].to_parquet(run_dir / "attribution_linked.parquet")
        attr["by_period"].to_parquet(run_dir / "attribution_by_period.parquet")
        meta = {
            "name": name,
            "created": pd.Timestamp.now().isoformat(timespec="seconds"),
            "provider": settings.data["price_provider"],
            "signals_asof": str(targets.index[-1].date()),
            "latest_target": targets.iloc[-1].round(4).to_dict(),
            "benchmark_weights": settings.benchmark_weights,
            "sleeves": settings.sleeves,
            "stats_attrs": stats.attrs,
            "attribution_check": attr["check"],
            "strategy": strat,
        }
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    return RunOutput(name, res, stats, ic, decay, attr, targets.iloc[-1], targets.index[-1], run_dir)
