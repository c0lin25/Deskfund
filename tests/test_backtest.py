import numpy as np
import pandas as pd

from deskfund.backtest.engine import run_backtest
from deskfund.data.providers.synthetic import SyntheticProvider
from deskfund.research.ic import rebalance_dates
from deskfund.research.signals import compute_signals


def _px():
    raw = SyntheticProvider().fetch(["SPY", "EFA", "EEM"], "2015-01-01", "2020-12-31")
    return raw.pivot(index="date", columns="ticker", values="adj_close").rename(
        columns={"SPY": "US", "EFA": "EAFE", "EEM": "EM"})


def test_holding_benchmark_has_zero_active_except_first_trade_cost():
    px = _px()
    bench = pd.Series({"US": 0.64, "EAFE": 0.25, "EM": 0.11})
    dates = rebalance_dates(px.index)
    targets = pd.DataFrame([bench] * len(dates), index=dates)
    res = run_backtest(px, targets, bench, cost_bps=5)
    r = res.returns
    no_cost = r["cost"] == 0
    assert np.allclose(r.loc[no_cost, "active"], 0, atol=1e-12)
    # Costs (initial buy-in + rebalancing drift back to policy) are the only source of active return
    c = r.loc[~no_cost]
    assert np.allclose(1 + c["port"], (1 + c["bench"]) * (1 - c["cost"]), atol=1e-12)


def test_turnover_is_one_way_and_bounded():
    px = _px()
    bench = pd.Series({"US": 0.64, "EAFE": 0.25, "EM": 0.11})
    dates = rebalance_dates(px.index)
    flip = [pd.Series({"US": 1.0, "EAFE": 0.0, "EM": 0.0}), pd.Series({"US": 0.0, "EAFE": 1.0, "EM": 0.0})]
    targets = pd.DataFrame([flip[i % 2] for i in range(len(dates))], index=dates)
    res = run_backtest(px, targets, bench)
    assert (res.turnover.iloc[1:] <= 1.0 + 1e-9).all()
    assert res.turnover.iloc[1:].mean() > 0.9  # near-full swap every month


def test_signals_have_no_lookahead():
    px = _px()
    cut = px.index[800]
    sig_a = compute_signals(px, {"momentum_12_1": 0.5, "trend_200d": 0.5})["composite"]
    shocked = px.copy()
    shocked.loc[shocked.index > cut] *= 3.0  # change only the future
    sig_b = compute_signals(shocked, {"momentum_12_1": 0.5, "trend_200d": 0.5})["composite"]
    pd.testing.assert_frame_equal(sig_a.loc[:cut], sig_b.loc[:cut])


def test_attribution_matches_backtest_active():
    from deskfund.attribution.brinson import attribute
    px = _px()
    bench = pd.Series({"US": 0.64, "EAFE": 0.25, "EM": 0.11})
    sig = compute_signals(px, {"momentum_3m": 1.0})["composite"]
    dates = [d for d in rebalance_dates(px.index) if sig.loc[d].notna().all()]
    targets = pd.DataFrame({d: (bench + 0.1 * sig.loc[d]).clip(lower=0) for d in dates}).T
    targets = targets.div(targets.sum(axis=1), axis=0)
    res = run_backtest(px, targets, bench, cost_bps=5)
    out = attribute(res.periods)
    eq = (1 + res.returns[["port", "bench"]]).prod() - 1
    assert np.isclose(out["check"]["port"], eq["port"], atol=1e-9)
    assert np.isclose(out["check"]["bench"], eq["bench"], atol=1e-9)
    assert abs(out["check"]["residual"]) < 1e-10
