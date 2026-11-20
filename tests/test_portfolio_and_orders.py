import numpy as np
import pandas as pd

from deskfund.data.store import Store
from deskfund.execution.orders import plan_orders
from deskfund.portfolio.construct import active_tilt, apply_risk_off, black_litterman

BENCH = pd.Series({"US": 0.64, "EAFE": 0.25, "EM": 0.11})


def test_active_tilt_respects_caps_and_budget():
    score = pd.Series({"US": -3.0, "EAFE": 0.5, "EM": 2.5})
    w = active_tilt(score, BENCH, scale=0.2, cap=0.15)
    assert np.isclose(w.sum(), 1.0)
    assert ((w - BENCH).abs() <= 0.15 + 1e-9).all()
    assert (w >= 0).all()
    assert w["EM"] > BENCH["EM"] and w["US"] < BENCH["US"]


def test_black_litterman_neutral_views_return_benchmark():
    rng = np.random.default_rng(1)
    rets = pd.DataFrame(rng.normal(0.0003, 0.01, (300, 3)), columns=BENCH.index)
    w = black_litterman(pd.Series(0.0, index=BENCH.index), BENCH, rets)
    assert np.allclose(w, BENCH, atol=1e-6)


def test_risk_off_moves_to_cash_only_when_all_trends_negative():
    w = BENCH.copy()
    on = apply_risk_off(w, pd.Series({"US": -0.1, "EAFE": -0.05, "EM": -0.2}), "CASH", 0.5)
    off = apply_risk_off(w, pd.Series({"US": 0.1, "EAFE": -0.05, "EM": -0.2}), "CASH", 0.5)
    assert np.isclose(on["CASH"], 0.5) and np.isclose(on.sum(), 1.0)
    assert np.isclose(off["CASH"], 0.0)


def test_plan_orders_sells_first_and_closes_dropped_names():
    orders = plan_orders({"SPY": 0.7, "EFA": 0.3}, {"SPY": 5000, "EEM": 3000, "EFA": 1000}, 10_000)
    sides = [o.side for o in orders]
    assert sides == sorted(sides, key=lambda s: s != "sell")
    eem = next(o for o in orders if o.symbol == "EEM")
    assert eem.close_all and eem.side == "sell"


def test_store_upsert_dedupes(tmp_path):
    s = Store(tmp_path)
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-01-03"]), "ticker": "SPY",
                       "open": 1, "high": 1, "low": 1, "close": 1, "adj_close": [1.0, 2.0], "volume": 1})
    s.write_prices(df)
    df2 = df.copy()
    df2["adj_close"] = [1.0, 2.5]
    s.write_prices(df2)
    wide = s.adj_close(["SPY"])
    assert len(wide) == 2 and wide["SPY"].iloc[-1] == 2.5
