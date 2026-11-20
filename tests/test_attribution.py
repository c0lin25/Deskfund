import numpy as np
import pandas as pd

from deskfund.attribution.brinson import attribute, brinson_fachler


def test_single_period_sums_to_active():
    wp = pd.Series({"US": 0.7, "EAFE": 0.2, "EM": 0.1})
    wb = pd.Series({"US": 0.6, "EAFE": 0.25, "EM": 0.15})
    rb = pd.Series({"US": 0.02, "EAFE": -0.01, "EM": 0.03})
    rp = pd.Series({"US": 0.025, "EAFE": -0.012, "EM": 0.02})  # different -> selection non-zero
    bf = brinson_fachler(wp, wb, rp, rb)
    active = (wp * rp).sum() - (wb * rb).sum()
    assert np.isclose(bf.values.sum(), active)
    assert not np.allclose(bf["selection"], 0)


def test_same_instrument_means_pure_allocation():
    wp = pd.Series({"A": 0.8, "B": 0.2})
    wb = pd.Series({"A": 0.5, "B": 0.5})
    r = pd.Series({"A": 0.05, "B": -0.02})
    bf = brinson_fachler(wp, wb, r, r)
    assert np.allclose(bf["selection"], 0) and np.allclose(bf["interaction"], 0)


def test_carino_linking_reconciles_over_many_periods():
    rng = np.random.default_rng(0)
    periods = []
    for i in range(60):
        r = pd.Series(rng.normal(0.01, 0.05, 3), index=["US", "EAFE", "EM"])
        wp = pd.Series(rng.dirichlet([5, 3, 2]), index=r.index)
        wb = pd.Series([0.64, 0.25, 0.11], index=r.index)
        periods.append({"start": i, "end": i + 1, "wp": wp, "wb": wb, "r": r, "cost": 0.0005})
    out = attribute(periods)
    assert abs(out["check"]["residual"]) < 1e-10
