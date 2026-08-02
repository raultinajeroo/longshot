"""Correction layer: OOS honesty, split integrity, PAV monotonicity."""

import numpy as np
import pytest

from longshot.correct import (
    fit_isotonic,
    run_correction,
    split_by_resolution_time,
)
from longshot.horizons import parse_horizon
from longshot.simulate import simulate_markets

HORIZONS = {"30d": parse_horizon("30d"), "14d": parse_horizon("14d")}


def test_isotonic_improves_oos_on_compressed():
    # Planted strong compression (c=0.55), n large: isotonic must beat the
    # raw price on the held-out time split with a CI excluding 0.
    markets = simulate_markets("compressed", 500, 101)
    res = run_correction(
        markets, horizons=["14d"], horizon_seconds=HORIZONS,
        method="isotonic", n_boot=300, seed=1,
    )
    r = res["horizons"]["14d"]["isotonic"]
    assert r["delta_brier"] < 0.0
    assert r["delta_brier_ci"][1] < 0.0
    assert r["verdict"] == "reliable improvement"


def test_calibrated_no_reliable_improvement():
    markets = simulate_markets("calibrated", 400, 202)
    res = run_correction(
        markets, horizons=["30d"], horizon_seconds=HORIZONS,
        method="both", n_boot=300, seed=1,
    )
    for meth in ("platt", "isotonic"):
        r = res["horizons"]["30d"][meth]
        assert (r["verdict"] == "no reliable improvement"
                or abs(r["delta_brier"]) < 0.005)


def test_split_respects_time_ordering():
    markets = simulate_markets("calibrated", 60, 5)
    train, test = split_by_resolution_time(markets, 0.6)
    assert len(train) == 36 and len(test) == 24
    assert max(m.resolved_ts for m in train) <= min(m.resolved_ts for m in test)
    assert {m.market_id for m in train}.isdisjoint(m.market_id for m in test)


def test_pav_monotonicity():
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 1, 300)
    y = (rng.random(300) < p ** 2).astype(float)  # non-monotone-ish noise
    model = fit_isotonic(p, y)
    vals = np.array(model.values)
    assert np.all(np.diff(vals) >= 0.0)
    # Prediction is non-decreasing in p too.
    grid = np.linspace(0, 1, 101)
    pred = model.predict(grid)
    assert np.all(np.diff(pred) >= 0.0)


def test_isotonic_predict_clamps_ends():
    p = np.array([0.2, 0.4, 0.6, 0.8])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    model = fit_isotonic(p, y)
    pred = model.predict(np.array([0.0, 1.0]))
    assert pred[0] == model.values[0]
    assert pred[1] == model.values[-1]


def test_run_correction_split_metadata():
    markets = simulate_markets("compressed", 100, 9)
    res = run_correction(
        markets, horizons=["30d"], horizon_seconds=HORIZONS,
        method="both", train_frac=0.6, n_boot=50, seed=1,
    )
    assert res["train_max_resolved_ts"] <= res["test_min_resolved_ts"]
    assert res["n_train_markets"] + res["n_test_markets"] == 100
