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
from longshot.types import MarketSeries, PricePoint

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


def test_isotonic_pools_equal_prices_before_fitting():
    p = np.array([0.2, 0.2, 0.8, 0.8])
    y = np.array([0.0, 1.0, 1.0, 1.0])
    for order in ([0, 1, 2, 3], [1, 0, 3, 2]):
        model = fit_isotonic(p[order], y[order])
        assert model.predict(np.array([0.2, 0.8])) == pytest.approx([0.5, 1.0])


def test_isotonic_preserves_weights_when_pooling_equal_prices():
    model = fit_isotonic(
        np.array([0.2, 0.4, 0.4, 0.4, 0.8]),
        np.array([1.0, 0.0, 0.0, 1.0, 1.0]),
    )
    assert model.predict(np.array([0.2, 0.4, 0.8])) == pytest.approx([0.5, 0.5, 1.0])


def _overlapping_markets():
    day = 86400
    return [MarketSeries(
        venue="test", market_id=f"m{i:03}", question="synthetic", category="test",
        created_ts=0, resolved_ts=(100 + i) * day, outcome=int(i % 3 == 0),
        volume=None, n_traders=None,
        # At the 10d target the carried price is actually 12d before T.
        series=(PricePoint((88 + i) * day, 0.3 if i % 2 == 0 else 0.7),),
    ) for i in range(80)]


def test_correction_excludes_labels_unavailable_at_first_test_observation():
    markets = _overlapping_markets()
    res = run_correction(
        markets, horizons=["10d"], horizon_seconds={"10d": 10 * 86400},
        n_boot=20,
    )
    entry = res["horizons"]["10d"]
    assert entry["n_train_before_purge"] == 48
    assert entry["n_train_purged"] == 12
    assert entry["n_train"] == 36
    assert entry["n_test"] == 32
    assert entry["test_min_observed_ts"] == 136 * 86400
    assert entry["train_max_resolved_ts"] < entry["test_min_observed_ts"]
    # Changing every unavailable label must leave fitted predictions unchanged.
    from dataclasses import replace
    changed = [replace(m, outcome=1 - m.outcome) if 36 <= i < 48 else m
               for i, m in enumerate(markets)]
    rerun = run_correction(
        changed, horizons=["10d"], horizon_seconds={"10d": 10 * 86400},
        n_boot=20,
    )
    for method in ("platt", "isotonic"):
        assert rerun["horizons"]["10d"][method]["brier"] == entry[method]["brier"]


def test_correction_skips_when_all_training_labels_arrive_too_late():
    from dataclasses import replace
    markets = [replace(m, resolved_ts=100 * 86400,
                       series=(PricePoint(88 * 86400, 0.5),))
               for m in _overlapping_markets()]
    res = run_correction(
        markets, horizons=["10d"], horizon_seconds={"10d": 10 * 86400}, n_boot=20,
    )
    entry = res["horizons"]["10d"]
    assert entry["n_train"] == 0
    assert entry["n_train_purged"] == 48
    assert entry["train_max_resolved_ts"] is None
    assert "after removing unavailable labels" in entry["skipped"]


def test_resolution_day_bootstrap_keeps_same_day_legs_together():
    from dataclasses import replace
    markets = _overlapping_markets()
    for i in range(48, 80):
        day = 200 if i < 64 else 201
        markets[i] = replace(markets[i], resolved_ts=day * 86400,
                             outcome=int(i >= 64),
                             series=(PricePoint((day - 12) * 86400, 0.5),))
    args = dict(horizons=["10d"], horizon_seconds={"10d": 10 * 86400},
                method="isotonic", n_boot=200)
    independent = run_correction(markets, **args)["horizons"]["10d"]
    grouped = run_correction(markets, **args, bootstrap_unit="resolution-day")["horizons"]["10d"]
    assert independent["n_test_groups"] == 32
    assert grouped["n_test_groups"] == 2
    assert grouped["isotonic"]["delta_brier"] == independent["isotonic"]["delta_brier"]
    old_ci, new_ci = (e["isotonic"]["delta_brier_ci"] for e in (independent, grouped))
    assert new_ci[1] - new_ci[0] > old_ci[1] - old_ci[0]
    # Both dates occupy a single fixed UTC week: no meaningful group CI.
    weekly = run_correction(markets, **args, bootstrap_unit="resolution-week")["horizons"]["10d"]
    assert weekly["n_test_groups"] == 1
    assert "at least 2 test resolution groups" in weekly["skipped"]


def test_run_correction_split_metadata():
    markets = simulate_markets("compressed", 100, 9)
    res = run_correction(
        markets, horizons=["30d"], horizon_seconds=HORIZONS,
        method="both", train_frac=0.6, n_boot=50, seed=1,
    )
    assert res["train_max_resolved_ts"] <= res["test_min_resolved_ts"]
    assert res["n_train_markets"] + res["n_test_markets"] == 100
