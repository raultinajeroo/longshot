"""Bias measures recover planted structure in simulated data."""

import numpy as np
import pytest

from longshot.bias import (
    compression_slope,
    logistic_calibration_slope,
    yes_price_inflation,
)
from longshot.horizons import build_panel, parse_horizon
from longshot.metrics import brier
from longshot.simulate import simulate_markets
from longshot.types import HorizonPoint, HorizonPanel


def _panel(mode, n, seed, horizon="30d", **sim_kwargs):
    markets = simulate_markets(mode, n, seed, **sim_kwargs)
    return build_panel(markets, parse_horizon(horizon), horizon)


def test_compression_recovers_planted_slope():
    # Planted compression c=0.55 -> expected slope 1/c ~= 1.82 (+-0.25),
    # and the bootstrap CI must cover it.
    panel = _panel("compressed", 400, 11)
    res = compression_slope(panel, min_per_bin=20, n_boot=300, seed=1)
    assert res["slope"] == pytest.approx(1.82, abs=0.25)
    assert res["ci_lo"] <= 1.82 <= res["ci_hi"]


def test_calibrated_slope_ci_covers_one():
    panel = _panel("calibrated", 400, 7)
    res = compression_slope(panel, min_per_bin=20, n_boot=300, seed=1)
    assert res["ci_lo"] <= 1.0 <= res["ci_hi"]


def test_logistic_slope_longshot_bias_below_one():
    # Longshots overpriced -> logistic slope of y on logit(p) < 1. Short
    # lives relative to the 30d horizon keep the planted +-0.08 shift
    # dominant over the terminal drift of the belief path.
    panel = _panel("longshot-bias", 400, 3, life_days=(45, 60))
    res = logistic_calibration_slope(panel, n_boot=300, seed=1)
    assert res["slope"] < 1.0
    assert res["ci_hi"] < 1.0


def test_logistic_slope_calibrated_near_one():
    panel = _panel("calibrated", 400, 7)
    res = logistic_calibration_slope(panel, n_boot=300, seed=1)
    assert res["ci_lo"] <= 1.0 <= res["ci_hi"]


def test_brier_symmetric_under_yes_no_flip():
    # Brier is invariant to (p, y) -> (1-p, 1-y); asymmetry is measured by
    # yes_price_inflation instead.
    panel = _panel("compressed", 120, 11)
    p = np.array([q.p for q in panel.points])
    y = np.array([q.outcome for q in panel.points], dtype=float)
    assert brier(p, y) == pytest.approx(brier(1 - p, 1 - y), abs=1e-15)


def test_yes_price_inflation_detects_shift():
    # Every price shifted +0.1 relative to outcomes -> gap ~ +0.1.
    pts = tuple(
        HorizonPoint(market_id=f"m{i}", category="c", volume=None,
                     resolved_ts=i, p=0.6, outcome=(1 if i % 2 else 0))
        for i in range(100)
    )
    panel = HorizonPanel(name="t", seconds=1, points=pts)
    res = yes_price_inflation(panel, n_boot=200, seed=1)
    assert res["c"]["gap"] == pytest.approx(0.1, abs=1e-12)
    assert res["c"]["ci_lo"] > 0.0
