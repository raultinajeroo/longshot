"""Known-answer tests for calibration metrics."""

import math

import numpy as np
import pytest

from longshot.binning import equal_width_bins
from longshot.metrics import (
    brier,
    climatology_brier,
    ece,
    log_loss,
    mce,
    murphy_decomposition,
)


def test_brier_known_answer():
    p = np.array([0.7, 0.2, 0.9, 0.4])
    y = np.array([1.0, 0.0, 0.0, 1.0])
    expected = np.mean((p - y) ** 2)  # hand-checkable
    assert brier(p, y) == pytest.approx(expected, abs=1e-15)
    assert brier(p, y) == pytest.approx((0.09 + 0.04 + 0.81 + 0.36) / 4)


def test_ece_known_answer():
    # Two bins by construction: p=0.25 (rate 0) and p=0.75 (rate 1).
    p = np.array([0.25] * 50 + [0.75] * 50)
    y = np.array([0.0] * 50 + [1.0] * 50)
    bins = equal_width_bins(p, y, n_bins=4, min_per_bin=10)
    # Bin [0,0.25]: meanp 0.25, rate 0; bin (0.5,0.75]: meanp 0.75, rate 1.
    assert ece(bins) == pytest.approx(0.25)
    assert mce(bins) == pytest.approx(0.25)


def test_murphy_identity_constant_bin_forecasts():
    # With one distinct price per bin, Brier == REL - RES + UNC exactly.
    rng = np.random.default_rng(0)
    levels = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    p = np.repeat(levels, 40)
    y = (rng.random(p.size) < p).astype(float)
    bins = equal_width_bins(p, y, n_bins=10, min_per_bin=1)
    d = murphy_decomposition(bins)
    direct = brier(p, y)
    identity = d["reliability"] - d["resolution"] + d["uncertainty"]
    assert identity == pytest.approx(direct, abs=1e-12)
    assert d["brier_binned"] == pytest.approx(direct, abs=1e-12)


def test_log_loss_clipping():
    # Perfectly wrong confident predictions must not produce inf.
    p = np.array([0.0, 1.0])
    y = np.array([1.0, 0.0])
    ll = log_loss(p, y)
    assert math.isfinite(ll)
    assert ll == pytest.approx(-math.log(1e-15), abs=1e-3)


def test_empty_input_errors():
    with pytest.raises(ValueError):
        brier(np.array([]), np.array([]))
    with pytest.raises(ValueError):
        log_loss(np.array([]), np.array([]))
    with pytest.raises(ValueError):
        ece(equal_width_bins(np.array([0.5]), np.array([1.0]),
                             n_bins=10, min_per_bin=30))
    with pytest.raises(ValueError):
        climatology_brier(np.array([]))


def test_climatology_baseline():
    y = np.array([0, 0, 1, 1, 1], dtype=float)
    ybar = 0.6
    assert climatology_brier(y) == pytest.approx(ybar * (1 - ybar))
    p = np.full(5, ybar)
    assert brier(p, y) == pytest.approx(climatology_brier(y))
