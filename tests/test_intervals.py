"""Wilson known values; bootstrap reproducibility and ordering."""

import numpy as np
import pytest

from longshot.intervals import bootstrap_ci, bootstrap_groups_ci, wilson_interval


def test_wilson_known_values():
    # p=0.5, n=100, z=1.96: Wilson interval is (0.4038, 0.5962).
    lo, hi = wilson_interval(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3)
    assert hi == pytest.approx(0.5962, abs=1e-3)
    # Edge cases: zero hits and all hits stay inside [0, 1].
    lo0, hi0 = wilson_interval(0, 10)
    assert lo0 == 0.0 and 0.0 < hi0 < 1.0
    lo1, hi1 = wilson_interval(10, 10)
    assert 0.0 < lo1 < 1.0 and hi1 == pytest.approx(1.0)
    # Degenerate n.
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_bootstrap_reproducible_by_seed():
    rng = np.random.default_rng(7)
    values = rng.normal(0.3, 0.1, 80)
    a = bootstrap_ci(values, n_boot=200, seed=99)
    b = bootstrap_ci(values, n_boot=200, seed=99)
    c = bootstrap_ci(values, n_boot=200, seed=100)
    assert a == b
    assert a != c


def test_bootstrap_ci_ordering_and_coverage():
    rng = np.random.default_rng(11)
    values = rng.normal(0.65, 0.05, 100)
    lo, hi = bootstrap_ci(values, n_boot=300, seed=5)
    assert lo <= float(np.mean(values)) <= hi
    assert hi - lo < 0.05  # SE of the mean is 0.005; CI must be tight


def test_bootstrap_groups_resamples_whole_groups():
    groups = [np.array([1.0, 1.0]), np.array([0.0, 0.0]), np.array([0.5, 0.5])]
    stat = lambda gs: float(np.mean(np.concatenate(gs)))  # noqa: E731
    lo, hi = bootstrap_groups_ci(groups, stat, n_boot=200, seed=3)
    assert lo <= 0.5 <= hi
    # Values must be multiples of whole-group means only: lo/hi in {0, 1/3,
    # 0.5, 2/3, 1} grid of group-mean averages.
    grid = {round(a / 3 + b / 3, 6) for a in (0.0, 0.5, 1.0) for b in (0.0, 0.5, 1.0)}
    assert round(lo, 6) in grid or lo in (0.0, 1.0)


def test_bootstrap_empty_raises():
    with pytest.raises(ValueError):
        bootstrap_ci([], n_boot=10, seed=1)
    with pytest.raises(ValueError):
        bootstrap_groups_ci([], lambda g: 0.0, n_boot=10, seed=1)
