"""Binning behavior: edges, sparse flags, Wilson coverage."""

import numpy as np
import pytest

from longshot.binning import equal_width_bins, quantile_bins
from longshot.intervals import wilson_interval


def test_equal_width_edges_and_counts():
    p = np.array([0.0, 0.05, 0.149, 0.5, 0.999, 1.0])
    y = np.array([0, 0, 1, 1, 0, 1], dtype=float)
    bins = equal_width_bins(p, y, n_bins=10, min_per_bin=1)
    assert len(bins) == 10
    assert bins[0].lo == 0.0 and bins[-1].hi == 1.0
    assert bins[0].n == 2  # 0.0, 0.05
    assert bins[1].n == 1  # 0.149
    assert bins[9].n == 2  # 0.999 and 1.0 (right edge inclusive)
    assert sum(b.n for b in bins) == len(p)
    assert bins[5].rate == pytest.approx(1.0)  # p=0.5, y=1


def test_sparse_flags():
    p = np.array([0.05] * 5 + [0.95] * 40)
    y = np.array([0] * 5 + [1] * 40, dtype=float)
    bins = equal_width_bins(p, y, n_bins=10, min_per_bin=10)
    assert bins[0].sparse is True   # n=5 < 10
    assert bins[9].sparse is False  # n=40
    mid = [b for b in bins[1:9]]
    assert all(b.n == 0 and b.sparse for b in mid)


def test_wilson_coverage_95pct():
    # Simulate 200 binomial experiments at p=0.3; Wilson 95% intervals
    # should cover the true value in ~95% of trials (tolerance +-6%).
    rng = np.random.default_rng(1234)
    true_p, n, trials = 0.3, 60, 200
    hits = 0
    for _ in range(trials):
        k = int(rng.binomial(n, true_p))
        lo, hi = wilson_interval(k, n)
        if lo <= true_p <= hi:
            hits += 1
    coverage = hits / trials
    assert 0.89 <= coverage <= 1.0


def test_quantile_bins_balanced_counts():
    rng = np.random.default_rng(5)
    p = np.concatenate([rng.beta(0.5, 0.5, 500), rng.uniform(0, 1, 500)])
    y = (rng.random(p.size) < p).astype(float)
    bins = quantile_bins(p, y, n_bins=10, min_per_bin=1)
    assert sum(b.n for b in bins) == p.size
    counts = [b.n for b in bins]
    assert max(counts) - min(counts) <= 2
    assert bins[0].lo == 0.0 and bins[-1].hi == 1.0
