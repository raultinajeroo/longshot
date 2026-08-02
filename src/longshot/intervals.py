"""Confidence intervals: Wilson score intervals and bootstrap helpers.

Wilson interval reference: Wilson, E. B. (1927), "Probable inference, the
law of succession, and statistical inference", JASA 22(158).
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

Z_95 = 1.959963984540054  # two-sided 95% standard normal quantile


def wilson_interval(k: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion with ``k`` of ``n`` hits.

    Returns (lower, upper). For ``n == 0`` returns (0.0, 1.0).
    """
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (float(max(0.0, center - half)), float(min(1.0, center + half)))


def bootstrap_ci(
    values: Sequence[float],
    stat_fn: Callable[[np.ndarray], float] | None = None,
    *,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI for a statistic over 1-D ``values``.

    Resamples observations with replacement ``n_boot`` times. The default
    statistic is the mean. Reproducible for a fixed ``seed``.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("bootstrap_ci: empty input")
    if stat_fn is None:
        stat_fn = lambda a: float(np.mean(a))  # noqa: E731
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    n = arr.size
    for b in range(n_boot):
        stats[b] = stat_fn(arr[rng.integers(0, n, n)])
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def bootstrap_groups_ci(
    groups: Sequence,
    stat_fn: Callable[[list], float],
    *,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI resampling whole groups (e.g. markets).

    Use this (rather than :func:`bootstrap_ci`) whenever observations within
    a group are dependent — e.g. all horizon points from one market.
    """
    if len(groups) == 0:
        raise ValueError("bootstrap_groups_ci: empty input")
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    n = len(groups)
    for b in range(n_boot):
        sample = [groups[i] for i in rng.integers(0, n, n)]
        stats[b] = stat_fn(sample)
    # Degenerate resamples (e.g. a resample with a constant predictor) can
    # make the statistic undefined; drop them. If more than a third of
    # resamples fail, the CI is not meaningful.
    stats = stats[np.isfinite(stats)]
    if stats.size < 2 * n_boot // 3:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))
