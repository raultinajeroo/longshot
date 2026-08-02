"""Binning of (predicted probability, outcome) pairs for calibration stats.

Equal-width bins on [0, 1] are the default (10 bins). Bins with fewer than
``min_per_bin`` observations are still computed but flagged ``sparse`` and
excluded from ECE/MCE and slope fits — sparse bins dominate naive
calibration estimators otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .intervals import wilson_interval


@dataclass(frozen=True)
class Bin:
    """Aggregate calibration statistics for one probability bin."""

    lo: float
    hi: float
    n: int
    mean_pred: float
    rate: float
    ci_lo: float  # Wilson 95% CI for the empirical rate
    ci_hi: float
    sparse: bool

    def to_dict(self) -> dict:
        """JSON-serializable representation."""
        return asdict(self)


def equal_width_bins(
    p: np.ndarray,
    y: np.ndarray,
    *,
    n_bins: int = 10,
    min_per_bin: int = 30,
) -> list[Bin]:
    """Bin (p, y) pairs into ``n_bins`` equal-width bins on [0, 1].

    ``p == 1.0`` falls into the last bin. Empty bins are returned with
    ``n == 0``, ``mean_pred``/``rate`` set to NaN, and ``sparse=True``.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    if p.size != y.size:
        raise ValueError("binning: p and y must have equal length")
    if n_bins < 1:
        raise ValueError("binning: n_bins must be >= 1")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    out: list[Bin] = []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n:
            k = int(y[mask].sum())
            mean_pred = float(p[mask].mean())
            rate = k / n
            ci_lo, ci_hi = wilson_interval(k, n)
        else:
            mean_pred, rate, ci_lo, ci_hi = float("nan"), float("nan"), 0.0, 1.0
        out.append(
            Bin(
                lo=float(edges[b]),
                hi=float(edges[b + 1]),
                n=n,
                mean_pred=mean_pred,
                rate=rate,
                ci_lo=ci_lo,
                ci_hi=ci_hi,
                sparse=n < min_per_bin,
            )
        )
    return out


def quantile_bins(
    p: np.ndarray,
    y: np.ndarray,
    *,
    n_bins: int = 10,
    min_per_bin: int = 30,
) -> list[Bin]:
    """Bin (p, y) pairs into bins with (approximately) equal counts.

    Edges are the empirical quantiles of ``p``. Useful when the price
    distribution is heavily skewed; equal-width remains the default for
    comparability across horizons.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    if p.size == 0:
        raise ValueError("binning: empty input")
    qs = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
    qs[0], qs[-1] = 0.0, 1.0
    qs = np.unique(qs)  # drop duplicated edges from ties
    if qs.size < 2:
        qs = np.array([0.0, 1.0])
    idx = np.clip(np.digitize(p, qs) - 1, 0, qs.size - 2)
    out: list[Bin] = []
    for b in range(qs.size - 1):
        mask = idx == b
        n = int(mask.sum())
        k = int(y[mask].sum())
        mean_pred = float(p[mask].mean()) if n else float("nan")
        rate = k / n if n else float("nan")
        ci_lo, ci_hi = wilson_interval(k, n) if n else (0.0, 1.0)
        out.append(
            Bin(
                lo=float(qs[b]),
                hi=float(qs[b + 1]),
                n=n,
                mean_pred=mean_pred,
                rate=rate,
                ci_lo=ci_lo,
                ci_hi=ci_hi,
                sparse=n < min_per_bin,
            )
        )
    return out
