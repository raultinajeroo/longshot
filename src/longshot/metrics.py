"""Calibration metrics.

Definitions (all for predicted probability p of a binary outcome y in {0,1}):

- Brier score: mean((p - y)^2). Lower is better; range [0, 1].
- Log-loss: -mean(y ln p + (1-y) ln(1-p)), with p clipped to
  [eps, 1-eps], eps = 1e-15.
- ECE (expected calibration error): sum over non-sparse bins of
  (n_b / N) * |rate_b - meanp_b|. See Naeini et al. (2015) and
  Guo et al. (2017) for the binning estimator.
- MCE: max over non-sparse bins of |rate_b - meanp_b|.
- Murphy decomposition (Murphy 1973, "A new vector partition of the
  probability score"): Brier = REL - RES + UNC, computed on the same bins:
    REL = sum_b (n_b/N)(meanp_b - rate_b)^2   (reliability)
    RES = sum_b (n_b/N)(rate_b - ybar)^2      (resolution)
    UNC = ybar(1 - ybar)                      (uncertainty)
  where ybar is the overall base rate. Sparse bins are excluded from the
  REL/RES sums and N is the count over included bins, so the identity
  holds exactly on the included subset.
"""

from __future__ import annotations

import math

import numpy as np

from .binning import Bin

EPS = 1e-15


def _check_panel(p: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    if p.size == 0:
        raise ValueError("metrics: empty panel")
    if p.size != y.size:
        raise ValueError("metrics: p and y must have equal length")
    return p, y


def brier(p: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error of probabilistic forecasts: mean((p - y)^2)."""
    p, y = _check_panel(p, y)
    return float(np.mean((p - y) ** 2))


def log_loss(p: np.ndarray, y: np.ndarray, *, eps: float = EPS) -> float:
    """Mean negative log-likelihood with probabilities clipped to [eps, 1-eps]."""
    p, y = _check_panel(p, y)
    pc = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(pc) + (1.0 - y) * np.log(1.0 - pc)))


def ece(bins: list[Bin]) -> float:
    """Expected calibration error over non-sparse, non-empty bins."""
    use = [b for b in bins if not b.sparse and b.n > 0]
    n_tot = sum(b.n for b in use)
    if n_tot == 0:
        raise ValueError("metrics: no non-sparse bins for ECE")
    return float(sum(b.n * abs(b.rate - b.mean_pred) for b in use) / n_tot)


def mce(bins: list[Bin]) -> float:
    """Maximum calibration error over non-sparse, non-empty bins."""
    use = [b for b in bins if not b.sparse and b.n > 0]
    if not use:
        raise ValueError("metrics: no non-sparse bins for MCE")
    return float(max(abs(b.rate - b.mean_pred) for b in use))


def murphy_decomposition(bins: list[Bin]) -> dict:
    """Murphy (1973) decomposition Brier = REL - RES + UNC on binned data.

    Uses non-sparse, non-empty bins only; UNC is the variance of the
    outcome over the included observations. The identity holds to
    floating-point precision on that subset (tested to 1e-12).
    """
    use = [b for b in bins if not b.sparse and b.n > 0]
    n_tot = sum(b.n for b in use)
    if n_tot == 0:
        raise ValueError("metrics: no non-sparse bins for Murphy decomposition")
    ybar = sum(b.n * b.rate for b in use) / n_tot
    rel = sum(b.n * (b.mean_pred - b.rate) ** 2 for b in use) / n_tot
    res = sum(b.n * (b.rate - ybar) ** 2 for b in use) / n_tot
    unc = ybar * (1.0 - ybar)
    return {
        "reliability": float(rel),
        "resolution": float(res),
        "uncertainty": float(unc),
        "brier_binned": float(rel - res + unc),
        "n": n_tot,
        "base_rate": float(ybar),
    }


def climatology_brier(y: np.ndarray, base_rate: float | None = None) -> float:
    """Brier score of the constant forecaster p = base_rate (default: mean y).

    This is the no-skill baseline; skill score is 1 - Brier/Brier_clim.
    """
    _, y = _check_panel(np.zeros(len(y)), y)
    ybar = float(np.mean(y)) if base_rate is None else float(base_rate)
    if not 0.0 <= ybar <= 1.0 or math.isnan(ybar):
        raise ValueError("metrics: base_rate out of [0, 1]")
    return float(np.mean((ybar - y) ** 2))
