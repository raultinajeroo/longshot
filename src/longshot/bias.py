"""Systematic-bias measures.

Compression index (calibration slope). Fit weighted least squares of
(rate_b - 0.5) on (meanp_b - 0.5) through the origin, weights n_b, over
non-sparse bins with |meanp_b - 0.5| >= 0.05 (bins hugging 0.5 carry no
signal and make the ratio estimator unstable — hence the exclusion, a
documented judgment call). Slope > 1 means the market is compressed
toward 0.5 (underconfident: extreme prices are not extreme enough — the
political-market finding in Le 2026); slope < 1 means overconfident.

Favorite-longshot slope. Newton-Raphson logistic regression of y on
logit(p) (p clipped to [1e-4, 1-1e-4]). Slope 1 = calibrated. Slope < 1
means longshots are overpriced relative to favorites (the classic
favorite-longshot bias).

YES/NO asymmetry. Recomputing metrics on the transformed panel
(p' = 1 - p, y' = 1 - y) leaves Brier, ECE, etc. identical by symmetry —
so asymmetry is measured instead as mean(p) - base_rate per category
("YES-price inflation": do YES shares trade richer than their realized
frequency?). Confidence intervals are market-level bootstrap.
"""

from __future__ import annotations

import numpy as np

from .binning import Bin
from .intervals import bootstrap_groups_ci
from .types import HorizonPanel

LOGIT_CLIP = 1e-4
MIN_SLOPE_DEVIATION = 0.05  # exclude bins with |meanp - 0.5| below this


def compression_slope_from_bins(bins: list[Bin]) -> float:
    """WLS-through-origin slope of (rate - 0.5) on (meanp - 0.5), weights n_b.

    Only non-sparse bins with |meanp - 0.5| >= MIN_SLOPE_DEVIATION are
    used. Returns NaN if no bins qualify.
    """
    num = den = 0.0
    for b in bins:
        if b.sparse or b.n == 0:
            continue
        x = b.mean_pred - 0.5
        if abs(x) < MIN_SLOPE_DEVIATION:
            continue
        num += b.n * x * (b.rate - 0.5)
        den += b.n * x * x
    if den == 0.0:
        return float("nan")
    return num / den


def compression_slope(
    panel: HorizonPanel,
    *,
    n_bins: int = 10,
    min_per_bin: int = 30,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Compression slope for a horizon panel, with a market-level bootstrap CI.

    Markets (not individual points) are resampled with replacement, since
    points within a market/horizon are not independent of the panel
    composition.
    """
    from .binning import equal_width_bins

    def stat(pts) -> float:
        p = np.array([q.p for q in pts])
        y = np.array([q.outcome for q in pts])
        bins = equal_width_bins(p, y, n_bins=n_bins, min_per_bin=min_per_bin)
        return compression_slope_from_bins(bins)

    points = list(panel.points)
    slope = stat(points)
    if not np.isfinite(slope) or len(points) < 2:
        return {"slope": slope, "ci_lo": float("nan"), "ci_hi": float("nan"),
                "n": len(points)}
    lo, hi = bootstrap_groups_ci(points, stat, n_boot=n_boot, seed=seed)
    return {"slope": float(slope), "ci_lo": lo, "ci_hi": hi, "n": len(points)}


def _logit(p: np.ndarray) -> np.ndarray:
    pc = np.clip(p, LOGIT_CLIP, 1.0 - LOGIT_CLIP)
    return np.log(pc / (1.0 - pc))


def fit_logistic(x: np.ndarray, y: np.ndarray, *, max_iter: int = 50,
                 tol: float = 1e-10) -> tuple[float, float]:
    """Newton-Raphson fit of y ~ sigmoid(a + b*x). Returns (intercept, slope).

    Raises ValueError if the fit does not converge or the design is
    degenerate (e.g. constant x).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or x.size != y.size:
        raise ValueError("fit_logistic: empty or mismatched input")
    if float(np.ptp(x)) == 0.0:
        raise ValueError("fit_logistic: constant predictor")
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(max_iter):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        w = np.maximum(mu * (1.0 - mu), 1e-10)
        grad = X.T @ (y - mu)
        h = (X * w[:, None]).T @ X
        try:
            step = np.linalg.solve(h, grad)
        except np.linalg.LinAlgError as exc:
            raise ValueError("fit_logistic: singular Hessian") from exc
        beta = beta + step
        if float(np.max(np.abs(step))) < tol:
            return (float(beta[0]), float(beta[1]))
    raise ValueError("fit_logistic: no convergence")


def logistic_calibration_slope(
    panel: HorizonPanel,
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Logistic regression of outcome on logit(price): the favorite-longshot slope.

    slope == 1 and intercept == 0 indicate calibration. slope < 1 is the
    favorite-longshot direction (longshots overpriced, favorites
    underpriced). CI is a market-level bootstrap; degenerate resamples are
    skipped.
    """
    def stat(pts) -> float:
        p = np.array([q.p for q in pts])
        y = np.array([q.outcome for q in pts], dtype=float)
        try:
            _, slope = fit_logistic(_logit(p), y)
        except ValueError:
            return float("nan")
        return slope

    points = list(panel.points)
    slope = stat(points)
    intercept = float("nan")
    if np.isfinite(slope):
        p = np.array([q.p for q in points])
        y = np.array([q.outcome for q in points], dtype=float)
        intercept, slope = fit_logistic(_logit(p), y)
        lo, hi = bootstrap_groups_ci(
            points, stat, n_boot=n_boot, seed=seed
        ) if len(points) >= 2 else (float("nan"), float("nan"))
    else:
        lo, hi = float("nan"), float("nan")
    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "ci_lo": lo,
        "ci_hi": hi,
        "n": len(points),
    }


def yes_price_inflation(
    panel: HorizonPanel,
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, dict]:
    """Per-category YES-price inflation: mean(p) - base rate, with bootstrap CI.

    Interpretation: > 0 means YES shares in this category traded, on
    average, richer than their realized frequency. Because Brier/ECE are
    symmetric under (p, y) -> (1-p, 1-y), this signed mean gap is the
    informative YES/NO asymmetry measure (see module docstring).
    """
    cats: dict[str, list] = {}
    for q in panel.points:
        cats.setdefault(q.category, []).append(q)
    out: dict[str, dict] = {}
    for cat, pts in sorted(cats.items()):
        gap = float(np.mean([q.p - q.outcome for q in pts]))
        if len(pts) >= 2:
            lo, hi = bootstrap_groups_ci(
                pts,
                lambda s: float(np.mean([q.p - q.outcome for q in s])),
                n_boot=n_boot,
                seed=seed,
            )
        else:
            lo, hi = float("nan"), float("nan")
        out[cat] = {"gap": gap, "ci_lo": lo, "ci_hi": hi, "n": len(pts)}
    return out
