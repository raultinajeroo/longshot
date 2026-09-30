"""Out-of-sample bias correction: Platt scaling and isotonic regression.

Protocol (honesty by construction):

1. Split *markets* by resolution time: the earliest ``train_frac`` of
   markets (by resolved_ts) form the train set, the rest the test set.
   No market appears in both. For each horizon, remove training labels
   resolving at or after the earliest TEST price observation. Ordering
   settlements alone does not establish when outcomes were available.
2. Fit the correction map on the remaining TRAIN panel, per horizon.
3. Evaluate raw vs corrected on TEST panels: delta-Brier, delta-logloss,
   ECE before/after, and a bootstrap CI for delta-Brier (whole markets by
   default; optional resolution-time blocks for dependence checks).
4. If the CI of delta-Brier includes 0, the verdict field is
   "no reliable improvement". Callers must surface this; the demo prints
   it verbatim. A correction that does not beat the raw market price
   out-of-sample is reported as such.

Methods:

- Platt scaling (Platt 1999): logistic regression y ~ sigmoid(a + b*logit(p)).
- Isotonic regression: pool-adjacent-violators on (p, y) pairs weighted
  equally, monotone non-decreasing; prediction by step interpolation with
  clamped ends (Zadrozny & Elkan 2002).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bias import _logit, fit_logistic
from .binning import equal_width_bins
from .horizons import build_panel
from .intervals import bootstrap_groups_ci
from .metrics import brier, ece, log_loss
from .types import HorizonPanel, MarketSeries


@dataclass(frozen=True)
class PlattModel:
    """Platt scaling map p -> sigmoid(a + b * logit(p))."""

    a: float
    b: float

    def predict(self, p: np.ndarray) -> np.ndarray:
        """Apply the Platt map elementwise."""
        z = self.a + self.b * _logit(np.asarray(p, dtype=float))
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


@dataclass(frozen=True)
class IsotonicModel:
    """Isotonic regression map as (knots, values), step interpolation.

    ``knots`` are increasing price levels; ``values[i]`` is the corrected
    probability for p in (knots[i-1], knots[i]]. Predictions are clamped
    to the end values outside the knot range.
    """

    knots: tuple[float, ...]
    values: tuple[float, ...]

    def predict(self, p: np.ndarray) -> np.ndarray:
        """Apply the isotonic map elementwise (clamped step interpolation)."""
        p = np.asarray(p, dtype=float)
        knots = np.asarray(self.knots)
        idx = np.searchsorted(knots, p, side="left")
        idx = np.clip(idx, 0, len(self.values) - 1)
        return np.asarray(self.values, dtype=float)[idx]


def fit_platt(p: np.ndarray, y: np.ndarray) -> PlattModel:
    """Fit Platt scaling: logistic regression of y on logit(p)."""
    a, b = fit_logistic(_logit(np.asarray(p, dtype=float)),
                        np.asarray(y, dtype=float))
    return PlattModel(a=a, b=b)


def fit_isotonic(p: np.ndarray, y: np.ndarray) -> IsotonicModel:
    """Fit monotone non-decreasing isotonic regression via PAVA.

    Equal prices are pooled first so labels at a tied price cannot depend
    on input order. Pool-adjacent-violators then merges adjacent blocks
    whose weighted means violate monotonicity.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    if p.size == 0 or p.size != y.size:
        raise ValueError("fit_isotonic: empty or mismatched input")
    order = np.argsort(p, kind="mergesort")
    ps, ys = p[order], y[order]
    knots, starts, counts = np.unique(ps, return_index=True, return_counts=True)
    sums = np.add.reduceat(ys, starts)
    # Each block carries its number of distinct knots as well as its weight.
    blocks: list[list[float]] = []  # [sum_y, observation_count, knot_count]
    for total, count in zip(sums, counts):
        blocks.append([float(total), float(count), 1])
        while len(blocks) >= 2:
            m1 = blocks[-2][0] / blocks[-2][1]
            m2 = blocks[-1][0] / blocks[-1][1]
            if m1 <= m2:
                break
            blocks[-2] = [blocks[-2][0] + blocks[-1][0],
                          blocks[-2][1] + blocks[-1][1],
                          blocks[-2][2] + blocks[-1][2]]
            blocks.pop()
    values: list[float] = []
    for sum_y, count, knot_count in blocks:
        values.extend([sum_y / count] * int(knot_count))
    return IsotonicModel(knots=tuple(float(k) for k in knots), values=tuple(values))


def split_by_resolution_time(
    markets: list[MarketSeries], train_frac: float = 0.6
) -> tuple[list[MarketSeries], list[MarketSeries]]:
    """Split markets by resolved_ts: earliest ``train_frac`` train, rest test.

    Ties are broken by market_id for determinism. Guarantees
    max(train.resolved_ts) <= min(test.resolved_ts).
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError("split: train_frac must be in (0, 1)")
    if len(markets) < 2:
        raise ValueError("split: need at least 2 markets")
    ordered = sorted(markets, key=lambda m: (m.resolved_ts, m.market_id))
    cut = max(1, min(len(ordered) - 1, int(round(len(ordered) * train_frac))))
    return ordered[:cut], ordered[cut:]


def _panel_xy(panel: HorizonPanel) -> tuple[np.ndarray, np.ndarray]:
    p = np.array([q.p for q in panel.points])
    y = np.array([q.outcome for q in panel.points], dtype=float)
    return p, y


def run_correction(
    markets: list[MarketSeries],
    *,
    horizons: list[str],
    horizon_seconds: dict[str, int],
    method: str = "both",
    train_frac: float = 0.6,
    n_bins: int = 10,
    min_per_bin: int = 30,
    n_boot: int = 1000,
    seed: int = 42,
    bootstrap_unit: str = "market",
) -> dict:
    """Fit corrections on the train split; evaluate honestly on the test split.

    Returns a JSON-serializable dict with per-horizon, per-method results.
    Per horizon h and method m: raw and corrected Brier/log-loss/ECE on
    the TEST panel, delta-Brier with the selected bootstrap unit, and the
    verdict ("reliable improvement" only if the CI excludes 0 on the
    improvement side; "no reliable improvement" otherwise).
    """
    if method not in ("platt", "isotonic", "both"):
        raise ValueError(f"correct: unknown method {method!r}")
    if bootstrap_unit not in ("market", "resolution-day", "resolution-week"):
        raise ValueError(f"correct: unknown bootstrap unit {bootstrap_unit!r}")
    train, test = split_by_resolution_time(markets, train_frac)
    methods = ["platt", "isotonic"] if method == "both" else [method]

    out = {
        "method": method,
        "split_protocol": "purged_before_test_observation",
        "bootstrap_unit": bootstrap_unit,
        "train_frac": train_frac,
        "n_train_markets": len(train),
        "n_test_markets": len(test),
        "train_max_resolved_ts": max(m.resolved_ts for m in train),
        "test_min_resolved_ts": min(m.resolved_ts for m in test),
        "horizons": {},
    }
    for h in horizons:
        train_panel = build_panel(train, horizon_seconds[h], h)
        test_panel = build_panel(test, horizon_seconds[h], h)
        n_before = len(train_panel)
        first_test_ts = min((q.observed_ts for q in test_panel.points), default=None)
        if first_test_ts is not None:
            train_panel = HorizonPanel(
                name=h, seconds=horizon_seconds[h],
                points=tuple(q for q in train_panel.points
                             if q.resolved_ts < first_test_ts),
            )
        p_tr, y_tr = _panel_xy(train_panel)
        p_te, y_te = _panel_xy(test_panel)
        grouped_indices: dict[int, list[int]] = {}
        for i, point in enumerate(test_panel.points):
            key = i if bootstrap_unit == "market" else point.resolved_ts // (
                86400 if bootstrap_unit == "resolution-day" else 7 * 86400
            )
            grouped_indices.setdefault(key, []).append(i)
        groups = list(grouped_indices.values())
        entry: dict = {
            "n_train": len(train_panel), "n_test": len(test_panel),
            "n_test_groups": len(groups),
            "n_train_before_purge": n_before,
            "n_train_purged": n_before - len(train_panel),
            "train_max_resolved_ts": max(
                (q.resolved_ts for q in train_panel.points), default=None),
            "test_min_observed_ts": first_test_ts,
        }
        if len(test_panel) < 20 or len(train_panel) < 20:
            entry["skipped"] = "too few train/test panel points (<20)"
            if entry["n_train_purged"]:
                entry["skipped"] += " after removing unavailable labels"
            out["horizons"][h] = entry
            continue
        if len(groups) < 2:
            entry["skipped"] = "need at least 2 test resolution groups for a bootstrap CI"
            out["horizons"][h] = entry
            continue
        entry["raw"] = {
            "brier": brier(p_te, y_te),
            "log_loss": log_loss(p_te, y_te),
            "ece": _safe_ece(p_te, y_te, n_bins, min_per_bin),
        }
        for m_name in methods:
            try:
                if m_name == "platt":
                    model = fit_platt(p_tr, y_tr)
                else:
                    model = fit_isotonic(p_tr, y_tr)
            except ValueError:
                entry[m_name] = {"skipped": "model fit failed on train panel"}
                continue
            p_hat = model.predict(p_te)
            d_brier = float(brier(p_hat, y_te) - brier(p_te, y_te))

            def stat(pts, _model=model, _p_te=p_te, _y_te=y_te) -> float:
                idx = [i for group in pts for i in group]
                return float(
                    brier(_model.predict(_p_te[idx]), _y_te[idx])
                    - brier(_p_te[idx], _y_te[idx])
                )

            lo, hi = bootstrap_groups_ci(
                groups, stat, n_boot=n_boot, seed=seed
            )
            entry[m_name] = {
                "brier": float(brier(p_hat, y_te)),
                "log_loss": float(log_loss(p_hat, y_te)),
                "ece": _safe_ece(p_hat, y_te, n_bins, min_per_bin),
                "delta_brier": d_brier,
                "delta_brier_ci": [lo, hi],
                "delta_log_loss": float(log_loss(p_hat, y_te) - log_loss(p_te, y_te)),
                "verdict": (
                    "reliable improvement"
                    if hi < 0.0
                    else ("reliable degradation" if lo > 0.0
                          else "no reliable improvement")
                ),
            }
        out["horizons"][h] = entry
    return out


def correction_timing_note(correction: dict) -> str:
    """Describe the timing check, including when a saved result lacks it."""
    if correction.get("split_protocol") != "purged_before_test_observation":
        return (
            "Legacy correction: training-label availability at test observation "
            "time was not checked. Rerun longshot correct to apply that check."
        )
    counts = "; ".join(
        f"{h}: {e['n_train']}/{e['n_train_purged']}"
        for h, e in correction["horizons"].items()
    )
    unit = correction.get("bootstrap_unit", "market")
    groups = ", ".join(f"{h}: {e.get('n_test_groups', e['n_test'])}"
                       for h, e in correction["horizons"].items())
    return (
        "At each horizon, training outcomes must resolve strictly before the "
        "earliest test price observation. Training panel counts (kept/purged): "
        + counts + f". Bootstrap unit: {unit}; test groups: {groups}. "
        + ("Related markets can make market-level intervals too narrow."
           if unit == "market" else
           "Resolution time blocks are a dependence sensitivity check, "
           "not verified independent events.")
    )


def _safe_ece(p: np.ndarray, y: np.ndarray, n_bins: int, min_per_bin: int) -> float:
    """ECE, or NaN when every bin is sparse."""
    try:
        return ece(equal_width_bins(p, y, n_bins=n_bins, min_per_bin=min_per_bin))
    except ValueError:
        return float("nan")
