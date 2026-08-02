"""Synthetic market generator (seeded), for fixtures and algorithm tests.

Each simulated market has a true probability q ~ Beta(2, 2) and an outcome
y ~ Bernoulli(q). The market's *belief* path is a Brownian bridge in logit
space from logit(q) at creation to logit(0.99) (if y=1) or logit(0.01)
(if y=0) at resolution, with a time warp r^gamma (gamma = 5) so most of
the move toward the truth happens late in the market's life, as in real
markets. The bridge noise scale (sigma = 0.35) was tuned so the
``calibrated`` mode measures a compression slope whose CI covers 1 at the
30-day horizon. The *observed price* applies the mode's bias transform to the
belief at every point; the final point is pinned to the terminal belief
(real markets converge to the outcome), which is a documented deviation
from applying the bias transform uniformly.

Modes:
- ``calibrated``: price = belief + N(0, 0.03) noise, clipped.
- ``compressed``: price = 0.5 + c * (belief - 0.5), c = 0.55 default —
  the chronic compression-toward-50% bias reported for political markets
  (Le 2026). Expected measured compression slope ~ 1/c ~ 1.8.
- ``longshot-bias``: price = belief + d if q < 0.5 else belief - d,
  d = 0.08 default, clipped to [0.01, 0.99] — longshots overpriced,
  favorites underpriced (classic favorite-longshot bias).

Output schema is identical to venue JSONL (venue = "simulated"), so
fixtures validate through store.load like any real download.
"""

from __future__ import annotations

import numpy as np

from .types import MarketSeries, PricePoint

MODES = ("calibrated", "compressed", "longshot-bias")
BASE_TS = 1735689600  # 2025-01-01T00:00:00Z


def _belief_path(
    rng: np.random.Generator,
    q: float,
    outcome: int,
    n_points: int,
    gamma: float = 5.0,
    sigma: float = 0.35,
) -> np.ndarray:
    """Logit-space Brownian bridge from logit(q) to the terminal belief."""
    terminal = 0.99 if outcome == 1 else 0.01
    r = np.linspace(0.0, 1.0, n_points)
    l0, l1 = np.log(q / (1 - q)), np.log(terminal / (1 - terminal))
    # Standard Brownian bridge: cumulative noise minus its linear trend.
    noise = np.concatenate([[0.0], rng.normal(0.0, 1.0, n_points - 1).cumsum()])
    bridge = noise - r * noise[-1]
    logit = l0 + (l1 - l0) * r**gamma + sigma * bridge
    logit[-1] = l1  # pin the terminal point exactly
    return 1.0 / (1.0 + np.exp(-logit))


def _apply_mode(
    belief: np.ndarray,
    mode: str,
    rng: np.random.Generator,
    *,
    compression: float,
    bias_shift: float,
    q: float,
) -> np.ndarray:
    if mode == "calibrated":
        p = belief + rng.normal(0.0, 0.03, belief.size)
    elif mode == "compressed":
        p = 0.5 + compression * (belief - 0.5)
    elif mode == "longshot-bias":
        p = belief + (bias_shift if q < 0.5 else -bias_shift)
    else:
        raise ValueError(f"simulate: unknown mode {mode!r}")
    p = np.clip(p, 0.01, 0.99)
    p[-1] = belief[-1]  # terminal convergence to the outcome side
    return p


def simulate_markets(
    mode: str,
    n_markets: int = 120,
    seed: int = 7,
    *,
    compression: float = 0.55,
    bias_shift: float = 0.08,
    n_points: int = 40,
    life_days: tuple[int, int] = (60, 120),
) -> list[MarketSeries]:
    """Generate ``n_markets`` synthetic resolved markets. Deterministic by seed.

    ``life_days`` bounds the uniform market lifetime; shorter lives (relative
    to an analysis horizon) mean the belief path has drifted less toward the
    terminal outcome, so planted price biases dominate at that horizon.
    """
    if mode not in MODES:
        raise ValueError(f"simulate: mode must be one of {MODES}")
    rng = np.random.default_rng(seed)
    markets: list[MarketSeries] = []
    for i in range(n_markets):
        q = float(rng.beta(2.0, 2.0))
        q = min(0.97, max(0.03, q))
        outcome = int(rng.random() < q)
        life = int(rng.integers(life_days[0] * 86400, life_days[1] * 86400))
        resolved_ts = BASE_TS + int(rng.integers(0, 300 * 86400))
        created_ts = resolved_ts - life
        belief = _belief_path(rng, q, outcome, n_points)
        price = _apply_mode(
            belief, mode, rng,
            compression=compression, bias_shift=bias_shift, q=q,
        )
        ts = created_ts + (r := np.linspace(0, life, n_points)).astype(int)
        series = tuple(
            PricePoint(ts=int(t), price=float(p_)) for t, p_ in zip(ts, price)
        )
        markets.append(
            MarketSeries(
                venue="simulated",
                market_id=f"sim-{mode}-{seed}-{i:04d}",
                question=f"Simulated {mode} market {i} (seed {seed})",
                category="simulated",
                created_ts=created_ts,
                resolved_ts=resolved_ts,
                outcome=outcome,
                volume=float(round(rng.lognormal(8.0, 1.0), 2)),
                n_traders=int(rng.integers(15, 400)),
                series=series,
                provenance={
                    "source": "longshot.simulate",
                    "mode": mode,
                    "seed": seed,
                    "compression": compression if mode == "compressed" else None,
                    "bias_shift": bias_shift if mode == "longshot-bias" else None,
                    "notes": "logit-space Brownian bridge, r^5 time warp",
                },
            )
        )
    return markets
