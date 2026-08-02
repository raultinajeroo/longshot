"""Horizon panels: market prices at fixed times before resolution.

For each market with resolution time T and each horizon h (seconds), the
panel price is a carry-forward (last-observation-carried-forward) value:

    p_h = price of the last PricePoint with ts <= T - h

Two guards keep the panel honest:

- Staleness bound: the carried price must be at most h old, i.e.
  ts >= T - 2h. Without this, a market whose last trade was weeks before
  resolution would contaminate short-horizon panels with ancient prices.
- Lifetime bound: h <= T - created_ts, i.e. the market must have existed
  at T - h. (With the standard creation-time prior point this is usually
  satisfied, but we check explicitly.)

Markets failing either guard are skipped for that horizon.
"""

from __future__ import annotations

from bisect import bisect_right
from typing import Iterable

from .types import HorizonPanel, HorizonPoint, MarketSeries

HORIZON_PRESETS: dict[str, int] = {
    "30d": 30 * 86400,
    "14d": 14 * 86400,
    "7d": 7 * 86400,
    "3d": 3 * 86400,
    "1d": 86400,
    "12h": 12 * 3600,
    "1h": 3600,
}

DEFAULT_HORIZONS = ["30d", "14d", "7d", "3d", "1d", "12h", "1h"]


def parse_horizon(name: str) -> int:
    """Parse a horizon name like "7d", "12h", "1h" into seconds."""
    name = name.strip().lower()
    if name in HORIZON_PRESETS:
        return HORIZON_PRESETS[name]
    unit = name[-1]
    mult = {"d": 86400, "h": 3600, "m": 60}.get(unit)
    if mult is None or not name[:-1].isdigit():
        raise ValueError(f"horizons: cannot parse horizon {name!r}")
    return int(name[:-1]) * mult


def build_panel(
    markets: Iterable[MarketSeries],
    horizon_seconds: int,
    name: str,
) -> HorizonPanel:
    """Build the (price, outcome) panel at ``horizon_seconds`` before resolution.

    Applies the carry-forward rule plus the staleness and lifetime bounds
    documented in the module docstring.
    """
    h = horizon_seconds
    points: list[HorizonPoint] = []
    for m in markets:
        T = m.resolved_ts
        if h > T - m.created_ts:
            continue  # market did not exist at T - h
        target = T - h
        ts_list = [pt.ts for pt in m.series]
        i = bisect_right(ts_list, target) - 1
        if i < 0:
            continue  # no observation at or before T - h
        pt = m.series[i]
        if pt.ts < T - 2 * h:
            continue  # staleness bound: price older than one horizon
        points.append(
            HorizonPoint(
                market_id=m.market_id,
                category=m.category,
                volume=m.volume,
                resolved_ts=T,
                p=pt.price,
                outcome=m.outcome,
            )
        )
    return HorizonPanel(name=name, seconds=h, points=tuple(points))


def build_panels(
    markets: Iterable[MarketSeries],
    horizons: list[str] | None = None,
) -> list[HorizonPanel]:
    """Build panels for several named horizons, preserving order."""
    markets = list(markets)
    horizons = horizons or DEFAULT_HORIZONS
    return [build_panel(markets, parse_horizon(h), h) for h in horizons]
