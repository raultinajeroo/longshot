"""Core data model for longshot.

A :class:`MarketSeries` is one resolved binary market: metadata plus the
probability-of-YES time series observed while it was trading. A
:class:`HorizonPoint` is one (market, horizon) observation: the price seen
``horizon`` seconds before resolution, paired with the realized outcome.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PricePoint:
    """One observation of the market-implied probability of YES.

    ``ts`` is unix seconds; ``price`` is in [0, 1].
    """

    ts: int
    price: float


@dataclass(frozen=True)
class MarketSeries:
    """A resolved binary market with its probability history.

    ``outcome`` is 1 if the market resolved YES, 0 if NO. ``series`` is
    sorted by ``ts`` ascending. ``category`` may be "uncategorized".
    ``provenance`` records where the data came from (fetched_at, source,
    api_base, notes) so any downstream number is traceable.
    """

    venue: str
    market_id: str
    question: str
    category: str
    created_ts: int
    resolved_ts: int
    outcome: int
    volume: float | None
    n_traders: int | None
    series: tuple[PricePoint, ...]
    provenance: dict = field(default_factory=dict)

    @property
    def life_seconds(self) -> int:
        """Trading lifetime from creation to resolution."""
        return self.resolved_ts - self.created_ts


@dataclass(frozen=True)
class HorizonPoint:
    """One (market, horizon) calibration observation."""

    market_id: str
    category: str
    volume: float | None
    resolved_ts: int
    p: float  # probability of YES, observed `horizon` before resolution
    outcome: int  # 1 = resolved YES, 0 = NO


@dataclass(frozen=True)
class HorizonPanel:
    """All markets' prices at a fixed time-to-resolution horizon."""

    name: str  # e.g. "7d"
    seconds: int  # horizon in seconds
    points: tuple[HorizonPoint, ...]

    def __len__(self) -> int:
        return len(self.points)
