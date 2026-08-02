"""Fixture venue: loads MarketSeries from local JSONL files.

This is the offline path used by ``longshot demo`` and the test suite:
any file in store format works — the bundled Manifold sample, simulator
output, or a previously fetched cache.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator

from ..store import iter_jsonl
from ..types import MarketSeries
from .base import VenueClient, register

DEFAULT_PATHS = ["data/bundled/manifold_resolved_sample.jsonl"]


@register
class FixtureClient(VenueClient):
    """Serves markets from local JSONL files in store format."""

    venue = "fixture"

    def __init__(self, paths: list[str] | None = None) -> None:
        self.paths = [Path(p) for p in (paths or DEFAULT_PATHS)]

    def fetch_resolved(
        self,
        max_markets: int = 250,
        seed: int = 42,
        progress_cb: Callable[[str], None] | None = None,
        **kwargs,
    ) -> Iterator[MarketSeries]:
        """Yield up to ``max_markets`` markets from the configured files."""
        n = 0
        for path in self.paths:
            for m in iter_jsonl(path):
                if n >= max_markets:
                    return
                n += 1
                yield m
