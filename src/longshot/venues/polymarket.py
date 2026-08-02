"""Polymarket collector (network-gated; no API key required).

Two APIs:

- Gamma (``https://gamma-api.polymarket.com``): market metadata.
  ``GET /markets?closed=true&limit=100&offset={n}`` — fields vary, so
  parsing is defensive. Outcome inference from terminal prices: the YES
  entry of ``outcomePrices`` >= 0.98 -> outcome 1, <= 0.02 -> outcome 0,
  anything in between is skipped (ambiguous terminal prices excluded — a
  documented judgment call).
- CLOB (``https://clob.polymarket.com``): price history.
  ``GET /prices-history?market={yes_token_id}&interval=max&fidelity=720``
  returns ``{"history": [{"t": .., "p": ..}]}`` at 12-hour granularity —
  the documented fidelity for closed markets; short-horizon panels built
  from Polymarket data inherit that granularity limit.

Base URLs are overridable via LONGSHOT_POLYMARKET_GAMMA and
LONGSHOT_POLYMARKET_CLOB (tests point them at a local stub server).
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
from datetime import datetime, timezone
from typing import Callable, Iterator

from ..store import provenance_meta
from ..types import MarketSeries, PricePoint
from .base import (
    POLITE_SLEEP_S,
    VenueClient,
    VenueUnavailableError,
    http_get_json,
    register,
)

DEFAULT_GAMMA_BASE = "https://gamma-api.polymarket.com"
DEFAULT_CLOB_BASE = "https://clob.polymarket.com"

YES_THRESHOLD = 0.98
NO_THRESHOLD = 0.02


def _parse_json_list(value: object) -> list:
    """Gamma encodes list fields as JSON strings; accept both forms."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _parse_iso_ts(value: object) -> int | None:
    """Parse an ISO-8601 timestamp into unix seconds; None when absent/bad."""
    if not value or not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def parse_gamma_market(raw: dict) -> dict | None:
    """Validate one closed Gamma market; None when unusable (see docstring)."""
    tokens = [str(t) for t in _parse_json_list(raw.get("clobTokenIds"))]
    prices = _parse_json_list(raw.get("outcomePrices"))
    if not tokens or not prices:
        return None
    try:
        yes_price = float(prices[0])
    except (TypeError, ValueError, IndexError):
        return None
    if yes_price >= YES_THRESHOLD:
        outcome = 1
    elif yes_price <= NO_THRESHOLD:
        outcome = 0
    else:
        return None  # ambiguous terminal price: excluded by policy
    resolved_ts = (
        _parse_iso_ts(raw.get("closedTime"))
        or _parse_iso_ts(raw.get("umaEndDate"))
        or _parse_iso_ts(raw.get("endDate"))
    )
    created_ts = (
        _parse_iso_ts(raw.get("createdAt")) or _parse_iso_ts(raw.get("startDate"))
    )
    if resolved_ts is None or created_ts is None or resolved_ts <= created_ts:
        return None
    volume = raw.get("volumeNum")
    if volume is None:
        try:
            volume = float(raw.get("volume") or 0.0)
        except (TypeError, ValueError):
            volume = None
    else:
        try:
            volume = float(volume)
        except (TypeError, ValueError):
            volume = None
    return {
        "id": str(raw.get("id")),
        "yes_token": tokens[0],
        "question": str(raw.get("question", "")),
        "outcome": outcome,
        "created_ts": created_ts,
        "resolved_ts": resolved_ts,
        "volume": volume,
    }


def parse_prices_history(data: object) -> tuple[PricePoint, ...]:
    """Parse a CLOB prices-history payload into a sorted series."""
    if not isinstance(data, dict) or not isinstance(data.get("history"), list):
        return ()
    points: list[PricePoint] = []
    for item in data["history"]:
        try:
            ts = int(item["t"])
            price = float(item["p"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0.0 <= price <= 1.0:
            points.append(PricePoint(ts=ts, price=price))
    points.sort(key=lambda p: p.ts)
    return tuple(points)


@register
class PolymarketClient(VenueClient):
    """Closed-market collector for Polymarket (keyless today)."""

    venue = "polymarket"

    def __init__(
        self,
        gamma_base: str | None = None,
        clob_base: str | None = None,
        max_calls: int = 5000,
    ) -> None:
        self.gamma_base = (
            gamma_base
            or os.environ.get("LONGSHOT_POLYMARKET_GAMMA")
            or DEFAULT_GAMMA_BASE
        ).rstrip("/")
        self.clob_base = (
            clob_base
            or os.environ.get("LONGSHOT_POLYMARKET_CLOB")
            or DEFAULT_CLOB_BASE
        ).rstrip("/")
        self.max_calls = max_calls
        self._calls = 0

    def _get(self, base: str, path: str, params: dict) -> object:
        if self._calls >= self.max_calls:
            raise VenueUnavailableError(
                self.venue, base, f"--max-calls guard tripped at {self.max_calls}"
            )
        self._calls += 1
        url = f"{base}{path}?{urllib.parse.urlencode(params)}"
        return http_get_json(url, venue=self.venue)

    def fetch_resolved(
        self,
        max_markets: int = 250,
        seed: int = 42,
        progress_cb: Callable[[str], None] | None = None,
        **kwargs,
    ) -> Iterator[MarketSeries]:
        """Yield closed Polymarket markets with CLOB price histories."""
        yielded = 0
        offset = 0
        while yielded < max_markets:
            data = self._get(self.gamma_base, "/markets", {
                "closed": "true", "limit": 100, "offset": offset,
            })
            if not isinstance(data, list):
                raise VenueUnavailableError(
                    self.venue, f"{self.gamma_base}/markets",
                    f"unexpected payload type {type(data).__name__}",
                )
            if not data:
                break
            offset += len(data)
            for raw in data:
                if yielded >= max_markets:
                    break
                parsed = parse_gamma_market(raw)
                if parsed is None:
                    continue
                history = self._get(self.clob_base, "/prices-history", {
                    "market": parsed["yes_token"],
                    "interval": "max",
                    "fidelity": 720,
                })
                series = parse_prices_history(history)
                if len(series) < 2:
                    continue
                yielded += 1
                if progress_cb and yielded % 25 == 0:
                    progress_cb(f"polymarket: fetched {yielded} markets...")
                yield MarketSeries(
                    venue=self.venue,
                    market_id=parsed["id"],
                    question=parsed["question"],
                    category="uncategorized",
                    created_ts=parsed["created_ts"],
                    resolved_ts=parsed["resolved_ts"],
                    outcome=parsed["outcome"],
                    volume=parsed["volume"],
                    n_traders=None,
                    series=series,
                    provenance=provenance_meta(
                        source="polymarket gamma markets + clob prices-history",
                        api_base=f"{self.gamma_base}, {self.clob_base}",
                        notes=(
                            "outcome inferred from terminal outcomePrices "
                            "(>=0.98 YES, <=0.02 NO); 12h history fidelity"
                        ),
                    ),
                )
                time.sleep(POLITE_SLEEP_S)
            time.sleep(POLITE_SLEEP_S)
