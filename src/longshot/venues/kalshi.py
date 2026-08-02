"""Kalshi collector (network-gated; market-data endpoints are keyless today).

- Settled markets: ``GET {base}/trade-api/v2/markets?status=settled&
  limit=200&cursor={c}`` -> ``markets[]`` with ``ticker``, ``title``,
  ``result`` ("yes"/"no"/""), ``volume``, ``close_time``,
  ``created_time``, ``series_ticker``.
- History: ``GET {base}/trade-api/v2/series/{series_ticker}/markets/
  {ticker}/candlesticks?period_interval=1440`` -> ``candlesticks[]`` with
  ``end_period_ts`` and ``price.close`` in cents (converted to [0, 1]).
  Daily candles are the coarsest documented granularity; short-horizon
  panels from Kalshi inherit that limit.

If Kalshi ever gates these endpoints, set KALSHI_API_KEY: when present the
client adds ``Authorization: Bearer <key>``; it is never required for
reads at the time of writing. Base URL override: LONGSHOT_KALSHI_BASE.
"""

from __future__ import annotations

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

DEFAULT_BASE = "https://api.elections.kalshi.com"


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


def parse_settled_market(raw: dict) -> dict | None:
    """Validate one settled Kalshi market; None when unusable."""
    result = raw.get("result")
    if result not in ("yes", "no"):
        return None  # "" and other terminal states carry no binary outcome
    ticker = raw.get("ticker")
    series_ticker = raw.get("series_ticker")
    if not ticker or not series_ticker:
        return None
    created_ts = _parse_iso_ts(raw.get("created_time"))
    resolved_ts = _parse_iso_ts(raw.get("close_time"))
    if created_ts is None or resolved_ts is None or resolved_ts <= created_ts:
        return None
    try:
        volume = float(raw.get("volume")) if raw.get("volume") is not None else None
    except (TypeError, ValueError):
        volume = None
    return {
        "ticker": str(ticker),
        "series_ticker": str(series_ticker),
        "question": str(raw.get("title", "")),
        "outcome": 1 if result == "yes" else 0,
        "created_ts": created_ts,
        "resolved_ts": resolved_ts,
        "volume": volume,
    }


def parse_candlesticks(data: object) -> tuple[PricePoint, ...]:
    """Parse a candlestick payload into a sorted series (cents -> [0, 1])."""
    if not isinstance(data, dict) or not isinstance(data.get("candlesticks"), list):
        return ()
    points: list[PricePoint] = []
    for c in data["candlesticks"]:
        try:
            ts = int(c["end_period_ts"])
            close = float(c["price"]["close"]) / 100.0
        except (KeyError, TypeError, ValueError):
            continue
        if 0.0 <= close <= 1.0:
            points.append(PricePoint(ts=ts, price=close))
    points.sort(key=lambda p: p.ts)
    return tuple(points)


@register
class KalshiClient(VenueClient):
    """Settled-market collector for Kalshi (keyless reads today)."""

    venue = "kalshi"

    def __init__(self, api_base: str | None = None, max_calls: int = 5000) -> None:
        self.api_base = (
            api_base or os.environ.get("LONGSHOT_KALSHI_BASE") or DEFAULT_BASE
        ).rstrip("/")
        self.api_key = os.environ.get("KALSHI_API_KEY")
        self.max_calls = max_calls
        self._calls = 0

    def _get(self, path: str, params: dict) -> object:
        if self._calls >= self.max_calls:
            raise VenueUnavailableError(
                self.venue, self.api_base,
                f"--max-calls guard tripped at {self.max_calls}",
            )
        self._calls += 1
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url = f"{self.api_base}{path}?{urllib.parse.urlencode(params)}"
        return http_get_json(url, venue=self.venue, extra_headers=headers)

    def fetch_resolved(
        self,
        max_markets: int = 250,
        seed: int = 42,
        progress_cb: Callable[[str], None] | None = None,
        **kwargs,
    ) -> Iterator[MarketSeries]:
        """Yield settled Kalshi markets with daily-candle price histories."""
        yielded = 0
        cursor: str | None = None
        while yielded < max_markets:
            params: dict = {"status": "settled", "limit": 200}
            if cursor:
                params["cursor"] = cursor
            data = self._get("/trade-api/v2/markets", params)
            if not isinstance(data, dict) or not isinstance(data.get("markets"), list):
                raise VenueUnavailableError(
                    self.venue, f"{self.api_base}/trade-api/v2/markets",
                    f"unexpected payload type {type(data).__name__}",
                )
            for raw in data["markets"]:
                if yielded >= max_markets:
                    break
                parsed = parse_settled_market(raw)
                if parsed is None:
                    continue
                candles = self._get(
                    f"/trade-api/v2/series/{parsed['series_ticker']}"
                    f"/markets/{parsed['ticker']}/candlesticks",
                    {"period_interval": 1440},
                )
                series = parse_candlesticks(candles)
                if len(series) < 2:
                    continue
                yielded += 1
                if progress_cb and yielded % 25 == 0:
                    progress_cb(f"kalshi: fetched {yielded} markets...")
                yield MarketSeries(
                    venue=self.venue,
                    market_id=parsed["ticker"],
                    question=parsed["question"],
                    category="uncategorized",
                    created_ts=parsed["created_ts"],
                    resolved_ts=parsed["resolved_ts"],
                    outcome=parsed["outcome"],
                    volume=parsed["volume"],
                    n_traders=None,
                    series=series,
                    provenance=provenance_meta(
                        source="kalshi v2 settled markets + candlesticks",
                        api_base=self.api_base,
                        notes="daily candles (1440 min); close price, cents/100",
                    ),
                )
                time.sleep(POLITE_SLEEP_S)
            cursor = data.get("cursor") or None
            if not cursor:
                break
            time.sleep(POLITE_SLEEP_S)
