"""Kalshi collector (network-gated; market-data endpoints are keyless today).

- Settled markets: ``GET {base}/trade-api/v2/markets?status=settled&
  limit=200&cursor={c}`` -> ``markets[]`` with ``ticker``, ``title``,
  ``result`` ("yes"/"no"/""), ``close_time``, ``created_time``.
- History: ``GET {base}/trade-api/v2/series/{series_ticker}/markets/
  {ticker}/candlesticks?period_interval=1440`` -> ``candlesticks[]`` with
  ``end_period_ts`` and a ``price`` object.
  Daily candles are the coarsest documented granularity; short-horizon
  panels from Kalshi inherit that limit.

Kalshi's fixed-point migration (confirmed live 2026-08-07) removed the
integer-cent fields rather than deprecating them, so parsing must read the
new names and treat the old ones as fallback:

- ``volume`` -> ``volume_fp``, a decimal *string* of contracts.
- ``price.close`` (cents) -> ``price.close_dollars``, a decimal string
  already denominated in dollars, so it needs no /100 scaling.
- ``series_ticker`` is no longer returned on market objects at all. It is
  the prefix of ``event_ticker`` (``KXHIGHNY-26AUG06`` -> ``KXHIGHNY``),
  which is what the candlesticks path needs.

Reading the old names alone yields zero usable markets, silently: the
parsers skip what they cannot read rather than raising.

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
    series_ticker = raw.get("series_ticker") or _series_from_event(
        raw.get("event_ticker")
    )
    if not ticker or not series_ticker:
        return None
    created_ts = _parse_iso_ts(raw.get("created_time"))
    resolved_ts = _parse_iso_ts(raw.get("close_time"))
    if created_ts is None or resolved_ts is None or resolved_ts <= created_ts:
        return None
    raw_volume = raw.get("volume_fp", raw.get("volume"))
    try:
        volume = float(raw_volume) if raw_volume is not None else None
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


def _series_from_event(event_ticker: object) -> str:
    """Derive the series ticker from an event ticker.

    Market objects stopped carrying ``series_ticker``; the candlesticks path
    still needs it. ``KXHIGHNY-26AUG06`` -> ``KXHIGHNY``.
    """
    if not isinstance(event_ticker, str):
        return ""
    return event_ticker.split("-", 1)[0]


def parse_candlesticks(data: object) -> tuple[PricePoint, ...]:
    """Parse a candlestick payload into a sorted series of [0, 1] prices."""
    if not isinstance(data, dict) or not isinstance(data.get("candlesticks"), list):
        return ()
    points: list[PricePoint] = []
    for c in data["candlesticks"]:
        try:
            ts = int(c["end_period_ts"])
            price = c["price"]
            if "close_dollars" in price:
                close = float(price["close_dollars"])  # already dollars
            else:
                close = float(price["close"]) / 100.0  # legacy integer cents
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
                # Markets living under a day (15-min crypto up/downs dominate
                # the settled stream) can never produce >=2 daily candles;
                # skip the doomed candlestick call for them.
                if parsed["resolved_ts"] - parsed["created_ts"] < 90000:
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
