"""Kalshi collector (network-gated; market-data endpoints are keyless today).

Retrieval goes series-first, never by trawling a global market stream. That
is not a stylistic choice; measured 2026-08-10, stream trawling returns
**nothing usable at any depth**:

===========================================  ==================  =========
stream                                       median market life  survivors
===========================================  ==================  =========
``/markets?status=settled`` (6000 rows)      679s                0
``  ... &mve_filter=exclude`` (200 rows)     7162s               0
``/historical/markets`` (200 rows)           ~1400s              0
``  ... &mve_filter=exclude`` (1200 rows)    5.7h (p90 15.5h)    0
===========================================  ==================  =========

Every undifferentiated stream is ordered by recency, and high-frequency
series own recency at every tier: sports micro-parlays, then hourly
commodity/temperature series, then per-game MLB props. A market needs to
live over ~25h to yield the two daily candles a calibration series needs,
and none of these come within two orders of magnitude. Excluding one layer
only exposes the next.

So the pipeline is:

1. ``GET {base}/trade-api/v2/series?category={c}`` -> the whole catalogue
   in **one uncursored call** (12620 series as of 2026-08-10). This is the
   only endpoint that escapes recency ordering. Drop ``frequency`` in
   {hourly, daily}; in Politics that removes 5 series out of 2138.
2. ``GET {base}/trade-api/v2/historical/markets?series_ticker={t}`` ->
   settled markets for one series. The *historical* tier is required, not
   optional: ``GET /historical/cutoff`` reported
   ``market_settled_ts = 2026-06-11T00:00:00Z``, and everything settled
   before that has left ``?status=settled`` entirely. Six sampled
   long-lived Politics series returned 0 rows from ``status=settled`` and
   real markets from the historical tier.
3. ``GET {base}/trade-api/v2/historical/markets/{ticker}/candlesticks
   ?period_interval=1440&start_ts={a}&end_ts={b}`` -> ``candlesticks[]``
   with ``end_period_ts`` and a ``price`` object. Daily candles are the
   coarsest documented granularity; short-horizon panels inherit that.

Two traps in step 3, both 400-on-violation and both found only by probing:
the ``/series/{s}/markets/{t}/candlesticks`` path **400s for historical
markets** (it is for live ones), and ``start_ts``/``end_ts`` are
**required** — ``period_interval`` alone 400s, and so does a range without
it. Only all three together succeed.

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
import random
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

# Daily candles: the coarsest interval, and the only one accepted alongside
# an explicit ts range on the historical candlesticks path.
CANDLE_PERIOD_MIN = 1440

# A market must outlive this to produce the >=2 daily candles a calibration
# series needs. 25h rather than 24h so a market spanning a single day
# boundary is not counted on the strength of one candle plus a sliver.
MIN_LIFETIME_S = 90_000

# Series whose markets are structurally too short-lived to ever qualify.
# Cheaper to skip by declared cadence than to discover per market.
SKIP_FREQUENCIES = frozenset({"hourly", "daily"})

# Categories worth walking by default. Kalshi splits political markets
# across two ("Politics" 2138 series, "Elections" 1538), and omitting
# either loses half the calibration set.
DEFAULT_CATEGORIES: tuple[str, ...] = ("Politics", "Elections")

_BUDGET_REMEDY = "raise --max-calls (the host is reachable; the budget ran out)"


class CallBudgetExceeded(VenueUnavailableError):
    """The ``--max-calls`` guard tripped.

    A subclass because the per-market handlers below swallow fetch failures
    to keep one bad market from killing a run — and must not swallow this
    one, which would spin the loop making zero further progress.
    """


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


def _candle_price(value: object) -> float | None:
    """One candle price as a probability in [0, 1]; None when unusable.

    **The type is the unit.** Kalshi's fixed-point migration turned these
    into decimal *strings* already denominated in dollars ("0.2100"); the
    legacy form was an integer number of cents (21). Scaling a string by
    1/100 is the bug that made every price fetched on 2026-08-10 exactly
    100x too small -- and it survived every range check, because a
    100x-shrunk probability is still a valid probability.
    """
    if isinstance(value, str):
        try:
            price = float(value)  # fixed-point: already dollars
        except ValueError:
            return None
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        price = float(value) / 100.0  # legacy integer cents
    else:
        return None  # None, dict, or anything else
    return price if 0.0 <= price <= 1.0 else None


def parse_candlesticks(data: object) -> tuple[PricePoint, ...]:
    """Parse a candlestick payload into a sorted series of [0, 1] prices.

    ``price.close`` is the last *traded* price and is null on any day the
    market did not trade -- routine for the long-lived political markets
    this collector targets, and enough to gut a series if those days are
    dropped. ``price.previous`` carries the last known trade forward, which
    is what the market was still saying on a silent day, so it is used as
    the fallback rather than leaving a hole at that horizon.
    """
    if not isinstance(data, dict) or not isinstance(data.get("candlesticks"), list):
        return ()
    points: list[PricePoint] = []
    for c in data["candlesticks"]:
        if not isinstance(c, dict):
            continue
        try:
            ts = int(c["end_period_ts"])
        except (KeyError, TypeError, ValueError):
            continue
        price = c.get("price")
        if not isinstance(price, dict):
            continue
        for field in ("close_dollars", "close", "previous"):
            close = _candle_price(price.get(field))
            if close is not None:
                points.append(PricePoint(ts=ts, price=close))
                break
    points.sort(key=lambda p: p.ts)
    return tuple(points)


@register
class KalshiClient(VenueClient):
    """Settled-market collector for Kalshi (keyless reads today)."""

    venue = "kalshi"

    def __init__(
        self,
        api_base: str | None = None,
        max_calls: int = 5000,
        categories: tuple[str, ...] = DEFAULT_CATEGORIES,
    ) -> None:
        self.api_base = (
            api_base or os.environ.get("LONGSHOT_KALSHI_BASE") or DEFAULT_BASE
        ).rstrip("/")
        self.api_key = os.environ.get("KALSHI_API_KEY")
        self.max_calls = max_calls
        self.categories = categories
        self._calls = 0
        # Drop counters, reported at the end of a run. The original failure
        # mode here was discarding 100% of every page in silence, so a run
        # that yields little must be able to say where everything went.
        self.stats: dict[str, int] = {}

    def _bump(self, key: str, n: int = 1) -> None:
        self.stats[key] = self.stats.get(key, 0) + n

    def _get(self, path: str, params: dict) -> object:
        if self._calls >= self.max_calls:
            raise CallBudgetExceeded(
                self.venue, self.api_base,
                f"--max-calls guard tripped at {self.max_calls}",
                remedy=_BUDGET_REMEDY,
            )
        self._calls += 1
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url = f"{self.api_base}{path}?{urllib.parse.urlencode(params)}"
        return http_get_json(url, venue=self.venue, extra_headers=headers)

    def candidate_series(self) -> list[tuple[str, str]]:
        """``(ticker, category)`` for series that can hold long-lived markets.

        One HTTP call: ``/series`` returns the whole catalogue uncursored.
        """
        out: list[tuple[str, str]] = []
        for category in self.categories or (None,):
            params = {"limit": 200}
            if category:
                params["category"] = category
            data = self._get("/trade-api/v2/series", params)
            if not isinstance(data, dict) or not isinstance(data.get("series"), list):
                raise VenueUnavailableError(
                    self.venue, f"{self.api_base}/trade-api/v2/series",
                    f"unexpected payload type {type(data).__name__}",
                )
            for s in data["series"]:
                if not isinstance(s, dict):
                    continue
                self._bump("series_seen")
                if str(s.get("frequency", "")).lower() in SKIP_FREQUENCIES:
                    self._bump("series_skipped_frequency")
                    continue
                ticker = s.get("ticker")
                if not ticker:
                    continue
                out.append((str(ticker), str(s.get("category", "uncategorized"))))
            time.sleep(POLITE_SLEEP_S)
        return out

    def _settled_in_series(self, series_ticker: str) -> Iterator[dict]:
        """Yield parsed, long-lived settled markets for one series."""
        cursor: str | None = None
        while True:
            params: dict = {"series_ticker": series_ticker, "limit": 200}
            if cursor:
                params["cursor"] = cursor
            data = self._get("/trade-api/v2/historical/markets", params)
            if not isinstance(data, dict) or not isinstance(data.get("markets"), list):
                return
            for raw in data["markets"]:
                self._bump("markets_seen")
                parsed = parse_settled_market(raw)
                if parsed is None:
                    self._bump("drop_unparsable")
                    continue
                if parsed["resolved_ts"] - parsed["created_ts"] < MIN_LIFETIME_S:
                    self._bump("drop_too_short")
                    continue
                yield parsed
            cursor = data.get("cursor") or None
            if not cursor:
                return
            time.sleep(POLITE_SLEEP_S)

    def _daily_candles(self, parsed: dict) -> tuple[PricePoint, ...]:
        """Daily candles for one settled market; empty when unavailable.

        A 400 here means this market has no usable history, not that the
        run is broken, so it is counted and skipped. ``CallBudgetExceeded``
        is deliberately not caught.
        """
        try:
            data = self._get(
                f"/trade-api/v2/historical/markets/{parsed['ticker']}"
                f"/candlesticks",
                {
                    "period_interval": CANDLE_PERIOD_MIN,
                    "start_ts": parsed["created_ts"] - 86400,
                    "end_ts": parsed["resolved_ts"] + 86400,
                },
            )
        except CallBudgetExceeded:
            raise
        except VenueUnavailableError:
            self._bump("drop_candle_error")
            return ()
        return parse_candlesticks(data)

    def fetch_resolved(
        self,
        max_markets: int = 250,
        seed: int = 42,
        progress_cb: Callable[[str], None] | None = None,
        **kwargs,
    ) -> Iterator[MarketSeries]:
        """Yield settled Kalshi markets with daily-candle price histories."""
        series_list = self.candidate_series()
        # Shuffle so a bounded run samples across the catalogue instead of
        # taking whatever sorts first, which would bias toward one series.
        random.Random(seed).shuffle(series_list)
        if progress_cb:
            progress_cb(
                f"kalshi: walking {len(series_list)} series "
                f"in {', '.join(self.categories) or 'all categories'}"
            )

        yielded = 0
        for series_ticker, category in series_list:
            if yielded >= max_markets:
                break
            self._bump("series_walked")
            for parsed in self._settled_in_series(series_ticker):
                if yielded >= max_markets:
                    break
                candles = self._daily_candles(parsed)
                if len(candles) < 2:
                    self._bump("drop_thin_history")
                    continue
                yielded += 1
                if progress_cb and yielded % 25 == 0:
                    progress_cb(f"kalshi: fetched {yielded} markets...")
                yield MarketSeries(
                    venue=self.venue,
                    market_id=parsed["ticker"],
                    question=parsed["question"],
                    category=category,
                    created_ts=parsed["created_ts"],
                    resolved_ts=parsed["resolved_ts"],
                    outcome=parsed["outcome"],
                    volume=parsed["volume"],
                    n_traders=None,
                    series=candles,
                    provenance=provenance_meta(
                        source="kalshi v2 historical markets + candlesticks",
                        api_base=self.api_base,
                        notes="daily candles (1440 min); close price in dollars",
                    ),
                )
                time.sleep(POLITE_SLEEP_S)

        if progress_cb:
            drops = ", ".join(
                f"{k}={v}" for k, v in sorted(self.stats.items()) if k.startswith("drop")
            )
            progress_cb(
                f"kalshi: {yielded} markets from {self.stats.get('series_walked', 0)} "
                f"series in {self._calls} calls" + (f" ({drops})" if drops else "")
            )
