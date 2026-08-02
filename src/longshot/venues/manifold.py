"""Manifold Markets client (play-money venue; fully supported, works offline
only in the sense that no API key is required — it does need the network).

Discovery: ``GET {base}/v0/search-markets?term={term}&filter=resolved&
contractType=BINARY&sort=score&limit=100`` over a fixed term list covering
major categories; results are deduped by market id.

History: ``GET {base}/v0/bets?contractId={id}&limit=1000[&before={bet_id}]``
paged descending by time. The probability series is built from filled,
non-cancelled, non-redemption bets as ``(createdTime, probAfter)``,
prepended by the creation-time prior ``(market.createdTime, 0.5)`` — every
Manifold market opens at the creator's implied probability, and 0.5 is our
documented neutral prior when the true opening probability is unavailable
from the lite-market payload. Times are converted ms -> s.

Filters: resolution in {YES, NO}; volume >= 100; uniqueBettorCount >= 10
when present; resolved_ts - created_ts >= 3 days; non-empty bet series.

Category assignment is a documented keyword heuristic (see CATEGORY_RULES).

When more markets pass the filters than ``max_markets``, a seeded
reservoir sample is drawn so the category mix survives.
"""

from __future__ import annotations

import os
import random
import re
import time
import urllib.parse
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

DEFAULT_API_BASE = "https://api.manifold.markets"

SEARCH_TERMS = [
    "fed", "election", "trump", "bitcoin", "crypto", "nba", "nfl", "ai",
    "openai", "ukraine", "china", "inflation", "oscars", "spacex", "climate",
    "supreme court", "gdp", "apple", "congress", "world cup",
]

MIN_VOLUME = 100.0
MIN_TRADERS = 10
MIN_LIFE_S = 3 * 86400

# Keyword heuristic, checked in order; first matching category wins.
CATEGORY_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("politics", ("election", "trump", "biden", "congress", "senate",
                  "supreme court", "president")),
    ("macro", ("fed", "inflation", "gdp", "rate", "fomc", "jobs report", "cpi")),
    ("crypto", ("bitcoin", "btc", "eth", "crypto", "solana")),
    ("sports", ("nba", "nfl", "mlb", "soccer", "world cup", "super bowl", "ufc")),
    ("tech_ai", ("ai", "openai", "gpt", "apple", "spacex", "tesla", "google",
                 "meta")),
    ("entertainment", ("oscar", "movie", "album", "taylor")),
]


def _keyword_matches(question_tokens: set[str], question_lower: str,
                     keyword: str) -> bool:
    """Keyword match rule (documented heuristic):

    - multi-word keywords ("supreme court"): substring match;
    - short keywords (<= 3 chars, e.g. "ai", "fed"): exact word match, so
      "ai" does not fire on "rain";
    - longer keywords: word-prefix match, so "oscar" matches "Oscars".
    """
    if " " in keyword:
        return keyword in question_lower
    if len(keyword) <= 3:
        return keyword in question_tokens
    return any(t.startswith(keyword) for t in question_tokens)


def classify_category(question: str) -> str:
    """Assign a category by keyword heuristic; "other" when nothing matches."""
    q = question.lower()
    tokens = set(re.findall(r"[a-z0-9]+", q))
    for category, keywords in CATEGORY_RULES:
        if any(_keyword_matches(tokens, q, k) for k in keywords):
            return category
    return "other"


def parse_search_market(raw: dict) -> dict | None:
    """Validate one lite-market search result; None if it fails the filters."""
    if raw.get("resolution") not in ("YES", "NO"):
        return None
    try:
        volume = float(raw.get("volume") or 0.0)
    except (TypeError, ValueError):
        return None
    if volume < MIN_VOLUME:
        return None
    traders = raw.get("uniqueBettorCount")
    if traders is not None and int(traders) < MIN_TRADERS:
        return None
    created_ms = raw.get("createdTime")
    resolved_ms = raw.get("resolutionTime")
    if not created_ms or not resolved_ms:
        return None
    if int(resolved_ms) - int(created_ms) < MIN_LIFE_S * 1000:
        return None
    return {
        "id": str(raw["id"]),
        "question": str(raw.get("question", "")),
        "outcome": 1 if raw["resolution"] == "YES" else 0,
        "created_ts": int(created_ms) // 1000,
        "resolved_ts": int(resolved_ms) // 1000,
        "volume": volume,
        "n_traders": None if traders is None else int(traders),
    }


def series_from_bets(bets: list[dict], created_ts: int) -> tuple[PricePoint, ...]:
    """Build the probability series from raw bet objects (see module docstring).

    Keeps filled, non-cancelled, non-redemption bets; sorts ascending by
    time; prepends the creation-time prior (created_ts, 0.5).
    """
    points = [PricePoint(ts=created_ts, price=0.5)]
    kept = [
        b for b in bets
        if b.get("isFilled") and not b.get("isCancelled")
        and not b.get("isRedemption")
        and isinstance(b.get("probAfter"), (int, float))
        and b.get("createdTime") is not None
    ]
    kept.sort(key=lambda b: int(b["createdTime"]))
    for b in kept:
        price = float(b["probAfter"])
        points.append(
            PricePoint(ts=int(b["createdTime"]) // 1000,
                       price=min(1.0, max(0.0, price)))
        )
    return tuple(points)


@register
class ManifoldClient(VenueClient):
    """Resolved-market collector for Manifold (no API key required)."""

    venue = "manifold"

    def __init__(self, api_base: str | None = None, max_calls: int = 5000) -> None:
        self.api_base = (
            api_base
            or os.environ.get("LONGSHOT_MANIFOLD_BASE")
            or DEFAULT_API_BASE
        ).rstrip("/")
        self.max_calls = max_calls
        self._calls = 0

    def _get(self, path: str, params: dict) -> object:
        if self._calls >= self.max_calls:
            raise VenueUnavailableError(
                self.venue, self.api_base,
                f"--max-calls guard tripped at {self.max_calls} requests",
            )
        self._calls += 1
        url = f"{self.api_base}{path}?{urllib.parse.urlencode(params)}"
        return http_get_json(url, venue=self.venue)

    def search_resolved(self, term: str, limit: int = 100) -> list[dict]:
        """One search page of resolved binary markets for ``term``."""
        data = self._get("/v0/search-markets", {
            "term": term,
            "filter": "resolved",
            "contractType": "BINARY",
            "sort": "score",
            "limit": limit,
        })
        if not isinstance(data, list):
            raise VenueUnavailableError(
                self.venue, f"{self.api_base}/v0/search-markets",
                f"unexpected payload type {type(data).__name__}",
            )
        return data

    def fetch_bets(self, market_id: str, max_bets: int = 4000) -> list[dict]:
        """Page through a market's bet history (descending) up to ``max_bets``."""
        bets: list[dict] = []
        before: str | None = None
        while len(bets) < max_bets:
            params: dict = {"contractId": market_id, "limit": 1000}
            if before is not None:
                params["before"] = before
            page = self._get("/v0/bets", params)
            if not isinstance(page, list):
                raise VenueUnavailableError(
                    self.venue, f"{self.api_base}/v0/bets",
                    f"unexpected payload type {type(page).__name__}",
                )
            if not page:
                break
            bets.extend(page)
            before = str(page[-1].get("id"))
            if len(page) < 1000:
                break
            time.sleep(POLITE_SLEEP_S)
        return bets[:max_bets]

    def fetch_resolved(
        self,
        max_markets: int = 250,
        max_bets_per_market: int = 4000,
        seed: int = 42,
        progress_cb: Callable[[str], None] | None = None,
        **kwargs,
    ) -> Iterator[MarketSeries]:
        """Yield resolved Manifold markets with bet-derived probability series.

        When discovery finds more than ``max_markets`` eligible markets, a
        seeded reservoir sample keeps the draw uniform over the discovered
        pool (so the category mix survives). Progress lines go to
        ``progress_cb`` every 25 markets.
        """
        seen: dict[str, dict] = {}
        for term in SEARCH_TERMS:
            for raw in self.search_resolved(term):
                parsed = parse_search_market(raw)
                if parsed is not None and parsed["id"] not in seen:
                    seen[parsed["id"]] = parsed
            time.sleep(POLITE_SLEEP_S)

        candidates = list(seen.values())
        if progress_cb:
            progress_cb(
                f"manifold: {len(candidates)} eligible markets discovered "
                f"across {len(SEARCH_TERMS)} search terms"
            )
        if len(candidates) > max_markets:
            rng = random.Random(seed)
            # Reservoir sample: uniform over the discovered pool.
            candidates = rng.sample(candidates, max_markets)

        fetched = 0
        for cand in candidates:
            bets = self.fetch_bets(cand["id"], max_bets=max_bets_per_market)
            series = series_from_bets(bets, cand["created_ts"])
            if len(series) < 2:
                continue  # prior point only: no real history
            fetched += 1
            if progress_cb and fetched % 25 == 0:
                progress_cb(f"manifold: fetched {fetched} markets...")
            yield MarketSeries(
                venue=self.venue,
                market_id=cand["id"],
                question=cand["question"],
                category=classify_category(cand["question"]),
                created_ts=cand["created_ts"],
                resolved_ts=cand["resolved_ts"],
                outcome=cand["outcome"],
                volume=cand["volume"],
                n_traders=cand["n_traders"],
                series=series,
                provenance=provenance_meta(
                    source="manifold v0 search-markets + bets",
                    api_base=self.api_base,
                    notes=(
                        "series = (createdTime, 0.5 prior) + filled bet "
                        "probAfter points; MKT/CANCEL resolutions excluded"
                    ),
                ),
            )
            time.sleep(POLITE_SLEEP_S)
