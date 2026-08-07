"""Venue parsers against recorded/canonical responses; offline error paths.

The Manifold JSON below was recorded from api.manifold.markets on
2026-07-31 (fields trimmed to those the parser reads). Polymarket and
Kalshi payloads are hand-written canonical shapes per their docs (those
hosts are unreachable from the build sandbox).
"""

import json
from pathlib import Path

import pytest

from longshot.venues.base import VenueUnavailableError
from longshot.venues.fixture import FixtureClient
from longshot.venues.kalshi import parse_candlesticks, parse_settled_market
from longshot.venues.manifold import (
    ManifoldClient,
    classify_category,
    parse_search_market,
    series_from_bets,
)
from longshot.venues.polymarket import (
    PolymarketClient,
    parse_gamma_market,
    parse_prices_history,
)

# Recorded from https://api.manifold.markets/v0/search-markets (2026-07-31).
MANIFOLD_SEARCH_JSON = """
[
 {"id": "6LtPIZQcRQ", "question": "Does the Fed rate change in July?",
  "resolution": "NO", "resolutionTime": 1785355982359,
  "createdTime": 1782483567726, "volume": 1968.757393440333,
  "probability": 0.0051444141566973035, "uniqueBettorCount": 27},
 {"id": "5SSCuOqnqn",
  "question": "Will a Kevin be nominated as the next fed chair by donald trump?",
  "resolution": "YES", "resolutionTime": 1770098321764,
  "createdTime": 1767819058471, "volume": 2132.2392547633426,
  "probability": 0.9891454741924915, "uniqueBettorCount": 20}
]
"""

# Recorded from https://api.manifold.markets/v0/bets (2026-07-31).
MANIFOLD_BETS_JSON = """
[
 {"id": "OEIE59QEIsqC", "createdTime": 1766170852421,
  "probBefore": 0.06444015378809555, "probAfter": 0.05853982229462849,
  "isFilled": true, "isRedemption": false, "isCancelled": false,
  "amount": 200},
 {"id": "Nctyt5CzS9AO", "createdTime": 1765942174094,
  "probBefore": 0.06447180985353859, "probAfter": 0.06444015378809555,
  "isFilled": true, "isRedemption": false, "isCancelled": false,
  "amount": 1},
 {"id": "O0Z8hNEsZE68", "createdTime": 1765849518224,
  "probBefore": 0.06450348871479372, "probAfter": 0.06447180985353859,
  "isFilled": true, "isRedemption": false, "isCancelled": false,
  "amount": 1}
]
"""


def test_manifold_search_parser_recorded():
    raw = json.loads(MANIFOLD_SEARCH_JSON)
    m0 = parse_search_market(raw[0])
    m1 = parse_search_market(raw[1])
    assert m0["outcome"] == 0 and m0["resolved_ts"] == 1785355982
    assert m0["n_traders"] == 27
    assert m1["outcome"] == 1
    # Filter rejections: MKT/CANCEL resolutions, low volume, short life.
    bad = dict(raw[0], resolution="MKT")
    assert parse_search_market(bad) is None
    bad = dict(raw[0], volume=5.0)
    assert parse_search_market(bad) is None
    bad = dict(raw[0], uniqueBettorCount=3)
    assert parse_search_market(bad) is None
    bad = dict(raw[0], resolutionTime=raw[0]["createdTime"] + 3600_000)
    assert parse_search_market(bad) is None


def test_manifold_bets_series_recorded():
    bets = json.loads(MANIFOLD_BETS_JSON)
    series = series_from_bets(bets, created_ts=1765849518)
    # Prior point at creation, then bets sorted ascending by time.
    assert series[0].ts == 1765849518 and series[0].price == 0.5
    assert [pt.ts for pt in series[1:]] == sorted(pt.ts for pt in series[1:])
    assert series[-1].price == pytest.approx(0.05853982229462849)
    assert series[1].ts == 1765849518  # ms -> s conversion
    # Redemptions and cancelled bets are excluded.
    dirty = bets + [dict(bets[0], id="x1", isRedemption=True),
                    dict(bets[0], id="x2", isCancelled=True),
                    dict(bets[0], id="x3", isFilled=False)]
    assert len(series_from_bets(dirty, created_ts=1765849518)) == len(series)


def test_manifold_category_classifier():
    assert classify_category("Will Trump win the election?") == "politics"
    assert classify_category("Will the Fed cut rates?") == "macro"
    assert classify_category("Bitcoin above 100k?") == "crypto"
    assert classify_category("Who wins the NBA finals?") == "sports"
    assert classify_category("Will OpenAI release GPT-6?") == "tech_ai"
    assert classify_category("Best picture at the Oscars?") == "entertainment"
    assert classify_category("Will it rain in Lisbon in March?") == "other"


def test_polymarket_parsers_canonical():
    market = {
        "id": "51234",
        "question": "Will it snow in NYC in January?",
        "clobTokenIds": "[\"111\", \"222\"]",
        "outcomes": "[\"Yes\", \"No\"]",
        "outcomePrices": "[\"0.997\", \"0.003\"]",
        "volumeNum": 45678.9,
        "createdAt": "2025-11-01T00:00:00Z",
        "closedTime": "2026-02-01T00:00:00Z",
    }
    parsed = parse_gamma_market(market)
    assert parsed["outcome"] == 1 and parsed["yes_token"] == "111"
    assert parsed["volume"] == pytest.approx(45678.9)
    # Ambiguous terminal prices are excluded.
    ambiguous = dict(market, outcomePrices="[\"0.5\", \"0.5\"]")
    assert parse_gamma_market(ambiguous) is None
    no = dict(market, outcomePrices="[\"0.01\", \"0.99\"]")
    assert parse_gamma_market(no)["outcome"] == 0
    history = parse_prices_history(
        {"history": [{"t": 1767225600, "p": 0.42}, {"t": 1767139200, "p": 0.4}]}
    )
    assert [pt.ts for pt in history] == [1767139200, 1767225600]
    assert parse_prices_history({"unexpected": 1}) == ()


def test_kalshi_parsers_canonical():
    market = {
        "ticker": "KXFED-26JAN-T4.50",
        "title": "Will the Fed cut rates in January 2026?",
        "result": "yes",
        "volume": 12345,
        "created_time": "2025-12-01T00:00:00Z",
        "close_time": "2026-01-28T18:00:00Z",
        "series_ticker": "KXFED",
    }
    parsed = parse_settled_market(market)
    assert parsed["outcome"] == 1 and parsed["ticker"].startswith("KXFED")
    assert parse_settled_market(dict(market, result="")) is None
    candles = parse_candlesticks({"candlesticks": [
        {"end_period_ts": 1767225600, "price": {"close": 42}},
        {"end_period_ts": 1767139200, "price": {"close": 40}},
    ]})
    assert [pt.price for pt in candles] == [0.40, 0.42]
    assert parse_candlesticks({"candlesticks": "nope"}) == ()


def test_kalshi_fixed_point_payloads():
    """Kalshi's fixed-point migration removed the integer-cent fields rather
    than deprecating them. Payloads recorded live 2026-08-07 from
    KXHIGHNY-26AUG06-T95: no `series_ticker`, no `volume`, no `price.close`.
    """
    market = {
        "ticker": "KXHIGHNY-26AUG06-T95",
        "event_ticker": "KXHIGHNY-26AUG06",
        "title": "Will the **high temp in NYC** be >95 on Aug 6, 2026?",
        "result": "no",
        "volume_fp": "1315.02",
        "created_time": "2026-08-05T09:30:52.385847Z",
        "close_time": "2026-08-07T04:59:00Z",
    }
    parsed = parse_settled_market(market)
    assert parsed is not None, "current payloads must not be silently dropped"
    # series_ticker is absent and must come from the event ticker prefix,
    # because the candlesticks path is keyed on it.
    assert parsed["series_ticker"] == "KXHIGHNY"
    assert parsed["outcome"] == 0
    assert parsed["volume"] == 1315.02

    # close_dollars is already dollars: no /100 scaling.
    candles = parse_candlesticks({"candlesticks": [
        {"end_period_ts": 1767225600, "price": {"close_dollars": "0.0100"}},
        {"end_period_ts": 1767139200, "price": {"close_dollars": "0.7300"}},
    ]})
    assert [pt.price for pt in candles] == [0.73, 0.01]


def test_kalshi_market_without_series_or_event_is_skipped():
    assert parse_settled_market({
        "ticker": "T", "result": "yes",
        "created_time": "2026-01-01T00:00:00Z",
        "close_time": "2026-02-01T00:00:00Z",
    }) is None


def test_fixture_venue_loads_bundled():
    bundled = Path("data/bundled/manifold_resolved_sample.jsonl")
    if not bundled.exists():
        pytest.skip("bundled data not fetched")
    client = FixtureClient([str(bundled)])
    markets = list(client.fetch_resolved(max_markets=1000))
    assert len(markets) >= 150
    assert all(m.venue == "manifold" for m in markets)
    capped = list(client.fetch_resolved(max_markets=10))
    assert len(capped) == 10


def test_unavailable_error_has_remedy_hint():
    client = ManifoldClient(api_base="http://127.0.0.1:9")
    with pytest.raises(VenueUnavailableError) as exc:
        client.search_resolved("fed")
    msg = str(exc.value)
    assert "manifold" in msg
    assert "blocked" in msg and "fixtures" in msg  # remedy hint

    poly = PolymarketClient(gamma_base="http://127.0.0.1:9",
                            clob_base="http://127.0.0.1:9")
    with pytest.raises(VenueUnavailableError) as exc2:
        list(poly.fetch_resolved(max_markets=1))
    assert "polymarket" in str(exc2.value)
