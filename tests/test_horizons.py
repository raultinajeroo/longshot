"""Horizon panel construction: carry-forward, staleness, lifetime bounds."""

from longshot.horizons import build_panel, parse_horizon
from longshot.types import MarketSeries, PricePoint

DAY = 86400


def _market(series, created=0, resolved=100 * DAY, outcome=1):
    return MarketSeries(
        venue="test", market_id="m1", question="q", category="cat",
        created_ts=created, resolved_ts=resolved, outcome=outcome,
        volume=1000.0, n_traders=50,
        series=tuple(PricePoint(ts, p) for ts, p in series),
        provenance={},
    )


def test_carry_forward_picks_last_point_before_target():
    m = _market([(0, 0.5), (85 * DAY, 0.7), (95 * DAY, 0.9)])
    panel = build_panel([m], 10 * DAY, "10d")  # target = 90d
    assert len(panel) == 1
    assert panel.points[0].p == 0.7  # last point at or before T - h
    assert panel.points[0].outcome == 1


def test_staleness_bound_rejects_ancient_prices():
    # Last price before target is 40 days old: older than one horizon (10d).
    m = _market([(0, 0.5), (50 * DAY, 0.7)])
    panel = build_panel([m], 10 * DAY, "10d")  # target = 90d, need ts >= 80d
    assert len(panel) == 0
    # But a 60d horizon accepts it (need ts >= T - 120d).
    panel60 = build_panel([m], 60 * DAY, "60d")
    assert len(panel60) == 1


def test_horizon_longer_than_life_skipped():
    m = _market([(0, 0.5), (50 * DAY, 0.7)], resolved=60 * DAY)
    assert len(build_panel([m], 90 * DAY, "90d")) == 0
    assert len(build_panel([m], 30 * DAY, "30d")) == 1


def test_no_point_before_target_skipped():
    # Series starts after the target time (no creation prior).
    m = _market([(95 * DAY, 0.9)])
    assert len(build_panel([m], 10 * DAY, "10d")) == 0


def test_panel_fields_and_parse():
    m = _market([(0, 0.5), (99 * DAY, 0.8)], outcome=0)
    panel = build_panel([m], 1 * DAY, "1d")
    pt = panel.points[0]
    assert pt.market_id == "m1"
    assert pt.category == "cat"
    assert pt.outcome == 0
    assert pt.p == 0.8
    assert parse_horizon("12h") == 12 * 3600
    assert parse_horizon("7d") == 7 * DAY
