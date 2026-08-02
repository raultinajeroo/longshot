"""JSONL store: round-trip, validation, downsampling."""

import json

import pytest

from longshot.store import (
    StoreError,
    downsample_series,
    load_jsonl,
    market_from_dict,
    market_to_dict,
    write_jsonl,
)
from longshot.types import MarketSeries, PricePoint


def _market(**over):
    base = dict(
        venue="test", market_id="m1", question="q", category="cat",
        created_ts=1000, resolved_ts=2000, outcome=1, volume=10.0,
        n_traders=5,
        series=(PricePoint(1000, 0.5), PricePoint(1500, 0.6),
                PricePoint(1999, 0.9)),
        provenance={"source": "test"},
    )
    base.update(over)
    return MarketSeries(**base)


def test_roundtrip_jsonl(tmp_path):
    path = tmp_path / "x.jsonl"
    markets = [_market(), _market(market_id="m2", outcome=0)]
    assert write_jsonl(path, markets) == 2
    back = load_jsonl(path)
    assert len(back) == 2
    assert market_to_dict(back[0]) == market_to_dict(markets[0])


def test_rejects_out_of_range_price():
    d = market_to_dict(_market())
    d["series"] = [[1000, 1.5]]
    with pytest.raises(StoreError):
        market_from_dict(d)


def test_rejects_bad_outcome_and_time_order():
    d = market_to_dict(_market())
    d["outcome"] = 2
    with pytest.raises(StoreError):
        market_from_dict(d)
    d = market_to_dict(_market())
    d["resolved_ts"] = d["created_ts"]
    with pytest.raises(StoreError):
        market_from_dict(d)


def test_unsorted_series_is_sorted_on_load():
    d = market_to_dict(_market())
    d["series"] = [[1999, 0.9], [1000, 0.5], [1500, 0.6]]
    m = market_from_dict(d)
    assert [pt.ts for pt in m.series] == [1000, 1500, 1999]


def test_empty_series_dropped(tmp_path):
    d = market_to_dict(_market())
    d["series"] = []
    assert market_from_dict(d) is None
    # A file containing only empty-series records is an error to load.
    path = tmp_path / "empty.jsonl"
    path.write_text(json.dumps(d) + "\n")
    with pytest.raises(StoreError):
        load_jsonl(path)


def test_invalid_json_line_raises(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{not json}\n")
    with pytest.raises(StoreError):
        load_jsonl(path)


def test_downsample_keeps_endpoints_and_final_window():
    series = tuple(PricePoint(1000 + i * 100, 0.5) for i in range(1000))
    out = downsample_series(series, max_points=200, keep_last_seconds=7200,
                            resolved_ts=series[-1].ts)
    assert len(out) <= 200
    assert out[0] == series[0] and out[-1] == series[-1]
    near = [pt for pt in out if pt.ts >= series[-1].ts - 7200]
    assert len(near) == len([pt for pt in series
                             if pt.ts >= series[-1].ts - 7200])
    assert [pt.ts for pt in out] == sorted(pt.ts for pt in out)
