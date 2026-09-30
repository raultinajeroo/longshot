"""The pmwatch contract uses observation times and explicitly labeled mids."""

import json

from longshot.analyze import format_digest, run_analysis
from longshot.horizons import build_panels
from longshot.publish import publish
from longshot.store import load_jsonl


def test_pmwatch_short_horizons_and_published_estimator(tmp_path):
    # Synthetic record in pmwatch export format, not a live observation.
    record = {
        "venue": "kalshi", "market_id": "synthetic-example",
        "question": "Synthetic contract?", "category": "uncategorized",
        "created_ts": 1767225600, "resolved_ts": 1767232800, "outcome": 1,
        "volume": None, "n_traders": None,
        "series": [[1767225600, 0.4], [1767229200, 0.6], [1767232500, 0.7]],
        "provenance": {
            "source": "pmwatch", "snapshot_source": "live",
            "price_estimator": "order_book_mid", "timestamp_source": "fetched_at",
            "created_ts_source": "first_observed", "label_source": "synthetic.test_label",
        },
    }
    path = tmp_path / "observations.jsonl"
    path.write_text(json.dumps(record) + "\n")
    markets = load_jsonl(path)
    panels = build_panels(markets, ["1d", "1h", "5m"])
    assert [len(panel) for panel in panels] == [0, 1, 1]
    assert panels[1].points[0].p == 0.6
    assert panels[2].points[0].p == 0.7
    assert markets[0].provenance == record["provenance"]

    analysis = run_analysis(markets, horizons=["1h", "5m"], n_boot=10)
    assert analysis["dataset"]["price_estimators"] == {"order_book_mid": 1}
    assert "order_book_mid" in format_digest(analysis)
    analysis_path = tmp_path / "analysis.json"
    analysis_path.write_text(json.dumps(analysis))
    publish(analysis_path, None, tmp_path / "site")
    for name in ("report.html", "README-summary.md", "provenance.json"):
        assert "order_book_mid" in (tmp_path / "site" / name).read_text()


def test_legacy_inputs_are_not_silently_labeled_as_midpoints(tmp_path):
    from longshot.simulate import simulate_markets

    markets = simulate_markets("calibrated", n_markets=30, seed=7)
    markets[0].provenance["price_estimator"] = None
    analysis = run_analysis(markets, horizons=["1d"], n_boot=10)
    assert analysis["dataset"]["price_estimators"] == {"unspecified": 30}
    assert "order_book_mid" not in format_digest(analysis)


def test_mixed_estimators_are_disclosed():
    from longshot.analyze import price_estimator_note

    note = price_estimator_note({"dataset": {"price_estimators": {
        "order_book_mid": 2, "unspecified": 3,
    }}})
    assert "order_book_mid (2)" in note
    assert "unspecified (3)" in note
    assert "pooled" in note
    assert "separately" in note
