"""End-to-end analysis on the bundled real-data sample."""

import json
import math
from pathlib import Path

import pytest

from longshot.analyze import run_analysis
from longshot.horizons import DEFAULT_HORIZONS, build_panel, parse_horizon
from longshot.store import load_jsonl

BUNDLED = Path("data/bundled/manifold_resolved_sample.jsonl")

pytestmark = pytest.mark.skipif(not BUNDLED.exists(),
                                reason="bundled data not fetched")


@pytest.fixture(scope="module")
def analysis():
    markets = load_jsonl(BUNDLED)
    return run_analysis(markets, n_boot=200, seed=1), markets


def test_all_horizons_present(analysis):
    a, _ = analysis
    assert list(a["horizons"].keys()) == DEFAULT_HORIZONS
    populated = [h for h in DEFAULT_HORIZONS if not a["horizons"][h].get("skipped")]
    assert len(populated) >= 3  # real data must populate several horizons


def test_bin_counts_sum_to_panel_size(analysis):
    a, markets = analysis
    for h, entry in a["horizons"].items():
        if entry.get("skipped"):
            continue
        panel = build_panel(markets, parse_horizon(h), h)
        assert len(panel) == entry["n"]
        assert sum(b["n"] for b in entry["bins"]) == entry["n"]


def test_categories_and_json_serializable(analysis):
    a, _ = analysis
    assert a["categories"]
    assert all(r["n"] > 0 for r in a["categories"])
    assert sum(r["n"] for r in a["categories"]) == \
        a["horizons"][a["reference_horizon"]]["n"]
    # NaN is emitted as NaN by json.dumps; ensure the round trip works.
    text = json.dumps(a)
    assert "horizons" in text


def test_metrics_sane_ranges(analysis):
    a, _ = analysis
    for h, entry in a["horizons"].items():
        if entry.get("skipped"):
            continue
        assert 0.0 <= entry["brier"] <= 1.0
        # ECE is NaN by design when every bin is sparse (small panels).
        assert math.isnan(entry["ece"]) or 0.0 <= entry["ece"] <= 1.0
        assert 0.0 <= entry["base_rate"] <= 1.0
        mur = entry["murphy"]
        if mur is None:  # all bins sparse at this horizon
            continue
        assert mur["reliability"] - mur["resolution"] + mur["uncertainty"] == \
            pytest.approx(mur["brier_binned"], abs=1e-12)
