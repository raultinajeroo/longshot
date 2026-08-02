"""HTML report contents: charts, caveats, citations, self-containment."""

import json
from pathlib import Path

import pytest

from longshot.analyze import run_analysis
from longshot.correct import run_correction
from longshot.horizons import parse_horizon
from longshot.report import render_html, write_report
from longshot.store import load_jsonl

FIXTURE = Path("fixtures/biased_markets.jsonl")


@pytest.fixture(scope="module")
def analysis():
    markets = load_jsonl(FIXTURE)
    return run_analysis(markets, n_boot=100, seed=1, label="test fixture")


def test_html_contains_horizons_citations_caveats(analysis):
    html = render_html(analysis)
    for horizon in ("30d", "7d", "1d"):
        assert horizon in html
    assert "Prophet Arena" in html
    assert "play-money" in html
    assert "not investment advice" in html
    assert "<svg" in html


def test_html_has_no_external_assets(analysis):
    html = render_html(analysis)
    assert "<script" not in html
    assert "src=\"http" not in html
    assert "href=\"http" not in html


def test_correction_section_and_verdict(tmp_path, analysis):
    markets = load_jsonl(FIXTURE)
    correction = run_correction(
        markets, horizons=["30d"],
        horizon_seconds={"30d": parse_horizon("30d")},
        method="both", n_boot=100, seed=1,
    )
    html = render_html(analysis, correction)
    assert "out-of-sample correction" in html
    assert "verdict" in html
    out = write_report(analysis, correction, tmp_path / "report.html")
    assert out.exists()
    sidecar = tmp_path / "report.json"
    assert sidecar.exists()
    assert json.loads(sidecar.read_text())["tool"] == "longshot"
