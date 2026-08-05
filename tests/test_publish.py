"""Tests for analysis.yaml loading, publish, and venue status."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from longshot.prereg import AnalysisConfigError, load_analysis_config
from longshot.publish import publish
from longshot.venues.status import is_manifold_only, venue_status

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO = REPO_ROOT / "examples" / "demo"


def run_cli(*argv: str):
    return subprocess.run(
        [sys.executable, "-m", "longshot", *argv],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=120,
    )


# --------------------------------------------------------------- analysis.yaml


def test_committed_analysis_yaml_loads():
    cfg = load_analysis_config(REPO_ROOT / "analysis.yaml")
    assert cfg["horizons"] == ["30d", "14d", "7d", "3d", "1d", "12h", "1h"]
    assert cfg["min_per_bin"] == 30
    assert cfg["correction"]["methods"] == ["platt", "isotonic"]
    assert "ci_includes_zero" in cfg["correction"]["verdict_rules"]


def test_analysis_config_validation(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("horizons: 30d\nbogus: 1\n")
    with pytest.raises(AnalysisConfigError, match="unknown key"):
        load_analysis_config(bad)
    with pytest.raises(AnalysisConfigError, match="not found"):
        load_analysis_config(tmp_path / "nope.yaml")


def test_analyze_with_config_matches_committed_demo(tmp_path):
    """--config analysis.yaml reproduces the committed bundled numbers."""
    out = tmp_path / "analysis.json"
    proc = run_cli(
        "analyze", "--input", "data/bundled/manifold_resolved_sample.jsonl",
        "--config", "analysis.yaml", "--out", str(out),
    )
    assert proc.returncode == 0, proc.stderr
    fresh = json.loads(out.read_text())
    committed = json.loads((DEMO / "analysis_bundled.json").read_text())
    for h in committed["params"]["horizons"]:
        old, new = committed["horizons"][h], fresh["horizons"][h]
        assert old["n"] == new["n"]
        assert old["brier"] == pytest.approx(new["brier"], abs=1e-9)
        slope = new["compression"]["slope"]
        if slope == slope:  # nan at sparse horizons: nothing to compare
            assert old["compression"]["slope"] == pytest.approx(slope, abs=1e-9)


# -------------------------------------------------------------------- publish


def test_publish_writes_all_artifacts(tmp_path):
    written = publish(
        DEMO / "analysis_bundled.json",
        DEMO / "correction_bundled.json",
        tmp_path / "site",
        title="test note",
    )
    names = {p.name for p in written}
    assert names == {
        "report.html", "report.json", "README-summary.md", "provenance.json",
    }

    summary = (tmp_path / "site" / "README-summary.md").read_text()
    # Caveats are unavoidable.
    assert "play-money" in summary
    assert "Thin support" in summary
    assert "inferred from terminal prices" in summary
    assert "multiple-comparison discipline" in summary
    assert "is investment advice" in summary
    # Manifold-only note and venue status are present.
    assert "Manifold-only methodology demonstration" in summary
    assert "parser-tested" in summary
    # Verdicts reported as computed.
    assert "no reliable improvement" in summary
    assert "reliable degradation" in summary
    # Key bundled numbers appear.
    assert "278" in summary and "1.059" in summary

    prov = json.loads((tmp_path / "site" / "provenance.json").read_text())
    assert prov["manifold_only_methodology_demo"] is True
    assert prov["inputs"]["analysis"]["sha256"]
    assert prov["inputs"]["correction"]["sha256"]
    statuses = {r["venue"]: r["status"] for r in prov["venue_status"]}
    assert "bundled sample" in statuses["manifold"]
    assert "NOT exercised live" in statuses["polymarket"]
    assert "NOT exercised live" in statuses["kalshi"]

    html = (tmp_path / "site" / "report.html").read_text()
    assert "caveats" in html


def test_publish_cli_end_to_end(tmp_path):
    proc = run_cli(
        "publish",
        "--analysis", str(DEMO / "analysis_bundled.json"),
        "--correction", str(DEMO / "correction_bundled.json"),
        "--out", str(tmp_path / "site"),
    )
    assert proc.returncode == 0, proc.stderr
    assert "wrote" in proc.stdout
    assert (tmp_path / "site" / "provenance.json").is_file()


def test_publish_cli_missing_input_exit_4(tmp_path):
    proc = run_cli(
        "publish", "--analysis", str(tmp_path / "nope.json"),
        "--out", str(tmp_path / "site"),
    )
    assert proc.returncode == 4
    assert "not found" in proc.stderr


# -------------------------------------------------------------- venue status


def test_venue_status_marks_unexercised_collectors():
    rows = {r["venue"]: r["status"] for r in venue_status()}
    assert "bundled sample" in rows["manifold"]
    assert "NOT exercised live" in rows["polymarket"]
    assert "NOT exercised live" in rows["kalshi"]


def test_venue_status_live_when_exercised():
    rows = {r["venue"]: r["status"] for r in venue_status({"polymarket"})}
    assert rows["polymarket"] == "exercised live in this run"
    assert is_manifold_only({"manifold"})
    assert not is_manifold_only({"manifold", "kalshi"})


def test_venues_cli():
    proc = run_cli("venues")
    assert proc.returncode == 0
    assert "manifold" in proc.stdout
    assert "NOT exercised live" in proc.stdout
