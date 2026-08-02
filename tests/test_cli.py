"""CLI behavior: exit codes and demo artifacts."""

import json
from pathlib import Path

import pytest

from longshot.cli import main

BUNDLED = Path("data/bundled/manifold_resolved_sample.jsonl")
FIXTURE = Path("fixtures/biased_markets.jsonl")


def test_analyze_fixture_exits_0(tmp_path, capsys):
    out = tmp_path / "a.json"
    code = main(["analyze", "--input", str(FIXTURE), "--min-per-bin", "10",
                 "--bootstrap", "100", "--out", str(out)])
    assert code == 0
    a = json.loads(out.read_text())
    assert a["dataset"]["n_markets"] == 120
    assert "Brier" in capsys.readouterr().out


def test_bad_input_exits_4(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json}\n")
    with pytest.raises(SystemExit) as exc:
        main(["analyze", "--input", str(bad)])
    assert exc.value.code == 4


def test_missing_input_exits_4():
    with pytest.raises(SystemExit) as exc:
        main(["analyze", "--input", "no/such/file.jsonl"])
    assert exc.value.code == 4


def test_simulate_cli(tmp_path):
    out = tmp_path / "sim.jsonl"
    assert main(["simulate", "--mode", "calibrated", "--markets", "10",
                 "--seed", "7", "--out", str(out)]) == 0
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 10
    assert json.loads(lines[0])["venue"] == "simulated"


@pytest.mark.skipif(not BUNDLED.exists(), reason="bundled data not fetched")
def test_demo_produces_artifacts(tmp_path):
    code = main(["demo", "--outdir", str(tmp_path), "--bootstrap", "100"])
    assert code == 0
    analysis = tmp_path / "analysis_bundled.json"
    report = tmp_path / "report_bundled.html"
    biased = tmp_path / "analysis_biased_fixture.json"
    assert analysis.exists() and report.exists() and biased.exists()
    html = report.read_text()
    assert "<svg" in html
    assert "not investment advice" in html
    a = json.loads(analysis.read_text())
    assert a["dataset"]["n_markets"] >= 150
