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


def test_correct_single_market_is_a_clean_input_error(tmp_path, capsys):
    single = tmp_path / "single.jsonl"
    single.write_text(FIXTURE.read_text().splitlines()[0] + "\n")
    out = tmp_path / "correction.json"
    out.write_text("keep existing output")
    assert main(["correct", "--input", str(single), "--out", str(out)]) == 4
    assert "at least 2 markets" in capsys.readouterr().err
    assert out.read_text() == "keep existing output"


def test_correct_discloses_resolution_bootstrap_unit(tmp_path, capsys):
    out = tmp_path / "correction.json"
    assert main(["correct", "--input", str(FIXTURE), "--out", str(out),
                 "--horizons", "14d", "--bootstrap", "20",
                 "--bootstrap-unit", "resolution-day"]) == 0
    result = json.loads(out.read_text())
    assert result["bootstrap_unit"] == "resolution-day"
    assert "resolution-day" in capsys.readouterr().out


def test_cost_screen_discloses_missing_quotes_in_cli_and_reports(tmp_path, capsys):
    from longshot.analyze import run_analysis
    from longshot.publish import render_summary
    from longshot.report import render_html
    from longshot.store import load_jsonl

    out = tmp_path / "costed.json"
    assert main(["correct", "--input", str(FIXTURE), "--out", str(out),
                 "--horizons", "14d", "--bootstrap", "20",
                 "--cost-per-contract", "0.01"]) == 0
    result = json.loads(out.read_text())
    assert result["cost_per_contract"] == 0.01
    assert result["horizons"]["14d"]["n_quoted_test"] == 0
    analysis = run_analysis(load_jsonl(FIXTURE), horizons=["14d"], n_boot=20)
    for text in (capsys.readouterr().out, render_html(analysis, result),
                 render_summary(analysis, result)):
        assert "0.0100" in text
        assert "quoted test" in text
        assert "retrospective" in text


def test_invalid_cost_keeps_existing_output(tmp_path, capsys):
    out = tmp_path / "existing.json"
    out.write_text("keep this")
    assert main(["correct", "--input", str(FIXTURE), "--out", str(out),
                 "--cost-per-contract", "nan"]) == 4
    assert "cost per contract" in capsys.readouterr().err
    assert out.read_text() == "keep this"


def test_quoted_payoff_survives_cli_and_publish(tmp_path, capsys):
    from dataclasses import replace

    from longshot.simulate import simulate_markets
    from longshot.store import write_jsonl

    # Synthetic quotes exercise the export contract; they are not venue data.
    markets = [replace(m, series=tuple(
        replace(p, bid=max(0, p.price - 0.02), ask=min(1, p.price + 0.02))
        for p in m.series)) for m in simulate_markets("calibrated", 400, 202)]
    source = tmp_path / "quoted.jsonl"
    write_jsonl(source, markets)
    analysis, correction = tmp_path / "analysis.json", tmp_path / "correction.json"
    args = ["--input", str(source), "--horizons", "30d", "--bootstrap", "20"]
    assert main(["analyze", *args, "--out", str(analysis)]) == 0
    assert main(["correct", *args, "--out", str(correction),
                 "--cost-per-contract", "0.01", "--bootstrap-unit", "resolution-week"]) == 0
    terminal = capsys.readouterr().out
    result = json.loads(correction.read_text())["horizons"]["30d"]["isotonic"]["quoted_payoff"]
    assert result["n_quoted"] >= 20 and result["n_selected"] > 0
    site = tmp_path / "site"
    assert main(["publish", "--analysis", str(analysis), "--correction", str(correction),
                 "--out", str(site)]) == 0
    for text in (terminal, (site / "report.html").read_text(),
                 (site / "README-summary.md").read_text()):
        assert f"selected {result['n_selected']}" in text
        assert f"{result['mean_payoff']:+.4f}" in text
        assert "resolution-week" in text and "retrospective" in text
    provenance = json.loads((site / "provenance.json").read_text())
    assert provenance["correction_protocol"]["cost_per_contract"] == 0.01


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
