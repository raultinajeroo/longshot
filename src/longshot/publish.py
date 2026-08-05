"""Publish a calibration research note: report, summary, provenance.

``longshot publish --analysis analysis.json [--correction correction.json]
--out site/`` writes four files into the output directory:

- ``report.html`` — the self-contained HTML report (same renderer as
  ``longshot report``);
- ``report.json`` — the analysis document the report was built from;
- ``README-summary.md`` — a plain-markdown summary of the dataset, the key
  numbers, the correction verdicts, and the caveats;
- ``provenance.json`` — what produced this note: input file hashes,
  longshot version, generation time, analysis parameters, and per-venue
  status.

Caveats are not optional: every publish carries the same honesty block
(play-money Manifold, thin support, inferred Polymarket outcomes,
carry-forward approximation, multiple-comparison discipline) in both the
summary and the HTML report. There is no flag to omit it.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .report import write_report
from .venues.status import MANIFOLD_ONLY_NOTE, is_manifold_only, venue_status

#: The honesty block, mirrored in report.py's HTML caveats section.
CAVEATS = [
    "Manifold is a play-money (Mana) market; incentives differ from "
    "real-money venues, so Manifold numbers are a methodology "
    "demonstration, not claims about real-money calibration.",
    "Thin support is flagged, not hidden: at this sample size most "
    "horizons have few non-sparse equal-width bins, so ECE/slope rest on "
    "thin support (marked where applicable).",
    "Polymarket outcomes are inferred from terminal prices (>= 0.98 YES / "
    "<= 0.02 NO); ambiguous terminals are excluded.",
    "Carry-forward pricing with a staleness bound (price at most one "
    "horizon old) is an approximation of the price you could have "
    "observed at the horizon.",
    "All slices are reported with intervals and nothing is cherry-picked; "
    "apply multiple-comparison discipline when reading across slices.",
    "This is a measurement tool. Nothing here is investment advice, and "
    "historical calibration says nothing about any single future market.",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fmt(x, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{digits}f}"


def render_summary(analysis: dict, correction: dict | None) -> str:
    """Markdown summary with key numbers, verdicts, venue status, caveats."""
    ds = analysis["dataset"]
    venues = set(ds.get("venues", {}))

    lines: list[str] = []
    lines.append("# longshot summary")
    lines.append("")
    lines.append(
        f"Calibration research note generated {analysis['generated_at']} by "
        f"longshot {analysis['version']}. Input: {analysis['label']} "
        f"({ds['n_markets']} markets: {ds['n_yes']} YES, {ds['n_no']} NO)."
    )
    lines.append("")
    if is_manifold_only(venues):
        lines.append(f"**{MANIFOLD_ONLY_NOTE}**")
        lines.append("")

    lines.append("## Calibration by horizon")
    lines.append("")
    lines.append("| horizon | n | Brier | ECE | skill | compression slope | 95% CI |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for h in analysis["params"]["horizons"]:
        e = analysis["horizons"][h]
        if e.get("skipped"):
            lines.append(f"| {h} | — | — | — | — | — | skipped |")
            continue
        c = e["compression"]
        thin = " *" if e.get("thin_support") else ""
        lines.append(
            f"| {h} | {e['n']} | {_fmt(e['brier'])} | {_fmt(e['ece'])} "
            f"| {_fmt(e.get('skill_vs_climatology'), 3)} | {_fmt(c['slope'], 3)} "
            f"| [{_fmt(c['ci_lo'], 3)}, {_fmt(c['ci_hi'], 3)}]{thin} |"
        )
    lines.append("")
    lines.append("(* thin support: fewer than 3 non-sparse bins)")
    lines.append("")

    if correction:
        lines.append("## Out-of-sample correction verdicts")
        lines.append("")
        lines.append(
            "Corrections are fit on the earliest "
            f"{correction['train_frac']:.0%} of markets by resolution date "
            "and evaluated on the later test split. Verdicts come from the "
            "bootstrap CI of delta-Brier and are reported as computed; the "
            "pipeline is never re-tuned until improvement appears."
        )
        lines.append("")
        lines.append("| horizon | method | dBrier | 95% CI | verdict |")
        lines.append("|---|---|---:|---|---|")
        for h, e in correction.get("horizons", {}).items():
            if e.get("skipped"):
                lines.append(f"| {h} | — | — | — | skipped |")
                continue
            for meth in ("platt", "isotonic"):
                r = e.get(meth)
                if not r or r.get("skipped"):
                    continue
                lines.append(
                    f"| {h} | {meth} | {r['delta_brier']:+.4f} "
                    f"| [{r['delta_brier_ci'][0]:+.4f}, "
                    f"{r['delta_brier_ci'][1]:+.4f}] | {r['verdict']} |"
                )
        lines.append("")

    lines.append("## Venue status")
    lines.append("")
    lines.append("| venue | status |")
    lines.append("|---|---|")
    # Publishing runs offline: no collector is exercised live at publish
    # time, whatever the dataset's provenance.
    for row in venue_status():
        lines.append(f"| {row['venue']} | {row['status']} |")
    lines.append("")

    lines.append("## Caveats (read these before quoting any number)")
    lines.append("")
    for caveat in CAVEATS:
        lines.append(f"- {caveat}")
    lines.append("")
    return "\n".join(lines)


def build_provenance(
    analysis: dict,
    correction: dict | None,
    *,
    analysis_path: Path,
    correction_path: Path | None,
) -> dict:
    """Provenance record: inputs (with hashes), params, venue status."""
    ds = analysis["dataset"]
    venues = set(ds.get("venues", {}))
    inputs = {
        "analysis": {
            "path": str(analysis_path),
            "sha256": _sha256(analysis_path),
        }
    }
    if correction_path is not None:
        inputs["correction"] = {
            "path": str(correction_path),
            "sha256": _sha256(correction_path),
        }
    return {
        "tool": "longshot",
        "version": __version__,
        "published_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "analysis_generated_at": analysis["generated_at"],
        "inputs": inputs,
        "params": analysis["params"],
        "dataset": {
            "label": analysis["label"],
            "n_markets": ds["n_markets"],
            "venues": ds.get("venues", {}),
        },
        "venue_status": venue_status(),
        "manifold_only_methodology_demo": is_manifold_only(venues),
        "caveats": CAVEATS,
        "note": (
            "all numbers in this note derive from the hashed inputs above; "
            "no live venue data was collected at publish time"
        ),
    }


def publish(
    analysis_path: str | Path,
    correction_path: str | Path | None,
    out_dir: str | Path,
    *,
    title: str = "longshot calibration report",
) -> list[Path]:
    """Write report.html, report.json, README-summary.md, provenance.json."""
    analysis_path = Path(analysis_path)
    correction_path = Path(correction_path) if correction_path else None
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    correction = (
        json.loads(correction_path.read_text(encoding="utf-8"))
        if correction_path
        else None
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    html_path = write_report(analysis, correction, out / "report.html", title=title)
    summary_path = out / "README-summary.md"
    summary_path.write_text(render_summary(analysis, correction), encoding="utf-8")
    provenance_path = out / "provenance.json"
    provenance_path.write_text(
        json.dumps(
            build_provenance(
                analysis, correction,
                analysis_path=analysis_path, correction_path=correction_path,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return [html_path, html_path.with_suffix(".json"), summary_path, provenance_path]
