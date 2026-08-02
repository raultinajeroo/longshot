"""Analysis orchestration: panels -> metrics -> bias -> AnalysisResult.

Produces one JSON-serializable document per input dataset. Every number in
the HTML report and the README tables comes from this structure.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Iterable

import numpy as np

from . import __version__
from .bias import (
    compression_slope,
    compression_slope_from_bins,
    logistic_calibration_slope,
    yes_price_inflation,
)
from .binning import equal_width_bins
from .horizons import DEFAULT_HORIZONS, build_panel, parse_horizon
from .metrics import (
    brier,
    climatology_brier,
    ece,
    log_loss,
    mce,
    murphy_decomposition,
)
from .types import HorizonPanel, MarketSeries

CATEGORY_MIN_PER_BIN = 5  # smaller than panel-level: category slices are thin


def _panel_arrays(panel: HorizonPanel) -> tuple[np.ndarray, np.ndarray]:
    p = np.array([q.p for q in panel.points])
    y = np.array([q.outcome for q in panel.points], dtype=float)
    return p, y


def _horizon_entry(
    panel: HorizonPanel,
    *,
    n_bins: int,
    min_per_bin: int,
    n_boot: int,
    seed: int,
) -> dict:
    p, y = _panel_arrays(panel)
    bins = equal_width_bins(p, y, n_bins=n_bins, min_per_bin=min_per_bin)
    entry: dict = {
        "horizon": panel.name,
        "seconds": panel.seconds,
        "n": len(panel),
        "base_rate": float(np.mean(y)) if len(panel) else float("nan"),
    }
    if len(panel) == 0:
        entry["skipped"] = "empty panel at this horizon"
        return entry
    entry["brier"] = brier(p, y)
    entry["log_loss"] = log_loss(p, y)
    n_nonsparse = sum(1 for b in bins if not b.sparse and b.n > 0)
    entry["n_nonsparse_bins"] = n_nonsparse
    # With a U-shaped price distribution (mass at the extremes), equal-width
    # binning can leave fewer than 3 supported bins; ECE/slope are still
    # computed per spec but flagged as thin so readers do not over-trust
    # them. See DECISIONS.md.
    entry["thin_support"] = n_nonsparse < 3
    try:
        entry["ece"] = ece(bins)
        entry["mce"] = mce(bins)
        entry["murphy"] = murphy_decomposition(bins)
    except ValueError:
        entry["ece"] = entry["mce"] = float("nan")
        entry["murphy"] = None
    clim = climatology_brier(y)
    entry["climatology_brier"] = clim
    entry["skill_vs_climatology"] = (
        1.0 - entry["brier"] / clim if clim > 0 else float("nan")
    )
    entry["bins"] = [b.to_dict() for b in bins]
    entry["compression"] = compression_slope(
        panel, n_bins=n_bins, min_per_bin=min_per_bin, n_boot=n_boot, seed=seed
    )
    entry["logistic"] = logistic_calibration_slope(panel, n_boot=n_boot, seed=seed)
    return entry


def _category_table(
    panel: HorizonPanel,
    *,
    n_bins: int = 10,
) -> list[dict]:
    """Per-category stats at the reference horizon (n, Brier, ECE, slope).

    Uses a smaller min_per_bin than the panel-level analysis; sparse ECEs
    are reported as NaN rather than hidden.
    """
    cats: dict[str, list] = {}
    for q in panel.points:
        cats.setdefault(q.category, []).append(q)
    rows = []
    for cat, pts in sorted(cats.items(), key=lambda kv: -len(kv[1])):
        p = np.array([q.p for q in pts])
        y = np.array([q.outcome for q in pts], dtype=float)
        bins = equal_width_bins(p, y, n_bins=n_bins,
                                min_per_bin=CATEGORY_MIN_PER_BIN)
        try:
            e = ece(bins)
        except ValueError:
            e = float("nan")
        slope = compression_slope_from_bins(bins)
        rows.append({
            "category": cat,
            "n": len(pts),
            "brier": brier(p, y),
            "ece": e,
            "slope": slope,
            "base_rate": float(np.mean(y)),
        })
    return rows


def run_analysis(
    markets: Iterable[MarketSeries],
    *,
    horizons: list[str] | None = None,
    n_bins: int = 10,
    min_per_bin: int = 30,
    n_boot: int = 1000,
    seed: int = 42,
    label: str | None = None,
) -> dict:
    """Run the full calibration/bias analysis. Returns a JSON-safe dict."""
    markets = list(markets)
    if not markets:
        raise ValueError("analyze: no markets in input")
    horizons = horizons or list(DEFAULT_HORIZONS)
    panels = [build_panel(markets, parse_horizon(h), h) for h in horizons]

    ref_name = "7d" if "7d" in horizons else horizons[len(horizons) // 2]
    ref_panel = next(p for p in panels if p.name == ref_name)

    venues: dict[str, int] = {}
    for m in markets:
        venues[m.venue] = venues.get(m.venue, 0) + 1

    return {
        "tool": "longshot",
        "version": __version__,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "label": label or ", ".join(f"{v}({n})" for v, n in sorted(venues.items())),
        "params": {
            "horizons": horizons,
            "n_bins": n_bins,
            "min_per_bin": min_per_bin,
            "n_bootstrap": n_boot,
            "seed": seed,
        },
        "dataset": {
            "n_markets": len(markets),
            "venues": venues,
            "n_yes": sum(m.outcome for m in markets),
            "n_no": sum(1 - m.outcome for m in markets),
            "resolved_ts_range": [
                min(m.resolved_ts for m in markets),
                max(m.resolved_ts for m in markets),
            ],
        },
        "reference_horizon": ref_name,
        "horizons": {
            p.name: _horizon_entry(
                p, n_bins=n_bins, min_per_bin=min_per_bin,
                n_boot=n_boot, seed=seed,
            )
            for p in panels
        },
        "categories": _category_table(ref_panel, n_bins=n_bins),
        "yes_price_inflation": yes_price_inflation(
            ref_panel, n_boot=n_boot, seed=seed
        ),
    }


def format_digest(analysis: dict, correction: dict | None = None) -> str:
    """Plain-ASCII summary table of an analysis (+ optional correction)."""
    ds = analysis["dataset"]
    lines = [
        f"longshot analysis: {analysis['label']}",
        f"markets: {ds['n_markets']} "
        f"(YES {ds['n_yes']}, NO {ds['n_no']})",
        "",
        f"{'horizon':<8}{'n':>6}{'Brier':>8}{'ECE':>8}{'skill':>8}"
        f"{'slope':>8}{'slope 95% CI':>18}",
        "-" * 62,
    ]
    for h in analysis["params"]["horizons"]:
        e = analysis["horizons"][h]
        if e.get("skipped"):
            lines.append(f"{h:<8}{e['n']:>6}  (skipped: {e['skipped']})")
            continue
        c = e["compression"]
        if math.isnan(c["slope"]):
            slope_s, ci = "-", "[all bins sparse]"
        else:
            slope_s, ci = f"{c['slope']:.3f}", f"[{c['ci_lo']:.2f}, {c['ci_hi']:.2f}]"
        ece_s = f"{e['ece']:.4f}" if not math.isnan(e["ece"]) else "-"
        thin = "*" if e.get("thin_support") else " "
        lines.append(
            f"{h:<8}{e['n']:>6}{e['brier']:>8.4f}{ece_s:>8}"
            f"{e['skill_vs_climatology']:>8.3f}{slope_s:>8}{ci:>18}{thin}"
        )
    lines.append("")
    lines.append("(* fewer than 3 non-sparse bins: ECE/slope rest on thin "
                 "support)")
    lines.append("slope > 1: prices compressed toward 0.5 (underconfident); "
                 "slope < 1: overconfident")
    if correction:
        lines.append("")
        lines.append("out-of-sample correction (test split by resolution date):")
        lines.append(f"{'horizon':<8}{'method':<10}{'dBrier':>9}"
                     f"{'95% CI':>20}  verdict")
        lines.append("-" * 72)
        for h in correction.get("horizons", {}):
            e = correction["horizons"][h]
            if e.get("skipped"):
                lines.append(f"{h:<8}  (skipped: {e['skipped']})")
                continue
            for meth in ("platt", "isotonic"):
                if meth not in e:
                    continue
                r = e[meth]
                if r.get("skipped"):
                    continue
                ci = f"[{r['delta_brier_ci'][0]:+.4f}, {r['delta_brier_ci'][1]:+.4f}]"
                lines.append(
                    f"{h:<8}{meth:<10}{r['delta_brier']:>+9.4f}{ci:>20}  "
                    f"{r['verdict']}"
                )
    return "\n".join(lines)
