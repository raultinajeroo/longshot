"""Self-contained HTML report: inline CSS, inline SVG, no JavaScript.

Charts: (1) reliability diagram at the reference horizon with Wilson CI
whiskers and greyed sparse bins; (2) compression slope by horizon with
bootstrap CIs; (3) Brier and ECE by horizon; (4) category table;
(5) out-of-sample correction section; (6) methods, caveats, citations.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

_CSS = """
:root { color-scheme: light; }
body { font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       margin: 0 auto; max-width: 960px; padding: 32px 24px; color: #2b2620;
       background: #faf7f2; line-height: 1.55; }
h1 { font-size: 1.45rem; margin: 0 0 4px; }
h2 { font-size: 1.08rem; margin: 30px 0 8px; color: #57493a; }
.meta { color: #8a7d6b; font-size: 0.85rem; margin-bottom: 20px; }
table { border-collapse: collapse; width: 100%; font-size: 0.85rem;
        background: #ffffff; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #e8e0d4; }
th { color: #57493a; font-weight: 600; background: #f3ede3; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.note { color: #8a7d6b; font-size: 0.82rem; }
.verdict-no { color: #8a6d1f; font-weight: 600; }
.verdict-yes { color: #2e6b3f; font-weight: 600; }
.caveats li { margin: 4px 0; }
svg { display: block; width: 100%; height: auto; background: #ffffff;
      border: 1px solid #e8e0d4; border-radius: 6px; }
"""


def _fmt(x: float | None, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{digits}f}"


def _reliability_svg(entry: dict) -> str:
    """Reliability diagram: 45-degree line, bin points, CI whiskers."""
    size, pad = 460, 46
    w = h = size

    def xy(vx: float, vy: float) -> tuple[float, float]:
        return (pad + vx * (w - 2 * pad), h - pad - vy * (h - 2 * pad))

    parts = [
        f'<svg viewBox="0 0 {w} {h}" role="img" '
        f'aria-label="reliability diagram at horizon {entry["horizon"]}">',
        f'<line x1="{pad}" y1="{h - pad}" x2="{w - pad}" y2="{pad}" '
        f'stroke="#b9ab93" stroke-width="1" stroke-dasharray="5 4"/>',
    ]
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        x, y = xy(t, 0.0)
        parts.append(f'<text x="{x:.1f}" y="{y + 16}" font-size="10" '
                     f'fill="#8a7d6b" text-anchor="middle">{t:.2f}</text>')
        x, y = xy(0.0, t)
        parts.append(f'<text x="{x - 8}" y="{y + 3}" font-size="10" '
                     f'fill="#8a7d6b" text-anchor="end">{t:.2f}</text>')
    for b in entry["bins"]:
        if b["n"] == 0:
            continue
        color = "#c9c2b4" if b["sparse"] else "#8a5a2b"
        cx, cy = xy(b["mean_pred"], b["rate"])
        _, ylo = xy(b["mean_pred"], b["ci_lo"])
        _, yhi = xy(b["mean_pred"], b["ci_hi"])
        r = 2.5 + 3.5 * math.sqrt(min(b["n"], 500) / 500)
        parts.append(
            f'<line x1="{cx:.1f}" y1="{ylo:.1f}" x2="{cx:.1f}" y2="{yhi:.1f}" '
            f'stroke="{color}" stroke-width="1"/>'
        )
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" '
                     f'fill="{color}"/>')
    parts.append(
        f'<text x="{w / 2}" y="{h - 6}" font-size="11" fill="#57493a" '
        f'text-anchor="middle">mean predicted probability</text>'
    )
    parts.append(
        f'<text x="14" y="{h / 2}" font-size="11" fill="#57493a" '
        f'text-anchor="middle" transform="rotate(-90 14 {h / 2})">'
        f'empirical frequency</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _slope_ci_svg(horizons: dict, order: list[str]) -> str:
    """Compression slope by horizon: point estimate + bootstrap CI."""
    entries = [(h, horizons[h]) for h in order
               if not horizons[h].get("skipped")
               and not math.isnan(horizons[h]["compression"]["slope"])]
    if not entries:
        return '<p class="note">no compression slope estimates available</p>'
    w, h, pad_l, pad_r = 920, 260, 60, 30
    vals = [e["compression"]["slope"] for _, e in entries]
    los = [e["compression"]["ci_lo"] for _, e in entries]
    his = [e["compression"]["ci_hi"] for _, e in entries]
    lo = min([1.0, min(v for v in los if not math.isnan(v))] + vals)
    hi = max([1.0, max(v for v in his if not math.isnan(v))] + vals)
    span = (hi - lo) or 1.0
    lo -= 0.08 * span
    span *= 1.16
    top, bot = 20, 40

    def y_of(v: float) -> float:
        return top + (1 - (v - lo) / span) * (h - top - bot)

    n = len(entries)
    step = (w - pad_l - pad_r) / max(n, 1)
    parts = [
        f'<svg viewBox="0 0 {w} {h}" role="img" '
        f'aria-label="compression slope by horizon">',
        f'<line x1="{pad_l}" y1="{y_of(1.0):.1f}" x2="{w - pad_r}" '
        f'y2="{y_of(1.0):.1f}" stroke="#b9ab93" stroke-width="1" '
        f'stroke-dasharray="5 4"/>',
        f'<text x="{pad_l - 8}" y="{y_of(1.0) + 3:.1f}" font-size="10" '
        f'fill="#8a7d6b" text-anchor="end">1.00</text>',
    ]
    for i, (name, e) in enumerate(entries):
        c = e["compression"]
        x = pad_l + step * (i + 0.5)
        parts.append(
            f'<line x1="{x:.1f}" y1="{y_of(c["ci_lo"]):.1f}" x2="{x:.1f}" '
            f'y2="{y_of(c["ci_hi"]):.1f}" stroke="#8a5a2b" stroke-width="1.4"/>'
        )
        parts.append(f'<circle cx="{x:.1f}" cy="{y_of(c["slope"]):.1f}" r="4" '
                     f'fill="#8a5a2b"/>')
        parts.append(f'<text x="{x:.1f}" y="{h - 22}" font-size="11" '
                     f'fill="#57493a" text-anchor="middle">{name}</text>')
        parts.append(f'<text x="{x:.1f}" y="{h - 8}" font-size="10" '
                     f'fill="#8a7d6b" text-anchor="middle">'
                     f'{c["slope"]:.2f}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _metrics_svg(horizons: dict, order: list[str]) -> str:
    """Brier and ECE by horizon (two series, one axis each half)."""
    entries = [(h, horizons[h]) for h in order if not horizons[h].get("skipped")]
    if not entries:
        return ""
    w, h, pad_l, pad_r = 920, 260, 50, 50
    briers = [e["brier"] for _, e in entries]
    eces = [e["ece"] for _, e in entries if not math.isnan(e["ece"])]
    vmax = max(briers + eces) * 1.15 or 0.3
    top, bot = 20, 40

    def y_of(v: float) -> float:
        return top + (1 - v / vmax) * (h - top - bot)

    n = len(entries)
    step = (w - pad_l - pad_r) / max(n - 1, 1)
    pts_b, pts_e = [], []
    parts = [
        f'<svg viewBox="0 0 {w} {h}" role="img" '
        f'aria-label="Brier and ECE by horizon">'
    ]
    for i, (name, e) in enumerate(entries):
        x = pad_l + step * i
        pts_b.append(f"{x:.1f},{y_of(e['brier']):.1f}")
        if not math.isnan(e["ece"]):
            pts_e.append(f"{x:.1f},{y_of(e['ece']):.1f}")
        parts.append(f'<text x="{x:.1f}" y="{h - 22}" font-size="11" '
                     f'fill="#57493a" text-anchor="middle">{name}</text>')
    parts.append(f'<polyline points="{" ".join(pts_b)}" fill="none" '
                 f'stroke="#8a5a2b" stroke-width="1.6"/>')
    parts.append(f'<polyline points="{" ".join(pts_e)}" fill="none" '
                 f'stroke="#3d6b8a" stroke-width="1.6"/>')
    for pt in pts_b:
        x, y = pt.split(",")
        parts.append(f'<circle cx="{x}" cy="{y}" r="3" fill="#8a5a2b"/>')
    for pt in pts_e:
        x, y = pt.split(",")
        parts.append(f'<circle cx="{x}" cy="{y}" r="3" fill="#3d6b8a"/>')
    parts.append(
        f'<text x="{pad_l}" y="14" font-size="11" fill="#8a5a2b">'
        f'— Brier</text>'
        f'<text x="{pad_l + 60}" y="14" font-size="11" fill="#3d6b8a">'
        f'— ECE</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def render_html(
    analysis: dict,
    correction: dict | None = None,
    *,
    title: str = "longshot calibration report",
) -> str:
    """Render the single-file HTML report for an analysis dict."""
    esc = html.escape
    order = analysis["params"]["horizons"]
    ref = analysis["reference_horizon"]
    ds = analysis["dataset"]

    cat_rows = "".join(
        "<tr>"
        f"<td>{esc(r['category'])}</td>"
        f"<td class='num'>{r['n']}</td>"
        f"<td class='num'>{_fmt(r['brier'])}</td>"
        f"<td class='num'>{_fmt(r['ece'])}</td>"
        f"<td class='num'>{_fmt(r['slope'], 2)}</td>"
        f"<td class='num'>{_fmt(r['base_rate'], 3)}</td>"
        "</tr>"
        for r in analysis["categories"]
    )

    corr_html = ""
    if correction:
        rows = []
        for h in correction.get("horizons", {}):
            e = correction["horizons"][h]
            if e.get("skipped"):
                continue
            for meth in ("platt", "isotonic"):
                r = e.get(meth)
                if not r or r.get("skipped"):
                    continue
                cls = ("verdict-yes" if r["verdict"] == "reliable improvement"
                       else "verdict-no")
                rows.append(
                    "<tr>"
                    f"<td>{esc(h)}</td><td>{meth}</td>"
                    f"<td class='num'>{_fmt(e['raw']['brier'])}</td>"
                    f"<td class='num'>{_fmt(r['brier'])}</td>"
                    f"<td class='num'>{r['delta_brier']:+.4f}</td>"
                    f"<td class='num'>[{r['delta_brier_ci'][0]:+.4f}, "
                    f"{r['delta_brier_ci'][1]:+.4f}]</td>"
                    f"<td class='{cls}'>{esc(r['verdict'])}</td>"
                    "</tr>"
                )
        corr_html = (
            "<h2>out-of-sample correction</h2>"
            "<p class='note'>Correction maps (Platt scaling, isotonic "
            "regression) are fit on the earliest "
            f"{correction['train_frac']:.0%} of markets by resolution date "
            "and evaluated on the later test split. When the bootstrap CI "
            "of the Brier change includes 0 the verdict is "
            "<b>no reliable improvement</b> — reported, never hidden.</p>"
            "<table><thead><tr><th>horizon</th><th>method</th>"
            "<th>raw Brier</th><th>corrected</th><th>&Delta;Brier</th>"
            "<th>95% CI</th><th>verdict</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )

    doc = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{esc(title)}</title>",
        f"<style>{_CSS}</style></head><body>",
        f"<h1>{esc(title)}</h1>",
        f"<div class='meta'>{ds['n_markets']} markets &middot; "
        f"{esc(analysis['label'])} &middot; generated "
        f"{esc(analysis['generated_at'])} by longshot "
        f"{esc(analysis['version'])}</div>",

        f"<h2>reliability diagram — {esc(ref)} before resolution</h2>",
        _reliability_svg(analysis["horizons"][ref]),
        "<p class='note'>Point area scales with bin count; whiskers are "
        "Wilson 95% intervals; grey points are sparse bins excluded from "
        "ECE.</p>",

        "<h2>compression slope by horizon</h2>",
        _slope_ci_svg(analysis["horizons"], order),
        "<p class='note'>Weighted least squares of (bin rate - 0.5) on "
        "(bin mean - 0.5) through the origin. Slope &gt; 1: prices "
        "compressed toward 0.5 (underconfident); &lt; 1: overconfident. "
        "Whiskers: market-level bootstrap 95% CI.</p>",

        "<h2>Brier and ECE by horizon</h2>",
        _metrics_svg(analysis["horizons"], order),

        f"<h2>categories — {esc(ref)} before resolution</h2>",
        "<table><thead><tr><th>category</th><th>n</th><th>Brier</th>"
        "<th>ECE</th><th>slope</th><th>base rate</th></tr></thead>"
        f"<tbody>{cat_rows}</tbody></table>",

        corr_html,

        "<h2>methods</h2>",
        "<p class='note'>Prices are carried forward to each horizon "
        "(last observation at or before T - h, at most one horizon old). "
        "Brier = mean((p - y)^2); ECE = count-weighted mean |rate - mean "
        "prediction| over non-sparse equal-width bins; Murphy decomposition "
        "Brier = REL - RES + UNC; compression slope via WLS through the "
        "origin; favorite-longshot slope via logistic regression of outcome "
        "on logit(price); correction by Platt scaling and isotonic "
        "regression (PAVA), fit and evaluated on disjoint time splits.</p>",

        "<h2>caveats</h2>",
        "<ul class='caveats'>"
        "<li>Manifold is a play-money (Mana) market; incentives differ from "
        "real-money venues.</li>"
        "<li>Polymarket outcomes are inferred from terminal prices "
        "(&ge;0.98 YES / &le;0.02 NO); ambiguous terminals are excluded.</li>"
        "<li>Category assignment is a keyword heuristic.</li>"
        "<li>Carry-forward pricing with a staleness bound (price at most "
        "one horizon old) is an approximation.</li>"
        "<li>All slices are reported with confidence intervals; no "
        "cherry-picking — treat patterns across many slices with "
        "multiple-comparison discipline.</li>"
        "<li><b>This is a measurement tool. It is not investment advice, "
        "and historical calibration says nothing about any single "
        "future market.</b></li>"
        "</ul>",

        "<h2>research context</h2>",
        "<p class='note'>This lab operationalizes measurement questions "
        "from Prophet Arena (Yang et al. 2025, arXiv:2510.17638), Halawi "
        "et al. 2024 (arXiv:2402.18563), Le 2026 (Decomposing Crowd "
        "Wisdom), Saguillo et al. (AFT 2025, arXiv:2508.03474), "
        "ForecastBench (Karger et al., ICLR 2025), and Zemani 2026 (SSRN). "
        "See README.md for full citations.</p>",

        "</body></html>",
    ]
    return "\n".join(doc)


def write_report(
    analysis: dict,
    correction: dict | None,
    out_path: str | Path,
    *,
    title: str = "longshot calibration report",
) -> Path:
    """Write report.html plus the analysis JSON alongside it."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_html(analysis, correction, title=title),
                        encoding="utf-8")
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(analysis, indent=2) + "\n",
                         encoding="utf-8")
    return out_path
