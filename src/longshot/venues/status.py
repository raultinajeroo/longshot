"""Honest per-venue status reporting.

The status of each collector is stated precisely: what has actually been
exercised, and where. A collector that has only been unit-tested against
recorded/canonical payloads is *parser-tested*, not "working"; it is
marked unavailable for live claims unless it was genuinely exercised
against the live API in the current run.
"""

from __future__ import annotations

#: Facts about the bundled sample (see data/bundled/SOURCES.md).
BUNDLED_MANIFOLD = "bundled sample of 278 markets (fetched 2026-07-31)"


def venue_status(exercised_live: set[str] | None = None) -> list[dict]:
    """One status row per known venue.

    ``exercised_live`` names venues whose collectors genuinely hit their
    live APIs in the current run (e.g. the venues present in the dataset
    being published, when that dataset came from a live fetch in this
    session). Everything else is reported by its strongest honest claim.
    """
    exercised_live = exercised_live or set()
    rows: list[dict] = []

    manifold_live = "manifold" in exercised_live
    rows.append({
        "venue": "manifold",
        "status": (
            "exercised live in this run"
            if manifold_live
            else f"working collector; {BUNDLED_MANIFOLD} committed"
        ),
        "key": "none",
    })

    for venue in ("polymarket", "kalshi"):
        if venue in exercised_live:
            status = "exercised live in this run"
        else:
            status = (
                "collector parser-tested against recorded/canonical "
                "payloads only; NOT exercised live in this run, so no live "
                "claims are made for it"
            )
        rows.append({"venue": venue, "status": status, "key": "none"})

    rows.append({
        "venue": "fixture",
        "status": "offline loader for any JSONL in store format",
        "key": "none",
    })
    return rows


def is_manifold_only(venues: set[str]) -> bool:
    """True when a dataset contains only Manifold (bundled or fetched)."""
    return venues == {"manifold"}


MANIFOLD_ONLY_NOTE = (
    "Manifold-only methodology demonstration: the Polymarket and Kalshi "
    "collectors were not exercised live in this run, so every number here "
    "describes the play-money Manifold sample only."
)
