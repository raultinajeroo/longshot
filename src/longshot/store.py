"""JSONL persistence for MarketSeries, with validation and provenance.

On-disk format: one JSON object per line with the MarketSeries fields;
``series`` is a list of [ts, price] pairs. Loading validates price ranges,
outcome values, and time ordering; series are re-sorted by ts; markets
with an empty series are dropped (a resolved market with no probability
history carries no calibration information).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

from .types import MarketSeries, PricePoint

REQUIRED_FIELDS = (
    "venue", "market_id", "question", "category", "created_ts",
    "resolved_ts", "outcome", "series",
)


class StoreError(ValueError):
    """Raised when a JSONL record fails validation."""


def market_to_dict(m: MarketSeries) -> dict:
    """Serialize a MarketSeries to the JSONL object format."""
    return {
        "venue": m.venue,
        "market_id": m.market_id,
        "question": m.question,
        "category": m.category,
        "created_ts": m.created_ts,
        "resolved_ts": m.resolved_ts,
        "outcome": m.outcome,
        "volume": m.volume,
        "n_traders": m.n_traders,
        "series": [[pt.ts, pt.price] for pt in m.series],
        "provenance": m.provenance,
    }


def market_from_dict(d: dict, *, line_no: int | None = None) -> MarketSeries | None:
    """Validate and build a MarketSeries from a JSONL object.

    Returns None when the record is dropped by policy (empty series after
    cleaning). Raises StoreError on hard validation failures.
    """
    where = f" (line {line_no})" if line_no is not None else ""
    missing = [f for f in REQUIRED_FIELDS if f not in d]
    if missing:
        raise StoreError(f"store: missing fields {missing}{where}")
    if d["outcome"] not in (0, 1):
        raise StoreError(f"store: outcome must be 0 or 1{where}")
    created = int(d["created_ts"])
    resolved = int(d["resolved_ts"])
    if resolved <= created:
        raise StoreError(f"store: resolved_ts must be after created_ts{where}")
    raw_series = d["series"]
    if not isinstance(raw_series, list):
        raise StoreError(f"store: series must be a list{where}")
    points: list[PricePoint] = []
    for item in raw_series:
        try:
            ts, price = int(item[0]), float(item[1])
        except (TypeError, ValueError, IndexError) as exc:
            raise StoreError(f"store: bad series point {item!r}{where}") from exc
        if not 0.0 <= price <= 1.0:
            raise StoreError(f"store: price {price} out of [0, 1]{where}")
        points.append(PricePoint(ts=ts, price=price))
    points.sort(key=lambda p: p.ts)
    if not points:
        return None  # policy: drop empty series
    return MarketSeries(
        venue=str(d["venue"]),
        market_id=str(d["market_id"]),
        question=str(d["question"]),
        category=str(d.get("category") or "uncategorized"),
        created_ts=created,
        resolved_ts=resolved,
        outcome=int(d["outcome"]),
        volume=None if d.get("volume") is None else float(d["volume"]),
        n_traders=None if d.get("n_traders") is None else int(d["n_traders"]),
        series=tuple(points),
        provenance=dict(d.get("provenance") or {}),
    )


def write_jsonl(path: str | Path, markets: Iterable[MarketSeries]) -> int:
    """Write markets to a JSONL file. Returns the number written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for m in markets:
            fh.write(json.dumps(market_to_dict(m), separators=(",", ":")) + "\n")
            n += 1
    return n


def load_jsonl(path: str | Path) -> list[MarketSeries]:
    """Load markets from a JSONL file, validating every record."""
    path = Path(path)
    if not path.exists():
        raise StoreError(f"store: no such file: {path}")
    out: list[MarketSeries] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as exc:
                raise StoreError(f"store: invalid JSON (line {line_no}): {exc}") from exc
            m = market_from_dict(d, line_no=line_no)
            if m is not None:
                out.append(m)
    if not out:
        raise StoreError(f"store: no valid markets in {path}")
    return out


def iter_jsonl(path: str | Path) -> Iterator[MarketSeries]:
    """Stream markets from a JSONL file (memory-friendly variant)."""
    path = Path(path)
    if not path.exists():
        raise StoreError(f"store: no such file: {path}")
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            m = market_from_dict(json.loads(line), line_no=line_no)
            if m is not None:
                yield m


def downsample_series(
    series: tuple[PricePoint, ...],
    max_points: int = 200,
    keep_last_seconds: int = 7200,
    resolved_ts: int | None = None,
) -> tuple[PricePoint, ...]:
    """Downsample a series to at most ``max_points`` by uniform-in-time stride.

    Always keeps the first and last point and every point within
    ``keep_last_seconds`` of ``resolved_ts`` (the near-resolution dynamics
    matter most for short horizons). The remaining budget is filled with
    evenly spaced indices from the rest. Returns the input unchanged when
    it already fits.
    """
    if len(series) <= max_points:
        return series
    cutoff = (resolved_ts if resolved_ts is not None else series[-1].ts) - keep_last_seconds
    keep = {0, len(series) - 1}
    keep.update(i for i, pt in enumerate(series) if pt.ts >= cutoff)
    budget = max(0, max_points - len(keep))
    pool = [i for i in range(len(series)) if i not in keep]
    if budget and pool:
        stride = len(pool) / budget
        keep.update(pool[int(k * stride)] for k in range(budget))
    return tuple(series[i] for i in sorted(keep))


def cache_dir() -> Path:
    """Resolve the local data cache directory (created on demand)."""
    base = os.environ.get("LONGSHOT_CACHE")
    path = Path(base) if base else Path("data") / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def provenance_meta(source: str, api_base: str | None = None,
                    notes: str | None = None) -> dict:
    """Standard provenance block stamped onto every fetched market."""
    return {
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": source,
        "api_base": api_base,
        "notes": notes,
    }
