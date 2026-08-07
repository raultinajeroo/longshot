#!/usr/bin/env python3
"""Convert downloaded Kaggle prediction-market datasets to longshot store JSONL.

Usage:
    uv run python scripts/kaggle_to_jsonl.py kalshi \
        --root data/kaggle/kalshi_jacksaleeby --out data/kaggle/kalshi.jsonl
    uv run python scripts/kaggle_to_jsonl.py polymarket \
        --root data/kaggle --out data/kaggle/polymarket.jsonl

Kalshi source: jacksaleeby/prediction-market-dataset (normalized/markets.csv
+ trades.csv + candles_1h.csv; labels via resolved_yes). Only markets with at
least one real trade or candle are kept — the dataset's ML features impute
price 0.5 for untraded markets, which would fabricate calibration data.

Polymarket source: sandeepkumarfromin/full-market-data-from-polymarket
(per-market price NDJSON, ~3h cadence) joined to
ismetsemedov/polymarket-prediction-markets (polymarket_markets.csv) on
conditionId for question text, creation time, volume, and the Yes-token
mapping. Outcomes are ground truth from ismetsemedov closed markets with
decisive terminal outcomePrices (>=0.98 / <=0.02) — closedTime is usually
empty and endDate is only the *nominal* end date, so resolved_ts is taken
as the moment the price series first enters the resolution band. Only
binary Yes/No markets whose series converges inside the sandeep collection
window are kept (a live longshot trading at 0.02 is NOT a resolution —
the market must also be closed in the metadata).
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

KEYWORD_CATEGORIES = [
    ("sports", ("nfl", "nba", "mlb", "nhl", "soccer", "tennis", "ufc",
                "super bowl", "world cup", "playoff", "lakers", "celtics",
                "yankees", "chiefs", " vs ", "game")),
    ("crypto", ("bitcoin", "btc", "ethereum", "eth", "solana", "sol",
                "crypto", "doge", "xrp")),
    ("politics", ("election", "president", "senate", "congress", "trump",
                  "biden", "harris", "governor", "mayor", "vote", "nominee")),
    ("macro", ("fed ", "fomc", "interest rate", "inflation", "cpi", "gdp",
               "recession", "jobs report", "unemployment", "treasury")),
    ("tech_ai", ("ai", "openai", "gpt", "anthropic", "google", "apple",
                 "microsoft", "nvidia", "spacex", "tesla")),
]


def parse_ts(s: str | None) -> int | None:
    if not s or not s.strip():
        return None
    s = s.strip()
    try:
        return int(float(s))
    except ValueError:
        pass
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except ValueError:
            continue
    try:  # ISO with fractional offset, e.g. 2026-08-03T01:51:25.139588+00:00
        return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return None


def categorize(question: str) -> str:
    q = question.lower()
    for cat, words in KEYWORD_CATEGORIES:
        if any(w in q for w in words):
            return cat
    return "other"


def parse_list(s: str | None) -> list | None:
    if not s or not s.strip():
        return None
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        try:
            return ast.literal_eval(s)
        except (SyntaxError, ValueError):
            return None


def downsample(points: list[list], max_points: int = 250) -> list[list]:
    """Uniform-stride downsample, always keeping first and last points."""
    if len(points) <= max_points:
        return points
    stride = len(points) / (max_points - 1)
    idx = sorted({0, len(points) - 1} | {int(k * stride) for k in range(max_points)})
    return [points[i] for i in idx]


def record(venue, market_id, question, created_ts, resolved_ts, outcome,
           series, volume, n_traders, source, notes):
    return {
        "venue": venue,
        "market_id": market_id,
        "question": question,
        "category": categorize(question),
        "created_ts": created_ts,
        "resolved_ts": resolved_ts,
        "outcome": outcome,
        "volume": volume,
        "n_traders": n_traders,
        "series": series,
        "provenance": {
            "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": source,
            "api_base": None,
            "notes": notes,
        },
    }


def convert_kalshi(root: Path) -> list[dict]:
    markets_csv = root / "normalized" / "markets.csv"
    trades_csv = root / "normalized" / "trades.csv"
    candles_csv = root / "normalized" / "candles_1h.csv"

    points: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with trades_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            ts = parse_ts(row.get("ts_utc"))
            try:
                price = float(row["price_yes"])
            except (TypeError, ValueError):
                continue
            if ts and 0.0 <= price <= 1.0:
                points[row["market_id"]].append((ts, price))
    with candles_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            ts = parse_ts(row.get("ts_utc"))
            close = (row.get("close_yes") or "").strip()
            if not ts or not close:
                continue
            price = float(close)
            if 0.0 <= price <= 1.0:
                points[row["market_id"]].append((ts, price))

    out, skipped_no_prices = [], 0
    with markets_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if (row.get("resolved_yes") or "").strip() not in ("0", "1"):
                continue
            pts = sorted(set(points.get(row["market_id"], [])))
            if not pts:
                skipped_no_prices += 1
                continue
            created = parse_ts(row.get("open_ts_utc"))
            # close_ts = last trading moment = effective resolution for
            # horizon panels (settle_ts can lag close by days).
            resolved = parse_ts(row.get("close_ts_utc")) or parse_ts(row.get("settle_ts_utc"))
            if not created or not resolved or resolved <= created:
                continue
            series = downsample([[ts, p] for ts, p in pts])
            try:
                volume = float(row.get("volume_usd_estimate") or 0) or None
            except ValueError:
                volume = None
            out.append(record(
                "kaggle_kalshi", row["market_id"], row.get("question") or "",
                created, resolved, int(row["resolved_yes"]), series, volume,
                None, "kaggle:jacksaleeby/prediction-market-dataset",
                "series from trades.csv + candles_1h.csv; resolved_ts=close_ts_utc"))
    print(f"kalshi: {len(out)} markets kept, "
          f"{skipped_no_prices} resolved markets had no trades/candles (skipped)",
          file=sys.stderr)
    return out


def _load_polymarket_meta(markets_csv: Path, wanted: set[str]) -> dict[str, dict]:
    """Metadata per conditionId for genuinely RESOLVED binary markets.

    Requires closed == True and terminal outcomePrices at an extreme
    (>=0.98 / <=0.02) — outcome is then ground truth from the snapshot.
    (A live favorite/longshot can trade at 0.02 without being resolved;
    only closed markets make the extreme-price rule valid.)
    """
    meta = {}
    with markets_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            cid = row.get("conditionId") or ""
            if cid not in wanted:
                continue
            if (row.get("closed") or "").strip() != "True":
                continue
            outcomes = parse_list(row.get("outcomes"))
            prices = parse_list(row.get("outcomePrices"))
            tokens = parse_list(row.get("clobTokenIds"))
            if not outcomes or not prices or len(outcomes) != 2 or "Yes" not in outcomes:
                continue
            try:
                prices = [float(p) for p in prices]
            except (TypeError, ValueError):
                continue
            hi, lo = max(prices), min(prices)
            if hi < 0.98 or lo > 0.02:
                continue  # closed without a decisive terminal price
            yes_idx = outcomes.index("Yes")
            created = parse_ts(row.get("createdAt"))
            if not created:
                continue
            try:
                volume = float(row.get("volumeNum") or row.get("volume") or 0) or None
            except ValueError:
                volume = None
            meta[cid] = {
                "question": row.get("question") or "",
                "created_ts": created,
                "outcome": 1 if prices.index(hi) == yes_idx else 0,
                "closed_ts": parse_ts(row.get("closedTime")),
                "yes_token": (tokens[yes_idx] if tokens and len(tokens) == 2 else None),
                "yes_index": yes_idx,
                "volume": volume,
            }
    return meta


def _resolution_ts(pts: list[tuple[int, float]], outcome: int) -> int | None:
    """First time the series enters the resolution band (>=0.98 for YES,
    <=0.02 for NO). None when the series never converges in-window — i.e.
    the collection window ended before the market resolved."""
    band = (lambda p: p >= 0.98) if outcome == 1 else (lambda p: p <= 0.02)
    for ts, p in pts:
        if band(p):
            return ts
    return None


def convert_polymarket(root: Path) -> list[dict]:
    raw_root = root / "polymarket_sandeep" / "Polymarket_dataset" / "Polymarket_dataset"
    filt_root = root / "polymarket_sandeep" / "filtered_4_ML" / "filtered_4_ML"
    markets_csv = root / "polymarket_ismetsemedov" / "polymarket_markets.csv"

    wanted = {d.name.split("=", 1)[1] for d in raw_root.iterdir() if d.name.startswith("market=")}
    print(f"polymarket: {len(wanted)} markets with raw price data; "
          f"scanning {markets_csv.name} for metadata...", file=sys.stderr)
    meta = _load_polymarket_meta(markets_csv, wanted)
    print(f"polymarket: {len(meta)} binary Yes/No markets matched",
          file=sys.stderr)

    out, dropped = [], {"no_series": 0, "no_convergence_in_window": 0, "bad_timing": 0}
    for cid, m in meta.items():
        pts: list[tuple[int, float]] = []
        if m["yes_token"]:
            token_file = raw_root / f"market={cid}" / "price" / f"token={m['yes_token']}.ndjson"
            if token_file.exists():
                with token_file.open(encoding="utf-8") as fh:
                    for chunk in fh.read().split("} {"):
                        chunk = chunk if chunk.startswith("{") else "{" + chunk
                        chunk = chunk if chunk.endswith("}") else chunk + "}"
                        try:
                            rec = json.loads(chunk)
                            pts.append((int(rec["t"]), float(rec["p"])))
                        except (json.JSONDecodeError, KeyError, ValueError):
                            continue
        if not pts:  # fallback: daily closes from filtered_4_ML
            mdir = filt_root / f"market={cid}"
            if mdir.is_dir():
                for f in sorted(mdir.glob("*.csv")):
                    day = parse_ts(f.stem + "T00:00:00Z")
                    with f.open(newline="", encoding="utf-8") as fh:
                        for row in csv.DictReader(fh):
                            if row.get("outcome") == str(m["yes_index"]):
                                try:
                                    pts.append((day, float(row["p_close"])))
                                except (TypeError, ValueError):
                                    pass
        pts = sorted({ts: p for ts, p in pts if 0.0 <= p <= 1.0}.items())
        if len(pts) < 5:
            dropped["no_series"] += 1
            continue
        # resolved_ts = when the series converged into the resolution band
        # (closedTime in the snapshot is usually empty; the band crossing is
        # the moment the market price effectively settled the question).
        resolved_ts = _resolution_ts(pts, m["outcome"])
        if resolved_ts is None:
            dropped["no_convergence_in_window"] += 1
            continue
        if resolved_ts <= m["created_ts"]:
            dropped["bad_timing"] += 1
            continue
        out.append(record(
            "kaggle_polymarket", cid, m["question"], m["created_ts"],
            resolved_ts, m["outcome"],
            downsample([[ts, p] for ts, p in pts]), m["volume"], None,
            "kaggle:sandeepkumarfromin/full-market-data-from-polymarket"
            "+ismetsemedov/polymarket-prediction-markets",
            "series=Yes-token 3h snapshots (sandeep) or daily closes "
            "(filtered_4_ML); outcome from closed-market terminal "
            "outcomePrices (ismetsemedov); resolved_ts=first 0.98/0.02 "
            "band crossing"))
    print(f"polymarket: {len(out)} markets kept, dropped: {dropped}", file=sys.stderr)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["kalshi", "polymarket"])
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    records = (convert_kalshi(args.root) if args.dataset == "kalshi"
               else convert_polymarket(args.root))
    if not records:
        print("error: conversion produced no markets", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")
    print(f"wrote {len(records)} markets to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
