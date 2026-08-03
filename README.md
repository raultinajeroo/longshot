# longshot

**Find where prediction markets are wrong, systematically.**

Calibration & bias analytics for prediction markets, built as a
continuation of recent research. longshot downloads resolved markets with
their probability histories, measures calibration at multiple horizons
before resolution, tests for systematic biases (compression toward 50%,
favorite-longshot slope, YES/NO asymmetry, category effects), fits
bias-correction maps on a train split and evaluates them honestly
out-of-sample, and renders a self-contained HTML report. It makes no
trading claims and no live-profit claims.

## 60-second demo (offline, no API keys)

```bash
pip install -e .
longshot demo
```

The demo analyzes the bundled sample of **278 real resolved Manifold
markets** (fetched 2026-07-31; see `data/bundled/SOURCES.md`), fits Platt
and isotonic corrections on the earliest 60% by resolution date, evaluates
them on the rest, renders `examples/demo/report_bundled.html`, and runs
the same pipeline on a fixture with planted bias. Actual output of the
committed run (`examples/demo/run_output.txt`):

```
longshot analysis: Manifold bundled resolved-market sample
markets: 278 (YES 120, NO 158)

horizon      n   Brier     ECE   skill   slope      slope 95% CI
--------------------------------------------------------------
30d        163  0.1323  0.0269   0.442   1.059      [0.97, 1.11]*
14d        176  0.1450  0.0140   0.412   0.970      [0.85, 1.08]*
7d         161  0.1321  0.0359   0.468   1.077      [1.06, 1.09]*
3d         174  0.0874  0.0208   0.647   1.044      [0.99, 1.08]*
1d         129  0.0862  0.0291   0.655   1.062      [1.05, 1.08]*
12h        111  0.0849  0.0317   0.658   1.068      [1.04, 1.09]*
1h          56  0.1114       -   0.503       - [all bins sparse]*

(* fewer than 3 non-sparse bins: ECE/slope rest on thin support)
slope > 1: prices compressed toward 0.5 (underconfident); slope < 1: overconfident

out-of-sample correction (test split by resolution date):
horizon method       dBrier              95% CI  verdict
------------------------------------------------------------------------
30d     platt       +0.0063  [-0.0080, +0.0202]  no reliable improvement
30d     isotonic    +0.0149  [-0.0058, +0.0337]  no reliable improvement
14d     platt       +0.0076  [-0.0096, +0.0277]  no reliable improvement
14d     isotonic    +0.0137  [-0.0161, +0.0436]  no reliable improvement
7d      platt       +0.0045  [-0.0099, +0.0189]  no reliable improvement
7d      isotonic    +0.0119  [-0.0174, +0.0410]  no reliable improvement
3d      platt       -0.0113  [-0.0253, +0.0024]  no reliable improvement
3d      isotonic    +0.0053  [-0.0191, +0.0341]  no reliable improvement
1d      platt       -0.0056  [-0.0179, +0.0075]  no reliable improvement
1d      isotonic    +0.0211  [+0.0046, +0.0404]  reliable degradation
12h     platt       -0.0042  [-0.0222, +0.0146]  no reliable improvement
12h     isotonic    +0.0241  [-0.0097, +0.0599]  no reliable improvement
1h        (skipped: too few train/test panel points (<20))

biased fixture (planted compression 0.55): slope at 30d = 1.78 [1.40, 2.19] (expect ~1.8 = 1/0.55)
```

Two honest headline findings. First, this Manifold sample is close to
calibrated: compression slopes sit within ~8% of 1.0 at every horizon, and
Brier skill vs climatology runs 0.41-0.66. Second, the correction layer
cannot beat the raw market price out-of-sample here: every horizon is
"no reliable improvement" (one "reliable degradation"). longshot reports
that verdict; it does not tune until the answer looks better. The
sanity-check anchor runs the other way: on the fixture with planted
compression (c = 0.55), the pipeline recovers slope 1.78 [1.40, 2.19]
against the expected 1/0.55 = 1.82.

## The research context

Recent work keeps rediscovering that the market price is a strong
forecaster, and mostly stops there. longshot is the measurement layer
those results point at.

- **Prophet Arena** (Yang et al. 2025, arXiv:2510.17638): 23 LLMs on
  1,367 Kalshi events tie markets on Brier (0.184-0.197 vs 0.187) and beat
  them on calibration (ECE 0.03-0.07 vs 0.069), yet no model breaks even
  on returns. The market price is the forecast to beat; measure it
  precisely.
- **Zemani 2026** (SSRN): frontier LLMs add almost no information beyond
  the market price. (Cited from the abstract; the SSRN full text was
  fetch-blocked at build time.)
- **Le 2026, "Decomposing Crowd Wisdom"**: political markets are
  chronically compressed toward 50%, and the effect does not replicate
  across platforms, so longshot ships per-venue replication by design
  (compression slope with bootstrap CIs per venue, per horizon).
- **Bartlett & O'Hara**: Kalshi YES-buyers' implied win rate is 60.9%
  vs 32.5% realized. Price-side calibration is an open measurement
  problem.
- **Halawi et al. 2024** (arXiv:2402.18563): human crowd Brier 0.149 vs
  best zero-shot LLM 0.208.
- **Saguillo et al.** (AFT 2025, arXiv:2508.03474): $40M of documented
  arbitrage. Efficiency gaps are real and measurable.
- **ForecastBench** (Karger et al., ICLR 2025): dynamic benchmark with
  public data.

longshot does not forecast. It measures the forecaster that already
exists (the market price) and asks where, when, and for whom it is
biased, with sample sizes and confidence intervals attached.

## Features

- **Horizon panels**: carry-forward prices at 30d/14d/7d/3d/1d/12h/1h
  before resolution, with a staleness bound (price at most one horizon
  old) and a lifetime bound.
- **Calibration metrics**: Brier, log-loss, ECE/MCE over equal-width bins
  with Wilson CIs and sparse-bin exclusion; Murphy decomposition
  (Brier = REL - RES + UNC, identity tested to 1e-12); climatology
  baseline and skill score.
- **Bias measures**: compression slope (WLS through the origin on
  (rate - 0.5) vs (meanp - 0.5), market-level bootstrap CI); logistic
  calibration slope on logit(p) (the favorite-longshot direction);
  per-category YES-price inflation with CIs.
- **Correction layer**: Platt scaling and isotonic regression (PAVA) fit
  on the earliest 60% of markets by resolution date, evaluated on the
  later test split: delta-Brier, delta-log-loss, ECE before/after,
  bootstrap CI of delta-Brier, and an explicit verdict. If the CI includes
  0 the verdict is `no reliable improvement`, and the demo prints it.
- **Venues**: Manifold (fully working, no key), Polymarket and Kalshi
  collectors (keyless, network-gated), a fixture venue for any local
  JSONL, and a seeded synthetic-market simulator with planted-bias modes.
- **Report**: single-file HTML, inline CSS and SVG, no JavaScript, no
  external assets: reliability diagram, slope-by-horizon chart, Brier/ECE
  chart, category table, correction section, methods, caveats, citations.

## Install and CLI

```bash
pip install -e .        # stdlib + numpy only, Python >= 3.11
```

```
longshot fetch --venue manifold --out data/manifold.jsonl [--max-markets 250] [--max-bets-per-market 4000] [--max-calls 5000] [--seed 42]
longshot fetch --venue polymarket|kalshi --out FILE [--max-markets N]
longshot analyze --input FILE [FILE ...] [--out analysis.json] [--horizons 30d,14d,7d,3d,1d,12h,1h] [--bins 10] [--min-per-bin 30] [--bootstrap 1000] [--seed 42]
longshot correct --input FILE [--method isotonic|platt|both] [--train-frac 0.6] [--out correction.json]
longshot report --analysis analysis.json [--correction correction.json] --out report.html [--title T]
longshot simulate --mode calibrated|compressed|longshot-bias --markets 120 --seed 7 --out fixtures/x.jsonl
longshot demo [--outdir examples/demo]
```

Exit codes: `0` ok, `2` usage, `3` venue unavailable (message includes a
remedy hint pointing at offline inputs), `4` empty/invalid input.

## Data model

One JSON object per line (JSONL):

```json
{"venue": "manifold", "market_id": "...", "question": "...", "category": "politics",
 "created_ts": 1767819058, "resolved_ts": 1770098321, "outcome": 1,
 "volume": 2132.24, "n_traders": 20,
 "series": [[1767819058, 0.5], [1767900000, 0.62]],
 "provenance": {"fetched_at": "2026-07-31T...", "source": "...", "api_base": "..."}}
```

`outcome` is 1 for YES, 0 for NO; prices are probabilities of YES in
[0, 1]. Loading validates ranges, sorts series, and drops empty series
(`store.py`).

## Venues

| Venue | Status | Key | Notes |
|---|---|---|---|
| Manifold | bundled sample + live collector | none | fully exercised from the build machine; bundled sample of 278 markets committed |
| Polymarket | live collector (Gamma + CLOB) | none | network-gated; parsers unit-tested against canonical payloads; outcome inferred from terminal prices; 12h history fidelity |
| Kalshi | live collector (v2 settled + candlesticks) | none today | network-gated; parsers unit-tested; daily candles; sends `Authorization: Bearer $KALSHI_API_KEY` only if set |
| fixture | offline | none | loads any JSONL in store format (bundled data, simulator output, prior fetches) |

Polymarket and Kalshi collectors were written defensively but **could not
be exercised against the live APIs from the build sandbox** (only
api.manifold.markets is reachable); their parsers are tested against
recorded/canonical payloads and their failure mode is a clear
`VenueUnavailableError` with a remedy hint.

## Methods, in formulas

- Horizon panel: `p_h = price of the last point with ts <= T - h`,
  requiring `ts >= T - 2h` and `h <= T - created`.
- Brier `= mean((p - y)^2)`; log-loss with clipping at 1e-15.
- ECE `= sum_b (n_b/N) |rate_b - meanp_b|` over non-sparse equal-width
  bins; MCE is the max bin gap; Wilson 95% CI per bin.
- Murphy: Brier = REL - RES + UNC with
  `REL = sum_b (n_b/N)(meanp_b - rate_b)^2`,
  `RES = sum_b (n_b/N)(rate_b - ybar)^2`, `UNC = ybar(1 - ybar)`.
- Compression slope: WLS through the origin of `(rate_b - 0.5)` on
  `(meanp_b - 0.5)`, weights `n_b`, bins with `|meanp - 0.5| < 0.05`
  excluded; slope > 1 = compressed toward 0.5 (the Le 2026 direction),
  < 1 = overconfident. Market-level bootstrap CI (B = 1000).
- Favorite-longshot slope: Newton-Raphson logistic regression of `y` on
  `logit(p)`; slope < 1 means longshots overpriced relative to favorites.
- Climatology: Brier of the constant forecaster `p = ybar`; skill
  `= 1 - Brier / Brier_clim`.
- Correction: train/test split by resolution date (no leakage, tested);
  Platt `sigmoid(a + b*logit(p))`; isotonic via pool-adjacent-violators
  with clamped step interpolation; verdict from the bootstrap CI of
  out-of-sample delta-Brier.

## Caveats

- **Manifold is play-money** (Mana). Incentives differ from real-money
  venues; treat the bundled numbers as methodology demonstration, not as
  claims about real-money calibration.
- **Thin support is flagged, not hidden.** Real price distributions are
  U-shaped; at this sample size most horizons have 1-2 non-sparse
  equal-width bins, so ECE/slope rest on thin support (marked `*` in the
  digest and `thin_support` in the JSON). Wider bins, quantile binning, or
  more data change the picture.
- Polymarket outcomes are **inferred from terminal prices** (>= 0.98 YES,
  <= 0.02 NO); ambiguous terminals are excluded.
- The **category classifier is a keyword heuristic**; category rows with
  small n carry wide intervals.
- Carry-forward pricing with a staleness bound is an approximation of the
  price you could have traded at.
- All slices are reported with intervals; nothing is cherry-picked, so
  apply multiple-comparison discipline when reading across slices.
- **This is a measurement tool. Nothing here is investment advice, and
  historical calibration says nothing about any single future market.**

## Data sources & attribution

- `data/bundled/manifold_resolved_sample.jsonl`: 278 real resolved
  Manifold markets fetched from the public API on 2026-07-31 (278 markets,
  69,043 bets, downsampled to <= 200 series points per market). Data (c)
  Manifold Markets, used under their public API; redistributed for
  research demonstration. Full endpoint/filter/stride details in
  `data/bundled/SOURCES.md`.
- `fixtures/`: synthetic markets from `longshot.simulate` (calibrated seed
  7, compressed seed 11, 120 markets each).

## Development

```bash
pip install -e ".[dev]"
python -m pytest -q     # 63 tests, all offline, seeded
```

CI runs the suite on Python 3.11 and 3.12 (`.github/workflows/ci.yml`).
Layout: `src/longshot/` with `venues/` (one module per venue), `metrics`,
`bias`, `horizons`, `correct`, `analyze`, `report`, `simulate`, `cli`.
Judgment calls are logged in `DECISIONS.md`.

To add a venue: subclass `venues.base.VenueClient` (implement
`fetch_resolved()` yielding `MarketSeries`, raise
`VenueUnavailableError` with a remedy hint on network failure), decorate
it with `@register`, and add parser tests with a recorded payload.

## License

MIT. See [LICENSE](LICENSE). Copyright (c) 2026 Raul Tinajero Olivas.
