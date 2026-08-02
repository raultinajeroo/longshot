# DECISIONS

Judgment calls made during the build, and why.

## Data model and panels

- **Creation-time prior (0.5).** Manifold lite-market payloads do not
  expose the opening probability, so every series is prepended with
  `(created_ts, 0.5)`. This makes long-horizon panels well-defined for
  every market; the price of the alternative (dropping markets whose first
  bet came late) was selection bias toward actively traded markets.
  Documented in `venues/manifold.py` and in each market's provenance notes.
- **Staleness bound.** A carried-forward price may be at most one horizon
  old (`ts >= T - 2h`). Without it, dormant markets contaminate short
  horizons with weeks-old prices. Cost: panel sizes shrink at short
  horizons (see the demo digest).
- **Store sorts rather than rejects unsorted series** (spec allowed
  either; §3 says "sorts series"), and drops empty-series markets instead
  of failing the whole file. Hard validation errors (price out of [0,1],
  outcome not in {0,1}, resolved <= created) raise `StoreError`.

## Statistics

- **min_per_bin = 30 default (per spec), with a thin-support flag.** Real
  resolved-market price distributions are U-shaped (mass near 0 and 1), so
  equal-width binning at this sample size leaves only 1-2 non-sparse bins
  at most horizons. Rather than silently change the estimator, every
  horizon entry carries `n_nonsparse_bins` and `thin_support` (< 3 bins),
  and the digest marks thin rows with `*`. Estimates are still computed
  exactly as specified.
- **Slope fits exclude bins with |meanp - 0.5| < 0.05** to keep the
  through-origin ratio estimator stable (spec §7). With thin support this
  can leave a single bin driving the slope; the bootstrap CI is still
  reported, and the thin-support flag is the honest signal.
- **Murphy decomposition is computed on the non-sparse-bin subset** so the
  identity REL - RES + UNC = Brier holds exactly (tested to 1e-12) on that
  subset; the subset base rate can therefore differ from the panel base
  rate.
- **Category stats use min_per_bin = 5** (categories are thin slices);
  NaN ECE/slope is reported as such, never hidden.
- **Bootstrap resamples whole markets** (not points) everywhere, because
  points from the same market are dependent. Correction delta-CIs resample
  test-panel points, which is equivalent at one point per market per
  horizon.
- **Seeds.** All demo/report numbers use seed 42 with 1000 bootstrap
  resamples; tests use smaller fixed seeds/resamples for speed. The
  simulator defaults (sigma = 0.35, gamma = 5) were tuned so the
  `calibrated` mode's compression-slope CI covers 1 at 30d and the
  `compressed` mode recovers the planted 1/0.55 slope at 30d.

## Venues

- **Manifold reservoir sampling.** When discovery finds more eligible
  markets than requested, a seeded uniform sample keeps the category mix
  representative (1456 eligible -> 280 sampled -> 278 with non-empty
  history).
- **Category classifier is a keyword heuristic**: multi-word keywords
  match as substrings, keywords of 3 chars or fewer match whole words only
  (so "ai" does not fire on "rain"), longer keywords match word prefixes
  (so "oscar" matches "Oscars"). Categories: politics, macro, crypto,
  sports, tech_ai, entertainment, else "other".
- **Polymarket ambiguous terminal prices are excluded** (YES price in
  (0.02, 0.98) at close). **Kalshi reads are keyless today**; the client
  adds `Authorization: Bearer $KALSHI_API_KEY` only if the env var is set.
- **Bundled series downsampling**: uniform-in-index stride to <= 200
  points, always keeping first/last and all points within 2h of
  resolution.

## Deviations from the spec

- `report` also writes a JSON sidecar (`report.html` -> `report.json`)
  because the spec asks the report path to "always emit the JSON analysis
  alongside" while also taking `--analysis` as input; the sidecar is a
  copy so the pair travels together.
- The demo writes `correction_bundled.json` and `report_bundled.json` in
  addition to the spec's listed artifacts (both are needed by the report
  and the digest).
- Demo analyzes the 120-market biased fixture with `min_per_bin = 10`
  (the default 30 leaves no non-sparse bins at that sample size).
- Correction skips horizons with < 20 train or test panel points (the
  1h horizon on the bundled sample) rather than fitting noise.
- Simulator: the final series point is pinned to the terminal belief
  (0.99/0.01) instead of applying the mode's bias transform there, since
  real markets converge to the outcome side; documented in `simulate.py`.
- Halawi et al. 2024 citation corrected to arXiv:2402.18563 per the
  research team's erratum.
