# Changelog

All notable changes to this project will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Both real-money collectors have now been exercised against their live
  APIs (2026-08-10). The venue table and the retrieval notes say what each
  one does and why the default ordering does not work on either.

### Fixed

- Correction training excludes outcomes that settled at or after the
  earliest test price observation at each horizon. Resolution ordering
  alone allowed look-ahead. Reports expose retained and removed counts,
  skipped horizons, and legacy results lacking the availability check.
- Isotonic calibration pools equal prices with their observation weights
  before PAVA, making predictions independent of tied rows' ordering.
- Empty requested horizon panels render an explanatory report instead of
  failing while drawing the reliability diagram.
- Correction on a one-market input reports exit code 4 with the minimum
  sample requirement instead of a traceback; existing output is preserved.
- **Kalshi collector returned zero markets.** It paged `/markets?status=settled`,
  which is dominated at every depth by markets too short-lived to produce
  two daily candles (median lifetime ~11 minutes over 6,000 measured rows),
  and discarded every one. Retrieval now walks the series catalogue and
  reads the historical tier, where the long-lived markets actually live.
- **Kalshi prices were 100x too small.** The historical candlesticks
  endpoint returns `price.close` as a decimal string already in dollars and
  omits `close_dollars` entirely, so the parser took its legacy
  integer-cents branch and divided dollars by 100. The type now determines
  the unit. The error passed every range check — a 100x-shrunk probability
  is still a probability — and was detectable only against outcomes:
  markets that settled YES had prices never exceeding 0.01.
- Kalshi candles now fall back to `price.previous` when `price.close` is
  null, which it is on any day a market did not trade. Long-lived political
  markets go quiet for days, and dropping those candles left holes at
  exactly the horizons the analysis measures.
- **HTTP 429 is now retried** with its own backoff ladder (5s, 15s, 45s)
  and honours `Retry-After`. It was previously grouped with other 4xx
  responses and abandoned immediately, so the documented retry behaviour
  did not apply to the one client error that means "try again".
- A tripped `--max-calls` guard no longer reports "this host may be blocked
  in your network". Budget exhaustion and an unreachable host are different
  events, and conflating them sent readers hunting a nonexistent fault.

### Added

- `longshot correct --bootstrap-unit resolution-day|resolution-week`
  resamples whole time blocks as an explicit dependence sensitivity check.
  Reports disclose the unit and group counts. The registered default remains
  market-level resampling; fewer than two test blocks cannot produce a CI.
- Direct pmwatch JSONL ingestion documented and tested at 1h and 5m
  horizons. Explicit price-estimator counts now appear in analysis and
  publication provenance, with midpoint and mixed-estimator notes in all
  human-readable reports.
- Pre-registered `analysis.yaml`: horizons, bins, min_per_bin, bootstrap,
  correction methods, and verdict rules declared before data is examined.
  `longshot analyze`/`correct` accept `--config`; explicit flags override.
- `longshot publish`: writes `report.html`, `report.json`,
  `README-summary.md`, and `provenance.json` (input sha256 hashes,
  parameters, per-venue status). Caveats are unavoidable in every
  artifact; Manifold-only inputs lead with a methodology-demonstration
  note.
- `longshot venues`: honest per-venue collector status (Manifold working +
  bundled; Polymarket/Kalshi parser-tested, marked not exercised live
  unless genuinely exercised in the run).
- `examples/demo/headline.md`: the two honest findings from the bundled
  sample and what generalizing them requires.
- CI step verifying the LICENSE file exists and is non-empty.
- `ROADMAP.md` for not-yet-implemented directions (kept out of README claims).
- `docs/IMPACT.md` stating honestly which metrics are and are not collected.
- GitHub issue templates for bug reports and feature requests.
- This changelog.

### Changed

- Runtime dependencies now include `pyyaml` (for `analysis.yaml`).

## [0.1.0] - 2026-08-01

### Added

- Initial public release: horizon panels, calibration metrics (Brier,
  log-loss, ECE/MCE, Murphy decomposition), bias measures (compression
  slope, favorite-longshot slope, category effects), Platt/isotonic
  correction layer evaluated out-of-sample, single-file HTML report,
  Manifold collector plus bundled sample, fixture venue, and a seeded
  synthetic-market simulator. Offline demo via `longshot demo`.
