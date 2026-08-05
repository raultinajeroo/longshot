# Changelog

All notable changes to this project will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
