# Changelog

All notable changes to this project will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- CI step verifying the LICENSE file exists and is non-empty.
- `ROADMAP.md` for not-yet-implemented directions (kept out of README claims).
- `docs/IMPACT.md` stating honestly which metrics are and are not collected.
- GitHub issue templates for bug reports and feature requests.
- This changelog.

## [0.1.0] - 2026-08-01

### Added

- Initial public release: horizon panels, calibration metrics (Brier,
  log-loss, ECE/MCE, Murphy decomposition), bias measures (compression
  slope, favorite-longshot slope, category effects), Platt/isotonic
  correction layer evaluated out-of-sample, single-file HTML report,
  Manifold collector plus bundled sample, fixture venue, and a seeded
  synthetic-market simulator. Offline demo via `longshot demo`.
