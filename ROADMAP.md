# Roadmap

Not-yet-implemented directions. Nothing on this page is a claim about
what longshot does today; the README describes current behavior.

## Near term

- Exercise the Polymarket and Kalshi collectors against their live APIs
  outside the build sandbox, and record real fetch fixtures from them
  (parsers are currently unit-tested against canonical payloads only).
- Larger bundled samples per venue, so equal-width bins stop being sparse
  at most horizons (thin support is currently flagged, not hidden).
- Quantile binning as an alternative to equal-width bins for calibration
  tables.

## Later

- Additional venues behind the `venues.base.VenueClient` interface.
- Per-category and per-venue replication reports across multiple venues
  in one HTML file.
- Optional persistence of analysis runs for comparing calibration across
  fetch dates.

## Explicitly out of scope

- Trading, order placement, or investment advice. longshot is a
  read-only research/measurement tool.
- Forecasting. longshot measures existing market prices; it does not
  produce forecasts.
