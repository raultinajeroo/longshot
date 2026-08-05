# Headline: what the bundled sample actually shows

Two honest findings from the committed demo
(`examples/demo/analysis_bundled.json`; 278 resolved Manifold markets,
play-money Mana, fetched 2026-07-31 — a bundled sample, not live data).

**1. This Manifold sample is close to calibrated.**
Compression slopes sit within ~8% of 1.0 at every measured horizon
(30d: 1.059 [0.97, 1.11]; 1d: 1.062 [1.05, 1.08]), and Brier skill vs the
climatology baseline runs 0.41–0.66. Prices here are neither strongly
over- nor underconfident. Caveat: most horizons have 1–2 non-sparse bins,
so these slopes rest on thin support (flagged `*` in the report).

**2. The correction layer cannot beat the raw market price out-of-sample.**
Platt and isotonic corrections, fit on the earliest 60% of markets by
resolution date and evaluated on the rest, show **no reliable improvement**
at 11 of 12 horizon/method pairs; the twelfth (1d isotonic) is a
**reliable degradation**. The pipeline reports that verdict as computed;
it is never re-tuned until the answer looks better. The sanity anchor
runs the other way: on a synthetic fixture with planted compression
(c = 0.55), the same pipeline recovers slope 1.78 [1.40, 2.19] against
the expected 1/0.55 = 1.82 — so "no improvement found" is not a dead
detector.

## What would be needed to generalize these findings

- **Real-money venues.** Manifold is play money. The Polymarket and Kalshi
  collectors exist and are parser-tested against recorded payloads, but
  have never been exercised against their live APIs as of this writing;
  until a live fetch is committed, these findings say nothing about
  real-money calibration.
- **More data per horizon.** Thin support means wider intervals and
  excluded bins. A sample an order of magnitude larger (or quantile
  binning) would firm up the slope estimates, especially at 12h/1h.
- **Pre-registered replication.** The questions and parameters are
  declared in `analysis.yaml` before seeing data; generalization means
  running exactly that configuration on a new venue or sample and
  reporting whatever verdicts come out — including "no reliable
  improvement".
- **Independent outcomes.** Polymarket outcomes would be inferred from
  terminal prices (>= 0.98 YES / <= 0.02 NO) unless a settlement source is
  wired in; that approximation needs validation before strong claims.
