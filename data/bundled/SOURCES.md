# Bundled data sources

## manifold_resolved_sample.jsonl

Real data fetched from the Manifold Markets public API on **2026-07-31**
from the build machine (api.manifold.markets verified reachable).

- **Markets**: 278 resolved binary markets (120 YES, 158 NO)
- **Bets consumed**: 69,043 (before series downsampling)
- **Series points after downsampling**: 27,503
- **Resolution window**: 2022-08-01 to 2026-07-23 (resolved_ts range
  1659344288-1784740868)
- **Category mix**: politics 69, tech_ai 65, sports 45, macro 44,
  crypto 23, entertainment 16, other 16
- **API calls**: 355; fetch runtime ~15 minutes

### Endpoints used

- `GET https://api.manifold.markets/v0/search-markets?term={term}&filter=resolved&contractType=BINARY&sort=score&limit=100`
  with the 20 terms in `longshot.venues.manifold.SEARCH_TERMS`
  (fed, election, trump, bitcoin, crypto, nba, nfl, ai, openai, ukraine,
  china, inflation, oscars, spacex, climate, supreme court, gdp, apple,
  congress, world cup), deduplicated by market id.
- `GET https://api.manifold.markets/v0/bets?contractId={id}&limit=1000&before={last_bet_id}`
  paged descending, at most 4000 bets per market.

### Filters

Resolution in {YES, NO} (MKT/CANCEL excluded); volume >= 100;
uniqueBettorCount >= 10 when present; resolved - created >= 3 days;
non-empty bet history. When more than 250 markets were eligible, a seeded
(seed 42) uniform sample was drawn from the discovered pool (1456 eligible
markets; 280 sampled, 278 survived the bet-history filter).

### Series construction and downsampling

Series = `(market.createdTime, 0.5)` prior point plus
`(bet.createdTime, bet.probAfter)` for filled, non-cancelled,
non-redemption bets; milliseconds converted to seconds. Series were
downsampled to at most 200 points per market by uniform-in-index stride,
always keeping the first point, the last point, and every point within
2 hours of resolution (see `longshot.store.downsample_series`).

### Caveats and terms

Manifold is a play-money (Mana) market — incentives differ from real-money
venues; see the README caveats. Data (c) Manifold Markets, used under
their public API (https://manifold.markets/terms); this redistributed
sample is for research demonstration only. Provenance fields
(`fetched_at`, `source`, `api_base`) are embedded per market in the JSONL.
