# MicroRCA Provenance

- Upstream repository: <https://github.com/elastisys/MicroRCA>
- Upstream HEAD checked by `git ls-remote` on 2026-06-05:
  `de334b2815742a248225cf6e242840a9d5fedc9a`
- Paper: "MicroRCA: Root Cause Localization of Performance Issues in
  Microservices", NOMS 2020, DOI: <https://doi.org/10.1109/NOMS47738.2020.9110353>

## Adaptation status

This directory contains a clean-room runner, not a verbatim vendored copy. The
upstream code is path-bound to its original CSV prefixes; UniSage needs a runner
that can consume scratch directories generated per sampled trace subset.

The runner preserves the experiment-relevant defaults:

- `alpha=0.55`
- `ad_threshold=0.045`
- `smoothing_window=12`

It consumes Nezha-lineage Hipster / Train Ticket data materialized by
`microrca_adapter.py`. AIOps21 is intentionally left as exploratory because its
metric vocabulary and topology format are not the published MicroRCA artifact's
input shape.

