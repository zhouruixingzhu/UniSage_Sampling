# MicroRCA Runner

This directory contains a clean-room, adapter-friendly runner for:

> MicroRCA: Root Cause Localization of Performance Issues in Microservices.

The upstream artifact is at <https://github.com/elastisys/MicroRCA>. UniSage does
not vendor the upstream script here; the runner keeps the published algorithm's
core structure (BIRCH anomaly detection over invocation latency, attributed
service graph, PageRank ranking) while accepting materialized CSVs from
`microrca_adapter.py`.

Default parameters are fixed to the upstream defaults used in the paper:

- `alpha = 0.55`
- `ad_threshold = 0.045`
- `smoothing_window = 12`

The adapter writes a case directory with:

- `latency_source.csv`
- `latency_destination.csv`
- `dependency.csv`
- `metrics/*.csv`
- `case_meta.json`

Run one materialized case:

```bash
python baselines/microrca/microrca_single_case.py --case-dir /path/to/case
```

