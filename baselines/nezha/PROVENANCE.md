# Nezha — vendored baseline provenance

Vendored copy of **Nezha** (FSE'23), used as a *downstream multimodal RCA evaluator*
for UniSage's sampling-quality experiments.

- **Upstream:** https://github.com/IntelligentDDS/Nezha
- **Commit:** `d8140101fdb4e7dfb60d3ef9f64706f382b68470` (2025-05-20, "Merge pull request #15")
- **Paper:** Yu, Chen, Li et al., "Nezha: Interpretable Fine-Grained Root Causes Analysis
  for Microservices on Multi-Modal Observability Data", FSE 2023.
- **License:** see `LICENSE` (upstream license, retained verbatim).
- The upstream `.git` was removed after vendoring; this directory is a plain snapshot.

## What is / isn't tracked in the UniSage repo

Tracked (small, needed to run): the `*.py` source, `log_template/` (drain3 `.ini` configs
+ `hipster.bin` / `ts.bin` template snapshots), `metric_threshold/` (precomputed per-pod
metric thresholds), and the upstream docs (`README.md`, `INSTALL.md`, `STATUS.md`,
`LICENSE`, `requirements.txt`).

**Gitignored** (large; re-fetchable by re-cloning upstream): `construct_data/` (~70 MB,
fault-free baseline), `rca_data/` (~2.8 GB, fault-suffering data), `log/` (runtime logs),
`FSE2023_Nezha.pdf`. See `../../.gitignore`.

To restore the data on a fresh checkout:
```
git clone https://github.com/IntelligentDDS/Nezha.git /tmp/nezha_upstream
cp -r /tmp/nezha_upstream/construct_data /tmp/nezha_upstream/rca_data baselines/nezha/
```

## Dataset lineage (key finding)

UniSage's Hipster and Train Ticket datasets are derived from **this very Nezha dataset**:
identical trace schema (`TraceID,SpanID,ParentID,PodName,OperationName,StartTimeUnixNano,
EndTimeUnixNano,Duration`), identical per-minute `HH_MM_trace.csv` file naming, identical
ground-truth keys (`inject_time,inject_timestamp,inject_pod,inject_type`), and the same
fault campaign (e.g. Nezha `2022-08-22` `inject_timestamp 1661140434` == UniSage Hipster
`batch1/03_53`). Nezha additionally ships the per-pod **metric** files that UniSage's
`metric/` dirs lack — these are recovered from here for the multimodal RCA evaluation.

## Environment

Runs in its own isolated conda env `nezha` (Python 3.11), invoked as a **subprocess** from
the UniSage harness (never imported into the `unisage` env — incompatible pins).

```
# NOTE: create with --override-channels -c https://repo.anaconda.com/pkgs/main because the
# machine's ~/.condarc points at a broken USTC mirror (malformed https// URL).
conda create -n nezha python=3.11 -y --override-channels -c https://repo.anaconda.com/pkgs/main
/opt/anaconda3/envs/nezha/bin/python -m pip install drain3==0.9.10 more_itertools psutil PyYAML tqdm matplotlib numpy pandas
```

Python 3.11 (not 3.9): the conda `pkgs/main` channel ships a `pip` (26.1.1) that uses
`@dataclass(slots=True)`, which only parses on Python 3.10+, so a 3.9 env's pip is dead on
arrival. 3.11 sidesteps that and runs Nezha fine.

`drain3` is pinned to **0.9.10** (the upstream pin) so the persisted `log_template/*.bin`
template snapshots load with a matching persistence format — template-id stability is
required for faithful AS@k/AIS@k reproduction. `numpy`/`pandas` are intentionally left
modern: the code uses no APIs removed since pandas 0.23 (verified — all `.append` calls are
`list.append`; no `.ix`/`pd.np`/`as_matrix`/`.iteritems`).

## Port notes

Upstream `requirements.txt` pins Python 3.6 + `numpy==1.15.4` / `pandas==0.23.4`, which do
not build on macOS arm64. INSTALL.md itself states "any python3 version should be fine".
Pinned `numpy==1.26.4` / `pandas==1.5.3` (the closest-to-upstream `1.15.4`/`0.23.4`
versions that build on py3.11) for fidelity — the default fresh install pulled
numpy 2.4 / pandas 3.0, which both crashes (see below) and risks behavioral drift.

Code changes applied to the vendored copy (kept minimal, each marked `[UniSage port]`):

- `alarm.py` `get_netwrok_metric`: `float(pod_reader.iloc[0])` -> `float(pod_reader.iloc[0, 0])`.
  `pod_reader` is a 1-column frame; under pandas >= 2 `.iloc[0]` is a Series and `float()`
  rejects it. Cell-selection is behavior-identical on old pandas. (Only triggered on the
  network-metric fallback path, so it surfaced ~45 faults into the first run.)

## Phase-1 reproduction status (PASS)

Run in env `nezha` (py3.11, pandas 1.5.3) from this directory:
`python ./main.py --ns <hipster|ts> --level <service|inner>`. Results land in
`log/<date>_nezha.log`. Reproduction vs upstream INSTALL.md targets:

| run | metric | upstream | reproduced |
|---|---|---|---|
| hipster service | AS@1 / AS@3 / AS@5 | 92.86 / 96.43 / 96.43 (56 faults) | **92.857 / 96.429 / 96.429 — EXACT** |
| ts service | AS@1 / AS@3 / AS@5 | 86.67 / 97.78 / 97.78 (45 faults) | **86.667 / 97.778 / 97.778 — EXACT** |
| hipster inner | AIS@1 / AIS@3 / AIS@5 | 92.86 / 96.43 / 96.43 | **92.857 / 96.429 / 96.429 — EXACT** |
| ts inner | AIS@1 / AIS@3 / AIS@5 | 86.67 / 97.78 / 97.78 | **86.667 / 97.778 / 97.778 — EXACT** |

**Phase-1 verdict: 4 / 4 reproductions EXACT.** The vendored copy + the one
`[UniSage port]` edit + py3.11 / pandas 1.5.3 / drain3 0.9.10 pins produce
upstream-faithful results across both datasets and both granularities.

Full hipster-service run ≈ 39 min wall (56 faults, ProcessPoolExecutor 64 workers) — the
per-fault cost (~40 s) drives Phase-3 sweep planning.
