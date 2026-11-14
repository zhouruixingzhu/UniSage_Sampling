# TraceRCA Provenance

- Upstream repository: <https://github.com/NetManAIOps/TraceRCA>
- Vendored commit (full source tree, `.git` stripped):
  `b9087c72d201fbb17d724114cd389de4854815ad`
  (verified via `git ls-remote` HEAD on 2026-06-22).
- Paper: Li, Zeyan, et al. "Practical Root Cause Localization for Microservice
  Systems via Trace Analysis." IEEE/ACM International Symposium on Quality of
  Service (IWQoS) 2021.
- License: the upstream repository ships no LICENSE file. The code is vendored
  here for academic reproduction only; cite the IWQoS'21 paper.

## What this directory is

The byte-for-byte upstream Python source as of the pinned commit, plus a thin
single-case runner `tracerca_single_case.py` added by UniSage. The upstream
study dataset (OneDrive / Tsinghua Cloud links in `README.md`) is **not**
vendored — UniSage feeds its own Hipster / Train Ticket / AIOps21 traces through
the adapter instead.

At this commit the only localization method present is association rule mining
(`run_localization_association_rule_mining_20210516.py`) — the paper's main
"Ours" method. The PageRank / RCSF / MEPFL / microscope baselines referenced in
the `Makefile` are not in this commit and are not needed.

## Adaptation status (clean-room runner, verbatim core)

Following the same convention as `baselines/microrca`: the
**localization core is reused verbatim**, the
anomaly-detection stage is reproduced faithfully with disclosed adaptations.

- **Localization (verbatim).** `tracerca_single_case.py` imports the upstream
  `TraceRCA` class from `run_localization_association_rule_mining_20210516.py`
  unmodified and calls it with the published defaults
  (`min_support_rate=0.1`, `k=100`, `enable_PRFL=False`). This is the paper's
  actual contribution and is byte-identical.
- **Invocation k-sigma AD (faithful re-implementation).** The runner reproduces
  `anomaly_detection_3sigma` from `run_anomaly_detection_invo.py`
  (`predict = max_f(|x_f - mean_f| > SIGMA*std_f)`, missing-token fallback to
  empirical stats) and the reference stats from
  `run_anomaly_detection_prepare_model.py` (`std = max(std, 0.1)`). Re-implemented
  rather than imported because upstream's vectorised `df.loc[(s,t), feats].values`
  assumes more than one row per edge; that breaks on the single-row edges that
  5%-budget sampling produces. Numeric behaviour is unchanged on multi-row edges.
  `SIGMA` = upstream Makefile default `1`.
- **Fisher feature selection (verbatim formula).** `stderr_criteria` from
  `run_selecting_features.py` is copied into the runner with a provenance
  comment, to avoid importing `seaborn`/`tqdm` at module load (those are only
  used for the upstream debug plots). Threshold = `FISHER` (Makefile default
  `3`).
- **Invocation construction (clean-room).** Upstream `run_invo_encoding.py`
  expects its own raw pickle schema and asserts every service is in the
  TT-specific `INVOLVED_SERVICES`. UniSage builds the invocation DataFrame
  directly from its own trace CSVs in `tracerca_adapter.py`
  (parent->child span join -> `source`/`target` services, `latency` from span
  duration), so the TT-specific encoder and its asserts are bypassed.

## Disclosed divergences (record in the paper's threats/footnotes)

1. **Feature set.** Upstream `trainticket_config.FEATURE_NAMES` defaults to
   `['latency', 'http_status']` and is TT-tuned. UniSage's traces carry latency
   reliably but not HTTP status, so the runner uses `['latency']` by default
   (an HTTP-status column, if absent, is constant and never flags anyway). This
   makes Fisher feature selection effectively a latency on/off decision, which
   is the dominant TraceRCA signal.
2. **`forbidden_names`.** Upstream hardcodes `frozenset({"gateway"})` (the Istio
   gateway is the ubiquitous TT entry node and never an injected service). For
   Hipster the analogous entry `frontend` **can** be the injected service, so
   forbidding it would zero out valid ground truth. The runner therefore makes
   `forbidden_names` a per-dataset parameter, defaulting to empty, and only sets
   `{"gateway"}` for Train Ticket to match upstream.
3. **Fresh dependency versions.** The env (`environment.yml`) is built from
   recent conda-forge/PyPI wheels (mlxtend, pandas, numpy, scikit-learn), not
   the original 2021 pins, which are unknown. `fpgrowth` is deterministic, so
   this should not change rankings, but it is unverified against the original
   lockfile.

## Environment

Dedicated conda env `tracerca` (Python 3.9). See `environment.yml`. Minimal
runtime deps actually exercised by the runner: `pandas, numpy, scikit-learn,
scipy, mlxtend, diskcache, loguru, click`. (`seaborn`, `tqdm`, `imblearn` are
imported only by upstream scripts the runner does not load.)
