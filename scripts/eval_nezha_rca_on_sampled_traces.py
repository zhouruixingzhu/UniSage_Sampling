"""Nezha (FSE'23) as a downstream multimodal RCA evaluator on baseline-sampled traces.

Companion to `scripts/eval_fair_rca_on_sampled_traces.py`. That harness holds UniSage's
internal trace-only RCA constant across samplers; this one swaps the RCA for
Nezha — a SOTA multimodal RCA that fuses logs + metrics + traces into event
graphs and ranks suspicious patterns. Both harnesses share the same retention
semantics (`native` per-budget Decision subset vs `topn` equal-N by score), so
their AC@1 / AC@3 / MRR numbers are directly comparable per (case, sampler,
budget, retention).

Three input-fidelity modes:

  Mode A -- trace-isolated. The retained TraceID set IS the variable; the
            `traceid` file Nezha iterates over is rewritten, while
            `trace/log/metric/` symlink full Nezha data. The apples-to-apples
            row against trace-only evaluators -- the only variable is which
            traces the downstream RCA sees.
  Mode B -- trace-derived multimodal (symmetric across samplers). Also filters
            `trace/<HH>_<MM>_trace.csv` and `log/<HH>_<MM>_log.csv` to
            retained-trace spans. Every sampler -- UniSage AND baselines --
            gets logs derived by the SAME rule: "logs whose SpanID belongs to
            a retained-trace span". Metrics stay full (UniSage has no metric
            sampler; per-window metric files already scope the time range).
            Symmetric and fair: tests whether the trace subset preserves
            downstream multimodal RCA under a uniform log-carry-forward rule.
  Mode C -- end-to-end multimodal (asymmetric BY SAMPLER, by design). For
            UniSage rows, logs come from UniSage's own
            `log_sampling_result.csv` (top-N by `LogSamplingProb` at the same
            budget as traces; SpanID recovered from the log body). Baseline
            rows fall back to Mode-B trace-derived logs (baselines have no
            separate log output, so their "natural multimodal carry-over" is
            logs-of-kept-traces). This represents each sampler's ACTUAL
            multimodal output; not a controlled comparison, but the
            deployment-realistic view. The mechanism must be made explicit in
            the paper.

Reuses building blocks from `scripts/eval_fair_rca_on_sampled_traces.py`:
`RANKING_COLUMNS`, `DECISION_PREFIXES`, `stable_uniform_score`, `budget_token`,
`as_bool`, `is_hipster_layout`, `GroundTruth`, `load_run_index`, `read_sample`,
`retained_ids`.

Nezha invocation goes via `nezha_adapter`: materialize a scratch `rca_data` dir
with the filtered `<HH>_<MM>_traceid.csv` (and trace/log files for Mode B),
then invoke `baselines/nezha/nezha_single_case.py` in the isolated `nezha`
conda env (py3.11 + pandas 1.5.3 + drain3 0.9.10). Per-cell wall time is
~30-60s, so this harness CACHES Nezha outputs by (case_key, mode,
frozenset(retained_ids)) -- many (sampler, budget) cells collapse onto the same
retained set, especially at budget=1.0 (the upper-bound invariant: every
sampler keeps all traces -> identical full-data Nezha ranking, the harness
correctness check).

Scope: Hipster + Train Ticket only. AIOps21 is documented out-of-scope in
`paper_impact/022_nezha_downstream_rca.md` §6 (zero log-trace linkage; schema
+ vocabulary mismatch; no `root_cause_aiops21.json`).

Prerequisites (one-off):

  1. Conda env `nezha` (py3.11, pandas 1.5.3, drain3 0.9.10) — see
     `baselines/nezha/PROVENANCE.md`. The named environment is discovered automatically;
     specify `--nezha-python /path/to/nezha/bin/python` or `NEZHA_PYTHON` if needed.
  2. `python -m baselines.nezha.adapter build-index` — produces
     `results/nezha/nezha_case_index.csv`.

Run from `UniSage/` in the `unisage` conda env.

Example A — smoke (a few cases, validates wiring, ~5 min/case-mode-budget):

    python scripts/eval_nezha_rca_on_sampled_traces.py \\
        --unisage results/v02_sampling_runs/unisage_trace_hip_tt_primary_target_raw_prop0_v02 \\
        --sieve  results/v02_sampling_runs/sieve_v02_hipster_train_ticket_primary_seed-1_FINAL \\
        --add-uniform --budgets 0.05 1.0 --modes A --retention topn \\
        --cases 03_53,04_52 --limit 2 \\
        --out results/nezha/hip_tt_seed1_smoke

Example B — full Hipster+TT Mode A sweep (publication numbers, ~20-30 h wall):

    python scripts/eval_nezha_rca_on_sampled_traces.py \\
        --unisage     results/v02_sampling_runs/unisage_trace_hip_tt_primary_target_raw_prop0_v02 \\
        --sifter      results/v02_sampling_runs/sifter_v02_hipster_train_ticket_primary_seed-1_FINAL \\
        --sieve       results/v02_sampling_runs/sieve_v02_hipster_train_ticket_primary_seed-1_FINAL \\
        --tracepicker results/v02_sampling_runs/tracepicker_v02_hip_tt_primary_portable_seed1_jsd \\
        --trastrainer <trastrainer_hip_tt_run_dir>  \\
        --add-uniform --budgets 0.01 0.025 0.05 0.1 0.2 1.0 --modes A --retention both \\
        --out results/nezha/hip_tt_seed1_mode_a_full

Example C — full Mode A + Mode B sweep (~40-60 h wall; add the multimodal row):

    # same as B but `--modes A B` and a separate --out dir.

Run from `nohup` / `screen` for long runs. Intermediate per-case CSV is
flushed after every case so a mid-sweep kill keeps work.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the existing harness's primitives so retention semantics stay identical.
from eval_fair_rca_on_sampled_traces import (
    DECISION_PREFIXES,
    MANIFEST_AUDIT_COLUMNS,
    RANKING_COLUMNS,
    GroundTruth,
    add_uniform_random_scores,
    filter_sample_to_manifest,
    is_hipster_layout,
    load_trace_manifest_ids,
    load_run_index,
    parse_manifest_index_arg,
    read_sample,
    retained_ids,
)

from baselines.nezha import adapter as nza


# --------------------------------------------------------------------------- #
# Ranking metrics
# --------------------------------------------------------------------------- #

def _first_pod_rank(ranked: list[dict], gt_pod: str) -> int:
    """1-indexed position of the first item whose `pod` matches `gt_pod`.

    Returns 0 if no match. This is the UniSage convention used by
    `eval_fair_rca_on_sampled_traces.agg()` so Nezha rows are directly
    comparable to trace-only evaluator rows.
    """
    for i, r in enumerate(ranked, 1):
        if r.get("pod") == gt_pod:
            return i
    return 0


def _nezha_native_rank(ranked: list[dict], gt_pod: str, inject_type: str,
                       ns: str, root_cause_dict: dict) -> int:
    """Replicate Nezha's tie-collapsed AS@k rank for one fault.

    From upstream `pattern_ranker.evaluation_pod`: walk `ranked`, increment
    `topk` only when (score, depth) changes from the previous item; on a match,
    return the current `topk`. The match condition mirrors upstream's two
    branches:
      * `len(root_cause) == 1` (resource_type root cause): require both
        `inject_pod` in `item.pod` AND the resource_type appears in
        `item.resource`.
      * `len(root_cause) == 2` (event-pair root cause): require `inject_pod`
        in `item.pod` (the event-pair template check needs the drain3 miner so
        is omitted here -- this is a slight upper bound on the rank for that
        case shape; the AC@k metric is unaffected).
    Returns 0 if no match within the result list.
    """
    inject_service = gt_pod.rsplit('-', 1)[0].rsplit('-', 1)[0]
    if inject_service not in root_cause_dict or inject_type not in root_cause_dict[inject_service]:
        return 0
    root_cause = str(root_cause_dict[inject_service][inject_type]).split("_")

    topk = 1
    for i, r in enumerate(ranked):
        if i > 0:
            prev = ranked[i - 1]
            if not (prev.get("score") == r.get("score") and prev.get("depth") == r.get("depth")):
                topk += 1
        pod_ok = gt_pod in str(r.get("pod") or "")
        if len(root_cause) == 1:
            res = str(r.get("resource") or "")
            if res and (root_cause[0] in res) and pod_ok:
                return topk
        else:
            if pod_ok:
                return topk
    return 0


# --------------------------------------------------------------------------- #
# Case orchestration
# --------------------------------------------------------------------------- #

def load_case_index(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # case_key matches the format used by load_run_index
    df["case_key"] = df["dataset"] + "/" + df["batch"].astype(str) + "/" + df["case"].astype(str)
    return df


def load_root_cause_dict(nezha_construct_dir: Path, ns: str) -> dict:
    rc_path = nezha_construct_dir / f"root_cause_{ns}.json"
    return json.loads(rc_path.read_text()) if rc_path.exists() else {}


def universe_trace_ids(sample: pd.DataFrame) -> list[str]:
    return sample["TraceID"].astype(str).tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    for name in ("sifter", "sieve", "tracepicker", "trastrainer", "unisage", "gleaner"):
        ap.add_argument(f"--{name}", type=str, default=None, help=f"{name} v0.2 run dir")
    ap.add_argument("--add-uniform", action="store_true",
                    help="add deterministic-hash uniform sampler (single shared sample frame)")
    ap.add_argument(
        "--uniform-seed",
        type=int,
        default=None,
        help="seed for RQ1-compatible Uniform random scores; omit for legacy TraceID-only hash",
    )
    ap.add_argument("--retention", choices=["native", "topn", "both"], default="both")
    ap.add_argument("--budgets", type=float, nargs="+",
                    default=[0.01, 0.025, 0.05, 0.1, 0.2, 1.0])
    ap.add_argument("--modes", nargs="+", choices=["A", "B", "C"], default=["A", "B"],
                    help="A=trace-isolated; B=trace-derived multimodal (symmetric); "
                         "C=end-to-end multimodal (UniSage uses log_sampling_result.csv, "
                         "baselines fall back to B)")
    ap.add_argument("--unisage-log-sampling-name", default="log_sampling_result.csv",
                    help="filename of UniSage's per-case log subset for Mode C "
                         "(found under data/<dataset>/batches/<batch>/<case>/)")
    ap.add_argument("--mode-c-log-policy", choices=["native", "rca_support"], default="native",
                    help="Mode-C UniSage log policy: native uses log_sampling_result.csv; "
                         "rca_support keeps raw logs linked to retained trace spans.")
    ap.add_argument("--data-root", type=str, default=".")
    ap.add_argument(
        "--trace-manifest-index",
        type=parse_manifest_index_arg,
        default=Path("results/label_protocol_v02/trace_manifest_index.csv"),
        help="v0.2 canonical trace manifest index; set to none/off to disable filtering",
    )
    ap.add_argument("--nezha-rca-dir", default="baselines/nezha/rca_data")
    ap.add_argument("--nezha-construct-dir", default="baselines/nezha/construct_data")
    ap.add_argument("--case-index", default="results/nezha/nezha_case_index.csv",
                    help="output of `python -m baselines.nezha.adapter build-index`")
    ap.add_argument("--nezha-python", default=None,
                    help="isolated nezha interpreter (or NEZHA_PYTHON; auto-discover named conda env)")
    ap.add_argument("--dataset-filter", type=str, default=None,
                    help="comma list, e.g. hipster,train_ticket")
    ap.add_argument("--cases", type=str, default=None,
                    help="comma list of case names (smoke)")
    ap.add_argument("--limit", type=int, default=None, help="cap number of cases (smoke)")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--timeout-sec", type=int, default=900,
                    help="per Nezha subprocess timeout")
    ap.add_argument("--keep-scratch", action="store_true",
                    help="don't delete scratch dirs after each case (debugging)")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip cases that already have rows in nezha_rca_per_case.csv; resume by case granularity.",
    )
    args = ap.parse_args()
    try:
        args.nezha_python = nza.resolve_nezha_python(args.nezha_python)
    except ValueError as exc:
        ap.error(str(exc))

    data_root = Path(args.data_root).resolve()
    nezha_rca_dir = Path(args.nezha_rca_dir).resolve()
    nezha_construct_dir = Path(args.nezha_construct_dir).resolve()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch_root = (out_dir / "_scratch").resolve()

    # Discover provided sampler runs (subset of the eval_fair_rca CLI).
    run_dirs = {n: getattr(args, n) for n in
                ("sifter", "sieve", "tracepicker", "trastrainer", "unisage", "gleaner")}
    run_dirs = {n: Path(p) for n, p in run_dirs.items() if p}
    if not run_dirs:
        ap.error("provide at least one sampler run dir")

    # Logical samplers each run contributes (UniSage run unfolds to multiple).
    run_samplers = {}
    for n in run_dirs:
        if n == "unisage":
            run_samplers[n] = [
                "unisage_combined",
                "unisage_edge",
                "unisage_analysis",
                "unisage_dynamic_pattern_guard",
            ]
        else:
            run_samplers[n] = [n]

    # Index every run, build case scope from union (status==ok).
    indices = {n: load_run_index(d) for n, d in run_dirs.items()}
    case_meta: dict[str, dict] = {}
    sample_paths: dict[str, dict[str, Path]] = {}
    for n, idx in indices.items():
        for _, row in idx.iterrows():
            ck = row["case_key"]
            case_meta.setdefault(ck, dict(
                dataset=row["dataset"], batch=str(row["batch"]), case=str(row["case"]),
                case_dir=Path(row["case_dir"]),
            ))
            sample_paths.setdefault(ck, {})[n] = Path(row["sample_path"])

    case_idx = load_case_index(Path(args.case_index))
    case_idx_by_key = {row["case_key"]: row for _, row in case_idx.iterrows()}

    # Apply filters / scope to Hipster + Train Ticket (Nezha-supported datasets only).
    nezha_supported = set(case_idx["case_key"])
    case_keys = sorted(k for k in case_meta if k in nezha_supported)
    if args.dataset_filter:
        keep = set(args.dataset_filter.split(","))
        case_keys = [ck for ck in case_keys if case_meta[ck]["dataset"] in keep]
    if args.cases:
        wanted = set(args.cases.split(","))
        case_keys = [ck for ck in case_keys if case_meta[ck]["case"] in wanted]
    if args.limit:
        case_keys = case_keys[: args.limit]

    manifest_ids_by_case = load_trace_manifest_ids(args.trace_manifest_index, data_root, set(case_keys))
    manifest_audit_rows: list[dict] = []

    modes = list(args.modes)
    retention_modes = ["native", "topn"] if args.retention == "both" else [args.retention]
    gt_resolver = GroundTruth(data_root)
    # cache (root_cause_dict) by ns
    rc_cache: dict[str, dict] = {}

    per_case_rows: list[dict] = []
    run_log: list[dict] = []
    # Resume support: skip cases whose per-case rows already exist.
    done_cases: set[tuple[str, str, str]] = set()
    if getattr(args, "resume", False):
        prior_csv = out_dir / "nezha_rca_per_case.csv"
        if prior_csv.exists():
            try:
                prior_df = pd.read_csv(prior_csv)
                per_case_rows = prior_df.to_dict("records")
                done_cases = {(str(r["dataset"]), str(r["batch"]), str(r["case"]))
                              for r in per_case_rows}
                print(f"[resume] loaded {len(per_case_rows)} prior rows, {len(done_cases)} done cases", flush=True)
            except Exception as exc:
                print(f"[resume] failed to load prior per-case csv: {exc}", flush=True)
        prior_log = out_dir / "nezha_run_log.csv"
        if prior_log.exists():
            try:
                run_log = pd.read_csv(prior_log).to_dict("records")
            except Exception:
                pass
    t_start_all = time.time()

    for ci, ck in enumerate(case_keys, 1):
        meta = case_meta[ck]
        dataset, case = meta["dataset"], meta["case"]
        if (str(dataset), str(meta["batch"]), str(case)) in done_cases:
            print(f"[{ci}/{len(case_keys)}] {ck}  [resume: skip done]", flush=True)
            continue
        case_dir = data_root / meta["case_dir"]
        gt = gt_resolver.inject_service(dataset, case, case_dir)
        if not gt:
            print(f"[{ci}/{len(case_keys)}] {ck}: NO_GT, skip", flush=True)
            continue
        if not is_hipster_layout(dataset):
            print(f"[{ci}/{len(case_keys)}] {ck}: AIOPS21_OUT_OF_SCOPE, skip", flush=True)
            continue

        nz_row = case_idx_by_key[ck]
        entry = nza.CaseEntry(
            dataset=dataset, batch=meta["batch"], case=case, ns=nz_row["ns"],
            nz_date=nz_row["nz_date"], nz_window=nz_row["nz_window"],
            inject_time=nz_row["inject_time"], inject_pod=nz_row["inject_pod"],
            inject_type=nz_row["inject_type"],
            inject_timestamp=str(nz_row["inject_timestamp"]),
        )
        if entry.ns not in rc_cache:
            rc_cache[entry.ns] = load_root_cause_dict(nezha_construct_dir, entry.ns)

        # Load every sample frame once per case.
        canonical_ids = manifest_ids_by_case.get(ck)
        loaded: dict[str, pd.DataFrame] = {}
        for run_name in run_dirs:
            sp = sample_paths.get(ck, {}).get(run_name)
            if sp is None:
                continue
            sample = read_sample(data_root / sp)
            if canonical_ids is not None:
                sample, audit = filter_sample_to_manifest(
                    sample,
                    canonical_ids,
                    dataset=dataset,
                    batch=str(meta["batch"]),
                    case=case,
                    run_name=run_name,
                )
                manifest_audit_rows.append(audit)
            loaded[run_name] = sample
        if not loaded:
            print(f"[{ci}/{len(case_keys)}] {ck}: NO_SAMPLE_FILES, skip", flush=True)
            continue
        any_sample = next(iter(loaded.values()))

        # Per-case Nezha cache. Key includes log_source so Mode C with
        # UniSage's log subset is a distinct cell from Mode C / Mode B with
        # trace-derived logs at the same retained_ids.
        # log_source is None (Modes A/B and Mode-C fallback) or the resolved
        # path to UniSage's log_sampling_result.csv plus the log budget.
        rank_cache: dict[tuple, list] = {}

        # Resolve UniSage log subset path for this case (used in Mode C only).
        unisage_log_csv = (case_dir / args.unisage_log_sampling_name)
        if not unisage_log_csv.exists():
            unisage_log_csv = None

        def get_ranking(mode: str, ids: frozenset,
                        unisage_log_path: Path | None, log_budget: float | None,
                        log_policy: str) -> list:
            cache_log_key = (str(unisage_log_path) if unisage_log_path else None,
                             log_budget if unisage_log_path else None, log_policy)
            key = (mode, cache_log_key, ids)
            if key in rank_cache:
                return rank_cache[key]
            log_tag = "ulog" if unisage_log_path else "tlog"
            scratch = scratch_root / (
                f"{entry.dataset}__{entry.batch}__{entry.case}__{mode}_{log_tag}__"
                f"{abs(hash(key))%(10**12):012d}"
            )
            if scratch.exists():
                shutil.rmtree(scratch)
            nza.materialize_case(
                entry, nezha_rca_dir, set(ids), mode, scratch,
                unisage_log_sampling_csv=unisage_log_path,
                log_budget=log_budget,
                log_policy=log_policy,
            )
            # Empty-after-filter guard: if the retained ID set has zero overlap
            # with Nezha's source traceid file (or the file is otherwise empty),
            # Nezha's pandas.read_csv crashes with EmptyDataError. Skip the
            # subprocess and record a graceful miss so the case-level row is
            # still emitted (rank=None) instead of blowing up the run.
            tid_target = scratch / entry.nz_date / "traceid" / f"{entry.nz_window}_traceid.csv"
            if tid_target.exists() and tid_target.stat().st_size == 0:
                wall = 0.0
                out = {
                    "ranked": [],
                    "error": "empty_after_nezha_filter",
                    "stderr_tail": "",
                }
                ranked = []
                run_log.append(dict(
                    case=ck, mode=mode, log_policy=log_policy,
                    log_source=("unisage" if unisage_log_path else "trace_derived"),
                    log_budget=log_budget if unisage_log_path else None,
                    n_ids=len(ids), n_ranked=0,
                    wall_sec=wall, error="empty_after_nezha_filter",
                ))
                print(f"  [nezha skip] {ck} mode={mode} n_ids={len(ids)} "
                      f"reason=empty_after_nezha_filter", flush=True)
                if not args.keep_scratch:
                    shutil.rmtree(scratch, ignore_errors=True)
                rank_cache[key] = ranked
                return ranked
            t0 = time.time()
            out = nza.run_nezha_subprocess(
                entry, scratch, nezha_construct_dir,
                nezha_python=args.nezha_python, timeout_sec=args.timeout_sec,
            )
            wall = time.time() - t0
            ranked = out.get("ranked") or []
            run_log.append(dict(
                case=ck, mode=mode, log_policy=log_policy,
                log_source=("unisage" if unisage_log_path else "trace_derived"),
                log_budget=log_budget if unisage_log_path else None,
                n_ids=len(ids), n_ranked=len(ranked),
                wall_sec=round(wall, 2), error=out.get("error"),
            ))
            if out.get("error"):
                print(f"  [nezha error] {ck} mode={mode} n_ids={len(ids)} "
                      f"err={out['error']} stderr_tail={out.get('stderr_tail', '')[-300:]!r}",
                      flush=True)
            if not args.keep_scratch:
                shutil.rmtree(scratch, ignore_errors=True)
            rank_cache[key] = ranked
            return ranked

        print(f"[{ci}/{len(case_keys)}] {ck}  gt={gt}  nz={entry.nz_date}/{entry.nz_window}",
              flush=True)

        # Build (sampler, sample_df) task list, plus uniform off any one sample.
        tasks: list[tuple[str, pd.DataFrame]] = []
        for run_name, samplers in run_samplers.items():
            sample = loaded.get(run_name)
            if sample is None:
                continue
            for sampler in samplers:
                tasks.append((sampler, sample))
        if args.add_uniform:
            uniform_sample = add_uniform_random_scores(
                any_sample,
                seed=args.uniform_seed,
                dataset=dataset,
                batch=str(meta["batch"]),
                case=case,
            )
            tasks.append(("uniform", uniform_sample))

        for sampler, sample in tasks:
            sampler_is_unisage = sampler.startswith("unisage")
            for ret_mode in retention_modes:
                for budget in args.budgets:
                    ids = retained_ids(sample, sampler, budget, ret_mode)
                    if ids is None:
                        continue
                    fs = frozenset(ids)
                    for mode in modes:
                        # Log source decision:
                        #   - Mode A: irrelevant (logs are full); log_path=None
                        #   - Mode B: trace-derived for everyone; log_path=None
                        #   - Mode C: UniSage rows use log_sampling_result.csv if
                        #     present; baselines fall back to trace-derived (None)
                        if mode == "C" and sampler_is_unisage and unisage_log_csv:
                            if args.mode_c_log_policy == "native":
                                log_path = unisage_log_csv
                                log_b = budget
                            else:
                                log_path = None
                                log_b = None
                        else:
                            log_path = None
                            log_b = None
                        log_policy = args.mode_c_log_policy if mode == "C" and sampler_is_unisage else "native"
                        ranked = get_ranking(mode, fs, log_path, log_b, log_policy)
                        rank_ac = _first_pod_rank(ranked, gt)
                        rank_as = _nezha_native_rank(
                            ranked, gt, entry.inject_type, entry.ns, rc_cache[entry.ns])
                        per_case_rows.append(dict(
                            dataset=dataset, batch=meta["batch"], case=case,
                            sampler=sampler, retention=ret_mode, budget=budget,
                            mode=mode,
                            log_policy=log_policy,
                            log_source=("unisage" if log_path else "trace_derived" if mode in {"B", "C"} else "full"),
                            n_kept=len(ids), gt=gt,
                            rank_ac=rank_ac, rank_as=rank_as,
                            top1=(ranked[0]["pod"] if ranked else None),
                            inject_type=entry.inject_type,
                        ))

        # Persist intermediate every case so a crash mid-sweep keeps work.
        pd.DataFrame(per_case_rows).to_csv(out_dir / "nezha_rca_per_case.csv", index=False)
        pd.DataFrame(run_log).to_csv(out_dir / "nezha_run_log.csv", index=False)

    if not per_case_rows:
        pd.DataFrame(manifest_audit_rows, columns=MANIFEST_AUDIT_COLUMNS).to_csv(
            out_dir / "nezha_manifest_filter_audit.csv", index=False
        )
        print("No rows produced.")
        return

    pd.DataFrame(manifest_audit_rows, columns=MANIFEST_AUDIT_COLUMNS).to_csv(
        out_dir / "nezha_manifest_filter_audit.csv", index=False
    )
    per_case = pd.DataFrame(per_case_rows)
    per_case.to_csv(out_dir / "nezha_rca_per_case.csv", index=False)

    def agg(g: pd.DataFrame) -> pd.Series:
        n = len(g)
        return pd.Series(dict(
            n_cases=n,
            n_kept_mean=g["n_kept"].mean(),
            ac1=(g["rank_ac"] == 1).mean(),
            ac3=g["rank_ac"].between(1, 3).mean(),
            mrr=(g.loc[g["rank_ac"] > 0, "rank_ac"].rdiv(1.0).sum() / n) if n else float("nan"),
            as1=(g["rank_as"] == 1).mean(),
            as3=g["rank_as"].between(1, 3).mean(),
            as5=g["rank_as"].between(1, 5).mean(),
        ))

    summary = (
        per_case.groupby(["dataset", "sampler", "retention", "budget", "mode"], group_keys=False)
        [["rank_ac", "rank_as", "n_kept"]]
        .apply(agg)
        .reset_index()
    )
    summary.to_csv(out_dir / "nezha_rca_summary.csv", index=False)

    # Pretty markdown tables -- one block per (dataset, mode, retention).
    lines = ["# Nezha RCA on baseline-sampled traces\n"]
    lines.append(f"Cases evaluated: {per_case['case'].nunique()} | "
                 f"Modes: {sorted(per_case['mode'].unique())} | "
                 f"Total Nezha invocations: {len(run_log)} | "
                 f"Total wall: {round((time.time()-t_start_all)/60, 1)} min\n")
    for (dataset, mode, retention), g in summary.groupby(["dataset", "mode", "retention"]):
        lines.append(f"\n## {dataset} -- mode={mode} retention={retention}\n")
        for metric in ("ac1", "ac3", "mrr", "as5"):
            lines.append(f"\n### {metric.upper()} by budget (rows=sampler)\n")
            pivot = g.pivot_table(index="sampler", columns="budget", values=metric).round(3)
            lines.append("```\n" + pivot.to_string() + "\n```")
        lines.append("")
    (out_dir / "nezha_rca_summary.md").write_text("\n".join(lines))

    print(f"\nWrote {out_dir}/nezha_rca_per_case.csv, nezha_rca_summary.csv, nezha_rca_summary.md")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
