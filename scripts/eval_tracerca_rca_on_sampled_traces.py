"""TraceRCA as a downstream RCA evaluator on sampled traces.

Mirrors `scripts/eval_microrca_rca_on_sampled_traces.py` / `scripts/eval_nezha_rca_on_sampled_traces.py`:
all samplers share the same retention semantics, budgets, manifest universe, and
localization metrics (AC@1 / AC@3 / MRR). TraceRCA is **trace-only** -- logs and
metrics are not consumed and are reported as N/A in the RQ2 modality table.

For each case we:
  * build the FULL fault-free historical invocation table once (the k-sigma
    reference; never sampled, matching the fair-RCA / MicroRCA / Nezha
    convention);
  * restrict abnormal spans to the fault window (case-minute based; see
    `tracerca_adapter.fault_window_spans`) and build the full abnormal
    invocation table once;
  * per (sampler, retention, budget) keep only retained TraceIDs, run the
    vendored TraceRCA (k-sigma AD + association-rule-mining localization), and
    score the injected service.

At budget>=1.0 every sampler collapses to the same retained set -> one TraceRCA
invocation shared across samplers (the upper-bound collapse correctness check).

Run from UniSage/. The driver imports (pandas etc.) come from the `unisage`
env; the per-case TraceRCA subprocess runs in the dedicated `tracerca` env
(auto-discovered by name; override with `--python /path/to/tracerca/bin/python`
or `TRACERCA_PYTHON`). The driver interpreter is never used as a fallback.

Smoke:
    python scripts/eval_tracerca_rca_on_sampled_traces.py \\
        --unisage results/v02_sampling_runs/unisage_trace_hip_tt_primary_target_raw_prop0_v02 \\
        --sieve results/v02_sampling_runs/sieve_v02_hipster_train_ticket_primary_seed-1_FINAL \\
        --add-uniform --uniform-seed 1 --budgets 0.05 1.0 --retention topn \\
        --cases 03_53,08_43 --limit 2 --out results/tracerca/smoke
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from baselines.tracerca import adapter as tra
from eval_fair_rca_on_sampled_traces import (
    MANIFEST_AUDIT_COLUMNS,
    GroundTruth,
    add_uniform_random_scores,
    filter_sample_to_manifest,
    load_run_index,
    load_spans,
    load_trace_manifest_ids,
    parse_manifest_index_arg,
    read_sample,
    retained_ids,
)

# Trace-only ranking/decision columns, identical to the other RCA harnesses.
RANKING_COLUMNS = {
    "unisage_combined": "CombinedSamplingProb",
    "unisage_edge": "EdgeCaseSamplingProb",
    "unisage_analysis": "AnomalySamplingProb",
    "sifter": "SifterLoss",
    "sieve": "SieveScore",
    "trastrainer": "TraStrainerCombined",
    "gleaner": "GleanerScore",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    for name in ("sifter", "sieve", "tracepicker", "trastrainer", "unisage", "gleaner"):
        ap.add_argument(f"--{name}", type=str, default=None, help=f"{name} v0.2 run dir")
    ap.add_argument("--add-uniform", action="store_true")
    ap.add_argument("--uniform-seed", type=int, default=None,
                    help="seed for RQ1-compatible Uniform random scores; omit for legacy TraceID hash")
    ap.add_argument("--retention", choices=["native", "topn", "both"], default="both")
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.01, 0.025, 0.05, 0.1, 0.2, 1.0])
    ap.add_argument("--data-root", type=str, default=".")
    ap.add_argument("--trace-manifest-index", type=parse_manifest_index_arg,
                    default=Path("results/label_protocol_v02/trace_manifest_index.csv"),
                    help="v0.2 canonical trace manifest index; set to none/off to disable filtering")
    ap.add_argument("--dataset-filter", type=str, default=None)
    ap.add_argument("--cases", type=str, default=None)
    ap.add_argument("--limit", type=int, default=None)
    # TraceRCA hyperparameters (upstream Makefile defaults)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--fisher", type=float, default=3.0)
    ap.add_argument("--min-support", type=float, default=0.1)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--features", default="latency", help="comma list of invocation features")
    ap.add_argument("--window-min", type=int, default=3, help="fault-window minutes (hipster/TT)")
    ap.add_argument("--python", default=None,
                    help="isolated tracerca interpreter (or TRACERCA_PYTHON; auto-discover named conda env)")
    ap.add_argument("--timeout-sec", type=int, default=300)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--keep-scratch", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="skip cases already present in tracerca_rca_per_case.csv")
    args = ap.parse_args()
    try:
        args.python = tra.resolve_tracerca_python(args.python)
    except ValueError as exc:
        ap.error(str(exc))

    data_root = Path(args.data_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch_root = out_dir / "_scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)
    features = [f.strip() for f in args.features.split(",") if f.strip()]

    run_dirs = {n: getattr(args, n) for n in ("sifter", "sieve", "tracepicker", "trastrainer", "unisage", "gleaner")}
    run_dirs = {n: Path(p) for n, p in run_dirs.items() if p}
    if not run_dirs:
        ap.error("provide at least one sampler run dir")

    run_samplers = {}
    for n in run_dirs:
        run_samplers[n] = (
            ["unisage_combined", "unisage_edge", "unisage_analysis", "unisage_dynamic_pattern_guard"]
            if n == "unisage" else [n]
        )

    indices = {n: load_run_index(d) for n, d in run_dirs.items()}
    case_meta: dict[str, dict] = {}
    sample_paths: dict[str, dict[str, Path]] = {}
    for n, idx in indices.items():
        for _, row in idx.iterrows():
            ck = row["case_key"]
            case_meta.setdefault(ck, dict(dataset=row["dataset"], batch=str(row["batch"]),
                                          case=str(row["case"]), case_dir=Path(row["case_dir"])))
            sample_paths.setdefault(ck, {})[n] = Path(row["sample_path"])

    case_keys = sorted(case_meta)
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
    modes = ["native", "topn"] if args.retention == "both" else [args.retention]
    gt_resolver = GroundTruth(data_root)
    per_case_rows: list[dict] = []
    run_log: list[dict] = []

    done_cases: set[tuple] = set()
    if args.resume and (out_dir / "tracerca_rca_per_case.csv").exists():
        prior = pd.read_csv(out_dir / "tracerca_rca_per_case.csv")
        per_case_rows = prior.to_dict("records")
        done_cases = {(str(r["dataset"]), str(r["batch"]), str(r["case"])) for r in per_case_rows}
        print(f"[resume] {len(per_case_rows)} prior rows, {len(done_cases)} done cases", flush=True)

    t_start = time.time()
    for ci, ck in enumerate(case_keys, 1):
        meta = case_meta[ck]
        dataset, batch, case = meta["dataset"], meta["batch"], meta["case"]
        if (str(dataset), str(batch), str(case)) in done_cases:
            print(f"[{ci}/{len(case_keys)}] {ck} [resume skip]", flush=True)
            continue
        case_dir = data_root / meta["case_dir"]
        gt = gt_resolver.inject_service(dataset, case, case_dir)
        if not gt:
            print(f"[{ci}/{len(case_keys)}] {ck}: NO_GT, skip", flush=True)
            continue

        abn_spans = load_spans(case_dir, dataset, normal=False)
        nrm_spans = load_spans(case_dir, dataset, normal=True)
        if abn_spans.empty or nrm_spans.empty:
            print(f"[{ci}/{len(case_keys)}] {ck}: NO_TRACE_DATA, skip", flush=True)
            continue
        abn_spans = tra.fault_window_spans(abn_spans, dataset, case, window_min=args.window_min)
        historical_invo = tra.build_invocations(nrm_spans)
        abnormal_invo_full = tra.build_invocations(abn_spans)
        forbidden = tra.default_forbidden(dataset)
        canonical_ids = manifest_ids_by_case.get(ck)

        loaded: dict[str, pd.DataFrame] = {}
        for run_name in run_dirs:
            sp = sample_paths.get(ck, {}).get(run_name)
            if sp is None:
                continue
            sample = read_sample(data_root / sp)
            if canonical_ids is not None:
                sample, audit = filter_sample_to_manifest(
                    sample, canonical_ids, dataset=dataset, batch=batch, case=case, run_name=run_name)
                manifest_audit_rows.append(audit)
            loaded[run_name] = sample
        if not loaded:
            print(f"[{ci}/{len(case_keys)}] {ck}: NO_SAMPLE_FILES, skip", flush=True)
            continue
        any_sample = next(iter(loaded.values()))
        print(f"[{ci}/{len(case_keys)}] {ck} gt={gt} window_invo={len(abnormal_invo_full)}", flush=True)

        rank_cache: dict[frozenset, dict] = {}

        def run_for(ids: set) -> dict:
            key = frozenset(ids)
            if key in rank_cache:
                return rank_cache[key]
            abn_invo = abnormal_invo_full[abnormal_invo_full["trace_id"].isin(ids)]
            scratch = scratch_root / f"{dataset}__{batch}__{case}__{abs(hash(key)) % (10**12):012d}"
            tra.materialize_case(abn_invo, historical_invo, scratch,
                                 meta=dict(dataset=dataset, batch=batch, case=case, gt=gt))
            t0 = time.time()
            out = tra.run_tracerca_subprocess(
                scratch, python_exe=args.python, sigma=args.sigma, fisher=args.fisher,
                min_support=args.min_support, k=args.k, features=features,
                forbidden=forbidden, timeout_sec=args.timeout_sec)
            wall = time.time() - t0
            diag = out.get("diagnostics") or {}
            run_log.append(dict(case=ck, n_ids=len(ids), n_window_invo=len(abn_invo),
                                n_ranked=len(out.get("ranked") or []),
                                n_abnormal_traces=diag.get("n_abnormal_traces"),
                                n_useful_edges=diag.get("n_useful_edges"),
                                wall_sec=round(wall, 2), error=out.get("error")))
            if out.get("error"):
                print(f"  [tracerca error] {ck} err={out['error']}", flush=True)
            if not args.keep_scratch:
                import shutil
                shutil.rmtree(scratch, ignore_errors=True)
            rank_cache[key] = out
            return out

        tasks: list[tuple[str, pd.DataFrame]] = []
        for run_name, samplers in run_samplers.items():
            sample = loaded.get(run_name)
            if sample is None:
                continue
            for sampler in samplers:
                tasks.append((sampler, sample))
        if args.add_uniform:
            uniform_sample = add_uniform_random_scores(any_sample, seed=args.uniform_seed,
                                                       dataset=dataset, batch=batch, case=case)
            tasks.append(("uniform", uniform_sample))

        for sampler, sample in tasks:
            for ret_mode in modes:
                for budget in args.budgets:
                    ids = retained_ids(sample, sampler, budget, ret_mode)
                    if ids is None:
                        continue
                    out = run_for(ids)
                    ranked = out.get("ranked") or []
                    rank = tra.first_service_rank(ranked, gt)
                    per_case_rows.append(dict(
                        dataset=dataset, batch=batch, case=case, sampler=sampler,
                        retention=ret_mode, budget=budget, n_kept=len(ids),
                        gt=gt, gt_service=tra.service_base(gt), rank=rank,
                        top1=(ranked[0] if ranked else None)))

        pd.DataFrame(per_case_rows).to_csv(out_dir / "tracerca_rca_per_case.csv", index=False)
        pd.DataFrame(run_log).to_csv(out_dir / "tracerca_run_log.csv", index=False)

    pd.DataFrame(manifest_audit_rows, columns=MANIFEST_AUDIT_COLUMNS).to_csv(
        out_dir / "tracerca_manifest_filter_audit.csv", index=False)
    if not per_case_rows:
        print("No rows produced.")
        return

    per_case = pd.DataFrame(per_case_rows)
    per_case.to_csv(out_dir / "tracerca_rca_per_case.csv", index=False)

    def agg(g: pd.DataFrame) -> pd.Series:
        n = len(g)
        return pd.Series(dict(
            n_cases=n, n_kept_mean=g["n_kept"].mean(),
            ac1=(g["rank"] == 1).mean(),
            ac3=g["rank"].between(1, 3).mean(),
            mrr=(g.loc[g["rank"] > 0, "rank"].rdiv(1.0).sum() / n) if n else float("nan")))

    summary = (per_case.groupby(["dataset", "sampler", "retention", "budget"], group_keys=False)[["rank", "n_kept"]]
               .apply(agg).reset_index())
    summary.to_csv(out_dir / "tracerca_rca_summary.csv", index=False)

    # budget=1.0 collapse: every sampler must yield the same top1 and rank per case.
    collapse_rows = []
    full = per_case[per_case["budget"] >= 1.0]
    for (dataset, batch, case, retention), g in full.groupby(["dataset", "batch", "case", "retention"]):
        tops = sorted(set(g["top1"].dropna().astype(str)))
        ranks = sorted(set(g["rank"].astype(int)))
        collapse_rows.append(dict(dataset=dataset, batch=batch, case=case, retention=retention,
                                  n_samplers=g["sampler"].nunique(), n_top1=len(tops),
                                  top1_values=";".join(tops), rank_values=";".join(map(str, ranks)),
                                  ok=(len(tops) <= 1 and len(ranks) <= 1)))
    pd.DataFrame(collapse_rows).to_csv(out_dir / "tracerca_collapse_check.csv", index=False)

    provenance = dict(params=dict(sigma=args.sigma, fisher=args.fisher, min_support=args.min_support,
                                  k=args.k, features=features, window_min=args.window_min,
                                  budgets=args.budgets, retention=args.retention),
                      run_dirs={k: str(v) for k, v in run_dirs.items()},
                      add_uniform=args.add_uniform, uniform_seed=args.uniform_seed,
                      elapsed_min=round((time.time() - t_start) / 60, 2))
    (out_dir / "tracerca_provenance.json").write_text(json.dumps(provenance, indent=2))

    lines = ["# TraceRCA RCA on sampled traces (trace-only)\n",
             f"Cases: {per_case['case'].nunique()} | Datasets: {sorted(per_case['dataset'].unique())} | "
             f"Invocations: {len(run_log)} | Wall: {round((time.time() - t_start) / 60, 1)} min\n"]
    for (dataset, retention), g in summary.groupby(["dataset", "retention"]):
        lines.append(f"\n## {dataset} -- retention={retention}\n")
        for metric in ("ac1", "ac3", "mrr"):
            lines.append(f"\n### {metric.upper()} by budget (rows=sampler)\n")
            pivot = g.pivot_table(index="sampler", columns="budget", values=metric).round(3)
            lines.append("```\n" + pivot.to_string() + "\n```")
        lines.append("")
    (out_dir / "tracerca_rca_summary.md").write_text("\n".join(lines))

    print(f"\nWrote {out_dir}/tracerca_rca_summary.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
