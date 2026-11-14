"""MicroRCA as a downstream RCA evaluator on sampled traces.

This harness mirrors `scripts/eval_nezha_rca_on_sampled_traces.py`:
all samplers share the same retention
semantics, budgets, and localization metrics (AC@1 / AC@3 / MRR).

Modes:
  * full_metric: MicroRCA sees Nezha-lineage full invocation latency and full
    resource metrics. This is the upper-bound reproduction sanity mode; it is
    not sampling-sensitive.
  * sampled_trace: retained TraceIDs rebuild edge-latency time series, while
    resource metrics remain full. This is the primary UniSage downstream-RCA
    mode because UniSage currently samples traces/logs, not metrics.

Smoke:
    python scripts/eval_microrca_rca_on_sampled_traces.py \\
        --unisage results/v02_sampling_runs/unisage_trace_hip_tt_primary_target_raw_prop0_v02 \\
        --sieve results/v02_sampling_runs/sieve_v02_hipster_train_ticket_primary_seed-1_FINAL \\
        --add-uniform --budgets 0.05 1.0 --retention topn \\
        --cases 03_53,08_43 --limit 2 \\
        --out results/microrca/smoke
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from baselines.microrca import adapter as mra
from eval_fair_rca_on_sampled_traces import (
    MANIFEST_AUDIT_COLUMNS,
    UNIFORM_RANDOM_SCORE_COL,
    add_uniform_random_scores,
    filter_sample_to_manifest,
    load_trace_manifest_ids,
    parse_manifest_index_arg,
)


RANKING_COLUMNS = {
    "unisage_combined": "CombinedSamplingProb",
    "unisage_edge": "EdgeCaseSamplingProb",
    "unisage_analysis": "AnomalySamplingProb",
    "sifter": "SifterLoss",
    "sieve": "SieveScore",
    "trastrainer": "TraStrainerCombined",
    "gleaner": "GleanerScore",
}
DECISION_PREFIXES = {
    "unisage_slot_gated_reserved": "SlotGatedReservedDecision",
    "unisage_dynamic_pattern_guard": "DynamicPatternGuardDecision",
    "sifter": "SifterDecision",
    "sieve": "SieveDecision",
    "trastrainer": "TraStrainerDecision",
    "tracepicker": "TracePickerDecision",
    "gleaner": "GleanerDecision",
}


def stable_uniform_score(trace_id: object) -> float:
    digest = hashlib.sha256(str(trace_id).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)


def budget_token(budget: float) -> str:
    return f"{budget:g}".replace("-", "m").replace(".", "p")


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "1.0", "yes"})


class GroundTruth:
    def __init__(self, data_root: Path):
        self.data_root = data_root
        self._labels: dict[Path, dict] = {}

    def inject_service(self, dataset: str, case: str, case_dir: Path) -> Optional[str]:
        if dataset not in {"hipster", "train_ticket"}:
            return None
        label_path = case_dir.parent / "label.json"
        if label_path not in self._labels:
            self._labels[label_path] = json.loads(label_path.read_text()) if label_path.exists() else {}
        return self._labels[label_path].get(case, {}).get("inject_pod")


def load_run_index(run_dir: Path) -> pd.DataFrame:
    idx = pd.read_csv(run_dir / "sample_index.csv")
    idx = idx[idx["status"] == "ok"].copy()
    idx["case_key"] = idx["dataset"] + "/" + idx["batch"].astype(str) + "/" + idx["case"].astype(str)
    return idx


def read_sample(sample_path: Path) -> pd.DataFrame:
    df = pd.read_csv(sample_path)
    df["TraceID"] = df["TraceID"].astype(str)
    return df


def _topn_ids(sample: pd.DataFrame, sampler: str, budget: float) -> Optional[set[str]]:
    n_total = len(sample)
    target_k = max(1, int(round(n_total * budget)))
    if sampler == "uniform":
        if UNIFORM_RANDOM_SCORE_COL in sample.columns:
            scored = sample.assign(_s=sample[UNIFORM_RANDOM_SCORE_COL].astype(float))
        else:
            scored = sample.assign(_s=sample["TraceID"].map(stable_uniform_score))
    else:
        col = RANKING_COLUMNS.get(sampler)
        if col is None or col not in sample.columns:
            return None
        scored = sample.dropna(subset=[col]).copy()
        scored["_s"] = scored[col].astype(float)
    scored["TraceID"] = scored["TraceID"].astype(str)
    k = min(len(scored), target_k)
    order = scored.sort_values(["_s", "TraceID"], ascending=[False, True])
    return set(order["TraceID"].head(k))


def _decision_ids(sample: pd.DataFrame, sampler: str, budget: float) -> Optional[set[str]]:
    prefix = DECISION_PREFIXES.get(sampler)
    if prefix is None:
        return None
    col = f"{prefix}_{budget_token(budget)}"
    if col not in sample.columns:
        return None
    return set(sample.loc[as_bool(sample[col]), "TraceID"].astype(str))


def retained_ids(sample: pd.DataFrame, sampler: str, budget: float, mode: str) -> Optional[set[str]]:
    if len(sample) == 0:
        return set()
    if budget >= 1.0:
        return set(sample["TraceID"].astype(str))
    if mode == "native":
        ids = _decision_ids(sample, sampler, budget)
        return ids if ids is not None else _topn_ids(sample, sampler, budget)
    if mode == "topn":
        ids = _topn_ids(sample, sampler, budget)
        return ids if ids is not None else _decision_ids(sample, sampler, budget)
    raise ValueError(f"unknown retention mode: {mode}")


def _load_case_index(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "case_key" not in df.columns:
        df["case_key"] = df["dataset"] + "/" + df["batch"].astype(str) + "/" + df["case"].astype(str)
    return df


def _ensure_case_index(data_root: Path, nezha_rca_dir: Path, case_index: Path) -> pd.DataFrame:
    if case_index.exists():
        return _load_case_index(case_index)
    return mra.write_case_index(data_root / "data", nezha_rca_dir, case_index)


def main() -> None:
    ap = argparse.ArgumentParser()
    for name in ("sifter", "sieve", "tracepicker", "trastrainer", "unisage", "gleaner"):
        ap.add_argument(f"--{name}", type=str, default=None, help=f"{name} v0.2 run dir")
    ap.add_argument("--add-uniform", action="store_true")
    ap.add_argument(
        "--uniform-seed",
        type=int,
        default=None,
        help="seed for RQ1-compatible Uniform random scores; omit for legacy TraceID-only hash",
    )
    ap.add_argument("--retention", choices=["native", "topn", "both"], default="both")
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.01, 0.025, 0.05, 0.1, 0.2, 1.0])
    ap.add_argument("--modes", nargs="+", choices=["full_metric", "sampled_trace"],
                    default=["full_metric", "sampled_trace"])
    ap.add_argument("--data-root", type=str, default=".")
    ap.add_argument(
        "--trace-manifest-index",
        type=parse_manifest_index_arg,
        default=Path("results/label_protocol_v02/trace_manifest_index.csv"),
        help="v0.2 canonical trace manifest index; set to none/off to disable filtering",
    )
    ap.add_argument("--nezha-rca-dir", default="baselines/nezha/rca_data")
    ap.add_argument("--case-index", default="results/microrca/microrca_case_index.csv")
    ap.add_argument("--python", default=None)
    ap.add_argument("--dataset-filter", type=str, default=None)
    ap.add_argument("--cases", type=str, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.55)
    ap.add_argument("--ad-threshold", type=float, default=0.045)
    ap.add_argument("--smoothing-window", type=int, default=12)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--timeout-sec", type=int, default=300)
    ap.add_argument("--keep-scratch", action="store_true")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip cases that already have rows in microrca_rca_per_case.csv; resume by case granularity.",
    )
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    nezha_rca_dir = Path(args.nezha_rca_dir).resolve()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch_root = (out_dir / "_scratch").resolve()
    scratch_root.mkdir(parents=True, exist_ok=True)

    run_dirs = {n: getattr(args, n) for n in ("sifter", "sieve", "tracepicker", "trastrainer", "unisage", "gleaner")}
    run_dirs = {n: Path(p) for n, p in run_dirs.items() if p}
    if not run_dirs:
        ap.error("provide at least one sampler run dir")

    run_samplers = {}
    for name in run_dirs:
        if name == "unisage":
            run_samplers[name] = [
                "unisage_combined",
                "unisage_edge",
                "unisage_analysis",
                "unisage_dynamic_pattern_guard",
            ]
        else:
            run_samplers[name] = [name]

    indices = {name: load_run_index(path) for name, path in run_dirs.items()}
    case_meta: dict[str, dict] = {}
    sample_paths: dict[str, dict[str, Path]] = {}
    for run_name, idx in indices.items():
        for _, row in idx.iterrows():
            ck = row["case_key"]
            case_meta.setdefault(ck, dict(
                dataset=row["dataset"],
                batch=str(row["batch"]),
                case=str(row["case"]),
                case_dir=Path(row["case_dir"]),
            ))
            sample_paths.setdefault(ck, {})[run_name] = Path(row["sample_path"])

    case_idx = _ensure_case_index(data_root, nezha_rca_dir, Path(args.case_index))
    case_idx_by_key = {row["case_key"]: row for _, row in case_idx.iterrows()}
    case_keys = sorted(k for k in case_meta if k in case_idx_by_key)
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

    retention_modes = ["native", "topn"] if args.retention == "both" else [args.retention]
    gt_resolver = GroundTruth(data_root)
    per_case_rows: list[dict] = []
    run_log: list[dict] = []
    # Resume support: skip cases whose per-case rows already exist.
    done_cases: set[tuple[str, str, str]] = set()
    if getattr(args, "resume", False):
        prior_csv = out_dir / "microrca_rca_per_case.csv"
        if prior_csv.exists():
            try:
                prior_df = pd.read_csv(prior_csv)
                per_case_rows = prior_df.to_dict("records")
                done_cases = {(str(r["dataset"]), str(r["batch"]), str(r["case"]))
                              for r in per_case_rows}
                print(f"[resume] loaded {len(per_case_rows)} prior rows, {len(done_cases)} done cases", flush=True)
            except Exception as exc:
                print(f"[resume] failed to load prior per-case csv: {exc}", flush=True)
        prior_log = out_dir / "microrca_run_log.csv"
        if prior_log.exists():
            try:
                run_log = pd.read_csv(prior_log).to_dict("records")
            except Exception:
                pass
    t_start = time.time()

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

        entry = mra.case_entry_from_row(case_idx_by_key[ck])
        canonical_ids = manifest_ids_by_case.get(ck)
        loaded: dict[str, pd.DataFrame] = {}
        for run_name in run_dirs:
            sample_path = sample_paths.get(ck, {}).get(run_name)
            if sample_path is not None:
                sample = read_sample(data_root / sample_path)
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

        print(f"[{ci}/{len(case_keys)}] {ck}  gt={gt}", flush=True)

        rank_cache: dict[tuple, dict] = {}

        def get_result(mode: str, ids: Optional[frozenset]) -> dict:
            key = (mode, ids if mode == "sampled_trace" else None)
            if key in rank_cache:
                return rank_cache[key]
            scratch = scratch_root / (
                f"{entry.dataset}__{entry.batch}__{entry.case}__{mode}__"
                f"{abs(hash(key)) % (10**12):012d}"
            )
            if scratch.exists():
                shutil.rmtree(scratch)
            retained = set(ids) if (mode == "sampled_trace" and ids is not None) else None
            mra.materialize_case(
                entry,
                unisage_data_root=data_root / "data",
                nezha_rca_dir=nezha_rca_dir,
                retained_ids=retained,
                mode=mode,
                scratch_root=scratch,
            )
            t0 = time.time()
            out = mra.run_microrca_subprocess(
                scratch,
                python_exe=args.python,
                alpha=args.alpha,
                ad_threshold=args.ad_threshold,
                smoothing_window=args.smoothing_window,
                top_k=args.top_k,
                timeout_sec=args.timeout_sec,
            )
            wall = time.time() - t0
            ranked = out.get("ranked") or []
            run_log.append(dict(
                case=ck,
                mode=mode,
                n_ids=(len(ids) if ids is not None else None),
                n_ranked=len(ranked),
                wall_sec=round(wall, 2),
                error=out.get("error"),
                n_anomalous_edges=(out.get("diagnostics") or {}).get("n_anomalous_edges"),
            ))
            if out.get("error"):
                print(f"  [microrca error] {ck} mode={mode} err={out['error']}", flush=True)
            if not args.keep_scratch:
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
            uniform_sample = add_uniform_random_scores(
                any_sample,
                seed=args.uniform_seed,
                dataset=dataset,
                batch=str(meta["batch"]),
                case=case,
            )
            tasks.append(("uniform", uniform_sample))

        for sampler, sample in tasks:
            for ret_mode in retention_modes:
                for budget in args.budgets:
                    ids = retained_ids(sample, sampler, budget, ret_mode)
                    if ids is None:
                        continue
                    fs = frozenset(ids)
                    for mode in args.modes:
                        result = get_result(mode, fs)
                        ranked = result.get("ranked") or []
                        rank = mra.first_service_rank(ranked, gt)
                        top1 = ranked[0]["service"] if ranked else None
                        top1_score = ranked[0].get("score") if ranked else None
                        per_case_rows.append(dict(
                            dataset=dataset,
                            batch=meta["batch"],
                            case=case,
                            sampler=sampler,
                            retention=ret_mode,
                            budget=budget,
                            mode=mode,
                            n_kept=len(ids),
                            gt=gt,
                            gt_service=mra.service_base(gt),
                            rank=rank,
                            top1=top1,
                            top1_score=top1_score,
                            inject_type=entry.inject_type,
                        ))

        pd.DataFrame(per_case_rows).to_csv(out_dir / "microrca_rca_per_case.csv", index=False)
        pd.DataFrame(run_log).to_csv(out_dir / "microrca_run_log.csv", index=False)

    if not per_case_rows:
        pd.DataFrame(manifest_audit_rows, columns=MANIFEST_AUDIT_COLUMNS).to_csv(
            out_dir / "microrca_manifest_filter_audit.csv", index=False
        )
        print("No rows produced.")
        return

    pd.DataFrame(manifest_audit_rows, columns=MANIFEST_AUDIT_COLUMNS).to_csv(
        out_dir / "microrca_manifest_filter_audit.csv", index=False
    )
    per_case = pd.DataFrame(per_case_rows)
    per_case.to_csv(out_dir / "microrca_rca_per_case.csv", index=False)

    def agg(group: pd.DataFrame) -> pd.Series:
        n = len(group)
        return pd.Series(dict(
            n_cases=n,
            n_kept_mean=group["n_kept"].mean(),
            ac1=(group["rank"] == 1).mean(),
            ac3=group["rank"].between(1, 3).mean(),
            mrr=(group.loc[group["rank"] > 0, "rank"].rdiv(1.0).sum() / n) if n else float("nan"),
        ))

    summary = (
        per_case
        .groupby(["dataset", "sampler", "retention", "budget", "mode"], group_keys=False)[["rank", "n_kept"]]
        .apply(agg)
        .reset_index()
    )
    summary.to_csv(out_dir / "microrca_rca_summary.csv", index=False)

    collapse_rows = []
    budget_full = per_case[per_case["budget"] >= 1.0]
    for (dataset, batch, case, retention, mode), group in budget_full.groupby(["dataset", "batch", "case", "retention", "mode"]):
        tops = sorted(set(group["top1"].dropna().astype(str)))
        ranks = sorted(set(group["rank"].astype(int)))
        collapse_rows.append(dict(
            dataset=dataset,
            batch=batch,
            case=case,
            retention=retention,
            mode=mode,
            n_samplers=group["sampler"].nunique(),
            n_top1=len(tops),
            top1_values=";".join(tops),
            rank_values=";".join(map(str, ranks)),
            ok=(len(tops) <= 1 and len(ranks) <= 1),
        ))
    pd.DataFrame(collapse_rows).to_csv(out_dir / "microrca_collapse_check.csv", index=False)

    provenance = {
        "params": {
            "alpha": args.alpha,
            "ad_threshold": args.ad_threshold,
            "smoothing_window": args.smoothing_window,
            "top_k": args.top_k,
            "budgets": args.budgets,
            "modes": args.modes,
            "retention": args.retention,
        },
        "run_dirs": {k: str(v) for k, v in run_dirs.items()},
        "add_uniform": args.add_uniform,
        "case_index": str(Path(args.case_index)),
        "elapsed_min": round((time.time() - t_start) / 60, 2),
    }
    (out_dir / "microrca_provenance.json").write_text(json.dumps(provenance, indent=2))

    lines = ["# MicroRCA RCA on sampled traces\n"]
    lines.append(
        f"Cases evaluated: {per_case['case'].nunique()} | "
        f"Modes: {sorted(per_case['mode'].unique())} | "
        f"Datasets: {sorted(per_case['dataset'].unique())} | "
        f"Total invocations: {len(run_log)} | "
        f"Total wall: {round((time.time() - t_start) / 60, 1)} min\n"
    )
    for (dataset, mode, retention), group in summary.groupby(["dataset", "mode", "retention"]):
        lines.append(f"\n## {dataset} -- mode={mode} retention={retention}\n")
        for metric in ("ac1", "ac3", "mrr"):
            lines.append(f"\n### {metric.upper()} by budget (rows=sampler)\n")
            pivot = group.pivot_table(index="sampler", columns="budget", values=metric).round(3)
            lines.append("```\n" + pivot.to_string() + "\n```")
        lines.append("")
    (out_dir / "microrca_rca_summary.md").write_text("\n".join(lines))

    print(f"\nWrote {out_dir}/microrca_rca_per_case.csv, microrca_rca_summary.csv, microrca_rca_summary.md")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
