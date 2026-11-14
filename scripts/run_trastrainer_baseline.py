"""Run TraStrainer as a version-stamped v0.2 trace-sampling baseline.

TraStrainer's upstream reference lives at
https://github.com/IntelligentDDS/TraStrainer (FSE 2024). It depends on
PyTorch + treelib + a per-(service, metric) LTSF_Linear checkpoint set. We
reimplement the algorithm in `baselines.trastrainer` using only numpy /
pandas so it runs on the same macOS arm64 / Python 3.9 environment as the
other UniSage v0.2 baselines, and so the per-trace behaviour stays
auditable. See `baselines/trastrainer.py` for the upstream mapping.

This runner mirrors `scripts/run_sifter_baseline.py`: it reads the frozen v0.2
trace manifest index, normalizes each case's raw trace frame, runs
TraStrainer once per target budget over the streaming order, and emits both
a continuous TraStrainerCombined column (the canonical pass's mean of
system-bias and diversity-bias rates, for ranker evaluation) and per-budget
TraStrainerDecision_<token> columns (faithful to TraStrainer's online AND/OR
voting at that target α). The companion v0.2 evaluator picks up both column
shapes.

Per-trace structure and feature extraction are cached across budgets — they
are α-independent — and reported as `extract_sec` in `sample_index.csv`.
`trastrainer_canonical_sec` is the sampling-only time of the canonical
budget pass (matches the column shape used for Sifter / TracePicker).
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import contextlib
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import sampling_labels as lp
from evaluate_trace_sampling import BUDGETS, budget_token
from build_trace_manifest import sha256_file
from baselines.trastrainer import (
    TraStrainerConfig,
    derive_service_keys,
    load_metric_keys_aiops21,
    run_trastrainer_over_trace_frame,
)


HERE = Path(__file__).resolve().parent.parent


def case_dir(data_root: Path, dataset: str, batch: str, case: str) -> Path:
    if dataset == "aiops21":
        return data_root / "aiops21" / "all" / case
    return data_root / dataset / "batches" / batch / case


def read_raw_trace_frame(cdir: Path, dataset: str) -> pd.DataFrame:
    if dataset == "aiops21":
        return pd.read_csv(cdir / "abnormal" / "traces.csv")
    return lp.read_benchmark_trace_files(cdir)


def metric_keys_for_case(cdir: Path, dataset: str) -> Tuple[List[Tuple[str, str]], str]:
    """Return (metric_keys, source) — empty list signals "derive from spans".

    For AIOps21 we read the per-case `abnormal/metrics.csv` and filter to
    the upstream `INCLUDED_METRICS` list. For Hipster / Train Ticket the
    local snapshot does not carry raw per-case metric CSVs, so we return an
    empty list and `derive_service_keys` synthesises one key per service —
    equivalent to upstream running with weight=1 across the trace-derived
    feature for each present service (the only mode reachable without
    LTSF_Linear checkpoints, which we do not have).
    """
    if dataset == "aiops21":
        path = cdir / "abnormal" / "metrics.csv"
        if path.is_file():
            keys = load_metric_keys_aiops21(path.as_posix())
            return keys, f"aiops21:{path.as_posix()}"
    return [], "synthesised_from_trace_services"


def file_fingerprint(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        if path.is_file():
            digest.update(path.as_posix().encode())
            digest.update(path.read_bytes())
        elif path.is_dir():
            for child in sorted(path.rglob("*.py")):
                digest.update(child.as_posix().encode())
                digest.update(child.read_bytes())
    return digest.hexdigest()[:12]


def git_head(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", path.as_posix(), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def dependency_versions(names: Iterable[str]) -> dict:
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def runtime_info() -> dict:
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "dependencies": dependency_versions(["numpy", "pandas"]),
    }


def run_fingerprint() -> str:
    paths: List[Path] = [
        HERE / "scripts/run_trastrainer_baseline.py",
        HERE / "scripts/evaluate_trace_sampling.py",
        HERE / "scripts/sampling_labels.py",
        HERE / "baselines",
    ]
    return file_fingerprint(paths)


def select_cases(
    index: pd.DataFrame,
    datasets: List[str],
    scope: str,
    limit: Optional[int],
    cases: Optional[List[str]] = None,
) -> pd.DataFrame:
    selected = index[index["dataset"].isin(datasets)].copy()
    if scope == "primary":
        selected = selected[selected["trace_label_scope"] == lp.PRIMARY_SCOPE]
    elif scope == "excluded":
        selected = selected[selected["trace_label_scope"] == lp.EXCLUDED_INFRA_SCOPE]
    elif scope != "all":
        raise ValueError(f"Unknown scope: {scope}")
    if cases:
        wanted = set(cases)
        keys = pd.DataFrame(
            {
                "case_key": selected["case"],
                "batch_case_key": selected["batch"].astype(str) + "/" + selected["case"].astype(str),
                "dataset_batch_case_key": (
                    selected["dataset"].astype(str)
                    + "/"
                    + selected["batch"].astype(str)
                    + "/"
                    + selected["case"].astype(str)
                ),
            },
            index=selected.index,
        )
        selected = selected[
            keys["case_key"].isin(wanted)
            | keys["batch_case_key"].isin(wanted)
            | keys["dataset_batch_case_key"].isin(wanted)
        ]
    if limit is not None:
        selected = selected.head(limit)
    return selected.reset_index(drop=True)


def write_sample(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})


def run_trastrainer_case(
    spans_normalized: pd.DataFrame,
    metric_keys: List[Tuple[str, str]],
    budgets: List[float],
    base_seed: int,
    warm_up_size: int,
    window_size: Optional[int],
    enforce_stream_cap: bool,
    canonical_budget_index: int,
) -> Tuple[pd.DataFrame, dict]:
    """Run TraStrainer once per budget; emit ranker + per-budget decisions.

    Returns `(per_trace_frame, timing)`. Structure and feature extraction
    happen once and are reused across budgets via shared caches.
    """
    timing = {
        "extract_sec": 0.0,
        "structure_sec": 0.0,
        "feature_sec": 0.0,
        "trastrainer_canonical_sec": 0.0,
        "trastrainer_ranker_sec": 0.0,
        "trastrainer_total_sec": 0.0,
        "n_traces_streamed_total": 0,
    }

    if spans_normalized.empty or not len(budgets):
        empty = pd.DataFrame(
            columns=[
                "TraceID",
                "TraStrainerSystemRate",
                "TraStrainerDiversityRate",
                "TraStrainerCombined",
                "TraStrainerStrict",
                "TraStrainerWarmUp",
                "TraStrainerProcessed",
                "TraStrainerCanonicalBudget",
            ]
            + [f"TraStrainerDecision_{budget_token(b)}" for b in budgets]
        )
        return empty, timing

    canonical_budget = budgets[canonical_budget_index]
    structure_cache: dict = {}
    feature_cache: dict = {}

    per_budget_decisions: dict = {}
    canonical_capped_elapsed = 0.0

    for bi, budget in enumerate(budgets):
        cfg = TraStrainerConfig(
            warm_up_size=warm_up_size,
            window_size=window_size,
            seed=base_seed * 10_000 + bi,
            enforce_stream_cap=enforce_stream_cap,
        )
        pass_start = time.perf_counter()
        frame, pass_timing = run_trastrainer_over_trace_frame(
            spans_normalized,
            alpha=budget,
            metric_keys=metric_keys or None,
            metric_weights=None,
            config=cfg,
            structure_cache=structure_cache,
            feature_cache=feature_cache,
        )
        elapsed = time.perf_counter() - pass_start
        timing["trastrainer_total_sec"] += elapsed
        if bi == canonical_budget_index:
            canonical_capped_elapsed = elapsed
        timing["structure_sec"] = pass_timing.get("structure_sec", timing["structure_sec"])
        timing["feature_sec"] = pass_timing.get("feature_sec", timing["feature_sec"])
        timing["extract_sec"] = timing["structure_sec"] + timing["feature_sec"]
        timing["n_traces_streamed_total"] += int(pass_timing.get("n_traces_streamed", 0))

        per_budget_decisions[budget] = (
            frame.set_index("TraceID")["TraStrainerDecision"].astype(bool).to_dict()
        )

    # Separate ranker pass: upstream's `cnt >= round(alpha*N)` cap can fire
    # before warm-up completes on small cases, leaving zero scored decisions
    # at small α. The continuous ranker view (TraStrainerCombined etc.)
    # is therefore produced from one extra pass at the canonical budget with
    # the cap disabled, so every trace gets a system-bias / diversity-bias
    # score for the v0.2 evaluator. Per-budget Decision columns remain
    # faithful to upstream's online sampling.
    ranker_cfg = TraStrainerConfig(
        warm_up_size=warm_up_size,
        window_size=window_size,
        seed=base_seed * 10_000 + canonical_budget_index,
        enforce_stream_cap=False,
    )
    ranker_start = time.perf_counter()
    ranker_frame, ranker_timing = run_trastrainer_over_trace_frame(
        spans_normalized,
        alpha=canonical_budget,
        metric_keys=metric_keys or None,
        metric_weights=None,
        config=ranker_cfg,
        structure_cache=structure_cache,
        feature_cache=feature_cache,
    )
    ranker_elapsed = time.perf_counter() - ranker_start
    timing["trastrainer_ranker_sec"] = ranker_elapsed
    timing["trastrainer_canonical_sec"] = canonical_capped_elapsed
    timing["trastrainer_total_sec"] += ranker_elapsed

    out = ranker_frame[
        [
            "TraceID",
            "TraStrainerSystemRate",
            "TraStrainerDiversityRate",
            "TraStrainerCombined",
            "TraStrainerStrict",
            "TraStrainerWarmUp",
            "TraStrainerProcessed",
        ]
    ].copy()
    out["TraStrainerCanonicalBudget"] = canonical_budget
    for budget in budgets:
        col = f"TraStrainerDecision_{budget_token(budget)}"
        out[col] = out["TraceID"].map(per_budget_decisions[budget]).fillna(False).astype(bool)
    return out, timing


def run_case(entry, args: argparse.Namespace, run_dir: Path) -> dict:
    cdir = case_dir(args.data_root, entry.dataset, entry.batch, entry.case)
    manifest = pd.read_csv(entry.manifest_path)
    sample_path = run_dir / "samples" / f"{entry.dataset}__{entry.batch}__{entry.case}.trastrainer.csv.gz"
    log_path = run_dir / "logs" / f"{entry.dataset}__{entry.batch}__{entry.case}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "protocol_version": lp.PROTOCOL_VERSION,
        "dataset": entry.dataset,
        "batch": entry.batch,
        "case": entry.case,
        "trace_label_scope": entry.trace_label_scope,
        "seed": args.seed,
        "case_dir": cdir.as_posix(),
        "sample_path": sample_path.as_posix(),
        "status": "ok",
        "error": "",
        "elapsed_sec": 0.0,
        "data_load_sec": 0.0,
        "normalize_sec": 0.0,
        "extract_sec": 0.0,
        "structure_sec": 0.0,
        "feature_sec": 0.0,
        "trastrainer_canonical_sec": 0.0,
        "trastrainer_ranker_sec": 0.0,
        "trastrainer_total_sec": 0.0,
        "n_ranked": 0,
        "n_traces_streamed_total": 0,
        "n_metric_keys": 0,
        "metric_keys_source": "",
        "sample_sha256": "",
        "input_spans": 0,
        "normalized_spans": 0,
        "n_traces": 0,
    }
    start = time.time()
    try:
        t = time.perf_counter()
        raw = read_raw_trace_frame(cdir, entry.dataset)
        row["data_load_sec"] = round(time.perf_counter() - t, 3)
        row["input_spans"] = int(len(raw))
        t = time.perf_counter()
        normalized = lp.normalize_trace_frame(raw, entry.dataset)
        row["normalize_sec"] = round(time.perf_counter() - t, 3)
        row["normalized_spans"] = int(len(normalized))
        metric_keys, source = metric_keys_for_case(cdir, entry.dataset)
        row["metric_keys_source"] = source
        if not metric_keys:
            metric_keys = derive_service_keys(normalized)
            row["metric_keys_source"] = "synthesised_from_trace_services"
        row["n_metric_keys"] = int(len(metric_keys))

        with log_path.open("w") as handle, contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
            ts_out, timing = run_trastrainer_case(
                normalized,
                metric_keys=metric_keys,
                budgets=list(args.budgets),
                base_seed=args.seed,
                warm_up_size=args.warm_up_size,
                window_size=args.window_size,
                enforce_stream_cap=not args.no_stream_cap,
                canonical_budget_index=args.canonical_budget_index,
            )
        row["extract_sec"] = round(timing.get("extract_sec", 0.0), 3)
        row["structure_sec"] = round(timing.get("structure_sec", 0.0), 3)
        row["feature_sec"] = round(timing.get("feature_sec", 0.0), 3)
        row["trastrainer_canonical_sec"] = round(timing.get("trastrainer_canonical_sec", 0.0), 3)
        row["trastrainer_ranker_sec"] = round(timing.get("trastrainer_ranker_sec", 0.0), 3)
        row["trastrainer_total_sec"] = round(timing.get("trastrainer_total_sec", 0.0), 3)
        row["n_traces_streamed_total"] = int(timing.get("n_traces_streamed_total", 0))
        row["n_traces"] = int(len(ts_out))

        base = manifest[["TraceID"]].copy()
        base["TraceID"] = base["TraceID"].astype(str)
        if ts_out.empty:
            sample = base.copy()
            sample["TraStrainerSystemRate"] = float("nan")
            sample["TraStrainerDiversityRate"] = float("nan")
            sample["TraStrainerCombined"] = float("nan")
            sample["TraStrainerStrict"] = False
            sample["TraStrainerWarmUp"] = False
            sample["TraStrainerProcessed"] = False
            sample["TraStrainerCanonicalBudget"] = args.budgets[args.canonical_budget_index]
            for budget in args.budgets:
                sample[f"TraStrainerDecision_{budget_token(budget)}"] = False
        else:
            ts_out = ts_out.copy()
            ts_out["TraceID"] = ts_out["TraceID"].astype(str)
            sample = base.merge(ts_out, on="TraceID", how="left")
            for budget in args.budgets:
                col = f"TraStrainerDecision_{budget_token(budget)}"
                sample[col] = sample[col].fillna(False).astype(bool)
            for bool_col in ("TraStrainerStrict", "TraStrainerWarmUp", "TraStrainerProcessed"):
                sample[bool_col] = sample[bool_col].fillna(False).astype(bool)
        row["n_ranked"] = int(sample["TraStrainerCombined"].notna().sum())
        for budget in args.budgets:
            col = f"TraStrainerDecision_{budget_token(budget)}"
            row[f"n_kept_{budget_token(budget)}"] = int(sample[col].sum())
            row[f"n_target_{budget_token(budget)}"] = int(
                min(len(sample), max(1, round(len(sample) * budget)))
            )
        write_sample(sample, sample_path)
        row["sample_sha256"] = sha256_file(sample_path)
    except Exception as exc:
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["elapsed_sec"] = round(time.time() - start, 3)
    return row


def render_run_summary(sample_index: pd.DataFrame, eval_summary: Path, run_dir: Path) -> str:
    lines = [
        "# v0.2 TraStrainer Baseline Run",
        "",
        "Version-stamped candidate baseline run over frozen v0.2 trace manifests.",
        "TraStrainer (Huang et al., FSE 2024) reimplemented in numpy/pandas;",
        "upstream PyTorch + LTSF_Linear stack omitted (no published checkpoints",
        "for our datasets). System-bias features come from the trace itself,",
        "matching upstream's behaviour when LTSF checkpoints are absent",
        "(metric weights default to 1; see baselines/trastrainer.py for the",
        "upstream mapping).",
        "",
        "Runtime columns: `trastrainer_canonical_sec` is one-α sampling time",
        "(matches the paper's Sec. 5 efficiency-table column);",
        "`trastrainer_total_sec` sums across all five budgets;",
        "`extract_sec` is the per-case structure + feature extraction cost.",
        "`elapsed_sec` is wall time per case including IO.",
        "",
        "## Run Status",
        "",
        "| dataset | status | cases | ranked traces | canonical sec total | total sec | mean sec/case | std sec/case | traces/sec |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for keys, group in sample_index.groupby(["dataset", "status"], sort=True):
        dataset, status = keys
        total = float(group["elapsed_sec"].sum())
        mean = float(group["elapsed_sec"].mean())
        std = float(group["elapsed_sec"].std(ddof=1)) if len(group) > 1 else float("nan")
        ranked = int(group["n_ranked"].sum())
        thr = ranked / total if total > 0 else float("nan")
        canonical_total = float(group.get("trastrainer_canonical_sec", pd.Series(dtype=float)).sum())
        lines.append(
            f"| {dataset} | {status} | {len(group)} | {ranked} | "
            f"{canonical_total:.1f} | {total:.1f} | {mean:.2f} | {std:.2f} | {thr:.1f} |"
        )
    lines.extend(["", "## Evaluator Summary", ""])
    if eval_summary.exists():
        lines.extend(eval_summary.read_text().splitlines()[4:])
    else:
        lines.append("Evaluator summary not found.")
    text = "\n".join(lines) + "\n"
    (run_dir / "run_summary.md").write_text(text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--manifest-index", type=Path, default=Path("results/label_protocol_v02/trace_manifest_index.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("results/v02_sampling_runs"))
    parser.add_argument("--datasets", nargs="+", default=["hipster", "train_ticket"])
    parser.add_argument("--scope", choices=["primary", "all", "excluded"], default="primary")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="Optional case filters: case, batch/case, or dataset/batch/case.",
    )
    parser.add_argument("--budgets", nargs="+", type=float, default=BUDGETS)
    parser.add_argument("--seed", type=int, default=1)
    # Defaults follow upstream sampler.py demo + paper §IV.
    parser.add_argument("--warm-up-size", type=int, default=10)
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="History window for system / diversity bias. Default int(1/alpha) per budget.",
    )
    parser.add_argument(
        "--no-stream-cap",
        action="store_true",
        help=(
            "Disable upstream's `cnt >= round(alpha*N)` early termination so "
            "every trace gets a decision. Default off (faithful to upstream)."
        ),
    )
    parser.add_argument(
        "--canonical-budget-index",
        type=int,
        default=2,
        help="Which budget pass supplies the ranker columns. Default index is 2 (zero-based).",
    )
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()

    if args.canonical_budget_index < 0 or args.canonical_budget_index >= len(args.budgets):
        raise SystemExit(
            f"--canonical-budget-index {args.canonical_budget_index} out of range for {len(args.budgets)} budgets"
        )

    manifest_index = pd.read_csv(args.manifest_index)
    selected = select_cases(manifest_index, args.datasets, args.scope, args.limit, args.cases)
    if selected.empty:
        raise SystemExit("No cases selected.")

    fingerprint = run_fingerprint()
    budget_part = "-".join(budget_token(b) for b in args.budgets)
    window_token = "wAuto" if args.window_size is None else f"w{args.window_size}"
    cap_token = "noCap" if args.no_stream_cap else "cap"
    run_id = args.run_id or (
        f"trastrainer_v02_{'_'.join(args.datasets)}_{args.scope}_seed-{args.seed}_"
        f"budgets-{budget_part}_warm{args.warm_up_size}_{window_token}_{cap_token}_"
        f"fp-{fingerprint}"
    )
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    provenance = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": lp.PROTOCOL_VERSION,
        "unisage_git_head": git_head(HERE),
        "code_fingerprint": fingerprint,
        "runtime": runtime_info(),
        "manifest_index": args.manifest_index.as_posix(),
        "manifest_index_sha256": sha256_file(args.manifest_index),
        "datasets": args.datasets,
        "scope": args.scope,
        "limit": args.limit,
        "cases": args.cases,
        "budgets": args.budgets,
        "seed": args.seed,
        "trastrainer_parameters": {
            "warm_up_size": args.warm_up_size,
            "window_size": args.window_size,  # None => upstream default int(1/alpha)
            "enforce_stream_cap": not args.no_stream_cap,
            "canonical_budget_index": args.canonical_budget_index,
            "ltsf_checkpoints": "absent_default_weight_1",
            "metric_source_aiops21": "abnormal/metrics.csv with INCLUDED_METRICS filter",
            "metric_source_hipster_train_ticket": "synthesised_one_key_per_service",
        },
        "implementation_note": (
            "TraStrainer reimplemented from Huang et al. (FSE 2024). Upstream "
            "PyTorch / LTSF_Linear forecasting stack omitted: published "
            "checkpoints do not cover our datasets, and the upstream code "
            "falls back to weight=1 per metric when no checkpoint exists. "
            "Per-trace system-bias feature comes from the trace itself "
            "(num_spans * avg_duration * (1 + n_failures) per service), "
            "matching `TraceProcessor.compute_feature_values`. Diversity-bias "
            "uses upstream multiset Jaccard similarity over sorted "
            "`depth-service-op-status-bucket` tokens with the running "
            "diversity_window normalisation. AND/OR voting follows upstream "
            "`judge`: AND when the current keep rate exceeds the target "
            "budget, OR otherwise."
        ),
        "runtime_fields_note": (
            "sample_index.csv columns trastrainer_canonical_sec / "
            "trastrainer_total_sec / extract_sec record sampling-only time "
            "per case for the Sec. 5 efficiency comparison. data_load_sec "
            "and normalize_sec are excluded from the sampling time, "
            "consistent with how the active paper times Sieve / Sifter / "
            "TraStrainer."
        ),
        "selected_cases": int(len(selected)),
        "sample_files_ignored_in_git": "results/v02_sampling_runs/*/samples/*.csv.gz",
    }
    (run_dir / "run_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    selected.to_csv(run_dir / "selected_cases.csv", index=False)

    rows = []
    for i, entry in enumerate(selected.itertuples(index=False), start=1):
        print(
            f"[trastrainer {i}/{len(selected)}] {entry.dataset}/{entry.batch}/{entry.case}",
            flush=True,
        )
        row = run_case(entry, args, run_dir)
        if row["status"] != "ok":
            print(f"  [error] {row['error']}", flush=True)
        rows.append(row)
        pd.DataFrame(rows).to_csv(run_dir / "sample_index.csv", index=False)
    sample_index = pd.DataFrame(rows)
    sample_index.to_csv(run_dir / "sample_index.csv", index=False)

    eval_dir = run_dir / "eval"
    cmd = [
        sys.executable,
        "scripts/evaluate_trace_sampling.py",
        "--manifest-index",
        args.manifest_index.as_posix(),
        "--sample-index",
        (run_dir / "sample_index.csv").as_posix(),
        "--output",
        eval_dir.as_posix(),
        "--rankings",
        "trastrainer",
        "trastrainer_decision",
        "uniform",
        "--budgets",
        *[str(b) for b in args.budgets],
        "--mode",
        "version_stamped_trastrainer_trace",
    ]
    subprocess.run(cmd, check=True, cwd=HERE)
    summary = render_run_summary(sample_index, eval_dir / "trace_eval_summary.md", run_dir)
    print(summary)
    print(f"Wrote run artifacts to {run_dir}")


if __name__ == "__main__":
    main()
