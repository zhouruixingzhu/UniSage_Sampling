"""Run Sifter as a version-stamped v0.2 trace-sampling baseline.

Sifter does not ship a public reference implementation, so this runner uses
`baselines.sifter`, which reimplements the algorithm from the SoCC 2019 paper
(Las-Casas et al.). The runner mirrors `scripts/run_tracepicker_baseline.py`: it
reads the frozen v0.2 trace manifest index, converts each case's raw trace
frame into Sifter's path inputs, runs Sifter at every target budget, and
writes both a continuous SifterLoss column (for ranker-style evaluation) and
per-budget SifterDecision_<token> columns (faithful to Sifter's online
sampling). The companion v0.2 evaluator picks up both column shapes.
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
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Iterable, List, Tuple
import platform

import numpy as np
import pandas as pd

import sampling_labels as lp
from evaluate_trace_sampling import BUDGETS, budget_token
from build_trace_manifest import sha256_file
from baselines.sifter import (
    SifterConfig,
    Sifter,
    extract_paths,
    trace_order_index,
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
        return subprocess.check_output(["git", "-C", path.as_posix(), "rev-parse", "HEAD"], text=True).strip()
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
        HERE / "scripts/run_sifter_baseline.py",
        HERE / "scripts/evaluate_trace_sampling.py",
        HERE / "scripts/sampling_labels.py",
        HERE / "baselines",
    ]
    return file_fingerprint(paths)


def select_cases(
    index: pd.DataFrame,
    datasets: List[str],
    scope: str,
    limit: int | None,
    cases: List[str] | None = None,
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


def run_sifter_case(
    spans_normalized: pd.DataFrame,
    budgets: List[float],
    cfg: SifterConfig,
    seed: int,
    canonical_budget_index: int = 0,
    max_paths_per_trace: int = 0,
) -> Tuple[pd.DataFrame, dict]:
    """Run Sifter once per budget (each with its own seeded RNG) over the
    streaming-ordered traces. The model dynamics are α-independent given the
    same seed and trace order, so SifterLoss/SifterProbability are recorded
    from the canonical pass and matched per TraceID; SifterDecision_<token>
    columns come from each pass's actual decisions at that target α.

    Returns (frame, timing) where `timing` contains per-budget sampling-only
    seconds (`sifter_canonical_sec`, `sifter_total_sec`, `extract_paths_sec`)
    that the Sec. 5 efficiency table relies on.
    """
    timing = {
        "extract_paths_sec": 0.0,
        "sifter_canonical_sec": 0.0,
        "sifter_total_sec": 0.0,
        "n_paths_total": 0,
    }

    order = trace_order_index(spans_normalized)
    if order.empty:
        empty = pd.DataFrame(
            columns=["TraceID", "SifterLoss", "SifterProbability", "SifterNumPaths"]
            + [f"SifterDecision_{budget_token(b)}" for b in budgets]
        )
        return empty, timing

    grouped = {tid: g for tid, g in spans_normalized.groupby("TraceID", sort=False)}
    # Pre-extract paths once; re-use across budgets. Path extraction is the
    # sampler's "feature engineering" cost and Sec. 5 lumps it under sampling.
    path_cache: dict[str, list[list[str]]] = {}
    t0 = time.perf_counter()
    for trace_id in order["TraceID"]:
        spans = grouped.get(trace_id)
        paths = (
            extract_paths(spans, path_length=cfg.path_length, pad_label=cfg.pad_label)
            if spans is not None
            else []
        )
        path_cache[str(trace_id)] = paths
        timing["n_paths_total"] += len(paths)
    timing["extract_paths_sec"] = time.perf_counter() - t0

    # Deterministic subsample for traces exceeding the path cap.
    n_capped = 0
    if max_paths_per_trace and max_paths_per_trace > 0:
        for tid, paths in path_cache.items():
            if len(paths) > max_paths_per_trace:
                rng = np.random.default_rng((hash(tid) & 0x7FFFFFFF) ^ seed)
                indices = rng.choice(len(paths), size=max_paths_per_trace, replace=False)
                indices.sort()
                path_cache[tid] = [paths[i] for i in indices]
                n_capped += 1
    timing["n_traces_path_capped"] = n_capped

    canonical_budget = budgets[canonical_budget_index]
    per_budget_decisions: dict[float, dict[str, bool]] = {b: {} for b in budgets}
    canonical_losses: dict[str, float] = {}
    canonical_probs: dict[str, float] = {}
    canonical_paths: dict[str, int] = {}

    for bi, budget in enumerate(budgets):
        cfg_pass = SifterConfig(
            path_length=cfg.path_length,
            embedding_dim=cfg.embedding_dim,
            window_size=cfg.window_size,
            learning_rate=cfg.learning_rate,
            pad_label=cfg.pad_label,
            seed=seed * 10_000 + bi,
        )
        sifter = Sifter(alpha=budget, config=cfg_pass)
        budget_start = time.perf_counter()
        for trace_id in order["TraceID"]:
            paths = path_cache[str(trace_id)]
            if paths:
                decision, loss, prob = sifter.sample(paths)
                n_paths = len(paths)
            else:
                decision, loss, prob, n_paths = False, float("nan"), float("nan"), 0
            per_budget_decisions[budget][str(trace_id)] = decision
            if bi == canonical_budget_index:
                canonical_losses[str(trace_id)] = loss
                canonical_probs[str(trace_id)] = prob
                canonical_paths[str(trace_id)] = n_paths
        elapsed = time.perf_counter() - budget_start
        timing["sifter_total_sec"] += elapsed
        if bi == canonical_budget_index:
            timing["sifter_canonical_sec"] = elapsed

    out = pd.DataFrame({"TraceID": order["TraceID"].astype(str)})
    out["SifterLoss"] = out["TraceID"].map(canonical_losses)
    out["SifterProbability"] = out["TraceID"].map(canonical_probs)
    out["SifterNumPaths"] = out["TraceID"].map(canonical_paths)
    out["SifterCanonicalBudget"] = canonical_budget
    for budget in budgets:
        col = f"SifterDecision_{budget_token(budget)}"
        out[col] = out["TraceID"].map(per_budget_decisions[budget]).fillna(False).astype(bool)
    return out, timing


def run_case(
    entry,
    args: argparse.Namespace,
    run_dir: Path,
) -> dict:
    cdir = case_dir(args.data_root, entry.dataset, entry.batch, entry.case)
    manifest = pd.read_csv(entry.manifest_path)
    sample_path = run_dir / "samples" / f"{entry.dataset}__{entry.batch}__{entry.case}.sifter.csv.gz"
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
        "extract_paths_sec": 0.0,
        "sifter_canonical_sec": 0.0,
        "sifter_total_sec": 0.0,
        "n_ranked": 0,
        "n_paths_total": 0,
        "n_traces_path_capped": 0,
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
        cfg = SifterConfig(
            path_length=args.path_length,
            embedding_dim=args.embedding_dim,
            window_size=args.window_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
        with log_path.open("w") as handle, contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
            sifter_out, timing = run_sifter_case(
                normalized,
                budgets=list(args.budgets),
                cfg=cfg,
                seed=args.seed,
                canonical_budget_index=args.canonical_budget_index,
                max_paths_per_trace=args.max_paths_per_trace,
            )
        row["extract_paths_sec"] = round(timing.get("extract_paths_sec", 0.0), 3)
        row["sifter_canonical_sec"] = round(timing.get("sifter_canonical_sec", 0.0), 3)
        row["sifter_total_sec"] = round(timing.get("sifter_total_sec", 0.0), 3)
        row["n_paths_total"] = int(timing.get("n_paths_total", 0))
        row["n_traces_path_capped"] = int(timing.get("n_traces_path_capped", 0))
        row["n_traces"] = int(len(sifter_out))
        if sifter_out.empty:
            sample = manifest[["TraceID"]].copy()
            sample["TraceID"] = sample["TraceID"].astype(str)
            sample["SifterLoss"] = float("nan")
            sample["SifterProbability"] = float("nan")
            sample["SifterNumPaths"] = 0
            sample["SifterCanonicalBudget"] = args.budgets[args.canonical_budget_index]
            for budget in args.budgets:
                sample[f"SifterDecision_{budget_token(budget)}"] = False
        else:
            base = manifest[["TraceID"]].copy()
            base["TraceID"] = base["TraceID"].astype(str)
            sample = base.merge(sifter_out, on="TraceID", how="left")
            for budget in args.budgets:
                col = f"SifterDecision_{budget_token(budget)}"
                sample[col] = sample[col].fillna(False).astype(bool)
            sample["SifterNumPaths"] = sample["SifterNumPaths"].fillna(0).astype(int)
        row["n_ranked"] = int(sample["SifterLoss"].notna().sum())
        for budget in args.budgets:
            col = f"SifterDecision_{budget_token(budget)}"
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
        "# v0.2 Sifter Baseline Run",
        "",
        "Version-stamped candidate baseline run over frozen v0.2 trace manifests.",
        "Sifter reimplemented from Las-Casas et al. (SoCC 2019) §5-§6 in numpy;",
        "no upstream reference code exists. Compare against TracePicker / Uniform / UniSage.",
        "",
        "Runtime columns: `sifter_canonical_sec` is one-α sampling time (matches",
        "the paper's Sec. 5 efficiency-table column); `sifter_total_sec` sums",
        "across all five budgets; `extract_paths_sec` is the per-case path-extraction cost.",
        "`elapsed_sec` is wall time per case including IO. mean/std/throughput use",
        "successful cases only; throughput is ranked_traces / total elapsed_sec.",
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
        canonical_total = float(group.get("sifter_canonical_sec", pd.Series(dtype=float)).sum())
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
    # Defaults follow Sifter SoCC '19 §6: N=5, P=10, k=50, lr=0.01.
    parser.add_argument("--path-length", type=int, default=5)
    parser.add_argument("--embedding-dim", type=int, default=10)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument(
        "--max-paths-per-trace",
        type=int,
        default=5000,
        help="Cap paths per trace (deterministic subsample seeded by trace_id). 0 = no cap.",
    )
    parser.add_argument(
        "--canonical-budget-index",
        type=int,
        default=2,
        help="Which budget's pass supplies SifterLoss/Probability. Default index is 2 (zero-based).",
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
    run_id = args.run_id or (
        f"sifter_v02_{'_'.join(args.datasets)}_{args.scope}_seed-{args.seed}_"
        f"budgets-{budget_part}_N{args.path_length}_P{args.embedding_dim}_k{args.window_size}_"
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
        "sifter_parameters": {
            "path_length_N": args.path_length,
            "embedding_dim_P": args.embedding_dim,
            "window_size_k": args.window_size,
            "learning_rate": args.learning_rate,
            "label_granularity": "service::operation",
            "path_extraction": "root_to_leaf_simple_paths_padded",
            "canonical_budget_index": args.canonical_budget_index,
        },
        "implementation_note": (
            "Sifter has no public reference code release. This baseline reimplements "
            "the SoCC 2019 algorithm (Las-Casas et al.) from §5-§6 in numpy: 2-layer "
            "label-prediction model with concatenated context embeddings, one batched "
            "backprop pass per trace (paper §5.4 semantics), k-window biased sampling "
            "per the paper's w_i = loss_i - min(loss); rho = w_{k+1}/sum(w) * (k+1) * "
            "alpha rule. Report this implementation in the manuscript."
        ),
        "runtime_fields_note": (
            "sample_index.csv columns sifter_canonical_sec / sifter_total_sec / "
            "extract_paths_sec record sampling-only time per case for the Sec. 5 "
            "efficiency comparison. data_load_sec and normalize_sec are excluded "
            "from the sampling time, consistent with how the active paper times "
            "Sieve / Sifter / TraStrainer."
        ),
        "selected_cases": int(len(selected)),
        "sample_files_ignored_in_git": "results/v02_sampling_runs/*/samples/*.csv.gz",
    }
    (run_dir / "run_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    selected.to_csv(run_dir / "selected_cases.csv", index=False)

    rows = []
    for i, entry in enumerate(selected.itertuples(index=False), start=1):
        print(f"[sifter {i}/{len(selected)}] {entry.dataset}/{entry.batch}/{entry.case}", flush=True)
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
        "sifter",
        "sifter_decision",
        "uniform",
        "--budgets",
        *[str(b) for b in args.budgets],
        "--mode",
        "version_stamped_sifter_trace",
    ]
    subprocess.run(cmd, check=True, cwd=HERE)
    summary = render_run_summary(sample_index, eval_dir / "trace_eval_summary.md", run_dir)
    print(summary)
    print(f"Wrote run artifacts to {run_dir}")


if __name__ == "__main__":
    main()
