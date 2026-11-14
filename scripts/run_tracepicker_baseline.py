"""Run TracePicker as a version-stamped v0.2 trace-sampling baseline.

TracePicker is an external baseline implementation. This runner keeps the
external code out of the UniSage tree, converts each frozen v0.2 case into
TracePicker's Trace/Span entities, runs TracePicker independently for each
budget, and writes decision columns that scripts/evaluate_trace_sampling.py can evaluate
without treating the sampler as a ranker. On macOS, it can replace geatpy's
unavailable native optimizer module with a pure-Python DE-style backend while
preserving TracePicker's encoder, quota allocator, and sampling objective.
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
import types
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
            if name == "geatpy" and name in sys.modules and hasattr(sys.modules[name], "__UNISAGE_PORTABLE_BACKEND__"):
                versions[name] = getattr(sys.modules[name], "__UNISAGE_PORTABLE_BACKEND__")
            else:
                versions[name] = "not-installed"
    return versions


def runtime_info() -> dict:
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "dependencies": dependency_versions(["numpy", "pandas", "networkx", "treelib", "geatpy"]),
    }


def run_fingerprint(tracepicker_root: Path) -> str:
    paths: List[Path] = [
        HERE / "scripts/run_tracepicker_baseline.py",
        HERE / "scripts/evaluate_trace_sampling.py",
        HERE / "scripts/sampling_labels.py",
        tracepicker_root / "core",
        tracepicker_root / "entity",
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


def install_portable_geatpy_backend() -> None:
    """Install a geatpy-compatible pure-Python optimizer backend.

    TracePicker imports geatpy only to optimize SampleProblem2. The official
    geatpy 2.x package does not ship the required Darwin/arm64 native core, so
    this backend provides the small API surface TracePicker uses and evaluates
    the same TracePicker objective with a DE-best/1/bin-style discrete search.
    """
    if "geatpy" in sys.modules:
        return

    ea = types.ModuleType("geatpy")
    ea.__UNISAGE_PORTABLE_BACKEND__ = "portable-de"

    class Problem:
        def __init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin):
            self.name = name
            self.M = M
            self.maxormins = maxormins
            self.Dim = Dim
            self.varTypes = varTypes
            self.lb = lb
            self.ub = ub
            self.lbin = lbin
            self.ubin = ubin

    class Population:
        def __init__(self, Encoding, NIND):
            self.Encoding = Encoding
            self.NIND = NIND

    class Algorithm:
        def __init__(self, problem, population, MAXGEN, logTras):
            self.problem = problem
            self.population = population
            self.MAXGEN = MAXGEN
            self.logTras = logTras

    def soea_DE_best_1_bin_templet(problem, population, MAXGEN=10, logTras=0):
        return Algorithm(problem, population, MAXGEN, logTras)

    def optimize(algorithm, **_kwargs):
        problem = algorithm.problem
        nind = max(1, int(algorithm.population.NIND))
        maxgen = max(1, int(algorithm.MAXGEN))
        dim = int(problem.Dim)
        lb = np.array(problem.lb, dtype=float)
        ub = np.array(problem.ub, dtype=float)
        if dim == 0:
            vars_ = np.empty((1, 0), dtype=int)
            objv = problem.evalVars(vars_)
            return {"Vars": vars_, "ObjV": objv}

        pop = problem.randomPhen(nind).astype(int)
        objv = problem.evalVars(pop).reshape(-1)
        f = 0.5
        cr = 0.9
        for _ in range(maxgen):
            best_vec = pop[int(np.argmin(objv))].astype(float)
            trials = []
            for i in range(nind):
                if nind >= 3:
                    choices = [idx for idx in range(nind) if idx != i]
                    r1, r2 = random.sample(choices, 2)
                    mutant = best_vec + f * (pop[r1].astype(float) - pop[r2].astype(float))
                else:
                    mutant = problem.randomPhen(1)[0].astype(float)
                cross = np.random.random(dim) < cr
                cross[np.random.randint(0, dim)] = True
                trial = np.where(cross, mutant, pop[i].astype(float))
                trial = np.rint(np.clip(trial, lb, ub)).astype(int)
                trials.append(trial)

            candidates = np.vstack([pop, np.array(trials, dtype=int), problem.randomPhen(nind).astype(int)])
            candidate_objv = problem.evalVars(candidates).reshape(-1)
            keep = np.argsort(candidate_objv)[:nind]
            pop = candidates[keep].astype(int)
            objv = candidate_objv[keep]

        best = int(np.argmin(objv))
        return {"Vars": pop[[best]], "ObjV": objv[[best]].reshape(-1, 1)}

    ea.Problem = Problem
    ea.Population = Population
    ea.soea_DE_best_1_bin_templet = soea_DE_best_1_bin_templet
    ea.optimize = optimize
    sys.modules["geatpy"] = ea


def import_tracepicker(tracepicker_root: Path, optimizer_backend: str):
    if not (tracepicker_root / "core" / "TracePicker.py").exists():
        raise FileNotFoundError(f"TracePicker.py not found under {tracepicker_root}")
    if optimizer_backend == "portable-de":
        install_portable_geatpy_backend()
    root = tracepicker_root.resolve().as_posix()
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from core.TracePicker import TracePicker  # type: ignore
        from entity.Trace import Span, Trace  # type: ignore
    except Exception as exc:
        if optimizer_backend == "official-geatpy":
            raise RuntimeError(
                "Could not import TracePicker with the official runtime. "
                "Install TracePicker's dependencies, especially a compatible geatpy build. "
                "Use --optimizer-backend portable-de for the macOS-compatible port."
            ) from exc
        raise

    return TracePicker, Span, Trace


def normalized_timestamp(value: object) -> int:
    try:
        if pd.isna(value):
            return 0
        return int(pd.Timestamp(value).value)
    except Exception:
        return 0


def traces_from_frame(raw: pd.DataFrame, dataset: str, Span, Trace) -> Tuple[List[object], dict]:
    frame = lp.normalize_trace_frame(raw, dataset)
    if frame.empty:
        return [], {"input_spans": 0, "normalized_spans": 0, "traces": 0, "multi_root_traces": 0}

    traces = []
    multi_root_traces = 0
    for trace_id, group in frame.sort_values(["TraceID", "Start"]).groupby("TraceID", sort=False):
        group = group.copy()
        span_ids = set(group["SpanID"].astype(str))
        parent_ids = group["ParentID"].astype(str)
        root_mask = parent_ids.str.lower().isin({"root", "nan", "none", ""}) | ~parent_ids.isin(span_ids)
        if not root_mask.any():
            root_idx = group.index[0]
        else:
            root_idx = group.index[root_mask][0]
            if int(root_mask.sum()) > 1:
                multi_root_traces += 1
        root_span_id = str(group.at[root_idx, "SpanID"])

        records = []
        for row in group.itertuples(index=False):
            span_id = str(row.SpanID)
            parent_id = str(row.ParentID)
            if span_id == root_span_id:
                parent_span_id = "-1"
            elif parent_id in span_ids and parent_id != span_id:
                parent_span_id = parent_id
            else:
                parent_span_id = root_span_id
            records.append(
                {
                    "span_id": span_id,
                    "parent_span_id": parent_span_id,
                    "start": row.Start,
                    "duration": row.SpanDurationMs,
                    "service": row.Service,
                    "operation": row.Operation,
                }
            )

        by_id = {record["span_id"]: record for record in records}
        ordered_records = []
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(span_id: str) -> None:
            if span_id in visited:
                return
            if span_id in visiting:
                return
            visiting.add(span_id)
            record = by_id[span_id]
            parent_span_id = record["parent_span_id"]
            if parent_span_id != "-1" and parent_span_id in by_id:
                visit(parent_span_id)
            visiting.remove(span_id)
            visited.add(span_id)
            ordered_records.append(record)

        for record in records:
            visit(record["span_id"])

        spans = []
        for record in ordered_records:
            spans.append(
                Span(
                    normalized_timestamp(record["start"]),
                    float(record["duration"]),
                    0,
                    str(trace_id),
                    record["span_id"],
                    record["parent_span_id"],
                    str(record["service"]),
                    str(record["service"]),
                    str(record["operation"]),
                )
            )
        if spans:
            traces.append(Trace(str(trace_id), spans, False))

    return traces, {
        "input_spans": int(len(raw)),
        "normalized_spans": int(len(frame)),
        "traces": int(len(traces)),
        "multi_root_traces": int(multi_root_traces),
    }


def run_tracepicker_for_budget(
    traces: List[object],
    TracePicker,
    sample_rate: float,
    seed: int,
    buffer_size: int,
    pool_height: int,
    comb_count: int,
) -> set[str]:
    np.random.seed(seed)
    random.seed(seed)
    picker = TracePicker(
        bufferSize=buffer_size,
        poolHeight=pool_height,
        sampleRate=sample_rate,
        combCount=comb_count,
    )
    sampled_ids: List[str] = []
    for i, trace in enumerate(traces):
        end_signal = i == len(traces) - 1
        ids, _, _, _ = picker.accept(trace, end_signal)
        sampled_ids.extend(str(trace_id) for trace_id in ids)
    return set(sampled_ids)


def budget_target_count(n_total: int, budget: float) -> int:
    if n_total <= 0:
        return 0
    return min(n_total, max(1, int(round(n_total * budget))))


def effective_sample_rate(n_total: int, target_count: int) -> float:
    if n_total <= 0 or target_count <= 0:
        return 0.0
    if target_count >= n_total:
        return 1.0
    return min(1.0, (target_count + 1e-9) / n_total)


def write_sample(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})


def run_case(
    entry,
    args: argparse.Namespace,
    run_dir: Path,
    TracePicker,
    Span,
    Trace,
) -> dict:
    cdir = case_dir(args.data_root, entry.dataset, entry.batch, entry.case)
    manifest = pd.read_csv(entry.manifest_path)
    sample_path = run_dir / "samples" / f"{entry.dataset}__{entry.batch}__{entry.case}.tracepicker.csv.gz"
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
        "n_ranked": 0,
        "sample_sha256": "",
        "input_spans": 0,
        "normalized_spans": 0,
        "multi_root_traces": 0,
    }
    start = time.time()
    try:
        raw = read_raw_trace_frame(cdir, entry.dataset)
        traces, stats = traces_from_frame(raw, entry.dataset, Span, Trace)
        row.update(stats)
        trace_ids = manifest["TraceID"].astype(str)
        sample = pd.DataFrame({"TraceID": trace_ids})
        raw_trace_ids = {trace.traceID for trace in traces}
        sample["TracePickerSeen"] = sample["TraceID"].isin(raw_trace_ids)
        with log_path.open("w") as handle, contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
            for budget in args.budgets:
                target_count = budget_target_count(len(sample), budget)
                sample_rate = effective_sample_rate(len(traces), target_count)
                selected = run_tracepicker_for_budget(
                    traces,
                    TracePicker,
                    sample_rate,
                    args.seed,
                    args.buffer_size,
                    args.pool_height,
                    args.comb_count,
                )
                column = f"TracePickerDecision_{budget_token(budget)}"
                sample[column] = sample["TraceID"].isin(selected)
                row[f"n_target_{budget_token(budget)}"] = int(target_count)
                row[f"n_kept_{budget_token(budget)}"] = int(sample[column].sum())
                row[f"effective_rate_{budget_token(budget)}"] = float(sample_rate)
        row["n_ranked"] = int(sample["TracePickerSeen"].sum())
        write_sample(sample, sample_path)
        row["sample_sha256"] = sha256_file(sample_path)
    except Exception as exc:
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["elapsed_sec"] = round(time.time() - start, 3)
    return row


def render_run_summary(sample_index: pd.DataFrame, eval_summary: Path, run_dir: Path) -> str:
    lines = [
        "# v0.2 TracePicker Baseline Run",
        "",
        "This is a version-stamped candidate baseline run over frozen v0.2 trace manifests.",
        "Runtime: `elapsed_sec` is wall time per case. mean/std/throughput below use",
        "successful cases only; throughput is ranked_traces / total elapsed_sec.",
        "",
        "## Run Status",
        "",
        "| dataset | status | cases | ranked traces | total sec | mean sec/case | std sec/case | traces/sec |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for keys, group in sample_index.groupby(["dataset", "status"], sort=True):
        dataset, status = keys
        total = float(group["elapsed_sec"].sum())
        mean = float(group["elapsed_sec"].mean())
        std = float(group["elapsed_sec"].std(ddof=1)) if len(group) > 1 else float("nan")
        ranked = int(group["n_ranked"].sum())
        thr = ranked / total if total > 0 else float("nan")
        lines.append(
            f"| {dataset} | {status} | {len(group)} | {ranked} | "
            f"{total:.1f} | {mean:.2f} | {std:.2f} | {thr:.1f} |"
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
    parser.add_argument("--tracepicker-root", type=Path, default=os.environ.get("TRACEPICKER_ROOT"))
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
    parser.add_argument("--buffer-size", type=int, default=4000)
    parser.add_argument("--pool-height", type=int, default=1000)
    parser.add_argument("--comb-count", type=int, default=100)
    parser.add_argument(
        "--optimizer-backend",
        choices=["portable-de", "official-geatpy"],
        default="portable-de",
        help="Use portable-de on macOS; use official-geatpy only where geatpy's native core is available.",
    )
    parser.add_argument(
        "--allow-geatpy-shim",
        action="store_true",
        help="Deprecated alias for --optimizer-backend portable-de.",
    )
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()

    if args.allow_geatpy_shim:
        args.optimizer_backend = "portable-de"
    if args.tracepicker_root is None:
        raise SystemExit("Provide --tracepicker-root or set TRACEPICKER_ROOT.")
    args.tracepicker_root = Path(args.tracepicker_root).resolve()
    TracePicker, Span, Trace = import_tracepicker(args.tracepicker_root, args.optimizer_backend)

    manifest_index = pd.read_csv(args.manifest_index)
    selected = select_cases(manifest_index, args.datasets, args.scope, args.limit, args.cases)
    if selected.empty:
        raise SystemExit("No cases selected.")

    fingerprint = run_fingerprint(args.tracepicker_root)
    budget_part = "-".join(budget_token(b) for b in args.budgets)
    run_id = args.run_id or (
        f"tracepicker_v02_{'_'.join(args.datasets)}_{args.scope}_seed-{args.seed}_"
        f"budgets-{budget_part}_opt-{args.optimizer_backend}_fp-{fingerprint}"
    )
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    provenance = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": lp.PROTOCOL_VERSION,
        "unisage_git_head": git_head(HERE),
        "tracepicker_git_head": git_head(args.tracepicker_root),
        "code_fingerprint": fingerprint,
        "tracepicker_root": args.tracepicker_root.as_posix(),
        "runtime": runtime_info(),
        "manifest_index": args.manifest_index.as_posix(),
        "manifest_index_sha256": sha256_file(args.manifest_index),
        "datasets": args.datasets,
        "scope": args.scope,
        "limit": args.limit,
        "cases": args.cases,
        "budgets": args.budgets,
        "seed": args.seed,
        "tracepicker_parameters": {
            "buffer_size": args.buffer_size,
            "pool_height": args.pool_height,
            "comb_count": args.comb_count,
            "optimizer_backend": args.optimizer_backend,
        },
        "optimizer_backend": args.optimizer_backend,
        "warning": (
            "TracePicker was run with the macOS-compatible portable-de optimizer backend "
            "because official geatpy 2.x native packages are unavailable on this platform. "
            "Report this implementation adaptation with the baseline."
            if args.optimizer_backend == "portable-de"
            else ""
        ),
        "selected_cases": int(len(selected)),
        "sample_files_ignored_in_git": "results/v02_sampling_runs/*/samples/*.csv.gz",
    }
    (run_dir / "run_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    selected.to_csv(run_dir / "selected_cases.csv", index=False)

    rows = []
    for i, entry in enumerate(selected.itertuples(index=False), start=1):
        print(f"[tracepicker {i}/{len(selected)}] {entry.dataset}/{entry.batch}/{entry.case}", flush=True)
        row = run_case(entry, args, run_dir, TracePicker, Span, Trace)
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
        "tracepicker",
        "uniform",
        "--budgets",
        *[str(b) for b in args.budgets],
        "--mode",
        "version_stamped_tracepicker_trace",
    ]
    subprocess.run(cmd, check=True, cwd=HERE)
    summary = render_run_summary(sample_index, eval_dir / "trace_eval_summary.md", run_dir)
    print(summary)
    print(f"Wrote run artifacts to {run_dir}")


if __name__ == "__main__":
    main()
