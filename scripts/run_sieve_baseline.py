"""Run Sieve as a version-stamped v0.2 trace-sampling baseline.

Sieve (Huang et al., ICWS 2021) ships an upstream Python reference at
https://github.com/IntelligentDDS/Sieve. We vendor that reference verbatim
(with the minimal patches documented in `baselines/sieve_upstream/__init__.py`)
and drive it through `baselines.sieve`, which mirrors the per-case adapter
shape used for the Sifter baseline.

The runner reads the frozen v0.2 trace manifest index, converts each case's
raw trace frame into Sieve's adapted Trace/Span shape, runs Sieve once per
case (Sieve's sampling rule is not parameterised by a target α), and writes
both a continuous `SieveScore` column (for ranker-style evaluation) and
per-budget `SieveDecision_<token>` columns produced by top-K selection on
that score. The companion v0.2 evaluator (extended with `sieve` and
`sieve_decision` rankings) picks up both column shapes.
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
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Iterable, List
import platform

import pandas as pd

import sampling_labels as lp
from evaluate_trace_sampling import BUDGETS, budget_token
from build_trace_manifest import sha256_file
from baselines.sieve import (
    SieveConfig,
    run_sieve_over_trace_frame,
    top_k_decision_by_score,
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
        HERE / "scripts/run_sieve_baseline.py",
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


def run_case(entry, args: argparse.Namespace, run_dir: Path) -> dict:
    cdir = case_dir(args.data_root, entry.dataset, entry.batch, entry.case)
    manifest = pd.read_csv(entry.manifest_path)
    sample_path = run_dir / "samples" / f"{entry.dataset}__{entry.batch}__{entry.case}.sieve.csv.gz"
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
        "n_traces": 0,
    }
    start = time.time()
    try:
        raw = read_raw_trace_frame(cdir, entry.dataset)
        row["input_spans"] = int(len(raw))
        normalized = lp.normalize_trace_frame(raw, entry.dataset)
        row["normalized_spans"] = int(len(normalized))
        cfg = SieveConfig(
            tree_num=args.tree_num,
            tree_size=args.tree_size,
            k=args.k,
            threshold=args.threshold,
            var_limit=args.var_limit,
            compact_interval=args.compact_interval,
            seed=args.seed,
        )
        with log_path.open("w") as handle, contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
            sieve_out = run_sieve_over_trace_frame(normalized, config=cfg)
        row["n_traces"] = int(len(sieve_out))

        base = manifest[["TraceID"]].copy()
        base["TraceID"] = base["TraceID"].astype(str)
        if sieve_out.empty:
            sample = base.copy()
            sample["SieveScore"] = float("nan")
            sample["SieveProbability"] = float("nan")
            sample["SieveNumSpans"] = 0
            for budget in args.budgets:
                sample[f"SieveDecision_{budget_token(budget)}"] = False
        else:
            sieve_out = sieve_out.copy()
            sieve_out["TraceID"] = sieve_out["TraceID"].astype(str)
            sample = base.merge(sieve_out, on="TraceID", how="left")
            sample["SieveNumSpans"] = sample["SieveNumSpans"].fillna(0).astype(int)
            ranked = sample.dropna(subset=["SieveScore"]).copy()
            for budget in args.budgets:
                col = f"SieveDecision_{budget_token(budget)}"
                target = max(1, int(round(len(sample) * budget))) if len(sample) else 0
                decision = pd.Series(False, index=sample.index)
                if target > 0 and not ranked.empty:
                    keep_idx = top_k_decision_by_score(ranked, target_count=target)
                    decision.loc[keep_idx[keep_idx].index] = True
                sample[col] = decision
        row["n_ranked"] = int(sample["SieveScore"].notna().sum())
        for budget in args.budgets:
            col = f"SieveDecision_{budget_token(budget)}"
            row[f"n_kept_{budget_token(budget)}"] = int(sample[col].sum())
            row[f"n_target_{budget_token(budget)}"] = int(
                min(len(sample), max(1, round(len(sample) * budget)))
            )
        write_sample(sample, sample_path)
        row["sample_sha256"] = sha256_file(sample_path)
    except Exception as exc:
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["traceback"] = traceback.format_exc()
    row["elapsed_sec"] = round(time.time() - start, 3)
    return row


def render_run_summary(sample_index: pd.DataFrame, eval_summary: Path, run_dir: Path) -> str:
    lines = [
        "# v0.2 Sieve Baseline Run",
        "",
        "Version-stamped candidate baseline run over frozen v0.2 trace manifests.",
        "Sieve vendored verbatim from IntelligentDDS/Sieve (Huang et al., ICWS 2021)",
        "with the minimal patches recorded in `baselines/sieve_upstream/__init__.py`.",
        "Compare against TracePicker / Sifter / UniSage / Uniform under the same v0.2 metric set.",
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
    ok_group = sample_index[sample_index["status"] == "ok"]
    if not ok_group.empty:
        lines.extend([
            "",
            "Per-case runtime is recorded in `sample_index.csv:elapsed_sec`. Mean/std above use "
            "successful cases only and exclude any case that errored out. Trace throughput is "
            "`ranked_traces / total_sec`; compare to Sifter / TracePicker / UniSage runs under "
            "the same env (recorded in `run_provenance.json:runtime`).",
        ])
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
    # Defaults follow Huang et al. ICWS 2021 §VI and the upstream README.
    parser.add_argument("--tree-num", type=int, default=50)
    parser.add_argument("--tree-size", type=int, default=128)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--var-limit", type=float, default=0.1)
    parser.add_argument("--compact-interval", type=int, default=128)
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()

    manifest_index = pd.read_csv(args.manifest_index)
    selected = select_cases(manifest_index, args.datasets, args.scope, args.limit, args.cases)
    if selected.empty:
        raise SystemExit("No cases selected.")

    fingerprint = run_fingerprint()
    budget_part = "-".join(budget_token(b) for b in args.budgets)
    run_id = args.run_id or (
        f"sieve_v02_{'_'.join(args.datasets)}_{args.scope}_seed-{args.seed}_"
        f"budgets-{budget_part}_T{args.tree_num}_S{args.tree_size}_k{args.k}_"
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
        "sieve_parameters": {
            "tree_num": args.tree_num,
            "tree_size": args.tree_size,
            "k": args.k,
            "threshold": args.threshold,
            "var_limit": args.var_limit,
            "compact_interval": args.compact_interval,
            "label_granularity": "service:operation",
        },
        "implementation_note": (
            "Sieve vendored verbatim from IntelligentDDS/Sieve (ICWS 2021 reference). "
            "Adapter `baselines/sieve.py` exposes a stable RandomState per RRCF tree "
            "(base_seed * 1000 + tree_idx) so the forest is reproducible *and* diverse. "
            "Per-trace SieveScore is the mean codisp across trees; SieveProbability is "
            "Sieve's native variance-windowed sampling probability, guarded against "
            "math.exp overflow. Per-budget SieveDecision_<token> columns are deterministic "
            "top-K-by-SieveScore selections at each target α, matching the tail-based "
            "evaluation convention used for the other v0.2 baselines."
        ),
        "selected_cases": int(len(selected)),
        "sample_files_ignored_in_git": "results/v02_sampling_runs/*/samples/*.csv.gz",
    }
    (run_dir / "run_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    selected.to_csv(run_dir / "selected_cases.csv", index=False)

    rows = []
    for i, entry in enumerate(selected.itertuples(index=False), start=1):
        print(f"[sieve {i}/{len(selected)}] {entry.dataset}/{entry.batch}/{entry.case}", flush=True)
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
        "sieve",
        "sieve_decision",
        "uniform",
        "--budgets",
        *[str(b) for b in args.budgets],
        "--mode",
        "version_stamped_sieve_trace",
    ]
    subprocess.run(cmd, check=True, cwd=HERE)
    summary = render_run_summary(sample_index, eval_dir / "trace_eval_summary.md", run_dir)
    print(summary)
    print(f"Wrote run artifacts to {run_dir}")


if __name__ == "__main__":
    main()
