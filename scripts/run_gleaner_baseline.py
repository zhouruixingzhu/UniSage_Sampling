"""Run Gleaner as a version-stamped v0.2 trace-sampling baseline.

Gleaner (Yang et al., arXiv:2604.16810v1, 2026) is a *log-enriched* online tail
sampler. This runner uses `baselines.gleaner`, which reimplements Gleaner's
EPS encoder + anomaly score + greedy-DPP selector from the paper in numpy. It
mirrors `scripts/run_sifter_baseline.py`: it reads the frozen v0.2 trace manifest
index, normalizes each case's raw traces, loads and attaches the case's parsed
logs, runs Gleaner at every target budget, and writes both a continuous
`GleanerScore` column (A(T), for ranker-style evaluation) and per-budget
`GleanerDecision_<token>` columns (the DPP-selected sets). The companion v0.2
evaluator picks up both column shapes.

Faithful-but-bounded adaptations to the offline, fixed-budget v0.2 setting are
documented in `paper_impact/019_gleaner_baseline.md`:
  * the alarm-driven quota allocator is dropped (no live alert stream; deriving
    alarms from the injected-fault window would leak the label) -> even quota
    across root-API groups;
  * span error status is derived from error-level logs (raw traces in these
    datasets carry no span status column);
  * parsed Hipster/Train-Ticket logs have no SpanID column, so log->span
    attribution recovers SpanID from the log Body text when present and
    otherwise falls back to the matching service's earliest span.
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
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Iterable, List
import platform

import pandas as pd

import sampling_labels as lp
from evaluate_trace_sampling import BUDGETS, budget_token
from build_trace_manifest import sha256_file
import baselines.gleaner as gl


HERE = Path(__file__).resolve().parent.parent
SPANID_RE = re.compile(r"SpanID:\s*([0-9a-fA-F]+)")


def case_dir(data_root: Path, dataset: str, batch: str, case: str) -> Path:
    if dataset == "aiops21":
        return data_root / "aiops21" / "all" / case
    return data_root / dataset / "batches" / batch / case


def read_raw_trace_frame(cdir: Path, dataset: str) -> pd.DataFrame:
    if dataset == "aiops21":
        return pd.read_csv(cdir / "abnormal" / "traces.csv")
    return lp.read_benchmark_trace_files(cdir)


def load_case_logs(cdir: Path, dataset: str) -> pd.DataFrame:
    """Return a common-schema log frame (gl.LOG_COLUMNS) for one case.

    Hipster/Train Ticket: reuse `lp.read_parsed_logs` (drain3-parsed templates +
    severity); recover SpanID from the log Body when present. AIOps21: adapt
    `abnormal/logs.csv` (already carries SpanId + SeverityText). Empty/sparse
    joins are kept, not dropped, so EPS encoding degrades gracefully.
    """
    if dataset == "aiops21":
        path = cdir / "abnormal" / "logs.csv"
        if not path.exists():
            return pd.DataFrame(columns=gl.LOG_COLUMNS)
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
        out = pd.DataFrame(
            {
                "TraceID": raw.get("TraceId", ""),
                "SpanID": raw.get("SpanId", ""),
                "Service": raw.get("ServiceName", "").map(lp.base_service),
                "log_level": raw.get("SeverityText", ""),
                "log_temp": raw.get("Body", ""),
            }
        )
    else:
        logs = lp.read_parsed_logs(cdir / "log" / "parsed")
        if logs.empty:
            return pd.DataFrame(columns=gl.LOG_COLUMNS)
        span = logs["Body"].astype(str).str.extract(SPANID_RE, expand=False).fillna("")
        out = pd.DataFrame(
            {
                "TraceID": logs["TraceID"].astype(str),
                "SpanID": span,
                "Service": logs["Service"].astype(str),
                "log_level": logs["log_level"].astype(str),
                "log_temp": logs["log_temp"].astype(str),
            }
        )
    out["TraceID"] = out["TraceID"].astype(str)
    out["SpanID"] = out["SpanID"].fillna("").astype(str)
    out["log_level"] = out["log_level"].fillna("").astype(str)
    out["log_temp"] = out["log_temp"].fillna("").astype(str)
    return out[gl.LOG_COLUMNS]


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
        HERE / "scripts/run_gleaner_baseline.py",
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


def make_config(args: argparse.Namespace) -> gl.GleanerConfig:
    return gl.GleanerConfig(
        w_err=args.w_err,
        w_lw=args.w_lw,
        w_le=args.w_le,
        w_perf=args.w_perf,
        latency_p90_mult=args.latency_p90_mult,
        quality_scale=args.quality_scale,
        dpp_candidate_cap=args.dpp_candidate_cap,
        batch_size=args.batch_size,
        seed=args.seed,
    )


def run_case(entry, args: argparse.Namespace, run_dir: Path) -> dict:
    cdir = case_dir(args.data_root, entry.dataset, entry.batch, entry.case)
    manifest = pd.read_csv(entry.manifest_path)
    sample_path = run_dir / "samples" / f"{entry.dataset}__{entry.batch}__{entry.case}.gleaner.csv.gz"
    log_path = run_dir / "logs" / f"{entry.dataset}__{entry.batch}__{entry.case}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    budgets = list(args.budgets)
    decision_cols = [f"GleanerDecision_{budget_token(b)}" for b in budgets]
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
        "log_load_sec": 0.0,
        "encode_sec": 0.0,
        "score_sec": 0.0,
        "dpp_sec": 0.0,
        "gleaner_sampling_sec": 0.0,
        "n_ranked": 0,
        "n_logs": 0,
        "n_logs_span_attributed": 0,
        "n_eps_events": 0,
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
        t = time.perf_counter()
        logs = load_case_logs(cdir, entry.dataset)
        row["log_load_sec"] = round(time.perf_counter() - t, 3)
        row["n_logs"] = int(len(logs))
        row["n_logs_span_attributed"] = int((logs["SpanID"].astype(str).str.len() > 0).sum()) if len(logs) else 0

        cfg = make_config(args)
        with log_path.open("w") as handle, contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
            gleaner_out, timing = gl.run_gleaner_over_trace_frame(
                normalized, logs, budgets=budgets, config=cfg, seed=args.seed
            )
        row["encode_sec"] = round(timing.get("encode_sec", 0.0), 3)
        row["score_sec"] = round(timing.get("score_sec", 0.0), 3)
        row["dpp_sec"] = round(timing.get("dpp_sec", 0.0), 3)
        row["gleaner_sampling_sec"] = round(
            timing.get("encode_sec", 0.0) + timing.get("score_sec", 0.0) + timing.get("dpp_sec", 0.0), 3
        )
        row["n_eps_events"] = int(timing.get("n_eps_events", 0))
        row["n_traces"] = int(len(gleaner_out))

        base = manifest[["TraceID"]].copy()
        base["TraceID"] = base["TraceID"].astype(str)
        if gleaner_out.empty:
            sample = base.copy()
            sample["GleanerScore"] = float("nan")
            sample["GleanerNumEvents"] = 0
            for col in decision_cols:
                sample[col] = False
        else:
            gleaner_out["TraceID"] = gleaner_out["TraceID"].astype(str)
            sample = base.merge(gleaner_out, on="TraceID", how="left")
            sample["GleanerNumEvents"] = sample["GleanerNumEvents"].fillna(0).astype(int)
            for col in decision_cols:
                sample[col] = sample[col].fillna(False).astype(bool)
        row["n_ranked"] = int(sample["GleanerScore"].notna().sum())
        for budget, col in zip(budgets, decision_cols):
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
        "# v0.2 Gleaner Baseline Run",
        "",
        "Version-stamped candidate baseline run over frozen v0.2 trace manifests.",
        "Gleaner reimplemented from Yang et al. (arXiv:2604.16810v1) Sec. 3 in numpy:",
        "EPS encoder + anomaly score + greedy-DPP selector, log-enriched.",
        "Compare against Sifter / Sieve / TraStrainer / TracePicker / Uniform / UniSage.",
        "",
        "Runtime columns: `gleaner_sampling_sec` = encode + score + dpp (the Sec. 5",
        "efficiency-table sampling time); `elapsed_sec` is wall time per case incl. IO.",
        "`n_logs_span_attributed` counts logs with a recovered/native SpanID.",
        "",
        "## Run Status",
        "",
        "| dataset | status | cases | ranked traces | sampling sec total | total sec | mean sec/case | traces/sec |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for keys, group in sample_index.groupby(["dataset", "status"], sort=True):
        dataset, status = keys
        total = float(group["elapsed_sec"].sum())
        mean = float(group["elapsed_sec"].mean())
        ranked = int(group["n_ranked"].sum())
        thr = ranked / total if total > 0 else float("nan")
        sampling_total = float(group.get("gleaner_sampling_sec", pd.Series(dtype=float)).sum())
        lines.append(
            f"| {dataset} | {status} | {len(group)} | {ranked} | "
            f"{sampling_total:.1f} | {total:.1f} | {mean:.2f} | {thr:.1f} |"
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
    # Gleaner hyperparameters (paper Sec. 3 defaults; w_perf is our documented choice).
    parser.add_argument("--w-err", type=float, default=5.0)
    parser.add_argument("--w-lw", type=float, default=1.0)
    parser.add_argument("--w-le", type=float, default=2.0)
    parser.add_argument("--w-perf", type=float, default=5.0)
    parser.add_argument("--latency-p90-mult", type=float, default=1.2)
    parser.add_argument("--quality-scale", type=float, default=1.0)
    parser.add_argument("--dpp-candidate-cap", type=int, default=400)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Online buffer size (#traces/window). seed=1 keeps canonical Start order; "
        "seed>1 permutes arrival order (trace-order perturbation variance protocol).",
    )
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()

    manifest_index = pd.read_csv(args.manifest_index)
    selected = select_cases(manifest_index, args.datasets, args.scope, args.limit, args.cases)
    if selected.empty:
        raise SystemExit("No cases selected.")

    fingerprint = run_fingerprint()
    budget_part = "-".join(budget_token(b) for b in args.budgets)
    run_id = args.run_id or (
        f"gleaner_v02_{'_'.join(args.datasets)}_{args.scope}_seed-{args.seed}_"
        f"budgets-{budget_part}_werr{args.w_err:g}_wle{args.w_le:g}_"
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
        "gleaner_parameters": {
            "w_err": args.w_err,
            "w_lw": args.w_lw,
            "w_le": args.w_le,
            "w_perf": args.w_perf,
            "latency_p90_mult": args.latency_p90_mult,
            "quality_scale": args.quality_scale,
            "dpp_candidate_cap": args.dpp_candidate_cap,
            "batch_size": args.batch_size,
            "anomaly_score": "sum_spans(w_err*1[error] + w_lw*|warn| + w_le*|error|) + w_perf*1[lat>mult*p90(group)]",
            "selection": "online_windowed(batch_size): even_quota_per_root_api_group + greedy_DPP_MAP(jaccard EPS) per buffer",
            "trace_order": "seed=1 canonical Start order (anchor); seed>1 permutes arrival order (variance protocol)",
            "log_to_span_attribution": "spanid_from_body_or_native; else earliest span of matching service",
        },
        "implementation_note": (
            "Gleaner (Yang et al., arXiv:2604.16810v1) reimplemented from Sec. 3 in numpy. "
            "Faithful-but-bounded adaptations to the offline fixed-budget v0.2 protocol: "
            "(1) the alarm-driven quota allocator is dropped (no live alert stream; deriving "
            "alarms from the injected-fault window would leak the ground-truth label) and "
            "replaced by even quota across root-API groups; (2) span error status is derived "
            "from error-level logs because these datasets' raw traces carry no span status "
            "field; (3) parsed Hipster/Train-Ticket logs lack a SpanID column, so log->span "
            "attribution recovers SpanID from the Body text or falls back to the matching "
            "service. Report these in the manuscript (paper_impact/020)."
        ),
        "runtime_fields_note": (
            "gleaner_sampling_sec = encode_sec + score_sec + dpp_sec records sampling-only "
            "time per case for the Sec. 5 efficiency comparison; data_load_sec / normalize_sec "
            "/ log_load_sec are excluded, consistent with the other baselines."
        ),
        "selected_cases": int(len(selected)),
        "sample_files_ignored_in_git": "results/v02_sampling_runs/*/samples/*.csv.gz",
    }
    (run_dir / "run_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    selected.to_csv(run_dir / "selected_cases.csv", index=False)

    rows = []
    for i, entry in enumerate(selected.itertuples(index=False), start=1):
        print(f"[gleaner {i}/{len(selected)}] {entry.dataset}/{entry.batch}/{entry.case}", flush=True)
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
        "gleaner",
        "gleaner_decision",
        "uniform",
        "--budgets",
        *[str(b) for b in args.budgets],
        "--mode",
        "version_stamped_gleaner_trace",
    ]
    subprocess.run(cmd, check=True, cwd=HERE)
    summary = render_run_summary(sample_index, eval_dir / "trace_eval_summary.md", run_dir)
    print(summary)
    print(f"Wrote run artifacts to {run_dir}")


if __name__ == "__main__":
    main()
