"""Run a version-stamped UniSage trace-sampling comparison against v0.2.

The runner executes UniSage for the cases listed in the frozen manifest index,
writes compact per-case sampling frames, then invokes scripts/evaluate_trace_sampling.py.
Per-case sample frames are regenerable and gitignored; compact indexes and
summaries are kept as provenance.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from aiomultiprocess import Pool


UNIFIED_BUDGETS = ["0.001", "0.005", "0.01", "0.025", "0.05", "0.075", "0.10", "0.20"]
UNIFIED_BUDGETS_ENV = ",".join(UNIFIED_BUDGETS)


def budget_args_from_env(name: str) -> List[str]:
    return [item.strip() for item in os.environ.get(name, UNIFIED_BUDGETS_ENV).split(",") if item.strip()]


# Set result-producing defaults before importing UniSage, because UniSage reads
# propagation weights/toggles at import time.
os.environ.setdefault("UNISAGE_EDGE_LAT", "raw")
os.environ.setdefault("UNISAGE_RCA_ATTR", "target")
# UniSage-v1 locked defaults (paper_impact/028): propagation guarded-ON,
# trace composition = dynamic_product_pattern_guard with floor 0.05,
# log composition = product (set via UNISAGE_LOG_COMPOSITION).
os.environ.setdefault("UNISAGE_PROPAGATE", "1")
os.environ.setdefault("UNISAGE_PROPAGATE_GUARD", "1")
os.environ.setdefault("UNISAGE_TRACE_COMPOSITION", "dynamic_product_pattern_guard")
os.environ.setdefault("UNISAGE_TRACE_RESERVE_MIN_SLOTS", "50")
os.environ.setdefault("UNISAGE_TRACE_PATTERN_GUARD_MIN_BUDGET", "0.05")
os.environ.setdefault("UNISAGE_TRACE_BUDGETS", UNIFIED_BUDGETS_ENV)
os.environ.setdefault("UNISAGE_LOG_COMPOSITION", "product")
os.environ.setdefault("UNISAGE_LOG_BUDGETS", UNIFIED_BUDGETS_ENV)

import UniSage  # noqa: E402
import sampling_labels as lp  # noqa: E402
from build_trace_manifest import sha256_file  # noqa: E402


SAMPLE_COLUMNS = [
    "TraceID",
    "CombinedSamplingProb",
    "EdgeCaseSamplingProb",
    "AnomalySamplingProb",
    "LatencySamplingProb",
    "TopologySamplingProb",
    "LatencyMaxZScore",
    "SamplingFallback",
]
OPTIONAL_SAMPLE_COLUMNS = [
    "TracePatternKey",
    "TracePatternCount",
    "StructuralPatternKey",
    "StructuralPatternCount",
    "ProductRankPercentile",
    "EdgeRankPercentile",
    "AnalysisRankPercentile",
    "AnalysisProductScore",
    "TraceCompositionStrategy",
    "TraceReserveMinSlots",
    "TracePatternGuardMinBudget",
]
OPTIONAL_SAMPLE_PREFIXES = (
    "SlotGatedReservedDecision_",
    "SlotGatedReservedMode_",
    "SlotGatedReservedTargetK_",
    "DynamicPatternGuardDecision_",
    "DynamicPatternGuardMode_",
    "DynamicPatternGuardTargetK_",
)


def case_dir(data_root: Path, dataset: str, batch: str, case: str) -> Path:
    if dataset == "aiops21":
        return data_root / "aiops21" / "all" / case
    return data_root / dataset / "batches" / batch / case


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


def run_fingerprint() -> str:
    paths: List[Path] = [
        Path("UniSage.py"),
        Path("sampling.py"),
        Path("utils.py"),
        Path("scripts/evaluate_trace_sampling.py"),
        Path("scripts/run_sampling_comparison.py"),
        Path("AD"),
        Path("RCL"),
    ]
    return file_fingerprint(paths)


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def dataset_type(dataset: str) -> str:
    return "aiops21" if dataset == "aiops21" else "hipster"


def compact_sample(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in SAMPLE_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    extra_columns = [
        c for c in OPTIONAL_SAMPLE_COLUMNS
        if c in out.columns and c not in SAMPLE_COLUMNS
    ]
    extra_columns.extend(
        c for c in sorted(out.columns)
        if any(c.startswith(prefix) for prefix in OPTIONAL_SAMPLE_PREFIXES)
        and c not in SAMPLE_COLUMNS
        and c not in extra_columns
    )
    columns = SAMPLE_COLUMNS + extra_columns
    return out[columns].drop_duplicates("TraceID").sort_values("TraceID").reset_index(drop=True)


def write_sample(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})


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


async def run_cases(args: argparse.Namespace, selected: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    rows = []
    # Resume support: if a prior partial sample exists, reuse it instead of
    # re-running the sampler. This makes long runs idempotent against external
    # interruption (sleep, OOM, kill). Algorithm bytes are unchanged.
    resume = bool(getattr(args, "resume", False))
    prior_index_path = run_dir / "sample_index.csv"
    prior_index: dict = {}
    if resume and prior_index_path.exists():
        try:
            prior_df = pd.read_csv(prior_index_path)
            for _, prow in prior_df.iterrows():
                key = (str(prow.get("dataset")), str(prow.get("batch")), str(prow.get("case")))
                prior_index[key] = prow.to_dict()
        except Exception as exc:
            print(f"[resume] failed to read prior sample_index.csv: {exc}", flush=True)
    async with Pool() as pool:
        for i, entry in enumerate(selected.itertuples(index=False), start=1):
            cdir = case_dir(args.data_root, entry.dataset, entry.batch, entry.case)
            sample_path = run_dir / "samples" / f"{entry.dataset}__{entry.batch}__{entry.case}.trace_sampling.csv.gz"
            key = (str(entry.dataset), str(entry.batch), str(entry.case))
            if resume and sample_path.exists():
                prior_row = prior_index.get(key)
                if prior_row is not None and prior_row.get("status") == "ok":
                    print(f"[run {i}/{len(selected)}] {entry.dataset}/{entry.batch}/{entry.case}  [resume: reuse]", flush=True)
                    rows.append(prior_row)
                    pd.DataFrame(rows).to_csv(prior_index_path, index=False)
                    continue
                # sample file exists without a prior index entry: trust the file, rebuild minimal row
                try:
                    n_ranked = int(pd.read_csv(sample_path, compression="gzip", usecols=["TraceID"]).shape[0])
                except Exception:
                    n_ranked = 0
                row = {
                    "protocol_version": lp.PROTOCOL_VERSION,
                    "dataset": entry.dataset,
                    "batch": entry.batch,
                    "case": entry.case,
                    "trace_label_scope": entry.trace_label_scope,
                    "case_dir": cdir.as_posix(),
                    "sample_path": sample_path.as_posix(),
                    "status": "ok",
                    "error": "",
                    "elapsed_sec": 0.0,
                    "n_ranked": n_ranked,
                    "sample_sha256": sha256_file(sample_path),
                }
                print(f"[run {i}/{len(selected)}] {entry.dataset}/{entry.batch}/{entry.case}  [resume: reattach sample]", flush=True)
                rows.append(row)
                pd.DataFrame(rows).to_csv(prior_index_path, index=False)
                continue
            print(f"[run {i}/{len(selected)}] {entry.dataset}/{entry.batch}/{entry.case}", flush=True)
            start = time.time()
            row = {
                "protocol_version": lp.PROTOCOL_VERSION,
                "dataset": entry.dataset,
                "batch": entry.batch,
                "case": entry.case,
                "trace_label_scope": entry.trace_label_scope,
                "case_dir": cdir.as_posix(),
                "sample_path": sample_path.as_posix(),
                "status": "ok",
                "error": "",
                "elapsed_sec": 0.0,
                "n_ranked": 0,
                "sample_sha256": "",
            }
            try:
                result = await UniSage.evaluate(cdir, pool, dataset_type(entry.dataset))
                if not result or result[3] is None or result[3].empty:
                    raise RuntimeError("UniSage.evaluate returned no trace sampling frame")
                sample = compact_sample(result[3])
                write_sample(sample, sample_path)
                row["n_ranked"] = len(sample)
                row["sample_sha256"] = sha256_file(sample_path)
            except Exception as exc:
                row["status"] = "error"
                row["error"] = f"{type(exc).__name__}: {exc}"
                print(f"  [error] {row['error']}", flush=True)
            row["elapsed_sec"] = round(time.time() - start, 3)
            rows.append(row)
            pd.DataFrame(rows).to_csv(prior_index_path, index=False)
    return pd.DataFrame(rows)


def render_run_summary(sample_index: pd.DataFrame, eval_summary: Path, run_dir: Path) -> str:
    lines = [
        "# v0.2 UniSage Sampling Run",
        "",
        "This is a version-stamped candidate run over frozen v0.2 trace manifests.",
        "",
        "## Run Status",
        "",
        "| dataset | status | cases | ranked traces | elapsed sec |",
        "|---|---|---:|---:|---:|",
    ]
    for keys, group in sample_index.groupby(["dataset", "status"], sort=True):
        dataset, status = keys
        lines.append(
            f"| {dataset} | {status} | {len(group)} | {int(group['n_ranked'].sum())} | "
            f"{group['elapsed_sec'].sum():.1f} |"
        )
    lines.extend(["", "## Evaluator Summary", ""])
    if eval_summary.exists():
        lines.extend(eval_summary.read_text().splitlines()[4:])
    else:
        lines.append("Evaluator summary not found.")
    text = "\n".join(lines) + "\n"
    (run_dir / "run_summary.md").write_text(text)
    return text


async def main_async(args: argparse.Namespace) -> None:
    manifest_index = pd.read_csv(args.manifest_index)
    selected = select_cases(manifest_index, args.datasets, args.scope, args.limit, args.cases)
    if selected.empty:
        raise SystemExit("No cases selected.")

    fingerprint = run_fingerprint()
    trace_composition = os.environ.get("UNISAGE_TRACE_COMPOSITION", "product")
    composition_token = f"_comp-{trace_composition}" if trace_composition not in {"", "product"} else ""
    rankings = list(dict.fromkeys(args.rankings))
    if trace_composition == "slot_gated_reserved_budget" and "slot_gated_reserved" not in rankings:
        rankings.append("slot_gated_reserved")
    if trace_composition == "dynamic_product_pattern_guard" and "dynamic_pattern_guard" not in rankings:
        rankings.append("dynamic_pattern_guard")
    evaluation_budgets = budget_args_from_env("UNISAGE_TRACE_BUDGETS")
    run_id = args.run_id or (
        f"unisage_trace_{'_'.join(args.datasets)}_{args.scope}_"
        f"attr-{os.environ.get('UNISAGE_RCA_ATTR')}_edge-{os.environ.get('UNISAGE_EDGE_LAT')}_"
        f"prop-{os.environ.get('UNISAGE_PROPAGATE')}{composition_token}_fp-{fingerprint}"
    )
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    provenance = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": lp.PROTOCOL_VERSION,
        "git_head": git_head(),
        "code_fingerprint": fingerprint,
        "manifest_index": args.manifest_index.as_posix(),
        "manifest_index_sha256": sha256_file(args.manifest_index),
        "datasets": args.datasets,
        "scope": args.scope,
        "limit": args.limit,
        "cases": args.cases,
        "env": {
            "UNISAGE_RCA_ATTR": os.environ.get("UNISAGE_RCA_ATTR"),
            "UNISAGE_EDGE_LAT": os.environ.get("UNISAGE_EDGE_LAT"),
            "UNISAGE_PROPAGATE": os.environ.get("UNISAGE_PROPAGATE"),
            "UNISAGE_PROPAGATE_GUARD": os.environ.get("UNISAGE_PROPAGATE_GUARD"),
            "UNISAGE_W_T": os.environ.get("UNISAGE_W_T", ""),
            "UNISAGE_W_L": os.environ.get("UNISAGE_W_L", ""),
            "UNISAGE_TRACE_COMPOSITION": trace_composition,
            "UNISAGE_TRACE_RESERVE_MIN_SLOTS": os.environ.get("UNISAGE_TRACE_RESERVE_MIN_SLOTS", ""),
            "UNISAGE_TRACE_PATTERN_GUARD_MIN_BUDGET": os.environ.get(
                "UNISAGE_TRACE_PATTERN_GUARD_MIN_BUDGET",
                "",
            ),
            "UNISAGE_TRACE_BUDGETS": os.environ.get("UNISAGE_TRACE_BUDGETS", ""),
            "UNISAGE_LOG_COMPOSITION": os.environ.get("UNISAGE_LOG_COMPOSITION", ""),
            "UNISAGE_LOG_BUDGETS": os.environ.get("UNISAGE_LOG_BUDGETS", ""),
        },
        "evaluation_rankings": rankings,
        "evaluation_budgets": evaluation_budgets,
        "selected_cases": int(len(selected)),
        "sample_files_ignored_in_git": "results/v02_sampling_runs/*/samples/*.csv.gz",
    }
    (run_dir / "run_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    selected.to_csv(run_dir / "selected_cases.csv", index=False)

    sample_index = await run_cases(args, selected, run_dir)
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
        "--mode",
        "version_stamped_unisage_trace",
        "--rankings",
        *rankings,
        "--budgets",
        *evaluation_budgets,
    ]
    subprocess.run(cmd, check=True)
    summary = render_run_summary(sample_index, eval_dir / "trace_eval_summary.md", run_dir)
    print(summary)
    print(f"Wrote run artifacts to {run_dir}")


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
    parser.add_argument("--run-id", default="")
    parser.add_argument(
        "--rankings",
        nargs="+",
        default=["combined", "edge", "analysis", "uniform"],
        help="Rankings passed to evaluate_trace_sampling; slot_gated_reserved is auto-added when enabled.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If a per-case sample file already exists under the run-id, reuse it instead of "
             "re-running the sampler. Makes long runs idempotent against external interruption.",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
