"""Materialize v0.2 log-retention manifests with hashes.

This script is method-blind: it reads parsed logs, fault-free reference logs,
raw traces, and external fault metadata only. It must not read log-sampling
outputs, RCA rankings, AD output, or UniSage probabilities.

The output complements `scripts/build_trace_manifest.py` on the trace side by
freezing log labels and diversity keys before method comparison.
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
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Set

import numpy as np
import pandas as pd

import sampling_labels as lp


DEFAULT_DATASETS = ["hipster", "train_ticket"]
LOG_BUDGETS = [0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.10, 0.20]
LOG_MANIFEST_COLUMNS = [
    "LogID",
    "TraceID",
    "Timestamp",
    "Service",
    "log_level",
    "template_hash",
    "service_template_hash",
    "reference_template_count",
    "reference_template_frequency",
    "log_template_unseen",
    "log_template_rare_le_0p1",
    "log_template_rare_le_1p0",
    "in_primary_window",
    "log_manifested",
    "log_manifested_broad",
    "log_injected_service_window",
    "linked_A_exposed",
    "diagnostic_pair_strict",
    "diagnostic_pair_broad",
    "level_error",
    "level_warn",
    "level_error_or_warn",
    "has_trace_link",
]
LOG_LABEL_COLUMNS = [
    "log_template_unseen",
    "log_template_rare_le_0p1",
    "log_template_rare_le_1p0",
    "in_primary_window",
    "log_manifested",
    "log_manifested_broad",
    "log_injected_service_window",
    "linked_A_exposed",
    "diagnostic_pair_strict",
    "diagnostic_pair_broad",
    "level_error",
    "level_warn",
    "level_error_or_warn",
    "has_trace_link",
]
LOG_KEY_COLUMNS = ["_timestamp_ns", "TraceID", "Service", "log_level", "log_temp", "Body"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: object, length: int = 16) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:length]


def source_hash(paths: Iterable[Path]) -> dict:
    return {path.as_posix(): sha256_file(path) for path in paths if path.exists()}


def normalize_logs_for_manifest(logs: pd.DataFrame) -> pd.DataFrame:
    """Normalize parsed logs and attach a deterministic per-row LogID.

    Duplicate log rows are kept distinct by a per-key occurrence index. This
    lets the evaluator count repeated diagnostic lines without requiring a raw
    row id in the dataset.
    """
    if logs is None or logs.empty:
        return pd.DataFrame(columns=["LogID", *LOG_KEY_COLUMNS])
    frame = logs.copy()
    frame["_source_row"] = np.arange(len(frame))
    for column in ["TraceID", "Service", "log_level", "log_temp", "Body"]:
        if column not in frame.columns:
            frame[column] = ""
    ts = pd.to_datetime(frame["Timestamp"], errors="coerce", utc=True)
    frame = frame.loc[ts.notna()].copy()
    if frame.empty:
        return pd.DataFrame(columns=["LogID", *LOG_KEY_COLUMNS])
    ts = ts.loc[frame.index]
    frame["Timestamp"] = ts.dt.tz_localize(None)
    frame["_timestamp_ns"] = ts.astype("int64").astype(str)
    frame["TraceID"] = frame["TraceID"].fillna("").astype(str)
    frame["Service"] = frame["Service"].fillna("").map(lp.base_service)
    frame["log_level"] = frame["log_level"].fillna("").astype(str).str.upper()
    frame["log_temp"] = frame["log_temp"].fillna("").astype(str)
    frame["Body"] = frame["Body"].fillna("").astype(str)
    key = frame[LOG_KEY_COLUMNS].astype(str).agg("||".join, axis=1)
    occurrence = frame.groupby(key, sort=False).cumcount().astype(str)
    frame["LogID"] = (key + "||" + occurrence).map(lambda value: stable_hash(value, 24))
    return frame.reset_index(drop=True)


def reference_template_counts(ref_logs: pd.DataFrame) -> tuple[Counter, int]:
    if ref_logs is None or ref_logs.empty:
        return Counter(), 0
    ref = normalize_logs_for_manifest(ref_logs)
    if ref.empty:
        return Counter(), 0
    counts = Counter(ref["log_temp"])
    return counts, len(ref)


def a_exposed_trace_ids(
    dataset: str,
    batch: str,
    case: str,
    case_dir: Path,
    info: dict,
    model: dict,
    strict_ids: Set[str],
    broad_ids: Set[str],
) -> Set[str]:
    _, traces = lp.trace_case_summary(
        dataset,
        batch,
        case,
        info.get("inject_type", ""),
        info["inject_pod"],
        lp.trace_features(lp.read_benchmark_trace_files(case_dir), dataset),
        model,
        lp.injection_time(info["inject_time"]),
        strict_ids,
        broad_ids,
    )
    if traces.empty or "A_exposed" not in traces.columns:
        return set()
    return set(traces.loc[traces["A_exposed"], "TraceID"].astype(str))


def manifest_frame(
    logs: pd.DataFrame,
    ref_counts: Counter,
    ref_total: int,
    injected_at: pd.Timestamp,
    injected_service: str,
    a_exposed_ids: Set[str],
) -> pd.DataFrame:
    frame = normalize_logs_for_manifest(logs)
    if frame.empty:
        return pd.DataFrame(columns=LOG_MANIFEST_COLUMNS)

    frame["template_hash"] = frame["log_temp"].map(stable_hash)
    frame["service_template_hash"] = (
        frame["Service"].astype(str) + "::" + frame["log_temp"].astype(str)
    ).map(stable_hash)
    frame["reference_template_count"] = frame["log_temp"].map(ref_counts).fillna(0).astype(int)
    frame["reference_template_frequency"] = frame["reference_template_count"] / max(1, ref_total)
    frame["log_template_unseen"] = frame["reference_template_count"] == 0
    for name, threshold in lp.LOG_FREQ_THRESHOLDS.items():
        frame[name] = frame["log_template_unseen"] | (frame["reference_template_frequency"] <= threshold)

    frame["in_primary_window"] = lp.primary_window_mask(frame["Timestamp"], injected_at)
    strict_symptom, broad_symptom = lp.symptom_log_masks(frame)
    frame["log_manifested"] = frame["in_primary_window"] & strict_symptom
    frame["log_manifested_broad"] = frame["in_primary_window"] & broad_symptom
    frame["log_injected_service_window"] = (
        frame["in_primary_window"] & frame["Service"].eq(lp.base_service(injected_service))
    )
    frame["linked_A_exposed"] = frame["TraceID"].isin(a_exposed_ids)
    frame["diagnostic_pair_strict"] = frame["linked_A_exposed"] & frame["log_manifested"]
    frame["diagnostic_pair_broad"] = frame["linked_A_exposed"] & frame["log_manifested_broad"]
    frame["level_error"] = frame["log_level"].isin({"ERROR", "FATAL"})
    frame["level_warn"] = frame["log_level"].isin({"WARN", "WARNING"})
    frame["level_error_or_warn"] = frame["level_error"] | frame["level_warn"]
    frame["has_trace_link"] = frame["TraceID"].astype(str).str.len() > 0
    for column in LOG_LABEL_COLUMNS:
        frame[column] = frame[column].fillna(False).astype(bool).astype(int)
    return frame[LOG_MANIFEST_COLUMNS].sort_values("LogID").reset_index(drop=True)


def write_manifest(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})


def case_index_row(dataset: str, batch: str, case: str, manifest_path: Path, frame: pd.DataFrame) -> dict:
    row = {
        "protocol_version": lp.PROTOCOL_VERSION,
        "dataset": dataset,
        "batch": batch,
        "case": case,
        "log_scope": "primary_log_scope",
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "n_logs": len(frame),
        "n_templates": int(frame["template_hash"].nunique()) if not frame.empty else 0,
        "n_service_templates": int(frame["service_template_hash"].nunique()) if not frame.empty else 0,
        "n_services": int(frame["Service"].nunique()) if not frame.empty else 0,
    }
    for column in LOG_LABEL_COLUMNS:
        row[column] = int(frame[column].sum()) if column in frame.columns else 0
    return row


def materialize_benchmark(
    data_root: Path,
    output: Path,
    dataset: str,
    limit: int | None = None,
    cases: set[str] | None = None,
) -> List[dict]:
    rows: List[dict] = []
    completed = 0
    for label_path in sorted((data_root / dataset / "batches").glob("batch*/label.json")):
        labels = json.loads(label_path.read_text())
        batch_dir = label_path.parent
        print(f"[log-manifest] {dataset}/{batch_dir.name}: preparing reference and {len(labels)} cases", flush=True)
        ref_trace = lp.trace_features(pd.read_csv(batch_dir / "fault_free" / "trace" / "normal_trace.csv"), dataset)
        trace_model = lp.reference_model(ref_trace)
        ref_logs = lp.read_parsed_logs(batch_dir / "fault_free" / "log" / "parsed")
        ref_counts, ref_total = reference_template_counts(ref_logs)
        for case_index, (case, info) in enumerate(labels.items(), start=1):
            if cases and case not in cases:
                continue
            case_dir = batch_dir / case
            if not case_dir.is_dir():
                continue
            logs = lp.read_parsed_logs(case_dir / "log" / "parsed")
            normalized = normalize_logs_for_manifest(logs)
            strict_mask, broad_mask = lp.symptom_log_masks(normalized)
            in_primary = lp.primary_window_mask(normalized["Timestamp"], lp.injection_time(info["inject_time"]))
            strict_ids = set(normalized.loc[in_primary & strict_mask, "TraceID"].astype(str))
            broad_ids = set(normalized.loc[in_primary & broad_mask, "TraceID"].astype(str))
            exposed_ids = a_exposed_trace_ids(
                dataset,
                batch_dir.name,
                case,
                case_dir,
                info,
                trace_model,
                strict_ids,
                broad_ids,
            )
            manifest = manifest_frame(
                logs,
                ref_counts,
                ref_total,
                lp.injection_time(info["inject_time"]),
                info["inject_pod"],
                exposed_ids,
            )
            path = output / "log_manifests" / f"{dataset}__{batch_dir.name}__{case}.log_manifest.csv.gz"
            write_manifest(manifest, path)
            rows.append(case_index_row(dataset, batch_dir.name, case, path, manifest))
            completed += 1
            if case_index == len(labels) or case_index % 10 == 0:
                print(f"[log-manifest] {dataset}/{batch_dir.name}: completed {case_index}/{len(labels)} cases", flush=True)
            if limit is not None and completed >= limit:
                return rows
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("results/label_protocol_v02"))
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--limit", type=int, default=None, help="optional smoke cap per invocation")
    parser.add_argument("--cases", type=str, default=None, help="comma-separated case names for smoke runs")
    args = parser.parse_args()

    selected_cases = {item.strip() for item in args.cases.split(",") if item.strip()} if args.cases else None
    rows: List[dict] = []
    for dataset in args.datasets:
        if dataset not in DEFAULT_DATASETS:
            print(f"[log-manifest] skipping {dataset}: primary log protocol is Hipster/Train Ticket only", flush=True)
            continue
        rows.extend(materialize_benchmark(args.data_root, args.output, dataset, args.limit, selected_cases))

    args.output.mkdir(parents=True, exist_ok=True)
    index = pd.DataFrame(rows)
    index_path = args.output / "log_manifest_index.csv"
    index.to_csv(index_path, index=False)
    provenance = {
        "protocol_version": lp.PROTOCOL_VERSION,
        "scope": "hipster_train_ticket_primary_log_scope",
        "budgets": LOG_BUDGETS,
        "source_hashes": source_hash(
            [
                Path("scripts/log_protocol_materialize.py"),
                Path("scripts/sampling_labels.py"),
            ]
        ),
        "index_sha256": sha256_file(index_path),
        "cases": int(len(index)),
        "log_rows": int(index["n_logs"].sum()) if not index.empty else 0,
        "manifests_ignored_in_git": "results/label_protocol_v02/log_manifests/*.csv.gz",
    }
    (args.output / "log_manifest_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        f"Wrote {len(index)} log manifests ({int(index['n_logs'].sum()) if not index.empty else 0} rows) "
        f"to {args.output / 'log_manifests'}"
    )
    print(f"Index: {index_path} sha256={provenance['index_sha256']}")


if __name__ == "__main__":
    main()
