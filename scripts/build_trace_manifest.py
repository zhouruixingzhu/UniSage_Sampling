"""Materialize v0.2 trace-label manifests with hashes.

This script is method-blind: it reads raw/reference telemetry and external
fault metadata only, then writes per-case trace manifests plus compact index
and provenance files. It must not read sampler, RCA, or AD outputs.
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
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

import sampling_labels as lp


MANIFEST_COLUMNS = [
    "TraceID",
    "pattern_hash",
    "root_api_hash",
    "LatencyMs",
    "trace_label_scope",
    "struct_unseen",
    "struct_rare_le_0p1",
    "struct_rare_le_1p0",
    "lat_tail_q95",
    "lat_tail_q99",
    "lat_tail_q995",
    "A_exposed",
    "A_manifested_latency",
    "A_manifested_log",
    "A_manifested",
    "A_manifested_broad",
]
LABEL_COLUMNS = [c for c in MANIFEST_COLUMNS if c not in {"TraceID", "pattern_hash", "root_api_hash", "LatencyMs", "trace_label_scope"}]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]


def source_hash(paths: Iterable[Path]) -> Dict[str, str]:
    return {path.as_posix(): sha256_file(path) for path in paths if path.exists()}


def manifest_frame(traces: pd.DataFrame, trace_label_scope: str) -> pd.DataFrame:
    out = traces.copy()
    out["pattern_hash"] = out["path_signature"].map(stable_hash)
    out["root_api_hash"] = out["root_api"].map(stable_hash)
    out["trace_label_scope"] = trace_label_scope
    for column in LABEL_COLUMNS:
        if column in out.columns:
            out[column] = out[column].fillna(False).astype(bool).astype(int)
    return out[MANIFEST_COLUMNS].sort_values("TraceID").reset_index(drop=True)


def write_manifest(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})


def case_index_row(dataset: str, batch: str, case: str, manifest_path: Path, frame: pd.DataFrame) -> dict:
    row = {
        "protocol_version": lp.PROTOCOL_VERSION,
        "dataset": dataset,
        "batch": batch,
        "case": case,
        "trace_label_scope": frame["trace_label_scope"].iloc[0] if not frame.empty else lp.PRIMARY_SCOPE,
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "n_traces": len(frame),
        "n_patterns": int(frame["pattern_hash"].nunique()) if not frame.empty else 0,
    }
    for column in LABEL_COLUMNS:
        row[column] = int(frame[column].sum()) if column in frame.columns else 0
    return row


def materialize_benchmark(data_root: Path, output: Path, dataset: str) -> List[dict]:
    rows: List[dict] = []
    for label_path in sorted((data_root / dataset / "batches").glob("batch*/label.json")):
        labels = json.loads(label_path.read_text())
        batch_dir = label_path.parent
        print(f"[manifest] {dataset}/{batch_dir.name}: preparing reference and {len(labels)} cases", flush=True)
        ref_trace = lp.trace_features(pd.read_csv(batch_dir / "fault_free" / "trace" / "normal_trace.csv"), dataset)
        model = lp.reference_model(ref_trace)
        ref_logs = lp.read_parsed_logs(batch_dir / "fault_free" / "log" / "parsed")
        ref_counts = lp.Counter(ref_logs["log_temp"]) if not ref_logs.empty else lp.Counter()
        for case_index, (case, info) in enumerate(labels.items(), start=1):
            case_dir = batch_dir / case
            if not case_dir.is_dir():
                continue
            logs = lp.read_parsed_logs(case_dir / "log" / "parsed")
            _, symptom_ids, broad_symptom_ids = lp.log_case_summary(
                dataset, batch_dir.name, case, logs, ref_counts, len(ref_logs), lp.injection_time(info["inject_time"])
            )
            _, traces = lp.trace_case_summary(
                dataset,
                batch_dir.name,
                case,
                info.get("inject_type", ""),
                info["inject_pod"],
                lp.trace_features(lp.read_benchmark_trace_files(case_dir), dataset),
                model,
                lp.injection_time(info["inject_time"]),
                symptom_ids,
                broad_symptom_ids,
            )
            manifest = manifest_frame(traces, lp.PRIMARY_SCOPE)
            path = output / "manifests" / f"{dataset}__{batch_dir.name}__{case}.trace_manifest.csv.gz"
            write_manifest(manifest, path)
            rows.append(case_index_row(dataset, batch_dir.name, case, path, manifest))
            if case_index == len(labels) or case_index % 10 == 0:
                print(f"[manifest] {dataset}/{batch_dir.name}: completed {case_index}/{len(labels)} cases", flush=True)
    return rows


def materialize_aiops(data_root: Path, output: Path) -> List[dict]:
    rows: List[dict] = []
    base = data_root / "aiops21" / "all"
    records = lp.read_aiops_metadata(base / "fault_injection.toml")
    print(f"[manifest] aiops21/all: preparing {len(records)} normal/abnormal case pairs", flush=True)
    for case_index, info in enumerate(records, start=1):
        case_dir = base / info["case"]
        normal_path = case_dir / "normal" / "traces.csv"
        abnormal_path = case_dir / "abnormal" / "traces.csv"
        if not normal_path.exists() or not abnormal_path.exists():
            continue
        _, traces = lp.trace_case_summary(
            "aiops21",
            "all",
            info["case"],
            info.get("chaos_type", ""),
            info["service"],
            lp.trace_features(pd.read_csv(abnormal_path), "aiops21"),
            lp.reference_model(lp.trace_features(pd.read_csv(normal_path), "aiops21")),
            None,
            set(),
            set(),
            aiops_partition=True,
        )
        scope = lp.PRIMARY_SCOPE if int(traces["A_exposed"].sum()) > 0 else lp.EXCLUDED_INFRA_SCOPE
        manifest = manifest_frame(traces, scope)
        path = output / "manifests" / f"aiops21__all__{info['case']}.trace_manifest.csv.gz"
        write_manifest(manifest, path)
        rows.append(case_index_row("aiops21", "all", info["case"], path, manifest))
        if case_index == len(records) or case_index % 10 == 0:
            print(f"[manifest] aiops21/all: completed {case_index}/{len(records)} cases", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("results/label_protocol_v02"))
    parser.add_argument("--datasets", nargs="+", default=["hipster", "train_ticket", "aiops21"])
    args = parser.parse_args()

    rows: List[dict] = []
    for dataset in args.datasets:
        if dataset == "aiops21":
            rows.extend(materialize_aiops(args.data_root, args.output))
        else:
            rows.extend(materialize_benchmark(args.data_root, args.output, dataset))

    index = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    index_path = args.output / "trace_manifest_index.csv"
    index.to_csv(index_path, index=False)
    excluded = index[index["trace_label_scope"] == lp.EXCLUDED_INFRA_SCOPE]
    excluded.to_csv(args.output / "trace_manifest_excluded_index.csv", index=False)

    provenance = {
        "protocol_version": lp.PROTOCOL_VERSION,
        "source_hashes": source_hash(
            [
                Path("scripts/sampling_labels.py"),
                Path("scripts/build_trace_manifest.py"),
            ]
        ),
        "index_sha256": sha256_file(index_path),
        "cases": int(len(index)),
        "trace_rows": int(index["n_traces"].sum()) if not index.empty else 0,
        "primary_scope_cases": int((index["trace_label_scope"] == lp.PRIMARY_SCOPE).sum()) if not index.empty else 0,
        "excluded_scope_cases": int((index["trace_label_scope"] == lp.EXCLUDED_INFRA_SCOPE).sum()) if not index.empty else 0,
        "manifests_ignored_in_git": "results/label_protocol_v02/manifests/*.csv.gz",
    }
    (args.output / "trace_manifest_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        f"Wrote {len(index)} trace manifests ({int(index['n_traces'].sum()) if not index.empty else 0} rows) "
        f"to {args.output / 'manifests'}"
    )
    print(f"Index: {index_path} sha256={provenance['index_sha256']}")


if __name__ == "__main__":
    main()
