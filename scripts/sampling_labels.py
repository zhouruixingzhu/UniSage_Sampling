"""Audit-only generator for independent sampling-quality labels.

This script reads raw/reference telemetry and external fault metadata only. It
must not read sampling results, anomaly detector output, or RCA output.

Usage:
    python scripts/sampling_labels.py
    python scripts/sampling_labels.py --datasets hipster train_ticket

Outputs are aggregate audit artifacts under results/label_protocol_v01/. Full
trace/log manifests are intentionally not emitted at this stage.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROTOCOL_VERSION = "v0.2"
PRIMARY_WINDOW_SECONDS = 60
MIN_API_REFERENCE_TRACES = 20
PRIMARY_SCOPE = "primary_trace_label_scope"
EXCLUDED_INFRA_SCOPE = "excluded_infrastructure_no_trace_exposure"
STRUCT_FREQ_THRESHOLDS = {"struct_rare_le_0p1": 0.001, "struct_rare_le_1p0": 0.01}
LATENCY_QUANTILES = {"lat_tail_q95": 0.95, "lat_tail_q99": 0.99, "lat_tail_q995": 0.995}
LOG_FREQ_THRESHOLDS = {"log_template_rare_le_0p1": 0.001, "log_template_rare_le_1p0": 0.01}
STRICT_SYMPTOM_TEXT_RE = re.compile(
    r"\b(?:error|exception|fail(?:ed|ure)?|timeout|unavailable)\b"
    r"|(?:HTTP/\d(?:\.\d)?\s+5\d\d\b|\bstatus\s*=\s*5\d\d\b)",
    re.IGNORECASE,
)
BROAD_SYMPTOM_TEXT_RE = re.compile(
    r"\b(?:error|warn(?:ing)?|exception|fail(?:ed|ure)?|timeout|unavailable)\b"
    r"|(?:HTTP/\d(?:\.\d)?\s+5\d\d\b|\bstatus\s*=\s*5\d\d\b)",
    re.IGNORECASE,
)
POD_SUFFIX_RE = re.compile(r"-[0-9a-f]{8,10}-[a-z0-9]{5}$")
TRACE_FEATURE_COLUMNS = ["TraceID", "Start", "LatencyMs", "root_api", "path_signature", "services"]


def base_service(value: object) -> str:
    return POD_SUFFIX_RE.sub("", str(value))


def clean_operation(value: object) -> str:
    return str(value).strip()


def read_aiops_metadata(path: Path) -> List[dict]:
    """Read the simple repeated-string-record TOML without a Python 3.11 dependency."""
    records: List[dict] = []
    current: Optional[dict] = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line == "[[chaos_injection]]":
            if current:
                records.append(current)
            current = {}
            continue
        if current is None or "=" not in line:
            continue
        key, value = (piece.strip() for piece in line.split("=", 1))
        current[key] = value.strip().strip('"')
    if current:
        records.append(current)
    return records


def normalize_trace_frame(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    if dataset == "aiops21":
        df = frame.rename(
            columns={
                "TraceId": "TraceID",
                "SpanId": "SpanID",
                "ParentSpanId": "ParentID",
                "ServiceName": "Service",
                "SpanName": "Operation",
            }
        ).copy()
        df["Start"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df["SpanDurationMs"] = pd.to_numeric(df["Duration"], errors="coerce") / 1_000_000.0
        df["End"] = df["Start"] + pd.to_timedelta(df["SpanDurationMs"], unit="ms")
        if "ParentServiceName" in df.columns:
            df["ParentService"] = df["ParentServiceName"].apply(
                lambda x: "" if pd.isna(x) else base_service(x)
            )
        else:
            df["ParentService"] = ""
    else:
        df = frame.rename(columns={"PodName": "Service", "OperationName": "Operation"}).copy()
        df["Start"] = pd.to_datetime(df["StartTimeUnixNano"], unit="ns", errors="coerce")
        df["End"] = pd.to_datetime(df["EndTimeUnixNano"], unit="ns", errors="coerce")
        df["SpanDurationMs"] = pd.to_numeric(df["Duration"], errors="coerce") / 1_000.0
        df["ParentService"] = ""
    required = ["TraceID", "SpanID", "ParentID", "Service", "Operation", "Start", "End"]
    df = df.dropna(subset=[c for c in required if c in df.columns])
    df["Service"] = df["Service"].apply(base_service)
    df["Operation"] = df["Operation"].apply(clean_operation)
    df["SpanID"] = df["SpanID"].astype(str)
    df["ParentID"] = df["ParentID"].astype(str)
    df["TraceID"] = df["TraceID"].astype(str)
    return df


def trace_features(raw: pd.DataFrame, dataset: str) -> pd.DataFrame:
    df = normalize_trace_frame(raw, dataset)
    if df.empty:
        return pd.DataFrame(columns=TRACE_FEATURE_COLUMNS)

    grouped = df.groupby("TraceID", sort=False)
    timing = grouped.agg(Start=("Start", "min"), End=("End", "max"))
    timing["LatencyMs"] = (timing["End"] - timing["Start"]).dt.total_seconds() * 1000.0

    service_values = (
        df[["TraceID", "Service"]]
        .drop_duplicates()
        .sort_values(["TraceID", "Service"])
        .groupby("TraceID")["Service"]
        .agg(",".join)
        .rename("services")
    )

    parent_services = df[["TraceID", "SpanID", "Service"]].rename(
        columns={"SpanID": "ParentID", "Service": "matched_parent_service"}
    )
    spans = df.merge(parent_services, on=["TraceID", "ParentID"], how="left")
    is_root = spans["ParentID"].str.lower().isin({"root", "nan", "none", ""}) | spans[
        "matched_parent_service"
    ].isna()
    if dataset == "aiops21":
        is_root = is_root | spans["ParentService"].eq("")
    spans["is_root"] = is_root
    spans["edge"] = np.where(
        spans["is_root"],
        "ROOT>" + spans["Service"] + ":" + spans["Operation"],
        spans["matched_parent_service"] + ">" + spans["Service"] + ":" + spans["Operation"],
    )
    paths = (
        spans[["TraceID", "edge"]]
        .drop_duplicates()
        .sort_values(["TraceID", "edge"])
        .groupby("TraceID")["edge"]
        .agg("||".join)
        .rename("path_signature")
    )
    root_spans = spans.loc[spans["is_root"], ["TraceID", "Service", "Operation"]].copy()
    root_spans["root_api_value"] = root_spans["Service"] + "|" + root_spans["Operation"]
    root_apis = (
        root_spans[["TraceID", "root_api_value"]]
        .drop_duplicates()
        .sort_values(["TraceID", "root_api_value"])
        .groupby("TraceID")["root_api_value"]
        .agg("||".join)
        .rename("root_api")
    )
    if len(root_apis) < len(timing):
        first_spans = df.sort_values(["TraceID", "Start"]).drop_duplicates("TraceID").set_index("TraceID")
        fallback_roots = first_spans["Service"] + "|" + first_spans["Operation"]
        root_apis = root_apis.reindex(timing.index).fillna(fallback_roots.reindex(timing.index))

    features = timing[["Start", "LatencyMs"]].join(root_apis).join(paths).join(service_values).reset_index()
    return features[TRACE_FEATURE_COLUMNS]


def read_benchmark_trace_files(case_dir: Path) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted((case_dir / "trace").glob("*_trace.csv"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def reference_model(reference: pd.DataFrame) -> dict:
    if reference.empty:
        raise ValueError("Cannot generate independent labels without fault-free/normal reference traces.")
    n = len(reference)
    pattern_counts = reference["path_signature"].value_counts().to_dict()
    root_groups = {
        root: group["LatencyMs"].to_numpy()
        for root, group in reference.groupby("root_api")
        if len(group) >= MIN_API_REFERENCE_TRACES
    }
    global_latency = reference["LatencyMs"].to_numpy()
    return {"n": n, "pattern_counts": pattern_counts, "root_groups": root_groups, "global_latency": global_latency}


def add_reference_flags(candidates: pd.DataFrame, model: dict) -> pd.DataFrame:
    out = candidates.copy()
    out["reference_path_count"] = out["path_signature"].map(model["pattern_counts"]).fillna(0).astype(int)
    out["reference_path_frequency"] = out["reference_path_count"] / max(1, model["n"])
    out["struct_unseen"] = out["reference_path_count"] == 0
    for name, threshold in STRUCT_FREQ_THRESHOLDS.items():
        out[name] = out["struct_unseen"] | (out["reference_path_frequency"] <= threshold)
    for name, quantile in LATENCY_QUANTILES.items():
        thresholds = []
        for row in out.itertuples(index=False):
            values = model["root_groups"].get(row.root_api, model["global_latency"])
            thresholds.append(float(np.quantile(values, quantile)))
        out[f"{name}_threshold_ms"] = thresholds
        out[name] = out["LatencyMs"] > out[f"{name}_threshold_ms"]
    return out


def read_parsed_logs(directory: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(directory.glob("*.csv")):
        frame = pd.read_csv(path)
        if frame.empty or "TraceID" not in frame.columns:
            continue
        frame = frame.copy()
        frame["Service"] = base_service(path.stem)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    logs = pd.concat(frames, ignore_index=True)
    if "log_level" not in logs.columns:
        logs["log_level"] = ""
    if "Body" not in logs.columns:
        logs["Body"] = ""
    if "log_temp" not in logs.columns:
        logs["log_temp"] = logs["Body"]
    logs["Timestamp"] = pd.to_datetime(logs["Timestamp"], errors="coerce", utc=True).dt.tz_localize(None)
    logs["TraceID"] = logs["TraceID"].astype(str)
    logs["log_level"] = logs["log_level"].fillna("").astype(str).str.upper()
    logs["Body"] = logs["Body"].fillna("").astype(str)
    logs["log_temp"] = logs["log_temp"].fillna("").astype(str)
    return logs.dropna(subset=["Timestamp"])


def symptom_log_masks(logs: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    if logs.empty:
        empty = pd.Series(dtype=bool)
        return empty, empty
    strict = logs["log_level"].isin({"ERROR", "FATAL"}) | logs["Body"].str.contains(
        STRICT_SYMPTOM_TEXT_RE, na=False
    )
    broad = logs["log_level"].isin({"ERROR", "WARN", "WARNING", "FATAL"}) | logs["Body"].str.contains(
        BROAD_SYMPTOM_TEXT_RE, na=False
    )
    return strict, broad


def injection_time(value: str) -> pd.Timestamp:
    # Recorded wall-clock strings align with raw telemetry; avoid inconsistent Unix fields.
    return pd.Timestamp(value)


def primary_window_mask(starts: pd.Series, injected_at: pd.Timestamp) -> pd.Series:
    return (starts >= injected_at) & (starts < injected_at + pd.Timedelta(seconds=PRIMARY_WINDOW_SECONDS))


def boolean_count(frame: pd.DataFrame, column: str) -> int:
    return int(frame[column].sum()) if column in frame.columns else 0


def trace_case_summary(
    dataset: str,
    batch: str,
    case: str,
    fault_type: str,
    injected_service: str,
    candidates: pd.DataFrame,
    model: dict,
    injected_at: Optional[pd.Timestamp],
    symptom_trace_ids: set,
    broad_symptom_trace_ids: set,
    aiops_partition: bool = False,
    trace_label_scope: str = PRIMARY_SCOPE,
) -> Tuple[dict, pd.DataFrame]:
    traces = add_reference_flags(candidates, model)
    services = traces["services"].str.split(",")
    traverses = services.apply(lambda values: base_service(injected_service) in set(values))
    if aiops_partition:
        in_primary = pd.Series(True, index=traces.index)
        in_sensitivity = in_primary
        window_mode = "dataset_abnormal_partition"
    else:
        in_primary = primary_window_mask(traces["Start"], injected_at)
        in_sensitivity = (traces["Start"] >= injected_at) & (traces["Start"] <= traces["Start"].max())
        window_mode = "post_injection_60s"
    traces["A_exposed"] = in_primary & traverses
    traces["A_exposed_sensitivity"] = in_sensitivity & traverses
    traces["linked_symptom_log"] = traces["TraceID"].isin(symptom_trace_ids)
    traces["linked_symptom_log_broad"] = traces["TraceID"].isin(broad_symptom_trace_ids)
    traces["A_manifested"] = traces["A_exposed"] & (traces["lat_tail_q99"] | traces["linked_symptom_log"])
    traces["A_manifested_broad"] = traces["A_exposed"] & (
        traces["lat_tail_q99"] | traces["linked_symptom_log_broad"]
    )
    traces["A_manifested_latency"] = traces["A_exposed"] & traces["lat_tail_q99"]
    traces["A_manifested_log"] = traces["A_exposed"] & traces["linked_symptom_log"]
    traces["A_manifested_log_broad"] = traces["A_exposed"] & traces["linked_symptom_log_broad"]
    row = {
        "dataset": dataset,
        "batch": batch,
        "case": case,
        "fault_type": fault_type,
        "injected_service": base_service(injected_service),
        "window_mode": window_mode,
        "trace_label_scope": trace_label_scope,
        "n_traces": len(traces),
        "struct_unseen": boolean_count(traces, "struct_unseen"),
        "struct_rare_le_0p1": boolean_count(traces, "struct_rare_le_0p1"),
        "struct_rare_le_1p0": boolean_count(traces, "struct_rare_le_1p0"),
        "lat_tail_q95": boolean_count(traces, "lat_tail_q95"),
        "lat_tail_q99": boolean_count(traces, "lat_tail_q99"),
        "lat_tail_q995": boolean_count(traces, "lat_tail_q995"),
        "A_exposed": boolean_count(traces, "A_exposed"),
        "A_exposed_sensitivity": boolean_count(traces, "A_exposed_sensitivity"),
        "A_manifested": boolean_count(traces, "A_manifested"),
        "A_manifested_broad": boolean_count(traces, "A_manifested_broad"),
        "A_manifested_latency": boolean_count(traces, "A_manifested_latency"),
        "A_manifested_log": boolean_count(traces, "A_manifested_log"),
        "A_manifested_log_broad": boolean_count(traces, "A_manifested_log_broad"),
        "A_manifested_and_struct_unseen": int((traces["A_manifested"] & traces["struct_unseen"]).sum()),
        "A_manifested_and_lat_tail_q99": int((traces["A_manifested"] & traces["lat_tail_q99"]).sum()),
    }
    return row, traces


def log_case_summary(
    dataset: str,
    batch: str,
    case: str,
    logs: pd.DataFrame,
    ref_template_counts: Counter,
    ref_total: int,
    injected_at: pd.Timestamp,
) -> Tuple[dict, set, set]:
    if logs.empty:
        return (
            {
                "dataset": dataset,
                "batch": batch,
                "case": case,
                "n_logs": 0,
                "log_template_unseen": 0,
                "log_template_rare_le_0p1": 0,
                "log_template_rare_le_1p0": 0,
                "log_manifested": 0,
                "log_manifested_broad": 0,
                "linked_manifested_trace_ids": 0,
                "linked_manifested_broad_trace_ids": 0,
            },
            set(),
            set(),
        )
    frame = logs.copy()
    frame["reference_template_count"] = frame["log_temp"].map(ref_template_counts).fillna(0).astype(int)
    frame["reference_template_frequency"] = frame["reference_template_count"] / max(1, ref_total)
    frame["log_template_unseen"] = frame["reference_template_count"] == 0
    for name, threshold in LOG_FREQ_THRESHOLDS.items():
        frame[name] = frame["log_template_unseen"] | (frame["reference_template_frequency"] <= threshold)
    in_primary = primary_window_mask(frame["Timestamp"], injected_at)
    strict_symptom, broad_symptom = symptom_log_masks(frame)
    frame["log_manifested"] = in_primary & strict_symptom
    frame["log_manifested_broad"] = in_primary & broad_symptom
    symptom_ids = set(frame.loc[frame["log_manifested"], "TraceID"])
    broad_symptom_ids = set(frame.loc[frame["log_manifested_broad"], "TraceID"])
    row = {
        "dataset": dataset,
        "batch": batch,
        "case": case,
        "n_logs": len(frame),
        "log_template_unseen": boolean_count(frame, "log_template_unseen"),
        "log_template_rare_le_0p1": boolean_count(frame, "log_template_rare_le_0p1"),
        "log_template_rare_le_1p0": boolean_count(frame, "log_template_rare_le_1p0"),
        "log_manifested": boolean_count(frame, "log_manifested"),
        "log_manifested_broad": boolean_count(frame, "log_manifested_broad"),
        "linked_manifested_trace_ids": len(symptom_ids),
        "linked_manifested_broad_trace_ids": len(broad_symptom_ids),
    }
    return row, symptom_ids, broad_symptom_ids


def benchmark_cases(data_root: Path, dataset: str) -> Iterable[dict]:
    for label_path in sorted((data_root / dataset / "batches").glob("batch*/label.json")):
        labels = json.loads(label_path.read_text())
        batch_dir = label_path.parent
        print(f"[audit] {dataset}/{batch_dir.name}: preparing reference and {len(labels)} cases", flush=True)
        ref_raw = pd.read_csv(batch_dir / "fault_free" / "trace" / "normal_trace.csv")
        ref_trace = trace_features(ref_raw, dataset)
        model = reference_model(ref_trace)
        ref_logs = read_parsed_logs(batch_dir / "fault_free" / "log" / "parsed")
        ref_counts = Counter(ref_logs["log_temp"]) if not ref_logs.empty else Counter()
        for case_index, (case, info) in enumerate(labels.items(), start=1):
            case_dir = batch_dir / case
            if not case_dir.is_dir():
                continue
            candidates = trace_features(read_benchmark_trace_files(case_dir), dataset)
            logs = read_parsed_logs(case_dir / "log" / "parsed")
            injected_at = injection_time(info["inject_time"])
            log_row, symptom_ids, broad_symptom_ids = log_case_summary(
                dataset, batch_dir.name, case, logs, ref_counts, len(ref_logs), injected_at
            )
            trace_row, _ = trace_case_summary(
                dataset,
                batch_dir.name,
                case,
                info.get("inject_type", ""),
                info["inject_pod"],
                candidates,
                model,
                injected_at,
                symptom_ids,
                broad_symptom_ids,
            )
            yield {"trace": trace_row, "log": log_row}
            if case_index == len(labels) or case_index % 10 == 0:
                print(f"[audit] {dataset}/{batch_dir.name}: completed {case_index}/{len(labels)} cases", flush=True)


def aiops_cases(data_root: Path) -> Iterable[dict]:
    base = data_root / "aiops21" / "all"
    records = read_aiops_metadata(base / "fault_injection.toml")
    print(f"[audit] aiops21/all: preparing {len(records)} normal/abnormal case pairs", flush=True)
    for case_index, info in enumerate(records, start=1):
        case_dir = base / info["case"]
        normal_path = case_dir / "normal" / "traces.csv"
        abnormal_path = case_dir / "abnormal" / "traces.csv"
        if not normal_path.exists() or not abnormal_path.exists():
            continue
        reference = trace_features(pd.read_csv(normal_path), "aiops21")
        candidates = trace_features(pd.read_csv(abnormal_path), "aiops21")
        row, _ = trace_case_summary(
            "aiops21",
            "all",
            info["case"],
            info.get("chaos_type", ""),
            info["service"],
            candidates,
            reference_model(reference),
            None,
            set(),
            set(),
            aiops_partition=True,
        )
        if row["A_exposed"] == 0:
            row["trace_label_scope"] = EXCLUDED_INFRA_SCOPE
        yield {"trace": row, "log": None}
        if case_index == len(records) or case_index % 10 == 0:
            print(f"[audit] aiops21/all: completed {case_index}/{len(records)} cases", flush=True)


def summarize_counts(frame: pd.DataFrame, denominator: str, columns: Sequence[str]) -> pd.DataFrame:
    rows = []
    for dataset, group in frame.groupby("dataset", sort=True):
        total = int(group[denominator].sum())
        row = {"dataset": dataset, "cases": len(group), denominator: total}
        for column in columns:
            count = int(group[column].sum())
            rates = group[column] / group[denominator].replace(0, np.nan)
            row[f"{column}_count"] = count
            row[f"{column}_micro_rate"] = count / total if total else np.nan
            row[f"{column}_macro_rate"] = float(rates.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def render_markdown(trace_summary: pd.DataFrame, log_summary: pd.DataFrame) -> str:
    descriptive_trace_cols = [
        "struct_unseen",
        "struct_rare_le_0p1",
        "struct_rare_le_1p0",
        "lat_tail_q99",
    ]
    problem_trace_cols = [
        "A_exposed",
        "A_manifested_latency",
        "A_manifested_log",
        "A_manifested",
        "A_manifested_broad",
    ]
    log_cols = [
        "log_template_unseen",
        "log_template_rare_le_0p1",
        "log_template_rare_le_1p0",
        "log_manifested",
        "log_manifested_broad",
    ]
    descriptive_ds = summarize_counts(trace_summary, "n_traces", descriptive_trace_cols)
    primary_trace = trace_summary[trace_summary["trace_label_scope"] == PRIMARY_SCOPE]
    primary_ds = summarize_counts(primary_trace, "n_traces", problem_trace_cols)
    excluded = trace_summary[trace_summary["trace_label_scope"] == EXCLUDED_INFRA_SCOPE]
    log_ds = summarize_counts(log_summary, "n_logs", log_cols) if not log_summary.empty else pd.DataFrame()
    lines = [
        "# Independent Label Protocol Audit v0.2",
        "",
        "- Frozen-scope audit run: no sampler outputs, RCA outputs, AD scores, or method probabilities are read.",
        f"- Hipster/Train Ticket primary problem window: post-injection {PRIMARY_WINDOW_SECONDS} seconds.",
        "- AIOps21 trace-label primary scope: `abnormal/` cases with observed injected-service trace exposure.",
        "- AIOps21 infrastructure injections without trace exposure are excluded from problem-related trace labels and listed separately.",
        "- Rare/path/latency reference statistics use fault-free or normal telemetry only.",
        "- Future method tables use per-case macro aggregation; micro rates below document label support only.",
        "- All-case descriptive trace strata remain eligible for label-free diversity analysis.",
        "",
        "## All-Case Descriptive Trace Strata",
        "",
        "| dataset | cases | traces | label | count | micro rate | macro rate |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for row in descriptive_ds.itertuples(index=False):
        for column in descriptive_trace_cols:
            lines.append(
                f"| {row.dataset} | {row.cases} | {row.n_traces} | `{column}` | "
                f"{getattr(row, column + '_count')} | {getattr(row, column + '_micro_rate'):.3%} | "
                f"{getattr(row, column + '_macro_rate'):.3%} |"
            )
    lines.extend(
        [
            "",
            "## Primary Problem-Related Trace-Label Scope",
            "",
            "| dataset | scope cases | traces | label | count | micro rate | macro rate |",
            "|---|---:|---:|---|---:|---:|---:|",
        ]
    )
    for row in primary_ds.itertuples(index=False):
        for column in problem_trace_cols:
            lines.append(
                f"| {row.dataset} | {row.cases} | {row.n_traces} | `{column}` | "
                f"{getattr(row, column + '_count')} | {getattr(row, column + '_micro_rate'):.3%} | "
                f"{getattr(row, column + '_macro_rate'):.3%} |"
            )
    lines.extend(["", "## Separately Reported AIOps21 Infrastructure Cases", ""])
    if excluded.empty:
        lines.append("No cases were excluded from problem-related trace-label scope.")
    else:
        lines.extend(
            [
                "These cases remain eligible for label-free diversity analysis but do not have an injected-service trace label.",
                "",
                "| injected service | cases | traces | latency-q99 count | latency-q99 rate |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for service, group in excluded.groupby("injected_service", sort=True):
            n_traces = int(group["n_traces"].sum())
            latency_count = int(group["lat_tail_q99"].sum())
            lines.append(
                f"| `{service}` | {len(group)} | {n_traces} | {latency_count} | "
                f"{latency_count / n_traces:.3%} |"
            )
        lines.append(
            f"| **total** | **{len(excluded)}** | **{int(excluded['n_traces'].sum())}** | "
            f"**{int(excluded['lat_tail_q99'].sum())}** | "
            f"**{excluded['lat_tail_q99'].sum() / excluded['n_traces'].sum():.3%}** |"
        )
    lines.extend(["", "## Log Symptom And Template Audit", ""])
    if log_ds.empty:
        lines.append("No log-label audit records were generated.")
    else:
        lines.extend(
            [
                "| dataset | cases | logs | label | count | micro rate | macro rate |",
                "|---|---:|---:|---|---:|---:|---:|",
            ]
        )
        for row in log_ds.itertuples(index=False):
            for column in log_cols:
                lines.append(
                    f"| {row.dataset} | {row.cases} | {row.n_logs} | `{column}` | "
                    f"{getattr(row, column + '_count')} | {getattr(row, column + '_micro_rate'):.3%} | "
                    f"{getattr(row, column + '_macro_rate'):.3%} |"
                )
    lines.extend(
        [
            "",
            "## Frozen Roles And Remaining Validation",
            "",
            "- Primary problem-related metric: `A_exposed` recall/proportion in the declared trace-label scope.",
            "- Secondary symptom metrics: latency-manifested and strict-log-manifested evidence separately;",
            "  their union and warning-inclusive form are sensitivity context only.",
            "- Structural/latency strata and log-template counts are descriptive or secondary, not a rebuilt omnibus Coverage label.",
            "- Before log diversity comparison, freeze a sampler-independent template preprocessing run;",
            "  ordinary unseen INFO templates must not be presented as anomaly labels.",
            "",
            "## Next Execution Gate",
            "",
            "Generate version/hash-stamped label manifests and implement method comparison against these frozen roles;",
            "do not reuse the lost paper Coverage values as evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("results/label_protocol_v02"))
    parser.add_argument("--datasets", nargs="+", default=["hipster", "train_ticket", "aiops21"])
    args = parser.parse_args()
    trace_rows: List[dict] = []
    log_rows: List[dict] = []
    for dataset in args.datasets:
        source = aiops_cases(args.data_root) if dataset == "aiops21" else benchmark_cases(args.data_root, dataset)
        for item in source:
            trace_rows.append(item["trace"])
            if item["log"] is not None:
                log_rows.append(item["log"])
    trace_summary = pd.DataFrame(trace_rows)
    log_summary = pd.DataFrame(log_rows)
    args.output.mkdir(parents=True, exist_ok=True)
    trace_summary.to_csv(args.output / "trace_case_audit.csv", index=False)
    log_summary.to_csv(args.output / "log_case_audit.csv", index=False)
    trace_summary[trace_summary["trace_label_scope"] == EXCLUDED_INFRA_SCOPE].to_csv(
        args.output / "aiops21_excluded_infrastructure_cases.csv", index=False
    )
    config = {
        "protocol_version": PROTOCOL_VERSION,
        "audit_only": True,
        "interpretation_status": "scope_and_metric_roles_frozen_before_method_comparison",
        "primary_window_seconds": PRIMARY_WINDOW_SECONDS,
        "problem_related_primary_metric": "A_exposed",
        "future_method_aggregation": "per_case_macro_mean_with_support_and_micro_provenance",
        "aiops21_primary_scope": "abnormal_partition_cases_with_observed_injected_service_trace_exposure",
        "aiops21_excluded_scope": EXCLUDED_INFRA_SCOPE,
        "aiops21_excluded_role": "separate_report_only_for_problem_related_trace_labels; eligible_for_label_free_diversity",
        "min_api_reference_traces": MIN_API_REFERENCE_TRACES,
        "struct_frequency_thresholds": STRUCT_FREQ_THRESHOLDS,
        "latency_quantiles": LATENCY_QUANTILES,
        "log_frequency_thresholds": LOG_FREQ_THRESHOLDS,
        "manifested_strict_rule": "A_exposed AND (lat_tail_q99 OR linked ERROR/FATAL or fault-text/HTTP-5xx log)",
        "manifested_broad_rule": "strict rule plus linked WARN/WARNING or warning-text log",
        "secondary_metrics": ["A_manifested_latency", "A_manifested_log", "A_manifested", "A_manifested_broad"],
        "descriptive_not_anomaly_labels": [
            "struct_unseen",
            "struct_rare_le_0p1",
            "struct_rare_le_1p0",
            "lat_tail_q99",
            "log_template_unseen",
            "log_template_rare_le_0p1",
            "log_template_rare_le_1p0",
        ],
        "forbidden_inputs": [
            "trace_sampling_result.csv",
            "log_sampling_result.csv",
            "final_ranking.csv",
            "rca_result/",
            "AD outputs or method probabilities",
        ],
    }
    (args.output / "protocol_config.json").write_text(json.dumps(config, indent=2) + "\n")
    report = render_markdown(trace_summary, log_summary)
    (args.output / "audit_summary.md").write_text(report)
    print(report)
    print(f"Wrote audit artifacts to {args.output}")


if __name__ == "__main__":
    main()
