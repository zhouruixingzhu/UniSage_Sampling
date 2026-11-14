"""UniSage <-> MicroRCA data bridge.

MicroRCA consumes service-level invocation latency, a dependency graph, and
resource metrics. Hipster and Train Ticket have Nezha-lineage metric artifacts
with exactly the right raw ingredients:

  * `metric/source_50.csv` and `metric/destination_50.csv` for full invocation
    latency;
  * `metric/dependency.csv` for service dependencies;
  * `metric/*_metric.csv` for per-pod resource time series.

For sampled downstream RCA, the sampling hook is trace latency: retained
TraceIDs are used to rebuild minute-level edge latency time series from the
case's raw traces, while resource metrics remain full because UniSage does not
currently sample metrics.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from baselines.nezha import adapter as nezha_adapter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MICRORCA_DIR = PROJECT_ROOT / "baselines" / "microrca"
MICRORCA_RUNNER = MICRORCA_DIR / "microrca_single_case.py"


@dataclass(frozen=True)
class CaseEntry:
    dataset: str
    batch: str
    case: str
    ns: str
    nz_date: str
    nz_window: str
    inject_time: str
    inject_pod: str
    inject_type: str
    inject_timestamp: str


def service_base(name: object) -> str:
    s = str(name or "").strip()
    s = s.removesuffix("_metric").removesuffix("_metric.csv").removesuffix(".csv")
    m = re.match(r"(.+)-[a-z0-9]{9,}-[a-z0-9]{5}$", s)
    return m.group(1) if m else s


def first_service_rank(ranked: list[dict], gt: str) -> int:
    """1-indexed rank of a ground-truth pod/service in a service-level list."""
    if not gt:
        return 0
    gt_service = service_base(gt)
    for i, row in enumerate(ranked, 1):
        service = service_base(row.get("service") or row.get("pod") or "")
        pod = str(row.get("pod") or "")
        if service == gt_service or pod == gt or pod.startswith(gt) or gt.startswith(pod):
            return i
    return 0


def build_case_index(unisage_data_root: Path,
                     nezha_rca_dir: Path,
                     datasets: Iterable[str] = ("hipster", "train_ticket")) -> pd.DataFrame:
    df = nezha_adapter.build_case_index(unisage_data_root, nezha_rca_dir, datasets=datasets)
    if df.empty:
        return df
    df["case_key"] = df["dataset"] + "/" + df["batch"].astype(str) + "/" + df["case"].astype(str)
    return df


def write_case_index(unisage_data_root: Path,
                     nezha_rca_dir: Path,
                     out_path: Path,
                     datasets: Iterable[str] = ("hipster", "train_ticket")) -> pd.DataFrame:
    df = build_case_index(unisage_data_root, nezha_rca_dir, datasets=datasets)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def case_entry_from_row(row: pd.Series) -> CaseEntry:
    return CaseEntry(
        dataset=str(row["dataset"]),
        batch=str(row["batch"]),
        case=str(row["case"]),
        ns=str(row["ns"]),
        nz_date=str(row["nz_date"]),
        nz_window=str(row["nz_window"]),
        inject_time=str(row["inject_time"]),
        inject_pod=str(row["inject_pod"]),
        inject_type=str(row["inject_type"]),
        inject_timestamp=str(row["inject_timestamp"]),
    )


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _link_or_copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def _standard_dependency(metric_dir: Path) -> pd.DataFrame:
    dep_path = metric_dir / "dependency.csv"
    if not dep_path.exists():
        return pd.DataFrame(columns=["Source", "Target", "Weight"])
    df = pd.read_csv(dep_path)
    if df.empty:
        return pd.DataFrame(columns=["Source", "Target", "Weight"])
    out = pd.DataFrame({
        "Source": df["Source"].map(service_base),
        "Target": df["Target"].map(service_base),
        "Weight": pd.to_numeric(df.get("Weight", 1.0), errors="coerce").fillna(1.0),
    })
    out = out[(out["Source"] != "") & (out["Target"] != "")]
    return out.groupby(["Source", "Target"], as_index=False)["Weight"].max()


def dependency_from_traces(traces: pd.DataFrame) -> pd.DataFrame:
    if traces.empty:
        return pd.DataFrame(columns=["Source", "Target", "Weight"])
    required = {"TraceID", "SpanID", "ParentID", "PodName"}
    if not required.issubset(traces.columns):
        return pd.DataFrame(columns=["Source", "Target", "Weight"])
    base = traces.copy()
    base["TraceID"] = base["TraceID"].astype(str)
    base["SpanID"] = base["SpanID"].astype(str)
    base["ParentID"] = base["ParentID"].astype(str)
    parents = base[["TraceID", "SpanID", "PodName"]].rename(
        columns={"SpanID": "_ParentSpanID", "PodName": "_ParentPod"}
    )
    child = base.merge(
        parents,
        left_on=["TraceID", "ParentID"],
        right_on=["TraceID", "_ParentSpanID"],
        how="left",
    ).dropna(subset=["_ParentPod"])
    if child.empty:
        return pd.DataFrame(columns=["Source", "Target", "Weight"])
    child["Source"] = child["_ParentPod"].map(service_base)
    child["Target"] = child["PodName"].map(service_base)
    child = child[(child["Source"] != "") & (child["Target"] != "") & (child["Source"] != child["Target"])]
    if child.empty:
        return pd.DataFrame(columns=["Source", "Target", "Weight"])
    out = (
        child.groupby(["Source", "Target"])
        .size()
        .reset_index(name="Weight")
        .sort_values(["Source", "Target"])
        .reset_index(drop=True)
    )
    return out


def _standardize_metric_file(src: Path) -> pd.DataFrame:
    df = pd.read_csv(src)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "ctn_cpu", "ctn_memory", "ctn_network",
                                     "node_cpu", "node_memory", "node_network"])
    ts_col = "TimeStamp" if "TimeStamp" in df.columns else "timestamp"
    out = pd.DataFrame({"timestamp": pd.to_numeric(df[ts_col], errors="coerce")})
    out["ctn_cpu"] = pd.to_numeric(
        df.get("CpuUsageRate(%)", df.get("CpuUsage(m)", 0.0)), errors="coerce"
    ).fillna(0.0)
    out["ctn_memory"] = pd.to_numeric(
        df.get("MemoryUsageRate(%)", df.get("MemoryUsage(Mi)", 0.0)), errors="coerce"
    ).fillna(0.0)
    net_rx = pd.to_numeric(df.get("NetworkReceiveBytes", 0.0), errors="coerce").fillna(0.0)
    net_tx = pd.to_numeric(df.get("NetworkTransmitBytes", 0.0), errors="coerce").fillna(0.0)
    out["ctn_network"] = net_rx + net_tx
    out["node_cpu"] = pd.to_numeric(df.get("NodeCpuUsageRate(%)", 0.0), errors="coerce").fillna(0.0)
    out["node_memory"] = pd.to_numeric(df.get("NodeMemoryUsageRate(%)", 0.0), errors="coerce").fillna(0.0)
    out["node_network"] = pd.to_numeric(df.get("NodeNetworkReceiveBytes", 0.0), errors="coerce").fillna(0.0)
    return out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def materialize_resource_metrics(metric_dir: Path, out_dir: Path) -> list[str]:
    _ensure_dir(out_dir)
    services = []
    for src in sorted(metric_dir.glob("*_metric.csv")):
        service = service_base(src.name)
        if not service:
            continue
        df = _standardize_metric_file(src)
        df.to_csv(out_dir / f"{service}.csv", index=False)
        services.append(service)
    return services


def _timestamps_from_traces(traces: pd.DataFrame) -> pd.DataFrame:
    if traces.empty or "StartTimeUnixNano" not in traces.columns:
        return pd.DataFrame({"timestamp": []})
    seconds = pd.to_numeric(traces["StartTimeUnixNano"], errors="coerce").dropna().astype(np.int64) // 10**9
    minutes = (seconds // 60 * 60).drop_duplicates().sort_values().to_numpy()
    return pd.DataFrame({"timestamp": minutes.astype(int)})


def _timestamps_from_metric_files(metric_dir: Path) -> pd.DataFrame:
    for path in sorted(metric_dir.glob("*_metric.csv")):
        df = pd.read_csv(path, nrows=2000)
        if df.empty:
            continue
        ts_col = "TimeStamp" if "TimeStamp" in df.columns else "timestamp"
        if ts_col not in df.columns:
            continue
        ts = pd.to_numeric(df[ts_col], errors="coerce").dropna().astype(int).drop_duplicates().sort_values()
        if not ts.empty:
            return pd.DataFrame({"timestamp": ts.to_numpy()})
    return pd.DataFrame({"timestamp": []})


def _read_latency_template(metric_dir: Path,
                           traces: Optional[pd.DataFrame] = None,
                           dependency: Optional[pd.DataFrame] = None) -> tuple[pd.DataFrame, list[str]]:
    source_path = metric_dir / "source_50.csv"
    dest_path = metric_dir / "destination_50.csv"
    if not source_path.exists() or not dest_path.exists():
        timestamps = _timestamps_from_traces(traces if traces is not None else pd.DataFrame())
        if timestamps.empty:
            timestamps = _timestamps_from_metric_files(metric_dir)
        dep_edges = []
        if dependency is not None and not dependency.empty:
            dep_edges = [f"{r.Source}_{r.Target}" for r in dependency.itertuples(index=False)]
        return timestamps, sorted(set(dep_edges))

    source = pd.read_csv(metric_dir / "source_50.csv")
    destination = pd.read_csv(metric_dir / "destination_50.csv")
    ts_col = "timestamp"
    timestamps = (
        pd.concat([source[ts_col], destination[ts_col]], ignore_index=True)
        .dropna()
        .astype(int)
        .drop_duplicates()
        .sort_values()
        .to_numpy()
    )
    edge_cols = sorted(
        (set(source.columns) | set(destination.columns))
        - {"time", "timestamp", "Time", "TimeStamp"}
    )
    return pd.DataFrame({"timestamp": timestamps}), edge_cols


def _trace_files(case_dir: Path) -> list[Path]:
    trace_dir = case_dir / "trace"
    if not trace_dir.is_dir():
        return []
    return [p for p in sorted(trace_dir.glob("*_trace.csv")) if p.name != "trace_stats.csv"]


def load_case_traces(case_dir: Path, retained_ids: Optional[set[str]]) -> pd.DataFrame:
    frames = []
    for path in _trace_files(case_dir):
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["TraceID"] = df["TraceID"].astype(str)
        if retained_ids is not None:
            df = df[df["TraceID"].isin(retained_ids)]
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=[
            "TraceID", "SpanID", "ParentID", "PodName", "StartTimeUnixNano",
            "EndTimeUnixNano", "Duration",
        ])
    return pd.concat(frames, ignore_index=True)


def _assign_nearest_timestamps(span_starts: pd.Series, timestamps: np.ndarray) -> np.ndarray:
    if len(span_starts) == 0:
        return np.array([], dtype=int)
    if len(timestamps) == 0:
        seconds = pd.to_numeric(span_starts, errors="coerce").fillna(0).to_numpy(dtype=np.int64) // 10**9
        return (seconds // 60 * 60).astype(int)
    seconds = pd.to_numeric(span_starts, errors="coerce").fillna(0).to_numpy(dtype=np.float64) / 1e9
    pos = np.searchsorted(timestamps, seconds)
    pos = np.clip(pos, 0, len(timestamps) - 1)
    prev_pos = np.clip(pos - 1, 0, len(timestamps) - 1)
    choose_prev = np.abs(seconds - timestamps[prev_pos]) <= np.abs(seconds - timestamps[pos])
    nearest = np.where(choose_prev, timestamps[prev_pos], timestamps[pos])
    return nearest.astype(int)


def trace_latency_frames(
    traces: pd.DataFrame,
    timestamps: pd.DataFrame,
    dependency: pd.DataFrame,
    edge_columns: Optional[Iterable[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build MicroRCA source/destination latency frames from retained spans.

    The source frame carries mean child-span latency in seconds for each
    cross-service edge. The destination frame is a zero frame with the same
    schema; the runner sums source+destination, so this keeps sampled trace
    latency single-counted.
    """
    ts = timestamps["timestamp"].astype(int).drop_duplicates().sort_values().to_numpy()
    dep_edges = {f"{r.Source}_{r.Target}" for r in dependency.itertuples(index=False)}
    columns = sorted(set(edge_columns or []) | dep_edges)
    source = pd.DataFrame({"timestamp": ts})
    destination = pd.DataFrame({"timestamp": ts})
    for col in columns:
        source[col] = 0.0
        destination[col] = 0.0

    diagnostics = {"n_spans": int(len(traces)), "n_edge_observations": 0, "n_edges_with_latency": 0}
    if traces.empty or len(ts) == 0:
        return source, destination, diagnostics

    required = {"TraceID", "SpanID", "ParentID", "PodName", "StartTimeUnixNano"}
    if not required.issubset(traces.columns):
        return source, destination, diagnostics

    base = traces.copy()
    base["TraceID"] = base["TraceID"].astype(str)
    base["SpanID"] = base["SpanID"].astype(str)
    base["ParentID"] = base["ParentID"].astype(str)
    parents = base[["TraceID", "SpanID", "PodName"]].rename(
        columns={"SpanID": "_ParentSpanID", "PodName": "_ParentPod"}
    )
    child = base.merge(
        parents,
        left_on=["TraceID", "ParentID"],
        right_on=["TraceID", "_ParentSpanID"],
        how="left",
    )
    child = child.dropna(subset=["_ParentPod"])
    if child.empty:
        return source, destination, diagnostics

    child["src"] = child["_ParentPod"].map(service_base)
    child["dst"] = child["PodName"].map(service_base)
    child = child[(child["src"] != "") & (child["dst"] != "") & (child["src"] != child["dst"])]
    if child.empty:
        return source, destination, diagnostics

    if {"StartTimeUnixNano", "EndTimeUnixNano"}.issubset(child.columns):
        start = pd.to_numeric(child["StartTimeUnixNano"], errors="coerce")
        end = pd.to_numeric(child["EndTimeUnixNano"], errors="coerce")
        child["_latency_sec"] = (end - start).clip(lower=0) / 1e9
    else:
        child["_latency_sec"] = pd.to_numeric(child.get("Duration", 0.0), errors="coerce").fillna(0.0) / 1e6
    child["_timestamp"] = _assign_nearest_timestamps(child["StartTimeUnixNano"], ts)
    child["_edge"] = child["src"].astype(str) + "_" + child["dst"].astype(str)
    child = child[child["_edge"].isin(columns)]
    diagnostics["n_edge_observations"] = int(len(child))
    if child.empty:
        return source, destination, diagnostics

    agg = (
        child.groupby(["_timestamp", "_edge"], as_index=False)["_latency_sec"]
        .mean()
        .rename(columns={"_timestamp": "timestamp"})
    )
    wide = agg.pivot_table(index="timestamp", columns="_edge", values="_latency_sec", aggfunc="mean")
    diagnostics["n_edges_with_latency"] = int(len(wide.columns))
    source = source.set_index("timestamp")
    for col in wide.columns:
        source.loc[wide.index.astype(int), col] = wide[col].astype(float)
    source = source.reset_index()
    return source, destination, diagnostics


def _case_dir(unisage_data_root: Path, entry: CaseEntry) -> Path:
    return unisage_data_root / entry.dataset / "batches" / entry.batch / entry.case


def materialize_case(
    entry: CaseEntry,
    unisage_data_root: Path,
    nezha_rca_dir: Path,
    retained_ids: Optional[set[str]],
    mode: str,
    scratch_root: Path,
) -> Path:
    if mode not in {"full_metric", "sampled_trace"}:
        raise ValueError(f"unknown MicroRCA mode: {mode!r}")

    scratch_root.mkdir(parents=True, exist_ok=True)
    metric_dir = nezha_rca_dir / entry.nz_date / "metric"
    case_dir = _case_dir(unisage_data_root, entry)
    all_traces = load_case_traces(case_dir, retained_ids=None)
    dependency = _standard_dependency(metric_dir)
    if dependency.empty:
        dependency = dependency_from_traces(all_traces)
    dependency.to_csv(scratch_root / "dependency.csv", index=False)
    services = materialize_resource_metrics(metric_dir, scratch_root / "metrics")

    has_full_mesh_latency = (metric_dir / "source_50.csv").exists() and (metric_dir / "destination_50.csv").exists()
    if mode == "full_metric" and has_full_mesh_latency:
        _link_or_copy_file(metric_dir / "source_50.csv", scratch_root / "latency_source.csv")
        _link_or_copy_file(metric_dir / "destination_50.csv", scratch_root / "latency_destination.csv")
        trace_diag = {"n_spans": None, "n_edge_observations": None, "n_edges_with_latency": None}
    else:
        timestamps, edge_cols = _read_latency_template(metric_dir, traces=all_traces, dependency=dependency)
        traces = all_traces if mode == "full_metric" else load_case_traces(case_dir, retained_ids)
        source, destination, trace_diag = trace_latency_frames(traces, timestamps, dependency, edge_cols)
        source.to_csv(scratch_root / "latency_source.csv", index=False)
        destination.to_csv(scratch_root / "latency_destination.csv", index=False)

    meta = {
        "dataset": entry.dataset,
        "batch": entry.batch,
        "case": entry.case,
        "mode": mode,
        "nz_date": entry.nz_date,
        "nz_window": entry.nz_window,
        "inject_pod": entry.inject_pod,
        "inject_service": service_base(entry.inject_pod),
        "inject_type": entry.inject_type,
        "inject_timestamp": entry.inject_timestamp,
        "n_retained_ids": None if retained_ids is None else len(retained_ids),
        "n_metric_services": len(services),
        "has_full_mesh_latency": has_full_mesh_latency,
        "trace_latency_diagnostics": trace_diag,
    }
    (scratch_root / "case_meta.json").write_text(json.dumps(meta, indent=2))
    return scratch_root


def run_microrca_subprocess(
    case_path: Path,
    python_exe: Optional[str] = None,
    alpha: float = 0.55,
    ad_threshold: float = 0.045,
    smoothing_window: int = 12,
    top_k: int = 20,
    timeout_sec: int = 300,
) -> dict:
    exe = python_exe or sys.executable
    cmd = [
        exe,
        str(MICRORCA_RUNNER),
        "--case-dir", str(case_path),
        "--alpha", str(alpha),
        "--ad-threshold", str(ad_threshold),
        "--smoothing-window", str(smoothing_window),
        "--top-k", str(top_k),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec, check=False)
    except subprocess.TimeoutExpired as exc:
        return {"ranked": [], "error": f"timeout after {timeout_sec}s", "stderr_tail": str(exc)}

    stdout = proc.stdout.strip()
    if proc.returncode != 0:
        return {
            "ranked": [],
            "error": f"returncode={proc.returncode}",
            "stdout_tail": stdout[-1000:],
            "stderr_tail": proc.stderr[-1000:],
        }
    try:
        return json.loads(stdout.splitlines()[-1])
    except Exception as exc:
        return {
            "ranked": [],
            "error": f"json parse failed: {exc}",
            "stdout_tail": stdout[-1000:],
            "stderr_tail": proc.stderr[-1000:],
        }


def _load_entry_from_index(case_index: Path, dataset: str, batch: str, case: str) -> CaseEntry:
    df = pd.read_csv(case_index)
    rows = df[(df["dataset"] == dataset) & (df["batch"].astype(str) == str(batch)) & (df["case"].astype(str) == str(case))]
    if rows.empty:
        raise SystemExit(f"case not found in {case_index}: {dataset}/{batch}/{case}")
    return case_entry_from_row(rows.iloc[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_idx = sub.add_parser("build-index")
    p_idx.add_argument("--data-root", default="data")
    p_idx.add_argument("--nezha-rca-dir", default="baselines/nezha/rca_data")
    p_idx.add_argument("--out", default="results/microrca/microrca_case_index.csv")

    p_mat = sub.add_parser("materialize")
    p_mat.add_argument("--data-root", default="data")
    p_mat.add_argument("--nezha-rca-dir", default="baselines/nezha/rca_data")
    p_mat.add_argument("--case-index", default="results/microrca/microrca_case_index.csv")
    p_mat.add_argument("--dataset", required=True)
    p_mat.add_argument("--batch", required=True)
    p_mat.add_argument("--case", required=True)
    p_mat.add_argument("--mode", choices=["full_metric", "sampled_trace"], default="sampled_trace")
    p_mat.add_argument("--out", required=True)

    args = ap.parse_args()
    if args.cmd == "build-index":
        df = write_case_index(Path(args.data_root), Path(args.nezha_rca_dir), Path(args.out))
        print(f"Wrote {args.out} ({len(df)} cases)")
    elif args.cmd == "materialize":
        entry = _load_entry_from_index(Path(args.case_index), args.dataset, args.batch, args.case)
        materialize_case(
            entry,
            unisage_data_root=Path(args.data_root),
            nezha_rca_dir=Path(args.nezha_rca_dir),
            retained_ids=None,
            mode=args.mode,
            scratch_root=Path(args.out),
        )
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
