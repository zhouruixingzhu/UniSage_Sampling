"""Run MicroRCA on one materialized UniSage scratch case.

The input directory is produced by `microrca_adapter.materialize_case` and
contains latency source/destination time series, a dependency graph, normalized
resource metrics, and case metadata. This runner intentionally avoids upstream
path assumptions while keeping MicroRCA's key steps: smooth invocation latency,
detect anomalous edges with BIRCH, build an attributed service graph, and rank
services with PageRank.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import Birch

DEFAULT_ALPHA = 0.55
DEFAULT_AD_THRESHOLD = 0.045
DEFAULT_SMOOTHING_WINDOW = 12

TIME_COLUMNS = {"time", "timestamp", "Time", "TimeStamp"}


def service_base(name: object) -> str:
    """Normalize pod/file names to MicroRCA service-level nodes."""
    s = str(name or "").strip()
    s = s.removesuffix("_metric.csv").removesuffix(".csv")
    m = re.match(r"(.+)-[a-z0-9]{9,}-[a-z0-9]{5}$", s)
    return m.group(1) if m else s


def _timestamp_column(df: pd.DataFrame) -> str:
    for col in ("timestamp", "TimeStamp", "time", "Time"):
        if col in df.columns:
            return col
    raise ValueError("missing timestamp/time column")


def _read_latency_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["timestamp"])
    ts_col = _timestamp_column(df)
    out = pd.DataFrame({"timestamp": pd.to_numeric(df[ts_col], errors="coerce")})
    for col in df.columns:
        if col in TIME_COLUMNS:
            continue
        out[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def load_latency(case_dir: Path) -> pd.DataFrame:
    """Return source+destination invocation latency, aligned by timestamp."""
    src = _read_latency_csv(case_dir / "latency_source.csv")
    dst = _read_latency_csv(case_dir / "latency_destination.csv")
    timestamps = sorted(set(src["timestamp"].astype(int)) | set(dst["timestamp"].astype(int)))
    out = pd.DataFrame({"timestamp": timestamps})
    edge_cols = sorted((set(src.columns) | set(dst.columns)) - {"timestamp"})
    for col in edge_cols:
        left = src[["timestamp", col]].copy() if col in src.columns else pd.DataFrame({"timestamp": timestamps, col: 0.0})
        right = dst[["timestamp", col]].copy() if col in dst.columns else pd.DataFrame({"timestamp": timestamps, col: 0.0})
        merged = (
            out[["timestamp"]]
            .merge(left, on="timestamp", how="left")
            .merge(right, on="timestamp", how="left", suffixes=("_src", "_dst"))
        )
        out[col] = merged[f"{col}_src"].fillna(0.0) + merged[f"{col}_dst"].fillna(0.0)
    return out


def load_dependency(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["Source", "Target", "Weight"])
    out = pd.DataFrame({
        "Source": df["Source"].map(service_base),
        "Target": df["Target"].map(service_base),
        "Weight": pd.to_numeric(df.get("Weight", 1.0), errors="coerce").fillna(1.0),
    })
    out = out[(out["Source"] != "") & (out["Target"] != "")]
    return out.groupby(["Source", "Target"], as_index=False)["Weight"].max()


def _nearest_index(timestamps: pd.Series, fault_ts: Optional[float]) -> int:
    if len(timestamps) == 0:
        return 0
    if fault_ts is None or not math.isfinite(fault_ts):
        return len(timestamps) - 1
    values = timestamps.astype(float).to_numpy()
    return int(np.argmin(np.abs(values - float(fault_ts))))


def _robust_scale(values: np.ndarray) -> float:
    q75, q25 = np.percentile(values, [75, 25])
    iqr = float(q75 - q25)
    if iqr > 1e-9:
        return iqr
    std = float(np.std(values))
    return std if std > 1e-9 else 1.0


def anomaly_strength(
    series: pd.Series,
    timestamps: pd.Series,
    fault_ts: Optional[float],
    ad_threshold: float = DEFAULT_AD_THRESHOLD,
    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
) -> tuple[float, bool]:
    """Return a non-negative anomaly strength and whether BIRCH split clusters."""
    raw = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    if len(raw) < 2 or raw.nunique(dropna=False) <= 1:
        return 0.0, False

    smoothed = raw.rolling(max(1, int(smoothing_window)), min_periods=1).mean()
    values = smoothed.to_numpy(dtype=float)
    center = float(np.median(values))
    scale = _robust_scale(values)
    z = (values - center) / scale

    split = False
    try:
        labels = Birch(threshold=ad_threshold, n_clusters=None).fit_predict(z.reshape(-1, 1))
        split = len(set(labels)) > 1
    except Exception:
        labels = np.zeros(len(z), dtype=int)

    fault_idx = _nearest_index(timestamps, fault_ts)
    lo = max(0, fault_idx - 2)
    hi = min(len(z), fault_idx + 3)
    near_peak = float(np.max(z[lo:hi])) if hi > lo else float(z[fault_idx])
    global_peak = float(np.max(z))
    strength = max(0.0, near_peak)

    if split and len(set(labels[lo:hi])) > 0:
        label_means = {lab: float(np.mean(z[labels == lab])) for lab in set(labels)}
        high_label = max(label_means, key=label_means.get)
        split = high_label in set(labels[lo:hi])

    # If the fault window is quiet but the case has a clear high-latency cluster,
    # keep a weak signal; this mirrors MicroRCA's artifact-level edge selection,
    # which is not tied to an injected timestamp argument.
    if strength <= 0.0 and split:
        strength = max(0.0, global_peak) * 0.25
    return float(strength), bool(split and strength > 0.0)


def edge_parts(edge_col: str) -> tuple[str, str]:
    if "_" not in edge_col:
        return edge_col, edge_col
    src, dst = edge_col.split("_", 1)
    return service_base(src), service_base(dst)


def load_metric_scores(
    metrics_dir: Path,
    fault_ts: Optional[float],
    ad_threshold: float = DEFAULT_AD_THRESHOLD,
    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    if not metrics_dir.is_dir():
        return scores
    for path in sorted(metrics_dir.glob("*.csv")):
        service = service_base(path.name)
        df = pd.read_csv(path)
        if df.empty:
            continue
        ts_col = _timestamp_column(df)
        timestamps = pd.to_numeric(df[ts_col], errors="coerce").fillna(0.0)
        best = 0.0
        for col in df.columns:
            if col in TIME_COLUMNS:
                continue
            vals = pd.to_numeric(df[col], errors="coerce")
            if vals.notna().sum() < 2:
                continue
            strength, split = anomaly_strength(vals, timestamps, fault_ts, ad_threshold, smoothing_window)
            if split:
                best = max(best, strength)
        if best > 0:
            scores[service] = float(best)
    return scores


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    max_v = max(scores.values())
    if max_v <= 0:
        return {k: 0.0 for k in scores}
    return {k: float(v) / max_v for k, v in scores.items()}


def rank_case_from_frames(
    latency: pd.DataFrame,
    dependency: pd.DataFrame,
    metric_scores: dict[str, float],
    fault_ts: Optional[float],
    alpha: float = DEFAULT_ALPHA,
    ad_threshold: float = DEFAULT_AD_THRESHOLD,
    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
) -> tuple[list[dict], dict]:
    timestamps = latency["timestamp"] if "timestamp" in latency.columns else pd.Series(dtype=float)
    edge_scores: dict[str, float] = {}
    for col in latency.columns:
        if col == "timestamp":
            continue
        strength, split = anomaly_strength(latency[col], timestamps, fault_ts, ad_threshold, smoothing_window)
        if split and strength > 0:
            edge_scores[col] = strength

    edge_norm = _normalize_scores(edge_scores)
    metric_norm = _normalize_scores(metric_scores)

    graph = nx.DiGraph()
    for _, row in dependency.iterrows():
        src = service_base(row["Source"])
        dst = service_base(row["Target"])
        if not src or not dst or src == dst:
            continue
        edge_key = f"{src}_{dst}"
        edge_signal = edge_norm.get(edge_key, 0.0)
        metric_signal = metric_norm.get(dst, 0.0)
        weight = 1.0 + alpha * edge_signal + (1.0 - alpha) * metric_signal
        graph.add_edge(src, dst, weight=float(weight))

    # Ensure services observed only in metrics/anomalous edges still rank.
    for service in metric_norm:
        graph.add_node(service)
    for edge_key in edge_norm:
        src, dst = edge_parts(edge_key)
        graph.add_edge(src, dst, weight=1.0 + alpha * edge_norm[edge_key])

    if graph.number_of_nodes() == 0:
        return [], {"n_anomalous_edges": 0, "n_metric_services": len(metric_norm)}

    personalization = {node: 0.0 for node in graph.nodes}
    for edge_key, value in edge_norm.items():
        src, dst = edge_parts(edge_key)
        if dst in personalization:
            personalization[dst] += value
        if src in personalization:
            personalization[src] += value * 0.25
    for service, value in metric_norm.items():
        if service in personalization:
            personalization[service] += value

    if sum(personalization.values()) <= 0:
        personalization = {node: 1.0 for node in graph.nodes}

    # Upstream MicroRCA runs PageRank over a graph whose direction lets evidence
    # flow through dependencies. The reverse view keeps a destination latency
    # anomaly from ranking only its immediate caller.
    pr_graph = graph.reverse(copy=True)
    try:
        pagerank = nx.pagerank(pr_graph, personalization=personalization, weight="weight", max_iter=200)
    except nx.PowerIterationFailedConvergence:
        pagerank = {node: 1.0 / graph.number_of_nodes() for node in graph.nodes}

    pers_norm = _normalize_scores(personalization)
    pr_norm = _normalize_scores(pagerank)
    ranked = []
    for service in graph.nodes:
        score = alpha * pr_norm.get(service, 0.0) + (1.0 - alpha) * pers_norm.get(service, 0.0)
        ranked.append({
            "service": service,
            "pod": service,
            "score": float(score),
            "pagerank": float(pagerank.get(service, 0.0)),
            "personalization": float(personalization.get(service, 0.0)),
        })
    ranked.sort(key=lambda r: (-r["score"], r["service"]))
    diagnostics = {
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "n_anomalous_edges": len(edge_norm),
        "n_metric_services": len(metric_norm),
        "top_anomalous_edges": sorted(edge_norm.items(), key=lambda kv: -kv[1])[:10],
    }
    return ranked, diagnostics


def run_case(
    case_dir: Path,
    alpha: float = DEFAULT_ALPHA,
    ad_threshold: float = DEFAULT_AD_THRESHOLD,
    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
    top_k: int = 20,
) -> dict:
    meta_path = case_dir / "case_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    fault_ts = meta.get("inject_timestamp")
    try:
        fault_ts = float(fault_ts) if fault_ts not in (None, "") else None
    except (TypeError, ValueError):
        fault_ts = None

    latency = load_latency(case_dir)
    dependency = load_dependency(case_dir / "dependency.csv")
    metric_scores = load_metric_scores(case_dir / "metrics", fault_ts, ad_threshold, smoothing_window)
    ranked, diagnostics = rank_case_from_frames(
        latency,
        dependency,
        metric_scores,
        fault_ts,
        alpha=alpha,
        ad_threshold=ad_threshold,
        smoothing_window=smoothing_window,
    )
    return {
        "ranked": ranked[:top_k],
        "diagnostics": diagnostics,
        "meta": meta,
        "params": {
            "alpha": alpha,
            "ad_threshold": ad_threshold,
            "smoothing_window": smoothing_window,
            "top_k": top_k,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-dir", required=True)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--ad-threshold", type=float, default=DEFAULT_AD_THRESHOLD)
    ap.add_argument("--smoothing-window", type=int, default=DEFAULT_SMOOTHING_WINDOW)
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()

    try:
        out = run_case(
            Path(args.case_dir),
            alpha=args.alpha,
            ad_threshold=args.ad_threshold,
            smoothing_window=args.smoothing_window,
            top_k=args.top_k,
        )
        print(json.dumps(out, ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({"error": str(exc), "ranked": []}), file=sys.stdout)
        raise


if __name__ == "__main__":
    main()

