"""Self-contained TraStrainer trace sampler.

TraStrainer is the baseline introduced by Huang et al. in "TraStrainer:
Adaptive Sampling for Distributed Traces with System Runtime State" (FSE
2024). The authors publish a reference implementation at
https://github.com/IntelligentDDS/TraStrainer that depends on PyTorch +
treelib + an LTSF_Linear forecasting model per `(service, metric)` pair.

This module reimplements the algorithm using only numpy + pandas so the
baseline can run on the same macOS arm64 / Python 3.9 environment used for
the other UniSage v0.2 baselines and so its behaviour stays auditable.

Faithfulness notes (mapped to upstream `TraStrainer.py`):

- ``compute_trace_features`` mirrors ``TraceProcessor.compute_feature_values``:
  per-service feature = ``num_spans * avg_duration * (1 + num_failures)``
  using span data from one trace. The upstream code never multiplies in any
  raw metric value — the `metrics` dict only supplies the *set* of
  ``(service, metric_name)`` keys that define the system-bias vector
  dimensionality, and the per-trace value is derived from the trace alone.

- ``trace_structure`` mirrors ``TraceProcessor.get_seq_span``: each span
  becomes ``"depth-service-op-status-duration_bucket"`` (duration bucket =
  ``int(duration / 1e4)`` to match the upstream tokenizer), and the bag is
  sorted ascending before comparison. Multiset Jaccard matches upstream's
  ``compute_jaccord_similarity`` (note the spelling) which iterates a copy
  of each spanline and pops matches.

- ``system_biased_filter`` reproduces the n-sigma rule and the
  ``tanh`` squashing applied to the weighted average n-sigma. Without
  LTSF_Linear checkpoints upstream falls back to weight=1 per metric
  (`compute_metrics_weights` early-returns); we expose `metric_weights`
  as an optional dict and default to all-1 weights, which matches a
  cold-start deployment of upstream as well.

- ``diversity_biased_filter`` reproduces the upstream rule: similarity
  score against the most-similar historical structure, weighted by that
  structure's count in the recent window, normalized by the running
  ``diversity_window``. When nothing matches we return 1.0 (upstream's
  unseen-structure branch).

- ``judge`` reproduces the AND-vs-OR voting: AND when the current keep
  rate exceeds the target budget (i.e. we are over budget and must be
  strict), OR otherwise.

- ``run`` streams traces in chronological order and stops the streaming
  pass at ``cnt >= round(budget * n_traces)``. Upstream uses the same
  termination ("sampling cost is bounded to budget*N traces"); traces
  after the break get a False decision in the v0.2 output frame.

Defaults follow the upstream `sampler.py` demo (warm_up=10, window=int(1/α))
and FSE 2024 §IV.

Public surface:

    TraStrainerConfig          — knob bag
    TraStrainer                — class with .step() per trace
    run_trastrainer_over_trace_frame(spans, alpha, ...)
    trace_order_index(spans)
    compute_trace_features(spans, service_keys)
    trace_structure(spans)
    smoke_self_test()

The runner ``scripts/run_trastrainer_baseline.py`` calls
``run_trastrainer_over_trace_frame`` once per budget over the streaming
order, and the v0.2 evaluator picks up the resulting columns through
``RANKING_COLUMNS["trastrainer"]`` and
``DECISION_COLUMN_PREFIXES["trastrainer_decision"]``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# Upstream uses fixed dataset-agnostic constants; mirror them here.
DEFAULT_WARM_UP = 10
# Duration is bucketed into 10us-equivalent integer buckets in upstream.
# Spans in the UniSage normalized frame carry SpanDurationMs (milliseconds),
# so the equivalent factor is `int(SpanDurationMs * 100)` (1 ms = 1e4 of
# upstream's bucket units when duration was in nanoseconds / 1e4).
DURATION_BUCKET_FACTOR = 100.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _depth_map(spans: pd.DataFrame) -> Dict[str, int]:
    """Compute depth (root=0) for every span in one trace.

    The normalized frame already carries ParentID and SpanID strings. Roots
    are spans whose ParentID is one of {"root", "nan", "none", ""} or whose
    parent is not in the trace.
    """
    span_ids = set(spans["SpanID"].astype(str))
    parent = {
        str(row.SpanID): (
            str(row.ParentID)
            if str(row.ParentID).lower() not in {"root", "nan", "none", ""}
            and str(row.ParentID) in span_ids
            else None
        )
        for row in spans.itertuples(index=False)
    }
    depth: Dict[str, int] = {}

    def lookup(sid: str) -> int:
        if sid in depth:
            return depth[sid]
        pid = parent.get(sid)
        if pid is None:
            depth[sid] = 0
        else:
            depth[sid] = lookup(pid) + 1
        return depth[sid]

    for sid in parent:
        lookup(sid)
    return depth


def trace_structure(spans: pd.DataFrame) -> List[str]:
    """Return upstream-style sorted span tokens for one trace.

    Each span contributes ``"depth-service-operation-status-bucket"``. The
    bag is sorted lexicographically so jaccard similarity over multisets is
    well-defined. ``status`` is hard-coded to ``success`` because upstream
    does the same when no status column is parsed (the normalized v0.2 frame
    drops error flags).
    """
    if spans.empty:
        return []
    depth = _depth_map(spans)
    tokens: List[str] = []
    for row in spans.itertuples(index=False):
        sid = str(row.SpanID)
        d = depth.get(sid, 0)
        # SpanDurationMs is the canonical normalized field.
        dur_bucket = int(round(float(getattr(row, "SpanDurationMs", 0.0)) * DURATION_BUCKET_FACTOR))
        tokens.append(
            f"{d}-{row.Service}-{row.Operation}-success-{dur_bucket}"
        )
    tokens.sort()
    return tokens


def compute_trace_features(
    spans: pd.DataFrame,
    service_keys: Sequence[Tuple[str, str]],
) -> Dict[Tuple[str, str], float]:
    """Per-trace system-bias feature vector keyed by ``(service, metric_name)``.

    Mirrors ``TraceProcessor.compute_feature_values`` upstream. The feature
    is identical across metric_name values for a given service (upstream
    behaviour); supplying multiple ``(service, metric_name)`` keys is how
    upstream raises the dimensionality of the system-bias term.
    """
    feature: Dict[Tuple[str, str], float] = {key: 0.0 for key in service_keys}
    if spans.empty:
        return feature
    grouped = spans.groupby("Service", sort=False)
    per_service: Dict[str, float] = {}
    for service, group in grouped:
        n_spans = len(group)
        if n_spans == 0:
            per_service[str(service)] = 0.0
            continue
        avg_duration = float(group["SpanDurationMs"].fillna(0.0).mean())
        # No failure flag in the v0.2 normalized frame; upstream uses status
        # field rarely populated either. Treat all as success.
        n_failures = 0
        per_service[str(service)] = n_spans * avg_duration * (1 + n_failures)
    for service, metric in service_keys:
        feature[(service, metric)] = per_service.get(service, 0.0)
    return feature


def multiset_jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    """Upstream ``compute_jaccord_similarity`` semantics on token multisets."""
    if not a and not b:
        return 0.0
    # Use Counter-style intersection to match the upstream pop loop.
    from collections import Counter

    ca = Counter(a)
    cb = Counter(b)
    inter = sum((ca & cb).values())
    union = sum((ca | cb).values())
    return inter / union if union else 0.0


def trace_order_index(spans: pd.DataFrame) -> pd.DataFrame:
    """Return one row per TraceID sorted by earliest Start (chronological)."""
    if spans.empty:
        return pd.DataFrame(columns=["TraceID", "Start"])
    return (
        spans.groupby("TraceID", sort=False)["Start"]
        .min()
        .reset_index()
        .sort_values(["Start", "TraceID"])
        .reset_index(drop=True)
    )


def derive_service_keys(
    spans: pd.DataFrame,
    metric_keys: Optional[Sequence[Tuple[str, str]]] = None,
) -> List[Tuple[str, str]]:
    """Pick the ``(service, metric_name)`` set defining the system-bias vector.

    If `metric_keys` is supplied (e.g. from a per-case metrics CSV), use it
    directly. Otherwise default to one synthetic key per service present in
    the case, named ``"trace_feature"``. This matches a cold-start upstream
    deployment with weight=1 per metric (the only mode reachable without
    LTSF_Linear checkpoints).
    """
    if metric_keys:
        return list(metric_keys)
    if spans.empty:
        return []
    services = sorted(str(s) for s in spans["Service"].dropna().unique())
    return [(svc, "trace_feature") for svc in services]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


@dataclass
class TraStrainerConfig:
    warm_up_size: int = DEFAULT_WARM_UP
    window_size: Optional[int] = None  # None = upstream rule int(1/alpha)
    seed: int = 1
    # Cap stream length to upstream's `cnt >= round(alpha * N)` budget cap.
    # Set to False to score every trace (no early termination).
    enforce_stream_cap: bool = True


@dataclass
class _RunStats:
    n_processed: int = 0
    n_kept: int = 0
    n_warm_up: int = 0
    strict_steps: int = 0
    relaxed_steps: int = 0


class TraStrainer:
    """Adaptive trace sampler: system-bias + diversity-bias with AND/OR voting."""

    def __init__(
        self,
        alpha: float,
        service_keys: Sequence[Tuple[str, str]],
        metric_weights: Optional[Dict[Tuple[str, str], float]] = None,
        config: Optional[TraStrainerConfig] = None,
    ):
        if alpha <= 0 or alpha > 1:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if config is None:
            config = TraStrainerConfig()
        self.alpha = float(alpha)
        self.cfg = config
        self.service_keys: List[Tuple[str, str]] = list(service_keys)
        if metric_weights is None:
            metric_weights = {}
        # Default weight=1 matches upstream's no-checkpoint branch.
        self.metric_weights: Dict[Tuple[str, str], float] = {
            key: float(metric_weights.get(key, 1.0)) for key in self.service_keys
        }
        # Window sizing: upstream uses int(1/alpha); enforce min 1.
        window = config.window_size if config.window_size is not None else int(round(1.0 / max(self.alpha, 1e-9)))
        self.window_size = max(int(window), 1)
        self.rng = np.random.default_rng(config.seed)
        self.history_trace_metrics: Dict[Tuple[str, str], Deque[float]] = {
            key: deque(maxlen=self.window_size) for key in self.service_keys
        }
        self.history_trace_structures: Deque[List[str]] = deque(maxlen=self.window_size)
        self.diversity_window: Deque[float] = deque(maxlen=self.window_size)
        self.stats = _RunStats()

    # ----- per-trace pieces -----

    @staticmethod
    def _tanh(x: float) -> float:
        # Numerically stable tanh; same shape as upstream.
        return float(np.tanh(x))

    def _system_rate(self, feature: Dict[Tuple[str, str], float]) -> float:
        if not self.metric_weights:
            return 0.0
        total = 0.0
        weight_sum = 0.0
        for key, weight in self.metric_weights.items():
            value = float(feature.get(key, 0.0))
            window = self.history_trace_metrics.get(key)
            if window is None or len(window) == 0:
                mean = 0.0
                std = 0.0
            else:
                arr = np.fromiter(window, dtype=np.float64)
                mean = float(arr.mean())
                std = float(arr.std())
            n_sigma = abs(value - mean) / (std + 1e-5)
            total += weight * n_sigma
            weight_sum += weight
        avg = total / weight_sum if weight_sum else 0.0
        return self._tanh(avg)

    def _diversity_rate(self, structure: List[str]) -> float:
        if not self.history_trace_structures:
            return 1.0
        best_sim = 0.0
        best_key: Optional[str] = None
        for hist in self.history_trace_structures:
            sim = multiset_jaccard(hist, structure)
            if sim > best_sim:
                best_sim = sim
                best_key = "+".join(hist)
        if best_key is None or best_sim <= 0:
            return 1.0
        counts: Dict[str, int] = {}
        for hist in self.history_trace_structures:
            k = "+".join(hist)
            counts[k] = counts.get(k, 0) + 1
        count = counts.get(best_key, 0)
        similarity_rate = round(best_sim * count, 2)
        if similarity_rate <= 0:
            return 1.0
        inv = 1.0 / similarity_rate
        self.diversity_window.append(inv)
        denom = float(sum(self.diversity_window))
        return inv / denom if denom > 0 else 1.0

    def _judge(self, sys_rate: float, div_rate: float, strict: bool) -> Tuple[bool, float, float]:
        sys_rand = float(self.rng.random())
        div_rand = float(self.rng.random())
        is_sys = sys_rand <= sys_rate
        is_div = div_rand <= div_rate
        if strict:
            return (is_sys and is_div), sys_rand, div_rand
        return (is_sys or is_div), sys_rand, div_rand

    def _update_history(
        self,
        structure: List[str],
        feature: Dict[Tuple[str, str], float],
    ) -> None:
        self.history_trace_structures.append(list(structure))
        for key in self.service_keys:
            self.history_trace_metrics[key].append(float(feature.get(key, 0.0)))

    # ----- public API -----

    def step(
        self,
        structure: List[str],
        feature: Dict[Tuple[str, str], float],
    ) -> Dict[str, float]:
        """Process one trace; return a row of result fields.

        Returns a dict with keys:
            decision      bool
            system_rate   float (NaN during warm-up)
            diversity_rate float (NaN during warm-up)
            strict        bool (whether AND voting was used)
            warm_up       bool
        """
        out = {
            "decision": False,
            "system_rate": float("nan"),
            "diversity_rate": float("nan"),
            "strict": False,
            "warm_up": False,
        }
        if self.stats.n_processed < self.cfg.warm_up_size:
            self.stats.n_warm_up += 1
            out["warm_up"] = True
            self._update_history(structure, feature)
            self.stats.n_processed += 1
            return out

        sys_rate = self._system_rate(feature)
        div_rate = self._diversity_rate(structure)
        strict = (
            (self.stats.n_kept / self.stats.n_processed) > self.alpha
            if self.stats.n_processed > 0
            else False
        )
        decision, _, _ = self._judge(sys_rate, div_rate, strict)
        if decision:
            self.stats.n_kept += 1
        if strict:
            self.stats.strict_steps += 1
        else:
            self.stats.relaxed_steps += 1
        self._update_history(structure, feature)
        self.stats.n_processed += 1
        out.update(
            {
                "decision": bool(decision),
                "system_rate": float(sys_rate),
                "diversity_rate": float(div_rate),
                "strict": bool(strict),
            }
        )
        return out


# ---------------------------------------------------------------------------
# Frame-level driver
# ---------------------------------------------------------------------------


def run_trastrainer_over_trace_frame(
    spans: pd.DataFrame,
    alpha: float,
    metric_keys: Optional[Sequence[Tuple[str, str]]] = None,
    metric_weights: Optional[Dict[Tuple[str, str], float]] = None,
    config: Optional[TraStrainerConfig] = None,
    structure_cache: Optional[Dict[str, List[str]]] = None,
    feature_cache: Optional[Dict[str, Dict[Tuple[str, str], float]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Run TraStrainer once at the given alpha over all traces in `spans`.

    `structure_cache` / `feature_cache` are optional precomputed per-trace
    inputs (keyed by TraceID) — the runner uses these to amortize tree /
    feature extraction across the per-budget passes.

    Returns `(per_trace_frame, timing)`. `per_trace_frame` carries one row
    per TraceID (in streaming order) with columns:
        TraceID
        TraStrainerSystemRate
        TraStrainerDiversityRate
        TraStrainerCombined        = (sys + div) / 2 (ranker view)
        TraStrainerDecision        bool
        TraStrainerStrict          bool (AND vote used)
        TraStrainerWarmUp          bool
        TraStrainerProcessed       bool (streamed before the budget cap)
    `timing` records `feature_sec`, `structure_sec`, `sampling_sec`,
    `extract_sec` (the sum of the first two), and `n_traces_streamed`.
    """
    cfg = config or TraStrainerConfig()
    timing = {
        "structure_sec": 0.0,
        "feature_sec": 0.0,
        "extract_sec": 0.0,
        "sampling_sec": 0.0,
        "n_traces_streamed": 0,
    }
    if spans.empty:
        return (
            pd.DataFrame(
                columns=[
                    "TraceID",
                    "TraStrainerSystemRate",
                    "TraStrainerDiversityRate",
                    "TraStrainerCombined",
                    "TraStrainerDecision",
                    "TraStrainerStrict",
                    "TraStrainerWarmUp",
                    "TraStrainerProcessed",
                ]
            ),
            timing,
        )

    order = trace_order_index(spans)
    grouped = {str(tid): g for tid, g in spans.groupby("TraceID", sort=False)}

    # Resolve service_keys union across the case so all per-trace vectors
    # share the same dimensions.
    service_keys = derive_service_keys(spans, metric_keys)

    # Pre-extract per-trace structure / feature (caching across budgets).
    import time as _time

    structures: Dict[str, List[str]] = {}
    features: Dict[str, Dict[Tuple[str, str], float]] = {}
    for trace_id in order["TraceID"].astype(str):
        if structure_cache is not None and trace_id in structure_cache:
            structures[trace_id] = structure_cache[trace_id]
        else:
            t0 = _time.perf_counter()
            structures[trace_id] = trace_structure(grouped.get(trace_id, spans.iloc[0:0]))
            timing["structure_sec"] += _time.perf_counter() - t0
            if structure_cache is not None:
                structure_cache[trace_id] = structures[trace_id]
        if feature_cache is not None and trace_id in feature_cache:
            features[trace_id] = feature_cache[trace_id]
        else:
            t0 = _time.perf_counter()
            features[trace_id] = compute_trace_features(
                grouped.get(trace_id, spans.iloc[0:0]), service_keys
            )
            timing["feature_sec"] += _time.perf_counter() - t0
            if feature_cache is not None:
                feature_cache[trace_id] = features[trace_id]
    timing["extract_sec"] = timing["structure_sec"] + timing["feature_sec"]

    # Stream
    sifter_like_cap = (
        max(1, int(round(alpha * len(order)))) if cfg.enforce_stream_cap else len(order)
    )
    sampler = TraStrainer(
        alpha=alpha,
        service_keys=service_keys,
        metric_weights=metric_weights,
        config=cfg,
    )

    rows = []
    streamed = 0
    sampling_start = _time.perf_counter()
    for trace_id in order["TraceID"].astype(str):
        if streamed >= sifter_like_cap:
            rows.append(
                {
                    "TraceID": trace_id,
                    "TraStrainerSystemRate": float("nan"),
                    "TraStrainerDiversityRate": float("nan"),
                    "TraStrainerCombined": float("nan"),
                    "TraStrainerDecision": False,
                    "TraStrainerStrict": False,
                    "TraStrainerWarmUp": False,
                    "TraStrainerProcessed": False,
                }
            )
            continue
        result = sampler.step(structures[trace_id], features[trace_id])
        sys_rate = result["system_rate"]
        div_rate = result["diversity_rate"]
        if np.isnan(sys_rate) or np.isnan(div_rate):
            combined = float("nan")
        else:
            combined = 0.5 * (sys_rate + div_rate)
        rows.append(
            {
                "TraceID": trace_id,
                "TraStrainerSystemRate": sys_rate,
                "TraStrainerDiversityRate": div_rate,
                "TraStrainerCombined": combined,
                "TraStrainerDecision": result["decision"],
                "TraStrainerStrict": result["strict"],
                "TraStrainerWarmUp": result["warm_up"],
                "TraStrainerProcessed": True,
            }
        )
        streamed += 1
    timing["sampling_sec"] = _time.perf_counter() - sampling_start
    timing["n_traces_streamed"] = streamed
    return pd.DataFrame(rows), timing


def load_metric_keys_aiops21(metrics_csv: "Optional[str]") -> List[Tuple[str, str]]:
    """Read the AIOps21 abnormal/metrics.csv and return its `(service, metric)` keys.

    Upstream `MetricProcessor.process_metrics` ships a hard-coded
    `INCLUDED_METRICS` filter for the new OTel format
    (`k8s.pod.cpu.usage`, etc.). The AIOps21 chaos dataset uses its own
    domain-specific MetricName values (e.g. `Mysql-MySQL_3306_Innodb buffer
    pool read requests`, `OSLinux-OSLinux_MEMORY_MEMORY_UserMem`), so the
    upstream filter would drop every row. The whole point of running
    TraStrainer on AIOps21 is to use *that* metric set, so we read every
    `(service, metric)` pair the file contains. Service is the
    `k8s_pod_name` (or `ServiceName` if present).
    """
    if metrics_csv is None:
        return []
    try:
        frame = pd.read_csv(metrics_csv, usecols=lambda c: c in {"ServiceName", "MetricName", "k8s_pod_name"})
    except Exception:
        return []
    if frame.empty:
        return []
    if "ServiceName" in frame.columns and frame["ServiceName"].notna().any():
        services = frame["ServiceName"].fillna("")
    elif "k8s_pod_name" in frame.columns:
        services = frame["k8s_pod_name"].fillna("")
    else:
        return []
    metrics = frame["MetricName"].fillna("") if "MetricName" in frame.columns else pd.Series([""], index=services.index)
    df = pd.DataFrame({"service": services.astype(str), "metric": metrics.astype(str)})
    df = df[(df["service"] != "") & (df["metric"] != "")]
    keys = sorted({(s, m) for s, m in zip(df["service"], df["metric"])})
    return keys


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------


def smoke_self_test() -> bool:
    """Synthetic check: an outlier trace should be more likely to be kept."""
    rng = np.random.default_rng(0)
    spans_rows = []
    # 60 common traces: 3 spans each over services {A,B,C}, ms~10
    for ti in range(60):
        tid = f"common-{ti}"
        for si, svc in enumerate("ABC"):
            spans_rows.append(
                {
                    "TraceID": tid,
                    "SpanID": f"{tid}-{si}",
                    "ParentID": "root" if si == 0 else f"{tid}-{si-1}",
                    "Service": svc,
                    "Operation": "op",
                    "Start": pd.Timestamp("2024-01-01") + pd.Timedelta(seconds=ti),
                    "SpanDurationMs": 10.0 + rng.standard_normal(),
                }
            )
    # 5 outlier traces: 6 spans, much higher latency
    for oi in range(5):
        tid = f"outlier-{oi}"
        for si in range(6):
            spans_rows.append(
                {
                    "TraceID": tid,
                    "SpanID": f"{tid}-{si}",
                    "ParentID": "root" if si == 0 else f"{tid}-{si-1}",
                    "Service": "Z" if si == 0 else "Q",
                    "Operation": "rare_op",
                    "Start": pd.Timestamp("2024-01-01") + pd.Timedelta(seconds=80 + oi),
                    "SpanDurationMs": 500.0 + rng.standard_normal(),
                }
            )
    spans = pd.DataFrame(spans_rows)
    cfg = TraStrainerConfig(warm_up_size=5, window_size=20, seed=0, enforce_stream_cap=False)
    frame, _ = run_trastrainer_over_trace_frame(spans, alpha=0.2, config=cfg)
    if frame.empty:
        return False
    keep_rate_outlier = frame[frame["TraceID"].str.startswith("outlier-")]["TraStrainerDecision"].mean()
    keep_rate_common = frame[frame["TraceID"].str.startswith("common-")]["TraStrainerDecision"].mean()
    return float(keep_rate_outlier) > float(keep_rate_common)


if __name__ == "__main__":
    ok = smoke_self_test()
    print("smoke_self_test:", "PASS" if ok else "FAIL")
