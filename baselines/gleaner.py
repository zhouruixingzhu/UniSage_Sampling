"""Self-contained Gleaner trace sampler.

Gleaner is the online tail sampler introduced by Yang, Fang, Zhang and He in
"Gleaner: A Semantically-Rich and Efficient Online Sampler for Microservice
Diagnostics" (arXiv:2604.16810v1, 2026). Unlike the trace-only baselines in
this directory, Gleaner is *log-enriched*: it encodes each trace together with
its associated intra-span log events as a lightweight Event-Pair Set (EPS),
scores each trace for anomaly relevance, and selects a diverse, anomaly-rich
subset with a Determinantal Point Process (DPP).

This module reimplements Gleaner's three-stage pipeline directly from the paper
using only numpy/pandas so that it runs on the same macOS arm64 / Python 3.9
environment used for the other UniSage baselines and stays auditable. It is
deliberately self-contained (no label-generation or evaluation imports); the companion
runner `scripts/run_gleaner_baseline.py` handles raw-data loading, log parsing,
manifest merging and version stamping.

Three stages (paper Sec. 3), per case:

1. Trace Encoder -> EPS + base anomaly score.
   Each span contributes a ``("SPAN", label)`` token and a structural
   ``("EDGE", parent_label, child_label)`` / ``("ROOT", label)`` pair. Each
   attributed log adds a ``("LOG", label, template)`` token plus an
   ``("ERR"/"WARN", label)`` marker by severity. The base score follows
   Equation 1: ``sum_spans (w_err*1[error] + w_lw*|warn logs| + w_le*|error
   logs|)``.
2. Quota Allocator -> even split of the budget across root-API groups. (The
   paper's alarm-driven boost is dropped in this offline fixed-budget setting;
   see `paper_impact/020_gleaner_baseline.md` for the rationale.) A trace-level
   performance term ``w_perf`` is added when end-to-end latency exceeds
   ``latency_p90_mult x p90`` of the trace's root-API group.
3. DPP Selector -> greedy MAP inference (Chen et al. 2018) per group over a
   kernel ``L_ij = q_i q_j * Jaccard(EPS_i, EPS_j)`` with quality ``q`` derived
   from the anomaly score; early-terminate when the marginal gain drops below
   ``dpp_epsilon``.

Selection is *online*: traces are consumed in arrival order and selected per
``batch_size`` buffer window, so the kept set depends on trace order. ``seed ==
1`` is the canonical Start-ascending order (the anchor run); ``seed > 1`` permutes
the arrival order, which is the revision's trace-order-perturbation variance
protocol (Gleaner's selection is deterministic given an order, so perturbing the
order -- not a model RNG -- is the meaningful robustness axis). The per-trace
``GleanerScore`` (= A(T)) is computed over the whole case and stays
order-independent.

Public API mirrors the other baselines:

    run_gleaner_over_trace_frame(spans, logs, budgets, config, seed) -> DataFrame

returning per-trace ``GleanerScore`` (= A(T), for ranker-style evaluation) and
per-budget ``GleanerDecision_<token>`` columns (the DPP-selected sets).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


ERROR_LEVELS = {"ERROR", "FATAL", "CRITICAL", "SEVERE", "ERR"}
WARN_LEVELS = {"WARN", "WARNING"}

# Common per-case log schema the sampler expects (produced by the runner).
LOG_COLUMNS = ["TraceID", "SpanID", "Service", "log_level", "log_temp"]


def budget_token(budget: float) -> str:
    """Match evaluate_trace_sampling.budget_token without importing it."""
    return f"{budget:g}".replace("-", "m").replace(".", "p")


@dataclass
class GleanerConfig:
    # Anomaly score weights (paper Sec. 3.2, Eq. 1 defaults).
    w_err: float = 5.0
    w_lw: float = 1.0
    w_le: float = 2.0
    # Performance-degradation term: added when latency > mult x group p90.
    # The paper specifies the 1.2 x p90 trigger but not the term's magnitude;
    # we set it to w_err and surface it as a config knob (documented choice).
    w_perf: float = 5.0
    latency_p90_mult: float = 1.2
    min_group_reference: int = 20  # fall back to global p90 below this group size
    # DPP selector.
    quality_scale: float = 1.0  # q_i = exp(0.5 * quality_scale * normalized A)
    dpp_epsilon: float = 1e-9
    dpp_candidate_cap: int = 400  # cap per-group similarity matrix for tractability
    # Online buffer (number of traces per arrival-order window). Gleaner is an
    # online tail sampler; selection runs per buffer, which makes the kept set
    # depend on trace arrival order. The variance protocol perturbs that order
    # (see `run_gleaner_over_trace_frame`).
    batch_size: int = 500
    seed: int = 1


def _label_for(service: object, operation: object) -> str:
    return f"{service}::{operation}"


def _root_span_ids(spans_t: pd.DataFrame, span_ids: set) -> List[str]:
    parent = spans_t["ParentID"].astype(str)
    root_mask = parent.str.lower().isin({"root", "nan", "none", ""}) | ~parent.isin(span_ids)
    if root_mask.any():
        return [str(s) for s in spans_t.loc[root_mask, "SpanID"]]
    # No structural root (cycle / detached): treat the earliest span as root.
    return [str(spans_t.sort_values("Start").iloc[0]["SpanID"])]


def root_api_of(spans_t: pd.DataFrame) -> str:
    """Root-span Service|Operation, used to group traces (SRE-aligned API view)."""
    spans_t = spans_t.sort_values("Start")
    span_ids = set(spans_t["SpanID"].astype(str))
    roots = _root_span_ids(spans_t, span_ids)
    by_id = {
        str(r.SpanID): (str(r.Service), str(r.Operation)) for r in spans_t.itertuples(index=False)
    }
    apis = sorted({f"{by_id[s][0]}|{by_id[s][1]}" for s in roots if s in by_id})
    if apis:
        return "||".join(apis)
    first = spans_t.iloc[0]
    return f"{first['Service']}|{first['Operation']}"


@dataclass
class TraceEncoding:
    trace_id: str
    eps: frozenset
    base_score: float
    root_api: str
    latency_ms: float

    @property
    def n_events(self) -> int:
        return len(self.eps)


def encode_trace(
    trace_id: str,
    spans_t: pd.DataFrame,
    logs_t: pd.DataFrame,
    config: GleanerConfig,
) -> TraceEncoding:
    """Encode one trace into its EPS and base anomaly score (no perf term yet).

    `spans_t` is the normalized span frame for a single trace; `logs_t` is the
    common-schema log frame for the same trace (may be empty). Logs are
    attributed to a span by SpanID when available, otherwise to the earliest
    span of the matching Service (a bounded adaptation, since parsed
    Hipster/Train-Ticket logs carry no SpanID column).
    """
    spans_t = spans_t.sort_values("Start")
    span_ids = set(spans_t["SpanID"].astype(str))
    label: Dict[str, str] = {}
    rep_span_of_service: Dict[str, str] = {}
    for row in spans_t.itertuples(index=False):
        sid = str(row.SpanID)
        label[sid] = _label_for(row.Service, row.Operation)
        rep_span_of_service.setdefault(str(row.Service), sid)

    eps: set = set()
    for row in spans_t.itertuples(index=False):
        sid = str(row.SpanID)
        pid = str(row.ParentID)
        eps.add(("SPAN", label[sid]))
        if pid in span_ids and pid != sid:
            eps.add(("EDGE", label[pid], label[sid]))
        else:
            eps.add(("ROOT", label[sid]))

    err_per_span: Dict[str, int] = defaultdict(int)
    warn_per_span: Dict[str, int] = defaultdict(int)
    has_err_span: set = set()
    if logs_t is not None and not logs_t.empty:
        for lg in logs_t.itertuples(index=False):
            lg_span = str(lg.SpanID) if isinstance(lg.SpanID, str) else ""
            sid = lg_span if lg_span in span_ids else rep_span_of_service.get(str(lg.Service))
            if sid is None:
                continue  # log for a service with no span in this trace
            template = str(lg.log_temp) if isinstance(lg.log_temp, str) and lg.log_temp else "<LOG>"
            eps.add(("LOG", label[sid], template))
            level = str(lg.log_level).upper()
            if level in ERROR_LEVELS:
                err_per_span[sid] += 1
                has_err_span.add(sid)
                eps.add(("ERR", label[sid]))
            elif level in WARN_LEVELS:
                warn_per_span[sid] += 1
                eps.add(("WARN", label[sid]))

    base = 0.0
    for sid in span_ids:
        base += config.w_err * (1.0 if sid in has_err_span else 0.0)
        base += config.w_lw * warn_per_span.get(sid, 0)
        base += config.w_le * err_per_span.get(sid, 0)

    start = pd.to_datetime(spans_t["Start"]).min()
    end = pd.to_datetime(spans_t["End"]).max() if "End" in spans_t.columns else start
    latency_ms = float((end - start).total_seconds() * 1000.0) if pd.notna(end) and pd.notna(start) else 0.0

    return TraceEncoding(
        trace_id=str(trace_id),
        eps=frozenset(eps),
        base_score=float(base),
        root_api=root_api_of(spans_t),
        latency_ms=latency_ms,
    )


def _jaccard_matrix(eps_list: Sequence[frozenset]) -> np.ndarray:
    n = len(eps_list)
    S = np.eye(n, dtype=float)
    for i in range(n):
        ei = eps_list[i]
        li = len(ei)
        for j in range(i + 1, n):
            ej = eps_list[j]
            if not ei or not ej:
                sim = 0.0
            else:
                inter = len(ei & ej)
                union = li + len(ej) - inter
                sim = inter / union if union else 0.0
            S[i, j] = S[j, i] = sim
    return S


def greedy_dpp_select(
    eps_list: Sequence[frozenset],
    scores: Sequence[float],
    target: int,
    config: GleanerConfig,
) -> List[int]:
    """Greedy MAP inference for a quality-diversity DPP (Chen et al. 2018).

    Returns indices (into `eps_list`) of the selected, diverse, high-anomaly
    subset of size up to `target`. Quality comes from the anomaly `scores`;
    diversity from Jaccard EPS similarity. If the DPP early-terminates before
    `target`, the remainder is filled with the highest-scoring unselected items
    so the per-budget kept count matches the other baselines.
    """
    n = len(eps_list)
    if target <= 0 or n == 0:
        return []
    if target >= n:
        return list(range(n))

    s = np.asarray(scores, dtype=float)
    smin, smax = float(s.min()), float(s.max())
    snorm = (s - smin) / (smax - smin) if smax > smin else np.zeros_like(s)
    q = np.exp(0.5 * config.quality_scale * snorm)
    S = _jaccard_matrix(eps_list)
    L = (q[:, None] * q[None, :]) * S

    cis = np.zeros((target, n), dtype=float)
    di2s = np.diag(L).astype(float).copy()
    selected: List[int] = []
    j = int(np.argmax(di2s))
    selected.append(j)
    while len(selected) < target:
        k = len(selected) - 1
        dj2 = di2s[j]
        if dj2 <= config.dpp_epsilon:
            break
        di_optimal = math.sqrt(dj2)
        ci_optimal = cis[:k, j]
        eis = (L[j, :] - ci_optimal @ cis[:k, :]) / di_optimal
        cis[k, :] = eis
        di2s = di2s - eis ** 2
        di2s[selected] = -np.inf
        j = int(np.argmax(di2s))
        if di2s[j] < config.dpp_epsilon:
            break
        selected.append(j)

    if len(selected) < target:
        chosen = set(selected)
        order = np.argsort(-s, kind="stable")
        for idx in order:
            if int(idx) not in chosen:
                selected.append(int(idx))
                chosen.add(int(idx))
                if len(selected) >= target:
                    break
    return selected[:target]


def _allocate_even_quota(group_sizes: Dict[str, int], target: int) -> Dict[str, int]:
    """Distribute `target` evenly across root-API groups, capped at group size.

    Even (not proportional) allocation is Gleaner's diversity-promoting
    normal-period policy: each API group gets an equal share so high-traffic
    endpoints cannot dominate the retained set. Leftover from groups smaller
    than their share is redistributed to groups with remaining capacity.
    """
    quota = {g: 0 for g in group_sizes}
    capacity = dict(group_sizes)
    remaining = min(target, sum(capacity.values()))
    while remaining > 0:
        active = [g for g in sorted(group_sizes) if quota[g] < capacity[g]]
        if not active:
            break
        share = max(1, remaining // len(active))
        for g in active:
            give = min(share, capacity[g] - quota[g], remaining)
            quota[g] += give
            remaining -= give
            if remaining == 0:
                break
    return quota


def run_gleaner_over_trace_frame(
    spans: pd.DataFrame,
    logs: pd.DataFrame,
    budgets: Sequence[float],
    config: GleanerConfig | None = None,
    seed: int = 1,
) -> Tuple[pd.DataFrame, dict]:
    """Run Gleaner across all traces in one case.

    `spans` is the normalized span frame (one row per span); `logs` is the
    common-schema per-case log frame (columns in LOG_COLUMNS; may be empty).
    Returns (frame, timing). `frame` has one row per TraceID with columns
    TraceID, GleanerScore, GleanerNumEvents and a GleanerDecision_<token>
    boolean column for each budget. `timing` records sampling-only seconds for
    the Sec. 5 efficiency table.

    `seed` selects the arrival order for the online windowed selection: seed=1
    is the canonical Start-ascending order; seed>1 permutes it (trace-order
    perturbation). It does not affect GleanerScore.
    """
    import time

    config = config or GleanerConfig()
    budgets = list(budgets)
    decision_cols = [f"GleanerDecision_{budget_token(b)}" for b in budgets]
    timing = {"encode_sec": 0.0, "score_sec": 0.0, "dpp_sec": 0.0, "n_eps_events": 0}

    if spans is None or spans.empty:
        empty = pd.DataFrame(columns=["TraceID", "GleanerScore", "GleanerNumEvents"] + decision_cols)
        return empty, timing

    if logs is None or logs.empty:
        logs = pd.DataFrame(columns=LOG_COLUMNS)
    logs_by_trace = {tid: g for tid, g in logs.groupby("TraceID", sort=False)} if not logs.empty else {}

    # ----- Stage 1: encode every trace (streaming order by earliest Start) -----
    order = (
        spans.groupby("TraceID", sort=False)["Start"].min().reset_index().sort_values(
            ["Start", "TraceID"]
        )["TraceID"].astype(str).tolist()
    )
    grouped_spans = {str(tid): g for tid, g in spans.groupby("TraceID", sort=False)}
    t0 = time.perf_counter()
    encodings: List[TraceEncoding] = []
    for tid in order:
        enc = encode_trace(tid, grouped_spans[tid], logs_by_trace.get(tid), config)
        encodings.append(enc)
        timing["n_eps_events"] += enc.n_events
    timing["encode_sec"] = time.perf_counter() - t0

    # ----- Stage 2: anomaly score with group-relative performance term -----
    t0 = time.perf_counter()
    by_group: Dict[str, List[int]] = defaultdict(list)
    for i, enc in enumerate(encodings):
        by_group[enc.root_api].append(i)
    global_p90 = float(np.percentile([e.latency_ms for e in encodings], 90)) if encodings else 0.0
    perf_threshold: Dict[str, float] = {}
    for g, idxs in by_group.items():
        lat = np.array([encodings[i].latency_ms for i in idxs], dtype=float)
        ref = float(np.percentile(lat, 90)) if len(idxs) >= config.min_group_reference else global_p90
        perf_threshold[g] = config.latency_p90_mult * ref
    a_scores = np.empty(len(encodings), dtype=float)
    for i, enc in enumerate(encodings):
        perf = config.w_perf if enc.latency_ms > perf_threshold[enc.root_api] else 0.0
        a_scores[i] = enc.base_score + perf
    timing["score_sec"] = time.perf_counter() - t0

    # ----- Stage 3: online windowed selection (trace-order sensitive) -----
    # Gleaner is an online tail sampler: traces are consumed in arrival order
    # and selected per buffer window. `seed == 1` uses the canonical
    # Start-ascending arrival order (the anchor run); `seed > 1` applies a
    # seeded permutation -- the variance protocol's *trace-order perturbation*.
    # Per window we split that window's budget share evenly across its root-API
    # groups and run greedy DPP within each group, so window composition (hence
    # the kept set) depends on order. Cases smaller than one buffer collapse to
    # a single window and are therefore order-insensitive by construction.
    n_total = len(encodings)
    decisions: Dict[str, np.ndarray] = {col: np.zeros(n_total, dtype=bool) for col in decision_cols}
    arrival = np.arange(n_total)
    if seed is not None and int(seed) != 1:
        arrival = np.random.default_rng(int(seed)).permutation(n_total)
    window_size = max(1, int(config.batch_size))
    windows = [arrival[s : s + window_size].tolist() for s in range(0, n_total, window_size)]

    def _select_pool(pool: List[int], need: int, decided: np.ndarray) -> None:
        if need <= 0 or not pool:
            return
        cand = sorted(pool, key=lambda i: (-a_scores[i], encodings[i].trace_id))
        capped = cand[: config.dpp_candidate_cap] if len(cand) > config.dpp_candidate_cap else cand
        for li in greedy_dpp_select(
            [encodings[i].eps for i in capped],
            [a_scores[i] for i in capped],
            target=min(need, len(capped)),
            config=config,
        ):
            decided[capped[li]] = True
        if int(sum(decided[i] for i in pool)) < need:  # cap < need: top up by score
            for i in cand:
                if not decided[i]:
                    decided[i] = True
                    if int(sum(decided[j] for j in pool)) >= need:
                        break

    t0 = time.perf_counter()
    for budget, col in zip(budgets, decision_cols):
        decided = decisions[col]
        for window in windows:
            target_w = max(1, int(round(len(window) * budget)))
            wg: Dict[str, List[int]] = defaultdict(list)
            for i in window:
                wg[encodings[i].root_api].append(i)
            quota = _allocate_even_quota({g: len(v) for g, v in wg.items()}, target_w)
            for g, idxs in wg.items():
                _select_pool(idxs, quota[g], decided)
    timing["dpp_sec"] = time.perf_counter() - t0

    out = pd.DataFrame(
        {
            "TraceID": [e.trace_id for e in encodings],
            "GleanerScore": a_scores,
            "GleanerNumEvents": [e.n_events for e in encodings],
        }
    )
    for col in decision_cols:
        out[col] = decisions[col]
    return out, timing


# ----- smoke test -----


def _make_common_trace(tid: str, t0: pd.Timestamp) -> pd.DataFrame:
    """A small two-span 'frontend -> catalog' trace."""
    return pd.DataFrame(
        [
            {"TraceID": tid, "SpanID": f"{tid}-a", "ParentID": "root", "Service": "frontend",
             "Operation": "GET", "Start": t0, "End": t0 + pd.Timedelta(milliseconds=20)},
            {"TraceID": tid, "SpanID": f"{tid}-b", "ParentID": f"{tid}-a", "Service": "catalog",
             "Operation": "List", "Start": t0 + pd.Timedelta(milliseconds=2),
             "End": t0 + pd.Timedelta(milliseconds=15)},
        ]
    )


def _make_anomaly_trace(tid: str, t0: pd.Timestamp) -> pd.DataFrame:
    """A structurally distinct 'frontend -> payment -> bank' trace."""
    return pd.DataFrame(
        [
            {"TraceID": tid, "SpanID": f"{tid}-a", "ParentID": "root", "Service": "frontend",
             "Operation": "GET", "Start": t0, "End": t0 + pd.Timedelta(milliseconds=80)},
            {"TraceID": tid, "SpanID": f"{tid}-c", "ParentID": f"{tid}-a", "Service": "payment",
             "Operation": "Charge", "Start": t0 + pd.Timedelta(milliseconds=5),
             "End": t0 + pd.Timedelta(milliseconds=70)},
            {"TraceID": tid, "SpanID": f"{tid}-d", "ParentID": f"{tid}-c", "Service": "bank",
             "Operation": "Auth", "Start": t0 + pd.Timedelta(milliseconds=10),
             "End": t0 + pd.Timedelta(milliseconds=60)},
        ]
    )


def smoke_self_test() -> bool:
    """An anomalous, structurally distinct trace with an error log must be
    selected at a tiny budget over many common duplicates."""
    base = pd.Timestamp("2026-01-01T00:00:00")
    span_frames = []
    log_rows = []
    for i in range(40):
        tid = f"common{i:02d}"
        span_frames.append(_make_common_trace(tid, base + pd.Timedelta(seconds=i)))
        log_rows.append({"TraceID": tid, "SpanID": f"{tid}-b", "Service": "catalog",
                         "log_level": "INFO", "log_temp": "listed <*> products"})
    anomaly_tid = "anom00"
    span_frames.append(_make_anomaly_trace(anomaly_tid, base + pd.Timedelta(seconds=41)))
    log_rows.append({"TraceID": anomaly_tid, "SpanID": f"{anomaly_tid}-d", "Service": "bank",
                     "log_level": "ERROR", "log_temp": "auth declined <*>"})

    spans = pd.concat(span_frames, ignore_index=True)
    logs = pd.DataFrame(log_rows, columns=LOG_COLUMNS)
    out, _ = run_gleaner_over_trace_frame(spans, logs, budgets=[0.05], config=GleanerConfig(seed=0))

    anom_row = out.loc[out["TraceID"] == anomaly_tid].iloc[0]
    top_score = anom_row["GleanerScore"] == out["GleanerScore"].max()
    selected = bool(anom_row["GleanerDecision_0p05"])
    return bool(top_score and selected)


if __name__ == "__main__":
    ok = smoke_self_test()
    print("smoke_self_test:", "PASS" if ok else "FAIL")
