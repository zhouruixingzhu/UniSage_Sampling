"""Self-contained Sieve trace sampler adapter for UniSage v0.2 baselines.

Sieve is the ICWS 2021 attention-based trace sampler by Huang, Chen, Yu, Chen,
and Zheng. The authors' Python reference implementation is vendored under
`baselines/sieve_upstream/` (see that package's `__init__.py` for the patch
list). This module exposes a thin adapter that:

  - converts a normalized UniSage span frame into the vendored
    `interface.SpanBase` / `interface.TraceBase` shape;
  - drives Sieve in the upstream's streaming order (per-case earliest span
    Start) and records the per-trace RRCF attention score plus Sieve's own
    sampling probability;
  - turns each target budget α into a deterministic top-K-by-score decision
    column so the v0.2 evaluator can compare Sieve to Sifter, TracePicker,
    UniSage, and uniform under the same metric pipeline.

Defaults follow Huang et al. ICWS 2021 §VI:
    tree_num=50, tree_size=128, k=50, threshold=0.3, var_limit=0.1,
    compact every 128 processed traces.

Defensive edge-case handling in this adapter (not in upstream Sieve):
  - **NaN-start spans**: spans with NaT/NaN Start are dropped at sort time
    (they have no meaningful arrival order); if this empties a trace, it
    produces `n_ranked=0` instead of crashing upstream.
  - **Zero-duration signals**: kept as-is (Sieve treats 0 as a valid value).
  - **Empty trace (no spans after normalization)**: return NaN scores and
    `n_ranked=0`; the upstream `score()` call is never reached.
"""

from __future__ import annotations

import logging
import math
import traceback as _traceback
from dataclasses import dataclass

import numpy as np
import pandas as pd
from typing import List

logger = logging.getLogger(__name__)

from baselines.sieve_upstream import sieve as _upstream_sieve
from baselines.sieve_upstream.interface import SpanBase, TraceBase


@dataclass
class SieveConfig:
    tree_num: int = 50
    tree_size: int = 128
    k: int = 50
    threshold: float = 0.3
    var_limit: float = 0.1
    compact_interval: int = 128
    seed: int = 1


class _AdaptedSpan(SpanBase):
    """Minimal SpanBase implementation backed by one row of a UniSage frame."""

    __slots__ = ("_sid", "_pid", "_label", "_dur")

    _ROOT_PID_SENTINELS = {"none", "nan", "", "root"}

    def __init__(self, sid: str, pid: str, label: str, duration_ms: float) -> None:
        self._sid = str(sid)
        pid_str = str(pid)
        self._pid = None if pid_str.lower() in self._ROOT_PID_SENTINELS else pid_str
        self._label = str(label)
        try:
            d = int(round(float(duration_ms)))
        except (TypeError, ValueError):
            d = 0
        self._dur = max(0, d)

    def getSpanId(self):
        return self._sid

    def getParentId(self):
        return self._pid

    def getSpanLabel(self):
        return self._label

    def getElapsedTime(self):
        return self._dur


class _AdaptedTrace(TraceBase):
    """Minimal TraceBase implementation backed by a list of _AdaptedSpan."""

    __slots__ = ("_tid", "_spans")

    def __init__(self, trace_id: str, spans: List[_AdaptedSpan]) -> None:
        self._tid = str(trace_id)
        self._spans = spans

    def getTraceId(self):
        return self._tid

    def getSpanNum(self):
        return len(self._spans)

    def getSpans(self):
        return self._spans


def trace_order_index(spans: pd.DataFrame) -> pd.DataFrame:
    """Return TraceID order sorted by earliest span Start (mirrors the
    streaming arrival order expected by Sieve).

    NaN/NaT Start spans are dropped silently — they have no meaningful
    arrival order. If this empties all of a trace's spans the trace
    will not appear in the returned index and will be handled as
    n_ranked=0 downstream.
    """
    if spans.empty:
        return pd.DataFrame(columns=["TraceID", "Start"])
    valid = spans.dropna(subset=["Start"])
    if len(valid) < len(spans):
        logger.warning("trace_order_index: dropped %d rows with NaN Start", len(spans) - len(valid))
    if valid.empty:
        return pd.DataFrame(columns=["TraceID", "Start"])
    return (
        valid.groupby("TraceID", sort=False)["Start"]
        .min()
        .reset_index()
        .sort_values(["Start", "TraceID"])
        .reset_index(drop=True)
    )


def _build_trace(trace_id: str, group: pd.DataFrame) -> _AdaptedTrace:
    spans = [
        _AdaptedSpan(
            sid=row.SpanID,
            pid=row.ParentID,
            label=f"{row.Service}:{row.Operation}",
            duration_ms=row.SpanDurationMs,
        )
        for row in group.itertuples(index=False)
    ]
    return _AdaptedTrace(trace_id, spans)


def _build_sieve(cfg: SieveConfig) -> _upstream_sieve.Sieve:
    """Instantiate a vendored Sieve with per-tree distinct, reproducible RNGs.

    The upstream constructor passes the same `seed` to every tree, so when an
    `int` seed is provided each tree's `np.random.RandomState` starts at the
    same state and produces identical cut sequences for the same data —
    collapsing the forest to one tree. We build with `seed=None` (each tree
    initialised against `np.random`) and then overwrite each tree's `rng`
    with `RandomState(cfg.seed * 1000 + tree_idx)` so the forest is both
    reproducible and genuinely diverse.
    """
    sieve = _upstream_sieve.Sieve(
        tree_num=cfg.tree_num,
        tree_size=cfg.tree_size,
        k=cfg.k,
        threshold=cfg.threshold,
        var_limit=cfg.var_limit,
        seed=None,
    )
    base = int(cfg.seed)
    for ti, tree in enumerate(sieve.rrcf):
        tree.rng = np.random.RandomState(base * 1000 + ti)
    return sieve


def _safe_probability(sieve: _upstream_sieve.Sieve, score: float) -> float:
    """Mirror upstream `getProbability(score=...)` with an `exp` overflow guard.

    Upstream uses `math.exp(avg(kScores) - kScores[-1])`; when a trace's score
    is far below the running mean, this exponent grows large and the bare
    `math.exp` overflows. We catch that, treat it as a vanishing sampling
    probability, and return 0.0. Anomalous traces (exponent very negative)
    are unaffected — those collapse the sigmoid toward 1.0 naturally.
    """
    try:
        return float(sieve.getProbability(trace=None, score=score))
    except OverflowError:
        # Keep upstream's internal kScores in sync: append the score and
        # apply the same windowing it would have done before the overflow.
        sieve.kScores.append(score)
        if len(sieve.kScores) > sieve.k:
            sieve.kScores = sieve.kScores[1:]
        return 0.0


def run_sieve_over_trace_frame(
    spans: pd.DataFrame,
    config: SieveConfig | None = None,
) -> pd.DataFrame:
    """Run vendored Sieve across all traces in `spans` and return per-trace
    rows with `TraceID, SieveScore, SieveProbability, SieveNumSpans`.

    Compaction runs every `config.compact_interval` *successfully scored*
    traces, matching the upstream README usage pattern.
    """
    cfg = config or SieveConfig()
    if spans.empty:
        return pd.DataFrame(
            columns=["TraceID", "SieveScore", "SieveProbability", "SieveNumSpans"]
        )

    order = trace_order_index(spans)
    groups = {tid: g for tid, g in spans.groupby("TraceID", sort=False)}
    sieve = _build_sieve(cfg)

    rows = []
    scored = 0
    for trace_id in order["TraceID"]:
        group = groups.get(trace_id)
        if group is None or group.empty:
            rows.append({
                "TraceID": str(trace_id),
                "SieveScore": float("nan"),
                "SieveProbability": float("nan"),
                "SieveNumSpans": 0,
            })
            continue
        trace = _build_trace(str(trace_id), group)
        # `score(trace)` encodes the trace, mutates RRCF state and kScores'
        # *input* indirectly via internal counters, then returns the mean
        # codisp. We then call getProbability with the explicit score so the
        # upstream does not re-encode the trace.
        # Wrap the upstream call in a try/except so a single trace encoding
        # failure (e.g. an upstream RRCF tree edge case) does not crash the
        # entire case. The failing trace gets NaN scores but the Sieve state
        # (kScores, tree) is NOT updated — compaction still runs on the same
        # interval to avoid diverging the `scored`↔`compact()` invariant more
        # than necessary.
        try:
            score = float(sieve.score(trace))
            prob = _safe_probability(sieve, score)
            rows.append({
                "TraceID": str(trace_id),
                "SieveScore": score,
                "SieveProbability": prob,
                "SieveNumSpans": int(len(group)),
            })
            scored += 1
        except Exception:
            logger.warning("Sieve score() failed for trace %s\n%s", trace_id, _traceback.format_exc())
            rows.append({
                "TraceID": str(trace_id),
                "SieveScore": float("nan"),
                "SieveProbability": float("nan"),
                "SieveNumSpans": int(len(group)),
            })
            # Do NOT increment scored — this trace did not update the RRCF
            # state and should not count toward compaction.
        if scored % cfg.compact_interval == 0:
            # Upstream `compact()` indexes into `sieve.dim2q`, which stays
            # `None` until the first non-duplicate insert creates a Branch.
            # On highly repetitive synthetic data (or very small cases) this
            # never happens, so calling compact would crash with
            # `TypeError: 'NoneType' object is not subscriptable`. Skip
            # compaction when the RRCF state has not yet produced a branch.
            if sieve.dim2q is not None:
                sieve.compact()

    return pd.DataFrame(rows)


def top_k_decision_by_score(
    frame: pd.DataFrame,
    target_count: int,
    score_column: str = "SieveScore",
) -> pd.Series:
    """Pick the top `target_count` TraceIDs by descending `score_column`."""
    if target_count <= 0 or frame.empty:
        return pd.Series(False, index=frame.index)
    ranked = frame.sort_values([score_column, "TraceID"], ascending=[False, True], na_position="last")
    keep = ranked.head(min(target_count, len(ranked))).index
    decision = pd.Series(False, index=frame.index)
    decision.loc[keep] = True
    return decision


def smoke_self_test() -> bool:
    """Synthetic test: a small set of structurally rare traces must rank
    above common ones after Sieve sees enough common traces to populate
    its RRCF window."""
    rows: List[dict] = []
    sid_counter = 0

    def add_trace(tid: str, services: List[str], durations: List[int], t_ms: int) -> None:
        nonlocal sid_counter
        parent = "ROOT"
        for s, d in zip(services, durations):
            sid = f"s{sid_counter}"
            sid_counter += 1
            rows.append({
                "TraceID": tid,
                "SpanID": sid,
                "ParentID": parent,
                "Service": s,
                "Operation": "op",
                "SpanDurationMs": d,
                "Start": pd.Timestamp("2026-01-01") + pd.Timedelta(milliseconds=t_ms),
            })
            parent = sid

    rng = np.random.default_rng(0)
    common_variants = [
        (["A", "B", "C"], [10, 20, 30]),
        (["A", "B", "D"], [10, 20, 25]),
        (["A", "E", "C"], [10, 18, 32]),
        (["A", "B", "C", "F"], [10, 20, 30, 5]),
    ]
    t = 0
    for i in range(80):
        services, durations = common_variants[i % len(common_variants)]
        jitter = [max(1, int(d + rng.normal(0, 1.5))) for d in durations]
        add_trace(f"common_{i:03d}", list(services), jitter, t)
        t += 100
    for i in range(5):
        add_trace(f"outlier_{i:03d}", ["A", "B", "Q_rare", "D_rare"], [10, 20, 999, 999], t)
        t += 100
    for i in range(80, 100):
        services, durations = common_variants[i % len(common_variants)]
        jitter = [max(1, int(d + rng.normal(0, 1.5))) for d in durations]
        add_trace(f"common_{i:03d}", list(services), jitter, t)
        t += 100

    spans = pd.DataFrame(rows)
    out = run_sieve_over_trace_frame(
        spans,
        SieveConfig(tree_num=10, tree_size=32, k=20, threshold=0.3, var_limit=0.1,
                    compact_interval=32, seed=0),
    )
    if out.empty or out["SieveScore"].notna().sum() == 0:
        return False
    out["is_outlier"] = out["TraceID"].str.startswith("outlier_")
    outlier_mean = out.loc[out["is_outlier"], "SieveScore"].mean()
    common_mean = out.loc[~out["is_outlier"], "SieveScore"].mean()
    return float(outlier_mean) > float(common_mean)


if __name__ == "__main__":
    ok = smoke_self_test()
    print("sieve smoke_self_test:", "PASS" if ok else "FAIL")
