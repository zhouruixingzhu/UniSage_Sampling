"""Post-hoc v0.2 trace-composition explorer for UniSage.

This script evaluates lightweight composition candidates over existing
version-stamped UniSage trace-sampling outputs. It does not rerun AD/RCL and
does not read baseline outputs or labels while constructing rankings. Labels
are used only by the frozen v0.2 evaluator metrics.

Run from ``UniSage/``. Example:

    python scripts/posthoc_trace_composition.py \
      --sample-index \
        results/v02_sampling_runs/unisage_trace_hipster_train_ticket_primary_attr-target_edge-raw_prop-0_fp-71658166498d/sample_index.csv \
        results/v02_sampling_runs/unisage_trace_aiops21_primary_attr-target_edge-raw_prop-0_fp-71658166498d/sample_index.csv \
        results/v02_sampling_runs/unisage_trace_aiops21_excluded_attr-target_edge-raw_prop-0_fp-730df5aaec67/sample_index.csv \
      --output results/posthoc_trace_composition_seed1
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

import sampling_labels as lp
from evaluate_trace_sampling import (
    BUDGETS,
    DESCRIPTIVE_LABELS,
    PROBLEM_LABELS,
    build_metric_row,
    label_stats,
    stable_uniform_score,
)
from build_trace_manifest import sha256_file


HERE = Path(__file__).resolve().parent.parent
REQUIRED_SAMPLE_COLUMNS = ["CombinedSamplingProb", "EdgeCaseSamplingProb", "AnomalySamplingProb"]
REFERENCE_STRATEGIES = {"current_product", "analysis_first", "edge_reference", "uniform_reference"}
DEFAULT_WEIGHTED_ANALYSIS = 0.7
DEFAULT_WEIGHTED_EDGE = 0.3


def build_fast_metric_row(base: pd.DataFrame, kept: pd.DataFrame, ranking: str, budget: float, n_ranked: int) -> dict:
    all_patterns = int(base["pattern_hash"].nunique())
    primary_scope = base["trace_label_scope"].iloc[0] == lp.PRIMARY_SCOPE
    k = len(kept)
    row = {
        "ranking": ranking,
        "budget": budget,
        "n_total": len(base),
        "n_ranked": n_ranked,
        "n_kept": k,
        "pattern_support": all_patterns,
        "pattern_kept": int(kept["pattern_hash"].nunique()) if k else 0,
        "pattern_coverage": (kept["pattern_hash"].nunique() / all_patterns) if all_patterns and k else np.nan,
        "pattern_entropy": np.nan,
        "pattern_jsd": np.nan,
        "pattern_benefit_cost": (kept["pattern_hash"].nunique() / k) if k else np.nan,
    }
    for label in DESCRIPTIVE_LABELS:
        row.update(label_stats(base, kept, label, enabled=True))
    for label in PROBLEM_LABELS:
        row.update(label_stats(base, kept, label, enabled=primary_scope))
    return row


def git_head(path: Path = HERE) -> str:
    try:
        return subprocess.check_output(["git", "-C", path.as_posix(), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def file_fingerprint(paths: Iterable[Path]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for path in paths:
        if path.is_file():
            digest.update(path.as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def resolve_path(path: object, root: Path = HERE) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else root / p


def budget_token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def rank_percentile(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series(0.0, index=values.index)
    ranks = numeric.rank(method="average", pct=True, na_option="bottom")
    return ranks.fillna(0.0).astype(float)


def case_trace_stats(merged: pd.DataFrame) -> dict:
    pa = pd.to_numeric(merged.get("AnomalySamplingProb"), errors="coerce").dropna()
    pe = pd.to_numeric(merged.get("EdgeCaseSamplingProb"), errors="coerce").dropna()
    prod = pd.to_numeric(merged.get("CombinedSamplingProb"), errors="coerce").dropna()

    def safe_iqr(series: pd.Series) -> float:
        if series.empty:
            return float("nan")
        return float(series.quantile(0.75) - series.quantile(0.25))

    def safe_spread(series: pd.Series) -> float:
        if series.empty:
            return float("nan")
        return float(series.max() - series.min())

    return {
        "pa_mean": float(pa.mean()) if not pa.empty else float("nan"),
        "pa_std": float(pa.std(ddof=0)) if not pa.empty else float("nan"),
        "pa_iqr": safe_iqr(pa),
        "pa_spread": safe_spread(pa),
        "pa_unique": int(pa.nunique()) if not pa.empty else 0,
        "pe_std": float(pe.std(ddof=0)) if not pe.empty else float("nan"),
        "product_std": float(prod.std(ddof=0)) if not prod.empty else float("nan"),
    }


def runtime_novelty_key(frame: pd.DataFrame) -> pd.Series:
    if "TracePatternKey" in frame.columns:
        key = frame["TracePatternKey"].fillna("").astype(str).str.strip()
    else:
        key = pd.Series("", index=frame.index)
    if "CurrentPath" in frame.columns:
        path = frame["CurrentPath"].fillna("").astype(str).str.strip()
        key = key.where(key != "", path)
    if "PodName" in frame.columns:
        pod = frame["PodName"].fillna("").astype(str).str.strip()
        key = key.where(key != "", pod)
    return key.where(key != "", frame["TraceID"].astype(str))


def add_derived_scores(merged: pd.DataFrame, lambdas: list[float], novelty_source: str) -> pd.DataFrame:
    out = merged.copy()
    out["_score_current_product"] = pd.to_numeric(out["CombinedSamplingProb"], errors="coerce")
    out["_score_analysis_first"] = pd.to_numeric(out["AnomalySamplingProb"], errors="coerce")
    out["_score_edge_reference"] = pd.to_numeric(out["EdgeCaseSamplingProb"], errors="coerce")
    analysis_score = out["_score_analysis_first"].fillna(0.0)
    edge_score = out["_score_edge_reference"].fillna(0.0)
    out["_score_max"] = pd.concat([analysis_score, edge_score], axis=1).max(axis=1)
    out["_score_weighted"] = (
        DEFAULT_WEIGHTED_ANALYSIS * analysis_score
        + DEFAULT_WEIGHTED_EDGE * edge_score
    ) / (DEFAULT_WEIGHTED_ANALYSIS + DEFAULT_WEIGHTED_EDGE)
    out["_score_uniform_reference"] = out["TraceID"].map(stable_uniform_score)

    out["_rank_product"] = rank_percentile(out["_score_current_product"])
    out["_rank_analysis"] = rank_percentile(out["_score_analysis_first"])
    out["_rank_edge"] = rank_percentile(out["_score_edge_reference"])
    out["_score_analysis_product"] = 0.5 * out["_rank_analysis"] + 0.5 * out["_rank_product"]

    if novelty_source == "runtime":
        out["_novelty_key"] = runtime_novelty_key(out)
    elif novelty_source == "manifest":
        out["_novelty_key"] = out["pattern_hash"].fillna("__MISSING_PATTERN__").astype(str)
    else:
        raise ValueError(f"Unknown novelty source: {novelty_source}")
    if "pattern_hash" in out.columns:
        out["_guard_pattern_key"] = out["pattern_hash"].fillna(out["_novelty_key"]).astype(str)
    else:
        out["_guard_pattern_key"] = out["_novelty_key"]
    pattern_counts = out["_novelty_key"].value_counts(dropna=False)
    out["_pattern_count"] = out["_novelty_key"].map(pattern_counts).fillna(len(out)).astype(float)
    out["_score_pattern_novel"] = rank_percentile(-out["_pattern_count"])
    out["_novelty_path_depth"] = out["_novelty_key"].astype(str).str.count(r"\s*->\s*").add(1).astype(float)
    for lam in lambdas:
        token = budget_token(lam)
        out[f"_score_rank_linear_l{token}"] = lam * out["_rank_analysis"] + (1.0 - lam) * out["_rank_edge"]
    return out


def top_by_score(frame: pd.DataFrame, score_column: str, k: int) -> pd.DataFrame:
    if k <= 0:
        return frame.iloc[0:0].copy()
    ranked = frame.dropna(subset=[score_column]).copy()
    ranked["_sort_score"] = pd.to_numeric(ranked[score_column], errors="coerce")
    ranked = ranked.sort_values(["_sort_score", "TraceID"], ascending=[False, True], kind="mergesort")
    return ranked.head(min(k, len(ranked))).drop(columns=["_sort_score"])


def take_new_by_score(frame: pd.DataFrame, score_column: str, k: int, used: set[str]) -> pd.DataFrame:
    if k <= 0:
        return frame.iloc[0:0].copy()
    available = frame[~frame["TraceID"].astype(str).isin(used)].copy()
    return top_by_score(available, score_column, k)


def take_pattern_novel(frame: pd.DataFrame, k: int, used: set[str]) -> pd.DataFrame:
    if k <= 0:
        return frame.iloc[0:0].copy()
    available = frame[~frame["TraceID"].astype(str).isin(used)].copy()
    if available.empty:
        return available
    available = available.sort_values(
        ["_pattern_count", "_rank_edge", "_rank_analysis", "TraceID"],
        ascending=[True, False, False, True],
        kind="mergesort",
    )
    selected = []
    seen_patterns = set()
    for idx, row in available.iterrows():
        pattern = row["_novelty_key"]
        if pattern in seen_patterns:
            continue
        selected.append(idx)
        seen_patterns.add(pattern)
        if len(selected) >= k:
            break
    if len(selected) < k:
        for idx in available.index:
            if idx in selected:
                continue
            selected.append(idx)
            if len(selected) >= k:
                break
    return available.loc[selected]


def reserved_budget_selection(
    frame: pd.DataFrame,
    k: int,
    analysis_pct: float = 70.0,
    edge_pct: float = 20.0,
    pattern_pct: float = 10.0,
    primary_score: str = "_score_analysis_product",
) -> pd.DataFrame:
    """Reserve budget across primary, edge, and pattern-novel buckets."""
    if k <= 0:
        return frame.iloc[0:0].copy()
    primary_n = min(k, max(1, int(round(k * analysis_pct / 100.0))))
    edge_n = min(k - primary_n, int(round(k * edge_pct / 100.0)))
    pattern_n = max(0, k - primary_n - edge_n)

    selected_parts = []
    used: set[str] = set()
    primary = take_new_by_score(frame, primary_score, primary_n, used)
    selected_parts.append(primary)
    used.update(primary["TraceID"].astype(str))

    edge = take_new_by_score(frame, "_rank_edge", edge_n, used)
    selected_parts.append(edge)
    used.update(edge["TraceID"].astype(str))

    pattern = take_pattern_novel(frame, pattern_n, used)
    selected_parts.append(pattern)
    used.update(pattern["TraceID"].astype(str))

    if len(used) < k:
        fill = take_new_by_score(frame, primary_score, k - len(used), used)
        selected_parts.append(fill)
    return pd.concat(selected_parts, ignore_index=False).drop_duplicates("TraceID").head(k)


def product_guard_pool(frame: pd.DataFrame, k: int, pool_multiplier: float) -> pd.DataFrame:
    if k <= 0:
        return frame.iloc[0:0].copy()
    pool_k = min(len(frame), max(k, int(math.ceil(k * pool_multiplier))))
    return top_by_score(frame, "_score_current_product", pool_k)


def guarded_reserved_budget_selection(
    frame: pd.DataFrame,
    k: int,
    ratio: tuple[float, float, float],
    pool_multiplier: float,
    primary_score: str = "_score_current_product",
) -> pd.DataFrame:
    """Reserve inside a product-anchored pool, then fill by product if needed."""
    if k <= 0:
        return frame.iloc[0:0].copy()
    pool = product_guard_pool(frame, k, pool_multiplier)
    guarded = reserved_budget_selection(
        pool,
        k,
        analysis_pct=ratio[0],
        edge_pct=ratio[1],
        pattern_pct=ratio[2],
        primary_score=primary_score,
    )
    used = set(guarded["TraceID"].astype(str))
    selected_parts = [guarded]
    if len(used) < k:
        fill = take_new_by_score(frame, "_score_current_product", k - len(used), used)
        selected_parts.append(fill)
    return pd.concat(selected_parts, ignore_index=False).drop_duplicates("TraceID").head(k)


def parse_ratio(value: str) -> tuple[float, float, float]:
    parts = re.split(r"[,/:]", value.strip())
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"Ratio must have three parts, got {value!r}")
    try:
        ratio = tuple(float(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid ratio {value!r}") from exc
    total = sum(ratio)
    if total <= 0:
        raise argparse.ArgumentTypeError(f"Ratio sum must be positive, got {value!r}")
    return tuple(100.0 * p / total for p in ratio)


def ratio_token(ratio: tuple[float, float, float]) -> str:
    def fmt(v: float) -> str:
        return f"{v:g}".replace(".", "p")

    return f"a{fmt(ratio[0])}e{fmt(ratio[1])}p{fmt(ratio[2])}"


def multiplier_token(value: float) -> str:
    return f"m{value:g}".replace(".", "p")


def primary_score_column(primary: str) -> str:
    if primary == "analysis_product":
        return "_score_analysis_product"
    if primary == "product":
        return "_score_current_product"
    raise ValueError(f"Unknown primary score: {primary}")


def product_context(frame: pd.DataFrame, k: int) -> dict:
    if k <= 0 or frame.empty:
        return {
            "all_patterns": 0,
            "product_novelty_cov": float("nan"),
            "edge_overlap": float("nan"),
            "pool1p25_novelty_cov": float("nan"),
            "pool1p5_novelty_cov": float("nan"),
            "pool2_novelty_cov": float("nan"),
            "pool3_novelty_cov": float("nan"),
            "path_like_fraction": float("nan"),
            "median_path_depth": float("nan"),
        }
    product = top_by_score(frame, "_score_current_product", k)
    edge = top_by_score(frame, "_score_edge_reference", k)
    guard_key = "_guard_pattern_key" if "_guard_pattern_key" in frame.columns else "_novelty_key"
    all_patterns = frame[guard_key].nunique(dropna=False)
    product_patterns = product[guard_key].nunique(dropna=False) if not product.empty else 0
    product_ids = set(product["TraceID"].astype(str))
    edge_ids = set(edge["TraceID"].astype(str))
    overlap = len(product_ids & edge_ids) / max(1, min(len(product_ids), len(edge_ids)))
    path_like = frame["_novelty_key"].astype(str).str.contains(r"\s*->\s*", regex=True).mean()
    median_depth = frame["_novelty_path_depth"].median() if "_novelty_path_depth" in frame.columns else float("nan")

    def pool_cov(multiplier: float) -> float:
        pool = product_guard_pool(frame, k, multiplier)
        patterns = pool[guard_key].nunique(dropna=False) if not pool.empty else 0
        return patterns / all_patterns if all_patterns else float("nan")

    return {
        "all_patterns": int(all_patterns),
        "product_novelty_cov": product_patterns / all_patterns if all_patterns else float("nan"),
        "edge_overlap": overlap,
        "pool1p25_novelty_cov": pool_cov(1.25),
        "pool1p5_novelty_cov": pool_cov(1.50),
        "pool2_novelty_cov": pool_cov(2.00),
        "pool3_novelty_cov": pool_cov(3.00),
        "path_like_fraction": float(path_like),
        "median_path_depth": float(median_depth),
    }


def dynamic_decision_suffix(ctx: dict) -> str:
    return (
        f"patterns{ctx.get('all_patterns', 0)}"
        f"_cov{ctx.get('product_novelty_cov', float('nan')):.3f}"
        f"_overlap{ctx.get('edge_overlap', float('nan')):.3f}"
        f"_pathlike{ctx.get('path_like_fraction', float('nan')):.3f}"
    )


def dynamic_ratio_selection(
    frame: pd.DataFrame,
    k: int,
    profile: str,
    reserve_min_slots: int,
) -> tuple[pd.DataFrame, str]:
    if k < reserve_min_slots:
        return top_by_score(frame, "_score_current_product", k), "product_small_budget"
    ctx = product_context(frame, k)
    cov = ctx["product_novelty_cov"]
    overlap = ctx["edge_overlap"]

    if profile == "dynamic_product_cov_guard":
        if not math.isnan(cov) and cov < 0.35:
            ratio = (70.0, 20.0, 10.0)
        elif not math.isnan(cov) and cov < 0.60:
            ratio = (85.0, 10.0, 5.0)
        else:
            ratio = (100.0, 0.0, 0.0)
    elif profile == "dynamic_product_overlap_guard":
        if not math.isnan(overlap) and overlap < 0.25:
            ratio = (85.0, 10.0, 5.0)
        else:
            ratio = (70.0, 20.0, 10.0)
    elif profile == "dynamic_product_conservative":
        if (not math.isnan(cov) and cov < 0.30 and not math.isnan(overlap) and overlap >= 0.25):
            ratio = (70.0, 20.0, 10.0)
        elif not math.isnan(cov) and cov < 0.60:
            ratio = (85.0, 10.0, 5.0)
        else:
            ratio = (100.0, 0.0, 0.0)
    else:
        raise ValueError(f"Unknown dynamic profile: {profile}")

    kept = reserved_budget_selection(
        frame,
        k,
        analysis_pct=ratio[0],
        edge_pct=ratio[1],
        pattern_pct=ratio[2],
        primary_score="_score_current_product",
    )
    return kept, f"{profile}_{ratio_token(ratio)}_cov{cov:.3f}_overlap{overlap:.3f}"


def dynamic_guarded_ratio_selection(
    frame: pd.DataFrame,
    k: int,
    profile: str,
    reserve_min_slots: int,
    budget: float,
    dynamic_min_budget: float,
) -> tuple[pd.DataFrame, str]:
    if profile.startswith("dynamic_product_pattern_") and budget < dynamic_min_budget:
        return top_by_score(frame, "_score_current_product", k), f"product_budget_floor{budget_token(dynamic_min_budget)}"
    if k < reserve_min_slots:
        return top_by_score(frame, "_score_current_product", k), "product_small_budget"

    ctx = product_context(frame, k)
    patterns = ctx["all_patterns"]
    cov = ctx["product_novelty_cov"]
    overlap = ctx["edge_overlap"]
    gain125 = ctx["pool1p25_novelty_cov"] - cov
    gain15 = ctx["pool1p5_novelty_cov"] - cov
    gain2 = ctx["pool2_novelty_cov"] - cov
    gain3 = ctx["pool3_novelty_cov"] - cov
    path_like = ctx["path_like_fraction"]
    ratio = (100.0, 0.0, 0.0)
    pool_multiplier = 1.0

    if profile == "dynamic_product_support_guard":
        if patterns >= 50:
            if cov < 0.25 and gain125 >= 0.03 and overlap >= 0.10:
                ratio = (95.0, 5.0, 0.0)
                pool_multiplier = 1.25
            elif cov < 0.40 and gain125 >= 0.05 and overlap >= 0.20:
                ratio = (95.0, 5.0, 0.0)
                pool_multiplier = 1.25
        elif patterns <= 20:
            if cov < 0.65 and gain2 >= 0.10:
                ratio = (85.0, 10.0, 5.0)
                pool_multiplier = 2.00
            elif cov < 0.90 and gain15 >= 0.05:
                ratio = (90.0, 5.0, 5.0)
                pool_multiplier = 1.50
        else:
            if cov < 0.35 and gain15 >= 0.05:
                ratio = (90.0, 10.0, 0.0)
                pool_multiplier = 1.50
    elif profile == "dynamic_product_support_balanced":
        if patterns >= 50:
            if cov < 0.30 and gain15 >= 0.05 and overlap >= 0.10:
                ratio = (90.0, 10.0, 0.0)
                pool_multiplier = 1.50
            elif cov < 0.45 and gain125 >= 0.03:
                ratio = (95.0, 5.0, 0.0)
                pool_multiplier = 1.25
        elif patterns <= 20:
            if cov < 0.75 and gain3 >= 0.10:
                ratio = (80.0, 15.0, 5.0)
                pool_multiplier = 3.00
            elif cov < 0.95 and gain2 >= 0.05:
                ratio = (85.0, 10.0, 5.0)
                pool_multiplier = 2.00
        else:
            if cov < 0.40 and gain2 >= 0.05:
                ratio = (85.0, 10.0, 5.0)
                pool_multiplier = 2.00
    elif profile == "dynamic_product_support_conservative":
        if patterns >= 50:
            if cov < 0.20 and gain125 >= 0.05 and overlap >= 0.20:
                ratio = (95.0, 5.0, 0.0)
                pool_multiplier = 1.25
        elif patterns <= 20:
            if cov < 0.80 and gain15 >= 0.05:
                ratio = (90.0, 10.0, 0.0)
                pool_multiplier = 1.50
        else:
            if cov < 0.25 and gain125 >= 0.05:
                ratio = (95.0, 5.0, 0.0)
                pool_multiplier = 1.25
    elif profile in {"dynamic_product_pattern_guard", "dynamic_product_pattern_guard_conservative"}:
        if patterns <= 20:
            if cov < 0.95 and not math.isnan(path_like) and path_like >= 0.50:
                ratio = (70.0, 20.0, 10.0) if profile == "dynamic_product_pattern_guard" else (85.0, 10.0, 5.0)
                kept = reserved_budget_selection(
                    frame,
                    k,
                    analysis_pct=ratio[0],
                    edge_pct=ratio[1],
                    pattern_pct=ratio[2],
                    primary_score="_score_current_product",
                )
                decision = f"{profile}_open_{ratio_token(ratio)}_{dynamic_decision_suffix(ctx)}"
                return kept, decision
    else:
        raise ValueError(f"Unknown guarded dynamic profile: {profile}")

    suffix = dynamic_decision_suffix(ctx)
    if ratio[1] == 0.0 and ratio[2] == 0.0:
        return top_by_score(frame, "_score_current_product", k), f"{profile}_product_{suffix}"

    kept = guarded_reserved_budget_selection(
        frame,
        k,
        ratio=ratio,
        pool_multiplier=pool_multiplier,
        primary_score="_score_current_product",
    )
    decision = (
        f"{profile}_{ratio_token(ratio)}_{multiplier_token(pool_multiplier)}"
        f"_{suffix}"
    )
    return kept, decision


def confidence_gate_selection(
    frame: pd.DataFrame,
    k: int,
    stats: dict,
    flat_std_threshold: float,
    flat_iqr_threshold: float,
) -> tuple[pd.DataFrame, str]:
    """Analysis-heavy when p_a is discriminative; edge/diversity-heavy otherwise."""
    pa_std = stats.get("pa_std", float("nan"))
    pa_iqr = stats.get("pa_iqr", float("nan"))
    is_flat = (
        (not math.isnan(pa_std) and pa_std < flat_std_threshold)
        or (not math.isnan(pa_iqr) and pa_iqr < flat_iqr_threshold)
    )
    if is_flat:
        score = 0.60 * frame["_rank_edge"] + 0.40 * frame["_score_pattern_novel"]
        work = frame.assign(_score_confidence_gate=score)
        return top_by_score(work, "_score_confidence_gate", k), "edge_diversity_fallback"

    score = 0.85 * frame["_rank_analysis"] + 0.15 * frame["_rank_edge"]
    work = frame.assign(_score_confidence_gate=score)
    return top_by_score(work, "_score_confidence_gate", k), "analysis_heavy"


def strategy_specs(
    lambdas: list[float],
    ratio_grid: list[tuple[float, float, float]],
    primary_scores: list[str],
    dynamic_profiles: list[str],
    guarded_pool_multipliers: list[float],
) -> list[dict]:
    specs = [
        {"name": "current_product", "kind": "score", "score": "_score_current_product", "family": "reference"},
        {"name": "max", "kind": "score", "score": "_score_max", "family": "candidate"},
        {"name": "weighted", "kind": "score", "score": "_score_weighted", "family": "candidate"},
        {"name": "analysis_first", "kind": "score", "score": "_score_analysis_first", "family": "reference"},
        {"name": "edge_reference", "kind": "score", "score": "_score_edge_reference", "family": "reference"},
        {"name": "uniform_reference", "kind": "score", "score": "_score_uniform_reference", "family": "reference"},
    ]
    for lam in lambdas:
        token = budget_token(lam)
        specs.append(
            {
                "name": f"rank_linear_l{token}",
                "kind": "score",
                "score": f"_score_rank_linear_l{token}",
                "family": "candidate",
                "lambda": lam,
            }
        )
    specs.extend(
        [
            {"name": "reserved_budget_a70_e20_p10", "kind": "reserved", "family": "candidate"},
            {"name": "slot_gated_reserved_budget", "kind": "slot_gated_reserved", "family": "candidate"},
            {"name": "confidence_gate", "kind": "confidence_gate", "family": "candidate"},
        ]
    )
    for primary in primary_scores:
        for ratio in ratio_grid:
            token = ratio_token(ratio)
            specs.append(
                {
                    "name": f"reserved_{primary}_{token}",
                    "kind": "ratio_reserved",
                    "family": "candidate",
                    "primary": primary,
                    "ratio": ratio,
                }
            )
            specs.append(
                {
                    "name": f"slot_gated_{primary}_{token}",
                    "kind": "slot_gated_ratio_reserved",
                    "family": "candidate",
                    "primary": primary,
                    "ratio": ratio,
                }
            )
    for profile in dynamic_profiles:
        kind = (
            "dynamic_guarded_ratio"
            if profile.startswith("dynamic_product_support_") or profile.startswith("dynamic_product_pattern_")
            else "dynamic_ratio"
        )
        specs.append({"name": profile, "kind": kind, "family": "candidate", "profile": profile})
    for primary in primary_scores:
        for ratio in ratio_grid:
            for multiplier in guarded_pool_multipliers:
                specs.append(
                    {
                        "name": f"guarded_{primary}_{multiplier_token(multiplier)}_{ratio_token(ratio)}",
                        "kind": "guarded_ratio_reserved",
                        "family": "candidate",
                        "primary": primary,
                        "ratio": ratio,
                        "pool_multiplier": multiplier,
                    }
                )
    return specs


def target_k(n_total: int, n_ranked: int, budget: float) -> int:
    if n_ranked <= 0:
        return 0
    return min(n_ranked, max(1, int(round(n_total * budget))))


def evaluate_strategy_case(
    base: pd.DataFrame,
    merged: pd.DataFrame,
    specs: list[dict],
    budgets: list[float],
    stats: dict,
    flat_std_threshold: float,
    flat_iqr_threshold: float,
    reserve_min_slots: int,
    dynamic_min_budget: float,
    fast_metrics: bool,
) -> tuple[list[dict], list[dict]]:
    rows = []
    decisions = []
    n_ranked = int(merged["TraceID"].notna().sum())
    for spec in specs:
        for budget in budgets:
            k = target_k(len(base), n_ranked, budget)
            if spec["kind"] == "score":
                kept = top_by_score(merged, spec["score"], k)
                decision = "score"
            elif spec["kind"] == "reserved":
                kept = reserved_budget_selection(merged, k)
                decision = "reserved_budget"
            elif spec["kind"] == "slot_gated_reserved":
                if k < reserve_min_slots:
                    kept = top_by_score(merged, "_score_current_product", k)
                    decision = "product_small_budget"
                else:
                    kept = reserved_budget_selection(merged, k)
                    decision = "reserved_budget"
            elif spec["kind"] == "ratio_reserved":
                ratio = spec["ratio"]
                kept = reserved_budget_selection(
                    merged,
                    k,
                    analysis_pct=ratio[0],
                    edge_pct=ratio[1],
                    pattern_pct=ratio[2],
                    primary_score=primary_score_column(spec["primary"]),
                )
                decision = f"{spec['primary']}_{ratio_token(ratio)}"
            elif spec["kind"] == "slot_gated_ratio_reserved":
                ratio = spec["ratio"]
                if k < reserve_min_slots:
                    kept = top_by_score(merged, "_score_current_product", k)
                    decision = "product_small_budget"
                else:
                    kept = reserved_budget_selection(
                        merged,
                        k,
                        analysis_pct=ratio[0],
                        edge_pct=ratio[1],
                        pattern_pct=ratio[2],
                        primary_score=primary_score_column(spec["primary"]),
                    )
                    decision = f"{spec['primary']}_{ratio_token(ratio)}"
            elif spec["kind"] == "guarded_ratio_reserved":
                ratio = spec["ratio"]
                if k < reserve_min_slots:
                    kept = top_by_score(merged, "_score_current_product", k)
                    decision = "product_small_budget"
                else:
                    kept = guarded_reserved_budget_selection(
                        merged,
                        k,
                        ratio=ratio,
                        pool_multiplier=spec["pool_multiplier"],
                        primary_score=primary_score_column(spec["primary"]),
                    )
                    decision = (
                        f"{spec['primary']}_{ratio_token(ratio)}"
                        f"_{multiplier_token(spec['pool_multiplier'])}"
                    )
            elif spec["kind"] == "dynamic_ratio":
                kept, decision = dynamic_ratio_selection(
                    merged,
                    k,
                    profile=spec["profile"],
                    reserve_min_slots=reserve_min_slots,
                )
            elif spec["kind"] == "dynamic_guarded_ratio":
                kept, decision = dynamic_guarded_ratio_selection(
                    merged,
                    k,
                    profile=spec["profile"],
                    reserve_min_slots=reserve_min_slots,
                    budget=budget,
                    dynamic_min_budget=dynamic_min_budget,
                )
            elif spec["kind"] == "confidence_gate":
                kept, decision = confidence_gate_selection(
                    merged,
                    k,
                    stats,
                    flat_std_threshold=flat_std_threshold,
                    flat_iqr_threshold=flat_iqr_threshold,
                )
            else:
                raise ValueError(f"Unknown strategy kind: {spec['kind']}")
            metric_builder = build_fast_metric_row if fast_metrics else build_metric_row
            row = metric_builder(base, kept, spec["name"], budget, n_ranked)
            row["strategy_family"] = spec["family"]
            row["decision_mode"] = decision
            rows.append(row)
            decisions.append(
                {
                    "strategy": spec["name"],
                    "budget": budget,
                    "decision_mode": decision,
                    "n_target": k,
                    "n_kept": len(kept),
                }
            )
    return rows, decisions


def summarize(per_case: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        c
        for c in per_case.columns
        if c
        not in {
            "dataset",
            "batch",
            "case",
            "trace_label_scope",
            "ranking",
            "strategy_family",
            "decision_mode",
        }
        and c != "budget"
        and pd.api.types.is_numeric_dtype(per_case[c])
    ]
    rows = []
    keys = ["dataset", "trace_label_scope", "ranking", "strategy_family", "budget"]
    for group_key, group in per_case.groupby(keys, sort=True):
        dataset, scope, ranking, family, budget = group_key
        row = {
            "dataset": dataset,
            "trace_label_scope": scope,
            "ranking": ranking,
            "strategy_family": family,
            "budget": budget,
            "cases": len(group),
        }
        for column in metric_cols:
            values = group[column].dropna()
            row[f"{column}_mean"] = float(values.mean()) if not values.empty else np.nan
            row[f"{column}_std"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            row[f"{column}_n"] = int(values.count())
        rows.append(row)
    return pd.DataFrame(rows)


def compare_against_current(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    metric_cols = [
        "A_exposed_recall_mean",
        "A_exposed_prop_mean",
        "pattern_coverage_mean",
        "pattern_entropy_mean",
        "pattern_jsd_mean",
    ]
    base_keys = ["dataset", "trace_label_scope", "budget"]
    current = summary[summary["ranking"] == "current_product"][base_keys + metric_cols].copy()
    current = current.rename(columns={c: f"current_{c}" for c in metric_cols})
    out = summary.merge(current, on=base_keys, how="left")
    for col in metric_cols:
        out[f"delta_{col}"] = out[col] - out[f"current_{col}"]
    return out


def promotion_table(compared: pd.DataFrame, focus_budget: float) -> pd.DataFrame:
    primary = compared[
        (compared["trace_label_scope"] == lp.PRIMARY_SCOPE)
        & np.isclose(compared["budget"], focus_budget)
        & (compared["strategy_family"] == "candidate")
    ].copy()
    if primary.empty:
        return primary
    grouped = []
    for ranking, group in primary.groupby("ranking", sort=True):
        datasets = sorted(group["dataset"].unique())
        grouped.append(
            {
                "ranking": ranking,
                "datasets": ",".join(datasets),
                "min_delta_A_exposed_recall": float(group["delta_A_exposed_recall_mean"].min()),
                "mean_delta_A_exposed_recall": float(group["delta_A_exposed_recall_mean"].mean()),
                "min_delta_pattern_coverage": float(group["delta_pattern_coverage_mean"].min()),
                "mean_delta_pattern_coverage": float(group["delta_pattern_coverage_mean"].mean()),
                "max_delta_pattern_jsd": float(group["delta_pattern_jsd_mean"].max()),
                "passes_no_recall_regression": bool((group["delta_A_exposed_recall_mean"] >= -1e-12).all()),
            }
        )
    return pd.DataFrame(grouped).sort_values(
        ["passes_no_recall_regression", "mean_delta_A_exposed_recall", "mean_delta_pattern_coverage"],
        ascending=[False, False, False],
    )


def support_weighted_table(per_case: pd.DataFrame, focus_budget: float) -> pd.DataFrame:
    primary = per_case[
        (per_case["trace_label_scope"] == lp.PRIMARY_SCOPE)
        & np.isclose(per_case["budget"], focus_budget)
    ].copy()
    required = {"A_exposed_support", "A_exposed_kept", "strategy_family"}
    if primary.empty or not required.issubset(primary.columns):
        return pd.DataFrame()

    keys = ["dataset", "batch", "case"]
    current = primary[primary["ranking"] == "current_product"][keys + ["A_exposed_support", "A_exposed_kept"]].copy()
    current = current.rename(
        columns={
            "A_exposed_support": "current_A_exposed_support",
            "A_exposed_kept": "current_A_exposed_kept",
        }
    )
    candidates = primary[primary["strategy_family"] == "candidate"].copy()
    merged = candidates.merge(current, on=keys, how="inner")
    rows = []
    for (ranking, dataset), group in merged.groupby(["ranking", "dataset"], sort=True):
        support = float(group["current_A_exposed_support"].sum())
        current_kept = float(group["current_A_exposed_kept"].sum())
        kept = float(group["A_exposed_kept"].sum())
        current_recall = current_kept / support if support else float("nan")
        recall = kept / support if support else float("nan")
        rows.append(
            {
                "ranking": ranking,
                "dataset": dataset,
                "cases": int(len(group)),
                "A_exposed_support_sum": support,
                "current_A_exposed_kept_sum": current_kept,
                "A_exposed_kept_sum": kept,
                "delta_A_exposed_kept": kept - current_kept,
                "current_weighted_A_exposed_recall": current_recall,
                "weighted_A_exposed_recall": recall,
                "weighted_delta_A_exposed_recall": recall - current_recall,
            }
        )
    weighted = pd.DataFrame(rows)
    if weighted.empty:
        return weighted

    primary_datasets = sorted(primary["dataset"].dropna().unique())
    aggregate_rows = []
    for ranking, group in weighted.groupby("ranking", sort=True):
        aggregate_rows.append(
            {
                "ranking": ranking,
                "dataset": "ALL_PRIMARY",
                "cases": int(group["cases"].sum()),
                "A_exposed_support_sum": float(group["A_exposed_support_sum"].sum()),
                "current_A_exposed_kept_sum": float(group["current_A_exposed_kept_sum"].sum()),
                "A_exposed_kept_sum": float(group["A_exposed_kept_sum"].sum()),
                "delta_A_exposed_kept": float(group["delta_A_exposed_kept"].sum()),
                "current_weighted_A_exposed_recall": (
                    float(group["current_A_exposed_kept_sum"].sum())
                    / float(group["A_exposed_support_sum"].sum())
                    if float(group["A_exposed_support_sum"].sum())
                    else float("nan")
                ),
                "weighted_A_exposed_recall": (
                    float(group["A_exposed_kept_sum"].sum()) / float(group["A_exposed_support_sum"].sum())
                    if float(group["A_exposed_support_sum"].sum())
                    else float("nan")
                ),
                "weighted_delta_A_exposed_recall": (
                    float(group["A_exposed_kept_sum"].sum() - group["current_A_exposed_kept_sum"].sum())
                    / float(group["A_exposed_support_sum"].sum())
                    if float(group["A_exposed_support_sum"].sum())
                    else float("nan")
                ),
            }
        )
        missing = sorted(set(primary_datasets) - set(group["dataset"]))
        if missing:
            aggregate_rows[-1]["missing_datasets"] = ",".join(missing)
    out = pd.concat([weighted, pd.DataFrame(aggregate_rows)], ignore_index=True)
    return out.sort_values(["ranking", "dataset"]).reset_index(drop=True)


def render_markdown(
    summary: pd.DataFrame,
    compared: pd.DataFrame,
    promoted: pd.DataFrame,
    weighted: pd.DataFrame,
    focus_budget: float,
) -> str:
    lines = [
        "# Post-hoc v0.2 Trace Composition Explorer",
        "",
        "This is an offline screening pass over existing UniSage sample frames. It does not use baseline outputs",
        "or v0.2 labels to construct any candidate ranking; labels are used only for evaluation.",
        "",
        f"## Focus Budget {focus_budget:g}",
        "",
    ]
    focus = compared[np.isclose(compared["budget"], focus_budget)].copy()
    show_cols = [
        "dataset",
        "trace_label_scope",
        "ranking",
        "strategy_family",
        "A_exposed_recall_mean",
        "delta_A_exposed_recall_mean",
        "A_exposed_prop_mean",
        "pattern_coverage_mean",
        "delta_pattern_coverage_mean",
        "pattern_jsd_mean",
        "delta_pattern_jsd_mean",
    ]
    if not focus.empty:
        primary = focus[focus["trace_label_scope"] == lp.PRIMARY_SCOPE]
        lines.extend(
            [
                "| dataset | strategy | family | A recall | ΔA | A prop | pattern cov | Δcov | JSD | ΔJSD |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in primary.sort_values(["dataset", "A_exposed_recall_mean"], ascending=[True, False]).itertuples(index=False):
            lines.append(
                f"| {row.dataset} | {row.ranking} | {row.strategy_family} | "
                f"{row.A_exposed_recall_mean:.3f} | {row.delta_A_exposed_recall_mean:+.3f} | "
                f"{row.A_exposed_prop_mean:.3f} | {row.pattern_coverage_mean:.3f} | "
                f"{row.delta_pattern_coverage_mean:+.3f} | {row.pattern_jsd_mean:.3f} | "
                f"{row.delta_pattern_jsd_mean:+.3f} |"
            )
    lines.extend(["", "## Promotion Screen", ""])
    if promoted.empty:
        lines.append("No candidate rows available for the promotion screen.")
    else:
        lines.extend(
            [
                "| strategy | datasets | min ΔA | mean ΔA | min Δcov | mean Δcov | max ΔJSD | no recall regression |",
                "|---|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in promoted.itertuples(index=False):
            lines.append(
                f"| {row.ranking} | {row.datasets} | {row.min_delta_A_exposed_recall:+.3f} | "
                f"{row.mean_delta_A_exposed_recall:+.3f} | {row.min_delta_pattern_coverage:+.3f} | "
                f"{row.mean_delta_pattern_coverage:+.3f} | {row.max_delta_pattern_jsd:+.3f} | "
                f"{row.passes_no_recall_regression} |"
            )
    lines.extend(["", "## Support-Weighted Screen", ""])
    if weighted.empty:
        lines.append("No support-weighted rows available.")
    else:
        all_primary = weighted[weighted["dataset"] == "ALL_PRIMARY"].copy()
        if not all_primary.empty:
            lines.extend(
                [
                    "| strategy | support | delta kept | weighted ΔA |",
                    "|---|---:|---:|---:|",
                ]
            )
            for row in all_primary.sort_values("weighted_delta_A_exposed_recall", ascending=False).head(12).itertuples(index=False):
                lines.append(
                    f"| {row.ranking} | {row.A_exposed_support_sum:.0f} | "
                    f"{row.delta_A_exposed_kept:+.0f} | {row.weighted_delta_A_exposed_recall:+.4f} |"
                )
    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- A candidate is safe to promote only if it does not regress `A_exposed` recall on any primary dataset.",
            "- The support-weighted screen is a guard against losing too many problem-related traces in large cases.",
            "- Pattern coverage and JSD are transparency axes: improvements are good, but they should not be optimized by using labels.",
            "- Winning candidates should be rerun as real sampler code before seeds 2-5 and fair-RCA evaluation.",
        ]
    )
    return "\n".join(lines) + "\n"


def load_case_inputs(entry, sample_lookup: dict[tuple[str, str, str], str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(resolve_path(entry.manifest_path))
    key = (entry.dataset, str(entry.batch), str(entry.case))
    sample_path = resolve_path(sample_lookup[key])
    sample = pd.read_csv(sample_path)
    manifest["TraceID"] = manifest["TraceID"].astype(str)
    sample["TraceID"] = sample["TraceID"].astype(str)
    missing = [c for c in REQUIRED_SAMPLE_COLUMNS if c not in sample.columns]
    if missing:
        raise ValueError(f"{sample_path} missing required columns: {missing}")
    return manifest, sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-index", type=Path, default=Path("results/label_protocol_v02/trace_manifest_index.csv"))
    parser.add_argument("--sample-index", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/posthoc_trace_composition_seed1"))
    parser.add_argument("--budgets", nargs="+", type=float, default=BUDGETS)
    parser.add_argument("--lambda-values", nargs="+", type=float, default=[0.60, 0.75, 0.90])
    parser.add_argument(
        "--ratio-grid",
        nargs="+",
        type=parse_ratio,
        default=[
            parse_ratio(v)
            for v in ["90,10,0", "85,10,5", "80,15,5", "75,20,5", "70,20,10", "60,30,10"]
        ],
        help="Reserved-budget ratios as analysis,edge,pattern triples; values are normalized to 100.",
    )
    parser.add_argument(
        "--primary-scores",
        nargs="+",
        choices=["analysis_product", "product"],
        default=["analysis_product"],
        help="Primary bucket score families to test for ratio-grid candidates.",
    )
    parser.add_argument(
        "--dynamic-profiles",
        nargs="+",
        default=[],
        choices=[
            "dynamic_product_cov_guard",
            "dynamic_product_overlap_guard",
            "dynamic_product_conservative",
            "dynamic_product_support_guard",
            "dynamic_product_support_balanced",
            "dynamic_product_support_conservative",
            "dynamic_product_pattern_guard",
            "dynamic_product_pattern_guard_conservative",
        ],
        help="Optional dynamic ratio profiles that select ratios from runtime score/novelty signals.",
    )
    parser.add_argument(
        "--guarded-pool-multipliers",
        nargs="+",
        type=float,
        default=[],
        help="Optional product top-M pool multipliers for fixed guarded ratio candidates.",
    )
    parser.add_argument(
        "--novelty-source",
        choices=["manifest", "runtime"],
        default="manifest",
        help="Use manifest pattern_hash or runtime-observable TracePatternKey/current path for novelty selection.",
    )
    parser.add_argument("--focus-budget", type=float, default=0.05)
    parser.add_argument("--flat-std-threshold", type=float, default=0.03)
    parser.add_argument("--flat-iqr-threshold", type=float, default=0.02)
    parser.add_argument(
        "--reserve-min-slots",
        type=int,
        default=50,
        help="Only enable the reserved-budget candidate when a case/budget keeps at least this many traces.",
    )
    parser.add_argument(
        "--dynamic-min-budget",
        type=float,
        default=0.05,
        help="For dynamic pattern-guard profiles, keep product top-k below this budget.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--fast-metrics",
        action="store_true",
        help="Skip entropy/JSD and compute only cheap gate metrics for broad candidate sweeps.",
    )
    args = parser.parse_args()

    manifest_index = pd.read_csv(resolve_path(args.manifest_index))
    sample_frames = []
    for path in args.sample_index:
        frame = pd.read_csv(resolve_path(path))
        if "status" in frame.columns:
            frame = frame[frame["status"] == "ok"].copy()
        frame["batch"] = frame["batch"].astype(str)
        frame["case"] = frame["case"].astype(str)
        frame["source_sample_index"] = resolve_path(path).as_posix()
        sample_frames.append(frame)
    sample_index = pd.concat(sample_frames, ignore_index=True)
    sample_index = sample_index.drop_duplicates(["dataset", "batch", "case"], keep="last")
    sample_lookup = {
        (row.dataset, str(row.batch), str(row.case)): row.sample_path
        for row in sample_index.itertuples(index=False)
    }

    selected = manifest_index.copy()
    selected["batch"] = selected["batch"].astype(str)
    selected["case"] = selected["case"].astype(str)
    selected = selected[
        selected.apply(lambda r: (r["dataset"], str(r["batch"]), str(r["case"])) in sample_lookup, axis=1)
    ].reset_index(drop=True)
    if args.limit is not None:
        selected = selected.head(args.limit)
    if selected.empty:
        raise SystemExit("No manifest cases matched the supplied sample indexes.")

    specs = strategy_specs(
        args.lambda_values,
        args.ratio_grid,
        args.primary_scores,
        args.dynamic_profiles,
        args.guarded_pool_multipliers,
    )
    all_rows = []
    decision_rows = []
    stats_rows = []
    skipped = []
    for i, entry in enumerate(selected.itertuples(index=False), start=1):
        print(f"[{i}/{len(selected)}] {entry.dataset}/{entry.batch}/{entry.case}", flush=True)
        try:
            manifest, sample = load_case_inputs(entry, sample_lookup)
            merged = manifest.merge(sample.drop_duplicates("TraceID"), on="TraceID", how="left")
            merged = add_derived_scores(merged, args.lambda_values, args.novelty_source)
            stats = case_trace_stats(merged)
            stats_rows.append(
                {
                    "dataset": entry.dataset,
                    "batch": entry.batch,
                    "case": entry.case,
                    "trace_label_scope": entry.trace_label_scope,
                    **stats,
                }
            )
            rows, decisions = evaluate_strategy_case(
                manifest,
                merged,
                specs,
                args.budgets,
                stats,
                flat_std_threshold=args.flat_std_threshold,
                flat_iqr_threshold=args.flat_iqr_threshold,
                reserve_min_slots=args.reserve_min_slots,
                dynamic_min_budget=args.dynamic_min_budget,
                fast_metrics=args.fast_metrics,
            )
            for row in rows:
                row.update(
                    {
                        "dataset": entry.dataset,
                        "batch": entry.batch,
                        "case": entry.case,
                        "trace_label_scope": entry.trace_label_scope,
                    }
                )
            for row in decisions:
                row.update(
                    {
                        "dataset": entry.dataset,
                        "batch": entry.batch,
                        "case": entry.case,
                        "trace_label_scope": entry.trace_label_scope,
                    }
                )
            all_rows.extend(rows)
            decision_rows.extend(decisions)
        except Exception as exc:
            skipped.append(
                {
                    "dataset": entry.dataset,
                    "batch": entry.batch,
                    "case": entry.case,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"  [skip] {skipped[-1]['reason']}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    per_case = pd.DataFrame(all_rows)
    summary = summarize(per_case) if not per_case.empty else pd.DataFrame()
    compared = compare_against_current(summary) if not summary.empty else pd.DataFrame()
    promoted = promotion_table(compared, args.focus_budget) if not compared.empty else pd.DataFrame()
    weighted = support_weighted_table(per_case, args.focus_budget) if not per_case.empty else pd.DataFrame()
    stats_df = pd.DataFrame(stats_rows)
    decisions_df = pd.DataFrame(decision_rows)
    skipped_df = pd.DataFrame(skipped, columns=["dataset", "batch", "case", "reason"])

    per_case.to_csv(args.output / "posthoc_trace_composition_per_case.csv", index=False)
    summary.to_csv(args.output / "posthoc_trace_composition_summary.csv", index=False)
    compared.to_csv(args.output / "posthoc_trace_composition_vs_current.csv", index=False)
    promoted.to_csv(args.output / "posthoc_trace_composition_promotion_screen.csv", index=False)
    weighted.to_csv(args.output / "posthoc_trace_composition_weighted_screen.csv", index=False)
    stats_df.to_csv(args.output / "posthoc_trace_composition_case_stats.csv", index=False)
    decisions_df.to_csv(args.output / "posthoc_trace_composition_decisions.csv", index=False)
    skipped_df.to_csv(args.output / "posthoc_trace_composition_skipped.csv", index=False)

    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": lp.PROTOCOL_VERSION,
        "git_head": git_head(),
        "script_fingerprint": file_fingerprint([HERE / "scripts/posthoc_trace_composition.py"]),
        "manifest_index": resolve_path(args.manifest_index).as_posix(),
        "manifest_index_sha256": sha256_file(resolve_path(args.manifest_index)),
        "sample_indexes": [resolve_path(p).as_posix() for p in args.sample_index],
        "budgets": args.budgets,
        "lambda_values": args.lambda_values,
        "weighted_variant": {
            "score": "0.7 * AnomalySamplingProb + 0.3 * EdgeCaseSamplingProb",
            "analysis_weight": DEFAULT_WEIGHTED_ANALYSIS,
            "edge_weight": DEFAULT_WEIGHTED_EDGE,
            "nan_handling": "missing analysis/edge scores are treated as 0.0 for this post-hoc variant",
        },
        "ratio_grid": [list(r) for r in args.ratio_grid],
        "primary_scores": args.primary_scores,
        "dynamic_profiles": args.dynamic_profiles,
        "guarded_pool_multipliers": args.guarded_pool_multipliers,
        "novelty_source": args.novelty_source,
        "flat_std_threshold": args.flat_std_threshold,
        "flat_iqr_threshold": args.flat_iqr_threshold,
        "reserve_min_slots": args.reserve_min_slots,
        "dynamic_min_budget": args.dynamic_min_budget,
        "focus_budget": args.focus_budget,
        "fast_metrics": args.fast_metrics,
        "selected_cases": int(len(selected)),
        "skipped_cases": int(len(skipped)),
        "note": "Post-hoc candidate screening only; do not cite as production sampler output until implemented and rerun.",
    }
    (args.output / "posthoc_trace_composition_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    report = render_markdown(summary, compared, promoted, weighted, args.focus_budget)
    (args.output / "posthoc_trace_composition_summary.md").write_text(report)
    print(report)
    print(f"Wrote post-hoc composition artifacts to {args.output}; skipped={len(skipped)}")


if __name__ == "__main__":
    main()
