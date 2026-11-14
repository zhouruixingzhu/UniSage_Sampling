"""Evaluate trace-sampling rankings against v0.2 manifests.

This script reads frozen trace manifests plus trace_sampling_result.csv files.
It does not generate labels. Results produced from existing case outputs should
be treated as smoke/provenance checks unless the sampling runs are themselves
version-stamped.
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
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import sampling_labels as lp


BUDGETS = [0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.10, 0.20]
RANKING_COLUMNS = {
    "combined": "CombinedSamplingProb",
    "edge": "EdgeCaseSamplingProb",
    "analysis": "AnomalySamplingProb",
    "latency": "LatencySamplingProb",
    "topology": "TopologySamplingProb",
    "sifter": "SifterLoss",
    "sieve": "SieveScore",
    "trastrainer": "TraStrainerCombined",
    "gleaner": "GleanerScore",
}
DECISION_COLUMN_PREFIXES = {
    "slot_gated_reserved": "SlotGatedReservedDecision",
    "dynamic_pattern_guard": "DynamicPatternGuardDecision",
    "tracepicker": "TracePickerDecision",
    "sifter_decision": "SifterDecision",
    "sieve_decision": "SieveDecision",
    "trastrainer_decision": "TraStrainerDecision",
    "gleaner_decision": "GleanerDecision",
}
DESCRIPTIVE_LABELS = ["struct_unseen", "struct_rare_le_0p1", "struct_rare_le_1p0", "lat_tail_q99"]
PROBLEM_LABELS = ["A_exposed", "A_manifested_latency", "A_manifested_log", "A_manifested", "A_manifested_broad"]


def stable_uniform_score(trace_id: object) -> float:
    digest = hashlib.sha256(str(trace_id).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)


def budget_token(budget: float) -> str:
    return f"{budget:g}".replace("-", "m").replace(".", "p")


def case_dir(data_root: Path, dataset: str, batch: str, case: str) -> Path:
    if dataset == "aiops21":
        return data_root / "aiops21" / "all" / case
    return data_root / dataset / "batches" / batch / case


def entropy(values: pd.Series) -> float:
    if values.empty:
        return float("nan")
    probs = values.value_counts(normalize=True)
    return float(-(probs * np.log2(probs)).sum())


def pattern_distribution(values: pd.Series) -> pd.Series:
    if values.empty:
        return pd.Series(dtype=float)
    normalized = values.fillna("__MISSING_PATTERN__").astype(str)
    return normalized.value_counts(normalize=True, sort=False)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))


def jensen_shannon_divergence(reference: pd.Series, observed: pd.Series) -> float:
    """Jensen-Shannon divergence between full and sampled pattern distributions.

    We use log base 2, so the value is bounded in [0, 1]. Lower means the
    sampled set is more distribution-faithful to the full case.
    """
    if reference.empty or observed.empty:
        return float("nan")
    ref_dist = pattern_distribution(reference)
    obs_dist = pattern_distribution(observed)
    support = ref_dist.index.union(obs_dist.index)
    p = ref_dist.reindex(support, fill_value=0.0).to_numpy(dtype=float)
    q = obs_dist.reindex(support, fill_value=0.0).to_numpy(dtype=float)
    midpoint = 0.5 * (p + q)
    value = 0.5 * kl_divergence(p, midpoint) + 0.5 * kl_divergence(q, midpoint)
    return float(np.clip(value, 0.0, 1.0))


def label_stats(frame: pd.DataFrame, kept: pd.DataFrame, label: str, enabled: bool = True) -> Dict[str, float]:
    if not enabled:
        return {f"{label}_support": np.nan, f"{label}_kept": np.nan, f"{label}_recall": np.nan, f"{label}_prop": np.nan}
    support = int(frame[label].sum()) if label in frame.columns else 0
    kept_count = int(kept[label].sum()) if label in kept.columns else 0
    return {
        f"{label}_support": support,
        f"{label}_kept": kept_count,
        f"{label}_recall": kept_count / support if support else np.nan,
        f"{label}_prop": kept_count / len(kept) if len(kept) else np.nan,
    }


def as_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) != 0
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "t", "yes", "y"})


def available_rankings(sample: Optional[pd.DataFrame], requested: List[str]) -> List[str]:
    rankings: List[str] = []
    if "uniform" in requested:
        rankings.append("uniform")
    if sample is None:
        return rankings
    for name in requested:
        column = RANKING_COLUMNS.get(name)
        if column and column in sample.columns:
            rankings.append(name)
            continue
        prefix = DECISION_COLUMN_PREFIXES.get(name)
        if prefix and any(c == prefix or c.startswith(f"{prefix}_") for c in sample.columns):
            rankings.append(name)
    return rankings


def decision_column(sample: pd.DataFrame, ranking: str, budget: float) -> Optional[str]:
    prefix = DECISION_COLUMN_PREFIXES[ranking]
    budget_specific = f"{prefix}_{budget_token(budget)}"
    if budget_specific in sample.columns:
        return budget_specific
    if prefix in sample.columns:
        return prefix
    return None


def build_metric_row(base: pd.DataFrame, kept: pd.DataFrame, ranking: str, budget: float, n_ranked: int) -> dict:
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
        "pattern_entropy": entropy(kept["pattern_hash"]) if k else np.nan,
        "pattern_jsd": jensen_shannon_divergence(base["pattern_hash"], kept["pattern_hash"]) if k else np.nan,
        "pattern_benefit_cost": (kept["pattern_hash"].nunique() / k) if k else np.nan,
    }
    for label in DESCRIPTIVE_LABELS:
        row.update(label_stats(base, kept, label, enabled=True))
    for label in PROBLEM_LABELS:
        row.update(label_stats(base, kept, label, enabled=primary_scope))
    return row


def evaluate_case(manifest: pd.DataFrame, sample: Optional[pd.DataFrame], rankings: List[str], budgets: List[float]) -> List[dict]:
    base = manifest.copy()
    if sample is not None:
        sample = sample.drop_duplicates("TraceID").copy()
        merged = base.merge(sample, on="TraceID", how="left")
    else:
        merged = base
    rows: List[dict] = []
    requested = available_rankings(sample, rankings)
    for ranking in requested:
        if ranking in DECISION_COLUMN_PREFIXES:
            if sample is None:
                continue
            for budget in budgets:
                column = decision_column(merged, ranking, budget)
                if column is None:
                    continue
                decision = as_bool_series(merged[column])
                kept = merged.loc[decision].sort_values("TraceID").reset_index(drop=True)
                n_ranked = int(merged[column].notna().sum())
                rows.append(build_metric_row(base, kept, ranking, budget, n_ranked))
            continue

        if ranking == "uniform":
            ranked = merged.copy()
            ranked["_rank_score"] = ranked["TraceID"].map(stable_uniform_score)
        else:
            column = RANKING_COLUMNS[ranking]
            ranked = merged.dropna(subset=[column]).copy()
            ranked["_rank_score"] = ranked[column].astype(float)
        ranked = ranked.sort_values(["_rank_score", "TraceID"], ascending=[False, True]).reset_index(drop=True)
        n_ranked = len(ranked)
        for budget in budgets:
            k = min(n_ranked, max(1, int(round(len(base) * budget)))) if n_ranked else 0
            kept = ranked.head(k)
            rows.append(build_metric_row(base, kept, ranking, budget, n_ranked))
    return rows


def summarize(per_case: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        c for c in per_case.columns
        if c not in {"dataset", "batch", "case", "trace_label_scope", "ranking"}
        and c not in {"budget"}
        and pd.api.types.is_numeric_dtype(per_case[c])
    ]
    rows = []
    for keys, group in per_case.groupby(["dataset", "ranking", "budget"], sort=True):
        dataset, ranking, budget = keys
        row = {"dataset": dataset, "ranking": ranking, "budget": budget, "cases": len(group)}
        for column in metric_cols:
            series = group[column].dropna()
            row[f"{column}_mean"] = float(series.mean()) if not series.empty else np.nan
            row[f"{column}_std"] = float(series.std(ddof=1)) if len(series) > 1 else np.nan
            row[f"{column}_n"] = int(series.count())
        rows.append(row)
    return pd.DataFrame(rows)


def render_summary(summary: pd.DataFrame, mode: str) -> str:
    if mode.startswith("version_stamped"):
        description = [
            "This file evaluates a version-stamped trace-sampling run against frozen v0.2 manifests.",
            "It is candidate paper evidence only for the datasets/cases covered by the associated sample index.",
        ]
    else:
        description = [
            "This file evaluates existing sampling outputs against frozen v0.2 manifests.",
            "Treat as a smoke/provenance check unless sampling runs are version-stamped.",
        ]
    lines = [
        f"# v0.2 Trace Sampling Evaluation ({mode})",
        "",
        *description,
        "",
        "| dataset | ranking | budget | cases | pattern cov. | pattern JSD | A_exposed recall | A_exposed prop |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.ranking} | {row.budget:.3f} | {row.cases} | "
            f"{getattr(row, 'pattern_coverage_mean'):.3f} | "
            f"{getattr(row, 'pattern_jsd_mean') if not pd.isna(getattr(row, 'pattern_jsd_mean')) else float('nan'):.3f} | "
            f"{getattr(row, 'A_exposed_recall_mean') if not pd.isna(getattr(row, 'A_exposed_recall_mean')) else float('nan'):.3f} | "
            f"{getattr(row, 'A_exposed_prop_mean') if not pd.isna(getattr(row, 'A_exposed_prop_mean')) else float('nan'):.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--manifest-index", type=Path, default=Path("results/label_protocol_v02/trace_manifest_index.csv"))
    parser.add_argument("--sample-index", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("results/label_protocol_v02/eval_existing_outputs_smoke"))
    parser.add_argument("--rankings", nargs="+", default=["combined", "edge", "analysis", "uniform"])
    parser.add_argument("--budgets", nargs="+", type=float, default=BUDGETS)
    parser.add_argument("--mode", default="existing_outputs_smoke")
    args = parser.parse_args()

    index = pd.read_csv(args.manifest_index)
    sample_lookup = {}
    skipped = []
    if args.sample_index is not None:
        sample_index = pd.read_csv(args.sample_index)
        if not sample_index.empty and "status" in sample_index.columns:
            failed = sample_index[sample_index["status"] != "ok"]
            for entry in failed.itertuples(index=False):
                skipped.append(
                    {
                        "dataset": entry.dataset,
                        "batch": entry.batch,
                        "case": entry.case,
                        "reason": f"sample_index_status_{entry.status}",
                    }
                )
            sample_index = sample_index[sample_index["status"] == "ok"].copy()
        index_keys = sample_index[["dataset", "batch", "case"]].drop_duplicates()
        index = index.merge(index_keys, on=["dataset", "batch", "case"], how="inner")
        for entry in sample_index.itertuples(index=False):
            sample_lookup[(entry.dataset, entry.batch, entry.case)] = entry.sample_path
    rows: List[dict] = []
    for entry in index.itertuples(index=False):
        manifest = pd.read_csv(entry.manifest_path)
        key = (entry.dataset, entry.batch, entry.case)
        sample_path = Path(sample_lookup[key]) if key in sample_lookup else (
            case_dir(args.data_root, entry.dataset, entry.batch, entry.case) / "trace_sampling_result.csv"
        )
        sample = pd.read_csv(sample_path) if sample_path.exists() else None
        if sample is None and any(r != "uniform" for r in args.rankings):
            skipped.append({"dataset": entry.dataset, "batch": entry.batch, "case": entry.case, "reason": "missing_trace_sampling_result"})
        for row in evaluate_case(manifest, sample, args.rankings, args.budgets):
            row.update(
                {
                    "dataset": entry.dataset,
                    "batch": entry.batch,
                    "case": entry.case,
                    "trace_label_scope": entry.trace_label_scope,
                }
            )
            rows.append(row)

    args.output.mkdir(parents=True, exist_ok=True)
    per_case = pd.DataFrame(rows)
    summary = summarize(per_case) if not per_case.empty else pd.DataFrame()
    per_case.to_csv(args.output / "trace_eval_per_case.csv", index=False)
    summary.to_csv(args.output / "trace_eval_summary.csv", index=False)
    pd.DataFrame(skipped, columns=["dataset", "batch", "case", "reason"]).to_csv(
        args.output / "trace_eval_skipped_cases.csv",
        index=False,
    )
    warning = (
        "Version-stamped run; validate skipped cases, repetitions, and baselines before manuscript use."
        if args.mode.startswith("version_stamped")
        else "Existing sampling outputs are not paper-result provenance unless generated by a stamped run."
    )
    provenance = {
        "protocol_version": lp.PROTOCOL_VERSION,
        "mode": args.mode,
        "manifest_index": args.manifest_index.as_posix(),
        "sample_index": args.sample_index.as_posix() if args.sample_index is not None else None,
        "rankings": args.rankings,
        "budgets": args.budgets,
        "skipped_cases": len(skipped),
        "warning": warning,
    }
    (args.output / "trace_eval_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    report = render_summary(summary, args.mode) if not summary.empty else "# v0.2 Trace Sampling Evaluation\n\nNo rows.\n"
    (args.output / "trace_eval_summary.md").write_text(report)
    print(report)
    print(f"Wrote evaluation artifacts to {args.output}; skipped cases with missing sampling output: {len(skipped)}")


if __name__ == "__main__":
    main()
