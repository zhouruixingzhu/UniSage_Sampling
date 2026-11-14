"""Evaluate log-sampling rankings against v0.2 log manifests.

This script reads frozen log manifests plus `log_sampling_result.csv` files.
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

import log_protocol_materialize as lm


BUDGETS = lm.LOG_BUDGETS
RANKING_COLUMNS = {
    "combined": "LogSamplingProb",
    "max": "LogSamplingProbMax",
    "product": "LogSamplingProbProduct",
    "weighted": "LogSamplingProbWeighted",
    "edge": "LogRaritySamplingProb",
    "analysis": "LogAnomalySamplingProb",
    "rarity_raw": "LogRaritySamplingProbRaw",
}
RESERVED_RANKING = "reserved_budget"
LABELS = [
    "log_manifested",
    "log_manifested_broad",
    "log_template_unseen",
    "log_template_rare_le_0p1",
    "log_template_rare_le_1p0",
    "log_injected_service_window",
    "linked_A_exposed",
    "diagnostic_pair_strict",
    "diagnostic_pair_broad",
    "level_error",
    "level_warn",
    "level_error_or_warn",
    "has_trace_link",
]


def stable_uniform_score(value: object) -> float:
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)


def budget_token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def case_dir(data_root: Path, dataset: str, batch: str, case: str) -> Path:
    return data_root / dataset / "batches" / batch / case


def entropy(values: pd.Series) -> float:
    if values.empty:
        return float("nan")
    probs = values.value_counts(normalize=True)
    return float(-(probs * np.log2(probs)).sum())


def distribution(values: pd.Series) -> pd.Series:
    if values.empty:
        return pd.Series(dtype=float)
    normalized = values.fillna("__MISSING__").astype(str)
    return normalized.value_counts(normalize=True, sort=False)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))


def jensen_shannon_divergence(reference: pd.Series, observed: pd.Series) -> float:
    if reference.empty or observed.empty:
        return float("nan")
    ref_dist = distribution(reference)
    obs_dist = distribution(observed)
    support = ref_dist.index.union(obs_dist.index)
    p = ref_dist.reindex(support, fill_value=0.0).to_numpy(dtype=float)
    q = obs_dist.reindex(support, fill_value=0.0).to_numpy(dtype=float)
    midpoint = 0.5 * (p + q)
    value = 0.5 * kl_divergence(p, midpoint) + 0.5 * kl_divergence(q, midpoint)
    return float(np.clip(value, 0.0, 1.0))


def label_stats(frame: pd.DataFrame, kept: pd.DataFrame, label: str) -> Dict[str, float]:
    support = int(frame[label].sum()) if label in frame.columns else 0
    kept_count = int(kept[label].sum()) if label in kept.columns else 0
    base_rate = support / len(frame) if len(frame) else np.nan
    prop = kept_count / len(kept) if len(kept) else np.nan
    return {
        f"{label}_support": support,
        f"{label}_kept": kept_count,
        f"{label}_recall": kept_count / support if support else np.nan,
        f"{label}_prop": prop,
        f"{label}_lift": prop / base_rate if support and base_rate > 0 and not pd.isna(prop) else np.nan,
    }


def sample_scores(sample: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if sample is None or sample.empty:
        return None
    normalized = lm.normalize_logs_for_manifest(sample)
    if normalized.empty:
        return None
    score_columns = [column for column in RANKING_COLUMNS.values() if column in sample.columns]
    score_columns.extend([
        column for column in sample.columns
        if column.startswith("LogReservedBudgetDecision_")
        or column.startswith("LogReservedBudgetRank_")
        or column.startswith("LogReservedBudgetScore_")
    ])
    score_columns = sorted(set(score_columns))
    out = normalized[["LogID"]].copy()
    source_rows = normalized["_source_row"].to_numpy(dtype=int)
    for column in score_columns:
        values = sample.iloc[source_rows][column]
        if column.startswith("LogReservedBudgetDecision_"):
            out[column] = values.fillna(False).astype(bool).to_numpy()
        else:
            out[column] = pd.to_numeric(values, errors="coerce").to_numpy()
    return out.drop_duplicates("LogID")


def available_rankings(scores: Optional[pd.DataFrame], requested: List[str]) -> List[str]:
    rankings: List[str] = []
    if "uniform" in requested:
        rankings.append("uniform")
    if scores is None:
        return rankings
    for name in requested:
        if name == RESERVED_RANKING:
            if any(column.startswith("LogReservedBudgetDecision_") for column in scores.columns):
                rankings.append(name)
            continue
        column = RANKING_COLUMNS.get(name)
        if column and column in scores.columns:
            rankings.append(name)
    return rankings


def build_metric_row(base: pd.DataFrame, kept: pd.DataFrame, ranking: str, budget: float, n_ranked: int) -> dict:
    k = len(kept)
    row = {
        "ranking": ranking,
        "budget": budget,
        "n_total": len(base),
        "n_ranked": n_ranked,
        "n_kept": k,
        "template_support": int(base["template_hash"].nunique()),
        "template_kept": int(kept["template_hash"].nunique()) if k else 0,
        "template_coverage": (
            kept["template_hash"].nunique() / base["template_hash"].nunique()
            if k and base["template_hash"].nunique()
            else np.nan
        ),
        "template_entropy": entropy(kept["template_hash"]) if k else np.nan,
        "template_jsd": jensen_shannon_divergence(base["template_hash"], kept["template_hash"]) if k else np.nan,
        "template_benefit_cost": kept["template_hash"].nunique() / k if k else np.nan,
        "service_template_support": int(base["service_template_hash"].nunique()),
        "service_template_kept": int(kept["service_template_hash"].nunique()) if k else 0,
        "service_template_coverage": (
            kept["service_template_hash"].nunique() / base["service_template_hash"].nunique()
            if k and base["service_template_hash"].nunique()
            else np.nan
        ),
        "service_support": int(base["Service"].nunique()),
        "service_kept": int(kept["Service"].nunique()) if k else 0,
        "service_coverage": kept["Service"].nunique() / base["Service"].nunique()
        if k and base["Service"].nunique()
        else np.nan,
        "service_entropy": entropy(kept["Service"]) if k else np.nan,
        "service_jsd": jensen_shannon_divergence(base["Service"], kept["Service"]) if k else np.nan,
        "level_jsd": jensen_shannon_divergence(base["log_level"], kept["log_level"]) if k else np.nan,
    }
    for label in LABELS:
        row.update(label_stats(base, kept, label))
    return row


def evaluate_case(manifest: pd.DataFrame, scores: Optional[pd.DataFrame], rankings: List[str], budgets: List[float]) -> List[dict]:
    base = manifest.copy()
    merged = base.merge(scores, on="LogID", how="left") if scores is not None else base
    rows: List[dict] = []
    for ranking in available_rankings(scores, rankings):
        if ranking == "uniform":
            ranked = merged.copy()
            ranked["_rank_score"] = ranked["LogID"].map(stable_uniform_score)
        elif ranking == RESERVED_RANKING:
            ranked = merged.copy()
        else:
            column = RANKING_COLUMNS[ranking]
            ranked = merged.dropna(subset=[column]).copy()
            ranked["_rank_score"] = ranked[column].astype(float)
        if ranking != RESERVED_RANKING:
            ranked = ranked.sort_values(["_rank_score", "LogID"], ascending=[False, True]).reset_index(drop=True)
            n_ranked = len(ranked)
        else:
            n_ranked = len(ranked)
        for budget in budgets:
            if ranking == RESERVED_RANKING:
                token = budget_token(float(budget))
                decision_col = f"LogReservedBudgetDecision_{token}"
                rank_col = f"LogReservedBudgetRank_{token}"
                score_col = f"LogReservedBudgetScore_{token}"
                if decision_col in ranked.columns:
                    kept = ranked.loc[ranked[decision_col].fillna(False).astype(bool)].copy()
                    if rank_col in kept.columns:
                        kept = kept.sort_values([rank_col, "LogID"], ascending=[True, True])
                    elif score_col in kept.columns:
                        kept = kept.sort_values([score_col, "LogID"], ascending=[False, True])
                elif score_col in ranked.columns:
                    candidate = ranked.dropna(subset=[score_col]).copy()
                    candidate["_rank_score"] = candidate[score_col].astype(float)
                    candidate = candidate.sort_values(["_rank_score", "LogID"], ascending=[False, True])
                    k = min(len(candidate), max(1, int(round(len(base) * budget)))) if len(candidate) else 0
                    kept = candidate.head(k)
                else:
                    kept = ranked.iloc[0:0].copy()
            else:
                k = min(n_ranked, max(1, int(round(len(base) * budget)))) if n_ranked else 0
                kept = ranked.head(k)
            rows.append(build_metric_row(base, kept, ranking, budget, n_ranked))
    return rows


def summarize(per_case: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        column
        for column in per_case.columns
        if column not in {"dataset", "batch", "case", "log_scope", "ranking", "budget"}
        and pd.api.types.is_numeric_dtype(per_case[column])
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
    if summary.empty:
        return f"# v0.2 Log Sampling Evaluation ({mode})\n\nNo evaluation rows were produced.\n"
    lines = [
        f"# v0.2 Log Sampling Evaluation ({mode})",
        "",
        "This file evaluates log-sampling rankings against frozen v0.2 log manifests.",
        "Existing case-dir outputs are smoke/provenance evidence unless regenerated as a version-stamped run.",
        "",
        "| dataset | ranking | budget | cases | template cov. | template JSD | strict-log recall | strict-log prop | rare<=1% recall | pair recall |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        def fmt(attr: str) -> str:
            value = getattr(row, attr, np.nan)
            return "nan" if pd.isna(value) else f"{value:.3f}"

        lines.append(
            f"| {row.dataset} | {row.ranking} | {row.budget:.3f} | {row.cases} | "
            f"{fmt('template_coverage_mean')} | {fmt('template_jsd_mean')} | "
            f"{fmt('log_manifested_recall_mean')} | {fmt('log_manifested_prop_mean')} | "
            f"{fmt('log_template_rare_le_1p0_recall_mean')} | "
            f"{fmt('diagnostic_pair_strict_recall_mean')} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--manifest-index", type=Path, default=Path("results/label_protocol_v02/log_manifest_index.csv"))
    parser.add_argument("--sample-index", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("results/label_protocol_v02/log_eval_existing_outputs_smoke"))
    parser.add_argument(
        "--rankings",
        nargs="+",
        default=["combined", "max", "product", "weighted", "reserved_budget", "edge", "analysis", "uniform"],
    )
    parser.add_argument("--budgets", nargs="+", type=float, default=BUDGETS)
    parser.add_argument("--mode", default="existing_outputs_smoke")
    args = parser.parse_args()

    index = pd.read_csv(args.manifest_index)
    sample_lookup: dict[tuple[str, str, str], Path] = {}
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
            sample_lookup[(entry.dataset, str(entry.batch), str(entry.case))] = Path(entry.sample_path)

    rows: List[dict] = []
    for entry in index.itertuples(index=False):
        manifest = pd.read_csv(entry.manifest_path)
        key = (entry.dataset, str(entry.batch), str(entry.case))
        sample_path = sample_lookup.get(key, case_dir(args.data_root, *key) / "log_sampling_result.csv")
        raw_sample = pd.read_csv(sample_path) if sample_path.exists() else None
        scores = sample_scores(raw_sample)
        if scores is None and any(ranking != "uniform" for ranking in args.rankings):
            skipped.append({"dataset": entry.dataset, "batch": entry.batch, "case": entry.case, "reason": "missing_log_sampling_result"})
        for row in evaluate_case(manifest, scores, args.rankings, args.budgets):
            row.update(
                {
                    "dataset": entry.dataset,
                    "batch": entry.batch,
                    "case": entry.case,
                    "log_scope": getattr(entry, "log_scope", "primary_log_scope"),
                }
            )
            rows.append(row)

    args.output.mkdir(parents=True, exist_ok=True)
    per_case = pd.DataFrame(rows)
    summary = summarize(per_case) if not per_case.empty else pd.DataFrame()
    per_case.to_csv(args.output / "log_eval_per_case.csv", index=False)
    summary.to_csv(args.output / "log_eval_summary.csv", index=False)
    pd.DataFrame(skipped, columns=["dataset", "batch", "case", "reason"]).to_csv(
        args.output / "log_eval_skipped_cases.csv", index=False
    )
    (args.output / "log_eval_summary.md").write_text(render_summary(summary, args.mode))
    provenance = {
        "mode": args.mode,
        "manifest_index": args.manifest_index.as_posix(),
        "sample_index": args.sample_index.as_posix() if args.sample_index else None,
        "rankings": args.rankings,
        "budgets": args.budgets,
        "cases": int(per_case[["dataset", "batch", "case"]].drop_duplicates().shape[0]) if not per_case.empty else 0,
        "skipped": len(skipped),
    }
    (args.output / "log_eval_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Wrote log evaluation to {args.output}")
    if skipped:
        print(f"Skipped {len(skipped)} cases; see log_eval_skipped_cases.csv")


if __name__ == "__main__":
    main()
