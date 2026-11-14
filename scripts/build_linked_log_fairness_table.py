"""Build RQ1 linked-log fairness evidence for trace samplers.

This script scores logs retained indirectly by trace samplers: for each
retained TraceID at a trace budget, all v0.2 log-manifest rows with that
TraceID are treated as retained logs. It reuses the frozen v0.2 log metrics
from scripts/log_protocol_eval.py without modifying shared evaluator code.
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
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import evaluate_trace_sampling as trace_eval
import log_protocol_eval as log_eval


BUDGETS = [0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.10, 0.20]
DATASETS = ["hipster", "train_ticket"]
FOCUS_METRICS = [
    "linked_A_exposed_recall",
    "log_manifested_recall",
    "log_template_unseen_recall",
    "log_template_rare_le_0p1_recall",
    "template_coverage",
    "template_jsd",
]
ANCHOR_RANKINGS = {
    "combined": "UniSage-v1-native-log",
    "uniform": "Uniform-log",
}


@dataclass(frozen=True)
class MethodSpec:
    method: str
    glob_pattern: str
    decision_prefix: str
    seed_regex: str
    require_name_contains: str = ""


METHOD_SPECS = [
    MethodSpec(
        method="Sifter-linked-log",
        glob_pattern="sifter_v02_hipster_train_ticket_primary_seed-*/sample_index.csv",
        decision_prefix="SifterDecision",
        seed_regex=r"seed-(\d+)",
    ),
    MethodSpec(
        method="Sieve-linked-log",
        glob_pattern="sieve_v02_hipster_train_ticket_primary_seed-*/sample_index.csv",
        decision_prefix="SieveDecision",
        seed_regex=r"seed-(\d+)",
    ),
    MethodSpec(
        method="TraStrainer-linked-log",
        glob_pattern="trastrainer_v02_hipster_train_ticket_primary_seed-*/sample_index.csv",
        decision_prefix="TraStrainerDecision",
        seed_regex=r"seed-(\d+)$",
    ),
    MethodSpec(
        method="TracePicker-linked-log",
        glob_pattern="tracepicker_v02_hip_tt_primary_portable_seed*/sample_index.csv",
        decision_prefix="TracePickerDecision",
        seed_regex=r"seed(\d+)$",
    ),
    MethodSpec(
        method="Gleaner-linked-log",
        glob_pattern="gleaner_v02_hipster_train_ticket_primary_seed-*/sample_index.csv",
        decision_prefix="GleanerDecision",
        seed_regex=r"seed-(\d+)",
        require_name_contains="budgets-0p001",
    ),
]


def budget_token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def stable_file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) != 0
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "t", "yes", "y"})


def fmt_float(value: object, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "nan"
    return f"{float(value):.{digits}f}"


def method_slug(method: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", method).strip("_").lower()


def load_index(path: Path, datasets: Iterable[str]) -> pd.DataFrame:
    index = pd.read_csv(path)
    out = index[index["dataset"].isin(list(datasets))].copy()
    return out.reset_index(drop=True)


def manifest_lookup(index: pd.DataFrame) -> Dict[Tuple[str, str, str], str]:
    return {
        (str(row.dataset), str(row.batch), str(row.case)): str(row.manifest_path)
        for row in index.itertuples(index=False)
    }


def source_path(root: Path, path_value: object) -> Path:
    path = Path(str(path_value))
    return path if path.is_absolute() else root / path


def discover_trace_sampler_runs(root: Path, log_index: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    runs_root = root / "results" / "v02_sampling_runs"
    wanted_keys = set(manifest_lookup(log_index))
    inventory_rows: List[dict] = []
    skipped_rows: List[dict] = []
    seen: set[Tuple[str, str, str, str, str]] = set()

    for spec in METHOD_SPECS:
        for index_path in sorted(runs_root.glob(spec.glob_pattern)):
            run_dir = index_path.parent
            if spec.require_name_contains and spec.require_name_contains not in run_dir.name:
                skipped_rows.append({
                    "method": spec.method,
                    "run_id": run_dir.name,
                    "reason": f"missing_name_token_{spec.require_name_contains}",
                })
                continue
            match = re.search(spec.seed_regex, run_dir.name)
            if not match:
                skipped_rows.append({
                    "method": spec.method,
                    "run_id": run_dir.name,
                    "reason": "seed_not_matched_or_nonfinal_name",
                })
                continue
            seed = match.group(1)
            try:
                sample_index = pd.read_csv(index_path)
            except Exception as exc:  # pragma: no cover - provenance path
                skipped_rows.append({
                    "method": spec.method,
                    "run_id": run_dir.name,
                    "reason": f"read_failed:{exc}",
                })
                continue
            if "status" in sample_index.columns:
                failed = sample_index[sample_index["status"] != "ok"].copy()
                for row in failed.itertuples(index=False):
                    skipped_rows.append({
                        "method": spec.method,
                        "run_id": run_dir.name,
                        "dataset": getattr(row, "dataset", ""),
                        "batch": getattr(row, "batch", ""),
                        "case": getattr(row, "case", ""),
                        "reason": f"sample_index_status_{getattr(row, 'status', '')}",
                    })
                sample_index = sample_index[sample_index["status"] == "ok"].copy()
            if "trace_label_scope" in sample_index.columns:
                sample_index = sample_index[sample_index["trace_label_scope"] == "primary_trace_label_scope"].copy()
            sample_index = sample_index[sample_index["dataset"].isin(DATASETS)].copy()
            for row in sample_index.itertuples(index=False):
                key = (str(row.dataset), str(row.batch), str(row.case))
                if key not in wanted_keys:
                    continue
                unique_key = (spec.method, seed, *key)
                if unique_key in seen:
                    skipped_rows.append({
                        "method": spec.method,
                        "seed": seed,
                        "run_id": run_dir.name,
                        "dataset": key[0],
                        "batch": key[1],
                        "case": key[2],
                        "reason": "duplicate_method_seed_case",
                    })
                    continue
                sample_path = source_path(root, getattr(row, "sample_path", ""))
                if not sample_path.exists():
                    skipped_rows.append({
                        "method": spec.method,
                        "seed": seed,
                        "run_id": run_dir.name,
                        "dataset": key[0],
                        "batch": key[1],
                        "case": key[2],
                        "reason": "missing_sample_path",
                        "sample_path": sample_path.as_posix(),
                    })
                    continue
                seen.add(unique_key)
                inventory_rows.append({
                    "method": spec.method,
                    "source_type": "trace_linked_sampler",
                    "budget_semantics": "trace_budget_linked_logs",
                    "seed": seed,
                    "repetition": np.nan,
                    "run_id": f"seed-{seed}",
                    "source_run": run_dir.name,
                    "sample_index_path": index_path.as_posix(),
                    "dataset": key[0],
                    "batch": key[1],
                    "case": key[2],
                    "sample_path": sample_path.as_posix(),
                    "sample_sha256": getattr(row, "sample_sha256", ""),
                    "decision_prefix": spec.decision_prefix,
                })

    return pd.DataFrame(inventory_rows), pd.DataFrame(skipped_rows)


def evaluate_linked_trace_sample(
    root: Path,
    inventory: pd.DataFrame,
    log_paths: Dict[Tuple[str, str, str], str],
    budgets: List[float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[dict] = []
    skipped: List[dict] = []
    manifest_cache: Dict[Tuple[str, str, str], pd.DataFrame] = {}

    for entry in inventory.itertuples(index=False):
        key = (str(entry.dataset), str(entry.batch), str(entry.case))
        if key not in manifest_cache:
            manifest_cache[key] = pd.read_csv(source_path(root, log_paths[key]))
        manifest = manifest_cache[key]
        try:
            sample = pd.read_csv(entry.sample_path)
        except Exception as exc:
            skipped.append({
                "method": entry.method,
                "run_id": entry.run_id,
                "dataset": entry.dataset,
                "batch": entry.batch,
                "case": entry.case,
                "reason": f"sample_read_failed:{exc}",
            })
            continue
        if "TraceID" not in sample.columns:
            skipped.append({
                "method": entry.method,
                "run_id": entry.run_id,
                "dataset": entry.dataset,
                "batch": entry.batch,
                "case": entry.case,
                "reason": "sample_missing_TraceID",
            })
            continue
        sample = sample.drop_duplicates("TraceID").copy()
        sample["TraceID"] = sample["TraceID"].astype(str)
        ranked_trace_ids = set(sample["TraceID"])
        ranked_logs = manifest[manifest["TraceID"].astype(str).isin(ranked_trace_ids)]
        for budget in budgets:
            column = f"{entry.decision_prefix}_{budget_token(budget)}"
            if column not in sample.columns:
                skipped.append({
                    "method": entry.method,
                    "run_id": entry.run_id,
                    "dataset": entry.dataset,
                    "batch": entry.batch,
                    "case": entry.case,
                    "budget": budget,
                    "reason": f"missing_decision_column_{column}",
                })
                continue
            kept_trace_ids = set(sample.loc[as_bool_series(sample[column]), "TraceID"])
            kept = manifest[manifest["TraceID"].astype(str).isin(kept_trace_ids)].copy()
            metric_row = log_eval.build_metric_row(
                manifest,
                kept,
                ranking=entry.method,
                budget=budget,
                n_ranked=len(ranked_logs),
            )
            metric_row.update({
                "dataset": entry.dataset,
                "batch": entry.batch,
                "case": entry.case,
                "method": entry.method,
                "source_type": entry.source_type,
                "budget_semantics": entry.budget_semantics,
                "seed": entry.seed,
                "repetition": entry.repetition,
                "run_id": entry.run_id,
                "source_run": entry.source_run,
                "sample_path": entry.sample_path,
                "sample_sha256": entry.sample_sha256,
                "n_trace_rows": len(sample),
                "n_trace_ranked": len(ranked_trace_ids),
                "n_trace_kept": len(kept_trace_ids),
                "n_ranked_logs_by_trace_link": len(ranked_logs),
            })
            rows.append(metric_row)

    return pd.DataFrame(rows), pd.DataFrame(skipped)


def evaluate_uniform_trace_linked(
    root: Path,
    trace_index: pd.DataFrame,
    log_paths: Dict[Tuple[str, str, str], str],
    budgets: List[float],
) -> pd.DataFrame:
    rows: List[dict] = []
    trace_index = trace_index[
        trace_index["dataset"].isin(DATASETS)
        & (trace_index["trace_label_scope"] == "primary_trace_label_scope")
    ].copy()
    for entry in trace_index.itertuples(index=False):
        key = (str(entry.dataset), str(entry.batch), str(entry.case))
        if key not in log_paths:
            continue
        trace_manifest = pd.read_csv(source_path(root, entry.manifest_path))
        log_manifest = pd.read_csv(source_path(root, log_paths[key]))
        ranked = trace_manifest[["TraceID"]].drop_duplicates().copy()
        ranked["TraceID"] = ranked["TraceID"].astype(str)
        ranked["_rank_score"] = ranked["TraceID"].map(trace_eval.stable_uniform_score)
        ranked = ranked.sort_values(["_rank_score", "TraceID"], ascending=[False, True]).reset_index(drop=True)
        ranked_trace_ids = set(ranked["TraceID"])
        ranked_logs = log_manifest[log_manifest["TraceID"].astype(str).isin(ranked_trace_ids)]
        for budget in budgets:
            k = min(len(ranked), max(1, int(round(len(trace_manifest) * budget)))) if len(ranked) else 0
            kept_trace_ids = set(ranked.head(k)["TraceID"])
            kept_logs = log_manifest[log_manifest["TraceID"].astype(str).isin(kept_trace_ids)].copy()
            metric_row = log_eval.build_metric_row(
                log_manifest,
                kept_logs,
                ranking="Uniform-trace-linked-log",
                budget=budget,
                n_ranked=len(ranked_logs),
            )
            metric_row.update({
                "dataset": entry.dataset,
                "batch": entry.batch,
                "case": entry.case,
                "method": "Uniform-trace-linked-log",
                "source_type": "trace_linked_sampler",
                "budget_semantics": "trace_budget_linked_logs",
                "seed": "seed_invariant",
                "repetition": np.nan,
                "run_id": "seed_invariant",
                "source_run": "v0.2_trace_manifest_stable_uniform",
                "sample_path": entry.manifest_path,
                "sample_sha256": entry.manifest_sha256,
                "n_trace_rows": len(trace_manifest),
                "n_trace_ranked": len(ranked_trace_ids),
                "n_trace_kept": len(kept_trace_ids),
                "n_ranked_logs_by_trace_link": len(ranked_logs),
            })
            rows.append(metric_row)
    return pd.DataFrame(rows)


def evaluate_unisage_trace_linked(
    root: Path,
    trace_index: pd.DataFrame,
    log_paths: Dict[Tuple[str, str, str], str],
    budgets: List[float],
    unisage_samples_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[dict] = []
    skipped: List[dict] = []
    trace_index = trace_index[
        trace_index["dataset"].isin(DATASETS)
        & (trace_index["trace_label_scope"] == "primary_trace_label_scope")
    ].copy()
    for entry in trace_index.itertuples(index=False):
        key = (str(entry.dataset), str(entry.batch), str(entry.case))
        if key not in log_paths:
            continue
        sample_file = unisage_samples_dir / f"{key[0]}__{key[1]}__{key[2]}.trace_sampling.csv.gz"
        if not sample_file.exists():
            skipped.append({
                "method": "UniSage-trace-linked-log",
                "run_id": "seed_invariant",
                "dataset": key[0],
                "batch": key[1],
                "case": key[2],
                "reason": "missing_unisage_sample_file",
                "sample_path": sample_file.as_posix(),
            })
            continue
        unisage_sample = pd.read_csv(sample_file)
        unisage_sample["TraceID"] = unisage_sample["TraceID"].astype(str)
        unisage_map = (
            unisage_sample.drop_duplicates("TraceID")
            .set_index("TraceID")["CombinedSamplingProb"]
            .to_dict()
        )
        trace_manifest = pd.read_csv(source_path(root, entry.manifest_path))
        log_manifest = pd.read_csv(source_path(root, log_paths[key]))
        ranked = trace_manifest[["TraceID"]].drop_duplicates().copy()
        ranked["TraceID"] = ranked["TraceID"].astype(str)
        ranked["_rank_score"] = ranked["TraceID"].map(unisage_map).fillna(0.0)
        ranked = ranked.sort_values(["_rank_score", "TraceID"], ascending=[False, True]).reset_index(drop=True)
        ranked_trace_ids = set(ranked["TraceID"])
        ranked_logs = log_manifest[log_manifest["TraceID"].astype(str).isin(ranked_trace_ids)]
        for budget in budgets:
            k = min(len(ranked), max(1, int(round(len(trace_manifest) * budget)))) if len(ranked) else 0
            kept_trace_ids = set(ranked.head(k)["TraceID"])
            kept_logs = log_manifest[log_manifest["TraceID"].astype(str).isin(kept_trace_ids)].copy()
            metric_row = log_eval.build_metric_row(
                log_manifest,
                kept_logs,
                ranking="UniSage-trace-linked-log",
                budget=budget,
                n_ranked=len(ranked_logs),
            )
            metric_row.update({
                "dataset": entry.dataset,
                "batch": entry.batch,
                "case": entry.case,
                "method": "UniSage-trace-linked-log",
                "source_type": "trace_linked_sampler",
                "budget_semantics": "trace_budget_linked_logs",
                "seed": "seed_invariant",
                "repetition": np.nan,
                "run_id": "seed_invariant",
                "source_run": "unisage_trace_dynamic_pattern_guard_seed1",
                "sample_path": sample_file.as_posix(),
                "sample_sha256": entry.manifest_sha256,
                "n_trace_rows": len(trace_manifest),
                "n_trace_ranked": len(ranked_trace_ids),
                "n_trace_kept": len(kept_trace_ids),
                "n_ranked_logs_by_trace_link": len(ranked_logs),
            })
            rows.append(metric_row)
    return pd.DataFrame(rows), pd.DataFrame(skipped)


def load_direct_log_anchors(root: Path, budgets: List[float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[pd.DataFrame] = []
    skipped: List[dict] = []
    for rep in range(1, 6):
        path = root / "results" / "v02_sampling_runs" / f"unisage_v1_rep{rep}_budget8" / "log_eval_clean" / "log_eval_per_case.csv"
        if not path.exists():
            skipped.append({
                "method": "direct_log_anchor",
                "run_id": f"rep-{rep}",
                "reason": "missing_anchor_per_case",
                "path": path.as_posix(),
            })
            continue
        frame = pd.read_csv(path)
        frame = frame[
            frame["dataset"].isin(DATASETS)
            & frame["ranking"].isin(ANCHOR_RANKINGS)
            & frame["budget"].isin(budgets)
        ].copy()
        frame["method"] = frame["ranking"].map(ANCHOR_RANKINGS)
        frame["source_type"] = "direct_log_anchor"
        frame["budget_semantics"] = "log_budget"
        frame["seed"] = np.nan
        frame["repetition"] = rep
        frame["run_id"] = f"rep-{rep}"
        frame["source_run"] = path.parent.parent.name
        frame["sample_path"] = path.as_posix()
        frame["sample_sha256"] = stable_file_hash(path)
        frame["n_trace_rows"] = np.nan
        frame["n_trace_ranked"] = np.nan
        frame["n_trace_kept"] = np.nan
        frame["n_ranked_logs_by_trace_link"] = np.nan
        rows.append(frame)
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(), pd.DataFrame(skipped))


def numeric_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else np.nan


def numeric_std(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.std(ddof=1)) if len(values) > 1 else np.nan


def aggregate_by_run(per_case: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    group_cols = [
        "dataset",
        "method",
        "budget",
        "run_id",
        "seed",
        "repetition",
        "source_type",
        "budget_semantics",
    ]
    for keys, group in per_case.groupby(group_cols, dropna=False, sort=True):
        row = dict(zip(group_cols, keys))
        row["cases"] = int(group[["dataset", "batch", "case"]].drop_duplicates().shape[0])
        row["n_total_logs_mean"] = numeric_mean(group["n_total"])
        row["n_kept_logs_mean"] = numeric_mean(group["n_kept"])
        if "n_trace_kept" in group.columns:
            row["n_trace_kept_mean"] = numeric_mean(group["n_trace_kept"])
        for metric in FOCUS_METRICS:
            row[metric] = numeric_mean(group[metric]) if metric in group.columns else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_over_runs(by_run: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    group_cols = ["dataset", "method", "budget", "source_type", "budget_semantics"]
    for keys, group in by_run.groupby(group_cols, sort=True):
        row = dict(zip(group_cols, keys))
        row["n_runs"] = int(group["run_id"].nunique())
        row["run_ids"] = ",".join(map(str, sorted(group["run_id"].astype(str).unique())))
        row["cases_min"] = int(group["cases"].min()) if not group.empty else 0
        row["cases_max"] = int(group["cases"].max()) if not group.empty else 0
        row["seed_invariant"] = bool(row["n_runs"] == 1 and "seed_invariant" in row["run_ids"])
        for metric in ["n_total_logs_mean", "n_kept_logs_mean", "n_trace_kept_mean", *FOCUS_METRICS]:
            if metric not in group.columns:
                continue
            row[f"{metric}_mean"] = numeric_mean(group[metric])
            row[f"{metric}_std"] = numeric_std(group[metric])
            row[f"{metric}_n"] = int(pd.to_numeric(group[metric], errors="coerce").dropna().shape[0])
        rows.append(row)
    return pd.DataFrame(rows)


def copy_anchor_aggregate_source(root: Path, output: Path) -> None:
    source = root / "results" / "v02_sampling_runs" / "unisage_v1_budget8_5rep_aggregate_log_hip_tt_primary" / "log_eval_5rep.csv"
    if not source.exists():
        return
    frame = pd.read_csv(source)
    frame = frame[frame["ranking"].isin(ANCHOR_RANKINGS)].copy()
    frame["method"] = frame["ranking"].map(ANCHOR_RANKINGS)
    frame.to_csv(output / "direct_log_anchor_existing_aggregate_source.csv", index=False)


def write_evidence_by_run(per_case: pd.DataFrame, output: Path) -> None:
    evidence_root = output / "evidence_by_run"
    for (method, run_id), group in per_case.groupby(["method", "run_id"], sort=True):
        path = evidence_root / method_slug(str(method)) / f"{method_slug(str(run_id))}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        group.to_csv(path, index=False)


def write_csv_with_columns(frame: pd.DataFrame, path: Path, columns: List[str]) -> None:
    if frame.empty:
        pd.DataFrame(columns=columns).to_csv(path, index=False)
    else:
        frame.to_csv(path, index=False)


def seed_availability(by_run: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    budget_slice = by_run[np.isclose(by_run["budget"].astype(float), 0.05)].copy()
    for (method, dataset), group in budget_slice.groupby(["method", "dataset"], sort=True):
        rows.append({
            "method": method,
            "dataset": dataset,
            "runs": ",".join(map(str, sorted(group["run_id"].astype(str).unique()))),
            "n_runs": int(group["run_id"].nunique()),
            "cases_min": int(group["cases"].min()),
            "cases_max": int(group["cases"].max()),
            "budget_semantics": group["budget_semantics"].iloc[0],
            "source_type": group["source_type"].iloc[0],
        })
    return pd.DataFrame(rows)


def expected_seed_note(availability: pd.DataFrame) -> List[str]:
    lines: List[str] = []
    for method in sorted(availability["method"].unique()):
        sub = availability[availability["method"] == method]
        runs = sorted(set(",".join(sub["runs"].astype(str)).split(",")))
        if runs == ["seed_invariant"]:
            note = "seed-invariant"
        else:
            note = ", ".join(runs)
        case_desc = "; ".join(
            f"{row.dataset}: {row.cases_min}-{row.cases_max} cases"
            for row in sub.itertuples(index=False)
        )
        lines.append(f"- {method}: {note}; {case_desc}.")
    return lines


def render_headline_table(aggregate: pd.DataFrame) -> str:
    focus = aggregate[np.isclose(aggregate["budget"].astype(float), 0.05)].copy()
    if focus.empty:
        return "No 5% rows were produced.\n"
    preferred = [
        "UniSage-v1-native-log",
        "Uniform-log",
        "Uniform-trace-linked-log",
        "Sieve-linked-log",
        "Sifter-linked-log",
        "TraStrainer-linked-log",
        "TracePicker-linked-log",
        "Gleaner-linked-log",
    ]
    focus["method_order"] = focus["method"].map({m: i for i, m in enumerate(preferred)}).fillna(99)
    focus = focus.sort_values(["dataset", "method_order", "method"])
    lines = [
        "| dataset | method | runs | budget semantics | linked_A recall | strict-log recall | unseen recall | rare<=0.1% recall | template cov. | template JSD |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in focus.itertuples(index=False):
        def cell(metric: str) -> str:
            mean = getattr(row, f"{metric}_mean", np.nan)
            std = getattr(row, f"{metric}_std", np.nan)
            if (
                pd.isna(std)
                or getattr(row, "seed_invariant", False)
                or math.isclose(float(std), 0.0, abs_tol=1e-12)
            ):
                return fmt_float(mean)
            return f"{fmt_float(mean)} +/- {fmt_float(std)}"

        lines.append(
            f"| {row.dataset} | {row.method} | {row.n_runs} | {row.budget_semantics} | "
            f"{cell('linked_A_exposed_recall')} | {cell('log_manifested_recall')} | "
            f"{cell('log_template_unseen_recall')} | {cell('log_template_rare_le_0p1_recall')} | "
            f"{cell('template_coverage')} | {cell('template_jsd')} |"
        )
    return "\n".join(lines) + "\n"


def missing_coverage_notes(availability: pd.DataFrame, expected_cases: Dict[str, int]) -> List[str]:
    expected_seeds = [f"seed-{seed}" for seed in range(1, 6)]
    trace_methods = [
        "Sieve-linked-log",
        "Sifter-linked-log",
        "TraStrainer-linked-log",
        "TracePicker-linked-log",
        "Gleaner-linked-log",
    ]
    lines: List[str] = []
    for method in trace_methods:
        for dataset in DATASETS:
            sub = availability[(availability["method"] == method) & (availability["dataset"] == dataset)]
            expected = expected_cases.get(dataset)
            if sub.empty:
                lines.append(f"- {method}: missing {dataset} entirely; expected {expected} primary cases.")
                continue
            row = sub.iloc[0]
            runs = set(str(row["runs"]).split(","))
            missing = [seed for seed in expected_seeds if seed not in runs]
            case_max = int(row["cases_max"])
            details = []
            if missing:
                details.append(f"missing seeds {', '.join(missing)}")
            if expected is not None and case_max < expected:
                details.append(f"case coverage {case_max}/{expected}")
            if details:
                lines.append(f"- {method} {dataset}: {'; '.join(details)}.")
    return lines or ["- No missing seed/case coverage detected for the expected Hipster+Train Ticket trace-sampler matrix."]


def render_summary(
    output: Path,
    per_case: pd.DataFrame,
    by_run: pd.DataFrame,
    aggregate: pd.DataFrame,
    availability: pd.DataFrame,
    skipped: pd.DataFrame,
    anchor_skipped: pd.DataFrame,
    expected_cases: Dict[str, int],
) -> str:
    coverage_notes = missing_coverage_notes(availability, expected_cases)
    has_missing_coverage = not (
        len(coverage_notes) == 1
        and "No missing seed/case coverage" in coverage_notes[0]
    )
    caveats = [
        "- AIOps21 and other long-tail baseline seeds are outside this Prompt A scope and are not included in this companion Hipster+Train Ticket fairness table.",
        "- Linked trace-sampler rows can retain more or fewer than the nominal log fraction because the budget is applied to traces first and logs are pulled by TraceID.",
    ]
    if has_missing_coverage and any("Sieve-linked-log" in note for note in coverage_notes):
        caveats.insert(
            0,
            "- Sieve-linked-log uses only the currently visible Hipster+Train Ticket primary seed/case matrix; see missing coverage details above.",
        )
    lines = [
        "# RQ1 Linked-Log Fairness Summary",
        "",
        "Scope: Hipster and Train Ticket primary v0.2 cases only.",
        "",
        "Metric semantics: trace-sampler rows retain logs by TraceID linkage from the sampler's retained traces; direct-log anchors use the original log budget. The frozen v0.2 log manifest has TraceID-level log links, so this table does not apply a SpanID-level filter.",
        "",
        f"Per-case rows: {len(per_case)}. By-run rows: {len(by_run)}. Aggregate rows: {len(aggregate)}.",
        "",
        "## 5% Headline",
        "",
        render_headline_table(aggregate).rstrip(),
        "",
        "## Seed and Run Availability",
        "",
        *expected_seed_note(availability),
        "",
        "Seed-invariant methods are reported as a single deterministic run; their aggregate std is left blank/NaN instead of written as fake +/- 0.000. UniSage-v1-native-log and Uniform-log are direct log-budget anchors from the existing 5-repetition clean log evaluation.",
        "",
        "## Missing Coverage Details",
        "",
        *coverage_notes,
        "",
        "## Caveats",
        "",
        *caveats,
        "",
        "## Output Files",
        "",
        f"- `{(output / 'linked_log_fairness_per_case.csv').as_posix()}`",
        f"- `{(output / 'linked_log_fairness_by_run.csv').as_posix()}`",
        f"- `{(output / 'linked_log_fairness_aggregate.csv').as_posix()}`",
        f"- `{(output / 'source_run_inventory.csv').as_posix()}`",
        f"- `{(output / 'direct_log_anchor_existing_aggregate_source.csv').as_posix()}`",
        f"- `{(output / 'evidence_by_run').as_posix()}/`",
        "",
    ]
    if not skipped.empty or not anchor_skipped.empty:
        lines.extend([
            "## Skipped Inputs",
            "",
        ])
        if not skipped.empty:
            reason_counts = skipped["reason"].value_counts().to_dict()
            lines.append(f"- Trace-linked skipped inputs: {reason_counts}.")
        if not anchor_skipped.empty:
            reason_counts = anchor_skipped["reason"].value_counts().to_dict()
            lines.append(f"- Direct-log anchor skipped inputs: {reason_counts}.")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("results/rq1_linked_log_fairness"))
    parser.add_argument("--log-manifest-index", type=Path, default=Path("results/label_protocol_v02/log_manifest_index.csv"))
    parser.add_argument("--trace-manifest-index", type=Path, default=Path("results/label_protocol_v02/trace_manifest_index.csv"))
    parser.add_argument(
        "--unisage-samples-dir",
        type=Path,
        default=Path("results/v02_sampling_runs/unisage_trace_hipster_train_ticket_primary_dynamic_pattern_guard_seed1/samples"),
    )
    parser.add_argument("--budgets", nargs="+", type=float, default=BUDGETS)
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)

    budgets = [float(value) for value in args.budgets]
    log_index = load_index(source_path(root, args.log_manifest_index), DATASETS)
    log_paths = manifest_lookup(log_index)
    trace_index = load_index(source_path(root, args.trace_manifest_index), DATASETS)
    expected_cases = {
        str(dataset): int(count)
        for dataset, count in log_index.groupby("dataset")[["batch", "case"]].size().items()
    }

    print("[linked-log] discovering trace sampler sample indexes", flush=True)
    inventory, discovery_skipped = discover_trace_sampler_runs(root, log_index)
    print(f"[linked-log] discovered {len(inventory)} method/case sample rows", flush=True)

    print("[linked-log] evaluating trace-linked logs", flush=True)
    linked_case, linked_skipped = evaluate_linked_trace_sample(root, inventory, log_paths, budgets)

    print("[linked-log] evaluating seed-invariant uniform trace-linked logs", flush=True)
    uniform_case = evaluate_uniform_trace_linked(root, trace_index, log_paths, budgets)

    print("[linked-log] evaluating seed-invariant UniSage trace-linked logs", flush=True)
    unisage_samples_dir = source_path(root, args.unisage_samples_dir)
    unisage_case, unisage_skipped = evaluate_unisage_trace_linked(
        root, trace_index, log_paths, budgets, unisage_samples_dir
    )
    print(f"[linked-log] UniSage trace-linked cases: {unisage_case[['dataset','batch','case']].drop_duplicates().shape[0] if not unisage_case.empty else 0}", flush=True)

    print("[linked-log] loading direct log anchors", flush=True)
    anchor_case, anchor_skipped = load_direct_log_anchors(root, budgets)

    per_case = pd.concat([linked_case, uniform_case, unisage_case, anchor_case], ignore_index=True, sort=False)
    by_run = aggregate_by_run(per_case)
    aggregate = aggregate_over_runs(by_run)
    availability = seed_availability(by_run)
    skipped = pd.concat([discovery_skipped, linked_skipped, unisage_skipped], ignore_index=True, sort=False)

    inventory.to_csv(output / "source_run_inventory.csv", index=False)
    write_csv_with_columns(skipped, output / "linked_log_fairness_skipped_inputs.csv", ["method", "run_id", "dataset", "batch", "case", "budget", "reason", "sample_path"])
    write_csv_with_columns(anchor_skipped, output / "direct_log_anchor_skipped_inputs.csv", ["method", "run_id", "reason", "path"])
    per_case.to_csv(output / "linked_log_fairness_per_case.csv", index=False)
    by_run.to_csv(output / "linked_log_fairness_by_run.csv", index=False)
    aggregate.to_csv(output / "linked_log_fairness_aggregate.csv", index=False)
    availability.to_csv(output / "seed_availability.csv", index=False)
    copy_anchor_aggregate_source(root, output)
    write_evidence_by_run(per_case, output)

    provenance = {
        "scope": {"datasets": DATASETS, "trace_label_scope": "primary_trace_label_scope"},
        "budgets": budgets,
        "log_manifest_index": source_path(root, args.log_manifest_index).as_posix(),
        "trace_manifest_index": source_path(root, args.trace_manifest_index).as_posix(),
        "trace_link_definition": "retain all log manifest rows whose TraceID is retained by the trace sampler",
        "metrics": FOCUS_METRICS,
        "rows": {
            "inventory": int(len(inventory)),
            "per_case": int(len(per_case)),
            "by_run": int(len(by_run)),
            "aggregate": int(len(aggregate)),
            "skipped": int(len(skipped) + len(anchor_skipped)),
        },
    }
    (output / "linked_log_fairness_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    summary = render_summary(output, per_case, by_run, aggregate, availability, skipped, anchor_skipped, expected_cases)
    (output / "linked_log_fairness_summary.md").write_text(summary)
    print(f"[linked-log] wrote outputs to {output}", flush=True)


if __name__ == "__main__":
    main()
