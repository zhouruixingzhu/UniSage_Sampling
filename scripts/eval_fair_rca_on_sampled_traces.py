"""Fair RCA on baseline-sampled traces (RevC-Q2 / S#6.1).

Isolates "paradigm win" from "implementation win": hold the RCA constant (UniSage's
trace RCL) and vary only WHICH traces each sampler retained at a budget. For each
(case, sampler, budget) we take the sampler's retained TraceID set, recompute the
abnormal trace aggregate from only those traces (the same self-latency / callee-
attribution / traffic-de-bias transforms via the real pipeline), run the SAME trace
RCL ranking against the full fault-free normal reference, and score localization
(AC@1 / AC@3 / MRR) against the injected service.

The RCA is identical across every sampler, so a localization gap is attributable to
the retained subset (the sampling paradigm), not to UniSage's own RCL implementation.

Trace-only RCL ranking is the primary result (sharpest isolation: log/metric stay out
of the loop, so they cannot mask trace-subset differences). A combined-ranking variant
is left as a follow-up.

Run from UniSage/ in the `unisage` conda env. Serial, one case at a time.

Example:
    python scripts/eval_fair_rca_on_sampled_traces.py \
        --sifter results/v02_sampling_runs/sifter_v02_hipster_train_ticket_primary_seed-1_FINAL \
        --sieve  results/v02_sampling_runs/sieve_v02_hipster_train_ticket_primary_seed-1_FINAL \
        --tracepicker results/v02_sampling_runs/tracepicker_v02_hip_tt_primary_portable_seed1_jsd \
        --unisage results/v02_sampling_runs/unisage_trace_hip_tt_primary_target_raw_prop0_v02 \
        --add-uniform --budgets 0.01 0.025 0.05 0.1 0.2 1.0 \
        --out results/fair_rca/hip_tt_primary_seed1
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import hashlib
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the EXACT pipeline RCA so it stays constant across samplers.
from AD.trace.trace_processing import normalize_trace_columns
from AD.utils.data_utils import TraceDurationCalculator
from RCL.trace_rcl import calculate_changes, sort_and_filter_data

# --- Retention semantics mirrored from scripts/evaluate_trace_sampling.py (keep in sync) ---
# We evaluate TWO retention modes so the fairness claim is robust to the N confound:
#
#   native  -- each baseline's ACTUAL retained subset. Sifter/Sieve/TraStrainer/
#              TracePicker use their per-budget Decision column; UniSage/uniform have
#              no decision column, so their native subset IS top-N by score. This is
#              faithful to what each real sampler stores, but N can differ across
#              samplers (e.g. Sifter's canonical-path quota under-fills: 125 vs a 160
#              target at 5%), so retention count is a confound.
#   topn    -- strict equal-N: every score-bearing sampler keeps the SAME
#              N = round(n_traces * budget) top-scoring traces (Sifter->SifterLoss,
#              Sieve->SieveScore, TraStrainer->TraStrainerCombined, UniSage->its probs,
#              uniform->stable hash). This removes the N confound but, for Sifter,
#              top-loss != Sifter's canonical-path selection. TracePicker has no score
#              and falls back to its Decision column in this mode too.
#
# A UniSage win in BOTH modes isolates the paradigm from both the RCA implementation
# (RCA is held constant) AND the retention count. topn matches evaluate_trace_sampling's
# primary rankings exactly, so its retained set == the diversity/coverage retained set.
RANKING_COLUMNS = {
    "unisage_combined": "CombinedSamplingProb",
    "unisage_edge": "EdgeCaseSamplingProb",
    "unisage_analysis": "AnomalySamplingProb",
    "sifter": "SifterLoss",
    "sieve": "SieveScore",
    "trastrainer": "TraStrainerCombined",
    "gleaner": "GleanerScore",
}
DECISION_PREFIXES = {
    "unisage_slot_gated_reserved": "SlotGatedReservedDecision",
    "unisage_dynamic_pattern_guard": "DynamicPatternGuardDecision",
    "sifter": "SifterDecision",
    "sieve": "SieveDecision",
    "trastrainer": "TraStrainerDecision",
    "tracepicker": "TracePickerDecision",
    "gleaner": "GleanerDecision",
}


UNIFORM_RANDOM_SCORE_COL = "_UniformRandomScore"


def stable_uniform_score(trace_id: object) -> float:
    digest = hashlib.sha256(str(trace_id).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)


def uniform_random_score(seed: int, dataset: str, batch: str, case: str, trace_id: object) -> float:
    material = f"{seed}\0{dataset}\0{batch}\0{case}\0{trace_id}".encode("utf-8")
    digest = hashlib.sha256(material).hexdigest()
    return int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)


def add_uniform_random_scores(
    sample: pd.DataFrame,
    *,
    seed: int | None,
    dataset: str,
    batch: str,
    case: str,
) -> pd.DataFrame:
    """Attach the RQ1-compatible seeded Uniform score when a seed is provided.

    Historical RCA smoke runs used the stable TraceID-only hash. Keeping that
    as the no-seed fallback preserves old command behavior, while formal RQ2
    runs can pass `--uniform-seed` to match `scripts/run_uniform_random.py`.
    """
    if seed is None:
        return sample
    out = sample.copy()
    out[UNIFORM_RANDOM_SCORE_COL] = out["TraceID"].map(
        lambda trace_id: uniform_random_score(seed, dataset, str(batch), str(case), trace_id)
    )
    return out


def budget_token(budget: float) -> str:
    return f"{budget:g}".replace("-", "m").replace(".", "p")


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "1.0", "yes"})


@contextmanager
def pushd(path: Path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def is_hipster_layout(dataset: str) -> bool:
    return dataset in ("hipster", "train_ticket")


# ----------------------------- ground truth -----------------------------------

def parse_fault_injection_toml(path: Path) -> dict:
    """case -> injected service, from aiops21 fault_injection.toml (array of tables).

    No tomllib/toml/tomli in the env; the file is a flat array of [[chaos_injection]]
    tables with `case = "..."` / `service = "..."`. Records without a `service`
    (trace-unobservable infra injections) are skipped — those are the v0.2 excluded
    scope and are evaluated separately, never here.
    """
    gt = {}
    case = service = None
    for raw in path.read_text().splitlines():
        s = raw.strip()
        if s.startswith("[[chaos_injection]]"):
            if case and service:
                gt[case] = service
            case = service = None
        elif s.startswith("case"):
            case = s.split("=", 1)[1].strip().strip('"')
        elif s.startswith("service"):
            service = s.split("=", 1)[1].strip().strip('"')
    if case and service:
        gt[case] = service
    return gt


class GroundTruth:
    """Lazily resolves the injected service per (dataset, case)."""

    def __init__(self, data_root: Path):
        self.data_root = data_root
        self._label_json: dict[Path, dict] = {}
        self._aiops: dict | None = None

    def inject_service(self, dataset: str, case: str, case_dir: Path) -> str | None:
        if is_hipster_layout(dataset):
            label_path = case_dir.parent / "label.json"
            if label_path not in self._label_json:
                import json

                self._label_json[label_path] = (
                    json.loads(label_path.read_text()) if label_path.exists() else {}
                )
            return self._label_json[label_path].get(case, {}).get("inject_pod")
        if self._aiops is None:
            toml_path = self.data_root / "data" / "aiops21" / "all" / "fault_injection.toml"
            self._aiops = parse_fault_injection_toml(toml_path) if toml_path.exists() else {}
        return self._aiops.get(case)


# ----------------------------- trace loading ----------------------------------

def load_spans(case_dir: Path, dataset: str, normal: bool) -> pd.DataFrame:
    if is_hipster_layout(dataset):
        if normal:
            f = case_dir.parent / "fault_free" / "trace" / "normal_trace.csv"
            df = pd.read_csv(f) if f.exists() else pd.DataFrame()
        else:
            tdir = case_dir / "trace"
            files = sorted(tdir.glob("*_trace.csv")) if tdir.is_dir() else []
            df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()
    else:
        sub = "normal" if normal else "abnormal"
        f = case_dir / sub / "traces.csv"
        df = pd.read_csv(f) if f.exists() else pd.DataFrame()
    if df.empty:
        return df
    df = normalize_trace_columns(df)
    df["TraceID"] = df["TraceID"].astype(str)
    return df


def rank_services(normal_rate: pd.DataFrame, abnormal_rate: pd.DataFrame, scratch: Path) -> list:
    """Run UniSage's trace RCL on a normal/abnormal aggregate pair -> ranked services.

    Mirrors RCL.trace_rcl.main: merge on (PodName, OperationName) with _normal/_abnormal
    suffixes (== read_and_merge_data), then calculate_changes + sort_and_filter_data
    (honors UNISAGE_RCA_ATTR=target). sort_and_filter_data writes rca_result/ in CWD, so
    it runs inside a throwaway scratch dir.
    """
    if normal_rate.empty or abnormal_rate.empty:
        return []
    # _calculate_abnormal returns (PodName, OperationName) as both a MultiIndex and
    # columns; the real pipeline round-trips through to_csv(index=False), so drop it.
    normal_rate = normal_rate.reset_index(drop=True)
    abnormal_rate = abnormal_rate.reset_index(drop=True)
    merged = pd.merge(
        normal_rate, abnormal_rate, on=["PodName", "OperationName"], suffixes=("_normal", "_abnormal")
    )
    if merged.empty:
        return []
    merged = calculate_changes(merged)
    scratch.mkdir(parents=True, exist_ok=True)
    with pushd(scratch):
        result = sort_and_filter_data(merged)
    return list(result["service_name"]) if not result.empty else []


# ----------------------------- retention --------------------------------------

def _topn_ids(sample: pd.DataFrame, sampler: str, budget: float) -> set | None:
    """Equal-N top-N by the sampler's priority score. None => no score column here."""
    n_total = len(sample)
    target_k = max(1, int(round(n_total * budget)))
    if sampler == "uniform":
        if UNIFORM_RANDOM_SCORE_COL in sample.columns:
            scored = sample.assign(_s=sample[UNIFORM_RANDOM_SCORE_COL].astype(float))
        else:
            scored = sample.assign(_s=sample["TraceID"].map(stable_uniform_score))
    else:
        col = RANKING_COLUMNS.get(sampler)
        if col is None or col not in sample.columns:
            return None
        scored = sample.dropna(subset=[col]).copy()
        scored["_s"] = scored[col].astype(float)
    scored["TraceID"] = scored["TraceID"].astype(str)
    k = min(len(scored), target_k)
    order = scored.sort_values(["_s", "TraceID"], ascending=[False, True])
    return set(order["TraceID"].head(k))


def _decision_ids(sample: pd.DataFrame, sampler: str, budget: float) -> set | None:
    """Native per-budget Decision subset. None => no decision column here."""
    prefix = DECISION_PREFIXES.get(sampler)
    if prefix is None:
        return None
    col = f"{prefix}_{budget_token(budget)}"
    if col not in sample.columns:
        return None
    return set(sample.loc[as_bool(sample[col]), "TraceID"].astype(str))


def retained_ids(sample: pd.DataFrame, sampler: str, budget: float, mode: str) -> set | None:
    """TraceIDs the sampler keeps at `budget` under retention `mode`.

    None => this sampler is not represented in this sample file. Callers must first
    project `sample` onto the canonical v0.2 trace-manifest universe. At budget >=
    1.0 every sampler then keeps ALL canonical traces (the upper-bound collapse:
    every sampler must yield the identical full-data ranking, the harness
    correctness check).

    native: baselines and budget-aware UniSage variants use their Decision column;
            score-only UniSage/uniform fall back to top-N. topn: every score-bearing sampler
            uses equal-N top-N; decision-only samplers fall back to their Decision column.
    """
    if len(sample) == 0:
        return set()
    if budget >= 1.0:
        return set(sample["TraceID"].astype(str))
    if mode == "native":
        ids = _decision_ids(sample, sampler, budget)
        return ids if ids is not None else _topn_ids(sample, sampler, budget)
    if mode == "topn":
        ids = _topn_ids(sample, sampler, budget)
        return ids if ids is not None else _decision_ids(sample, sampler, budget)
    raise ValueError(f"unknown retention mode: {mode}")


# ----------------------------- driver -----------------------------------------

def load_run_index(run_dir: Path) -> pd.DataFrame:
    idx = pd.read_csv(run_dir / "sample_index.csv")
    idx = idx[idx["status"] == "ok"].copy()
    idx["case_key"] = idx["dataset"] + "/" + idx["batch"].astype(str) + "/" + idx["case"].astype(str)
    return idx


def read_sample(sample_path: Path) -> pd.DataFrame:
    df = pd.read_csv(sample_path)  # pandas auto-handles .gz
    df["TraceID"] = df["TraceID"].astype(str)
    return df


MANIFEST_AUDIT_COLUMNS = [
    "dataset",
    "batch",
    "case",
    "run_name",
    "n_rows_before",
    "n_rows_after",
    "n_trace_ids_before",
    "n_trace_ids_after",
    "n_manifest_trace_ids",
    "n_trace_ids_dropped",
    "n_manifest_ids_missing_from_sample",
]


def _resolve_path(path: Path, data_root: Path) -> Path:
    if path.is_absolute() or path.exists():
        return path
    return data_root / path


def load_trace_manifest_ids(
    manifest_index: Path | None,
    data_root: Path,
    case_keys: set[str] | None = None,
) -> dict[str, set[str]]:
    """Load v0.2 canonical TraceID sets keyed as dataset/batch/case."""
    if manifest_index is None:
        return {}
    index_path = _resolve_path(manifest_index, data_root)
    if not index_path.exists():
        raise FileNotFoundError(f"trace manifest index not found: {index_path}")

    index = pd.read_csv(index_path)
    out: dict[str, set[str]] = {}
    for row in index.itertuples(index=False):
        ck = f"{row.dataset}/{str(row.batch)}/{str(row.case)}"
        if case_keys is not None and ck not in case_keys:
            continue
        manifest_path = _resolve_path(Path(row.manifest_path), data_root)
        manifest = pd.read_csv(manifest_path, usecols=["TraceID"])
        out[ck] = set(manifest["TraceID"].astype(str))
    return out


def filter_sample_to_manifest(
    sample: pd.DataFrame,
    canonical_ids: set[str],
    *,
    dataset: str,
    batch: str,
    case: str,
    run_name: str,
) -> tuple[pd.DataFrame, dict]:
    sample_ids = set(sample["TraceID"].astype(str))
    before_rows = len(sample)
    filtered = sample[sample["TraceID"].astype(str).isin(canonical_ids)].copy()
    filtered_ids = set(filtered["TraceID"].astype(str))
    audit = {
        "dataset": dataset,
        "batch": batch,
        "case": case,
        "run_name": run_name,
        "n_rows_before": before_rows,
        "n_rows_after": len(filtered),
        "n_trace_ids_before": len(sample_ids),
        "n_trace_ids_after": len(filtered_ids),
        "n_manifest_trace_ids": len(canonical_ids),
        "n_trace_ids_dropped": len(sample_ids - canonical_ids),
        "n_manifest_ids_missing_from_sample": len(canonical_ids - sample_ids),
    }
    return filtered, audit


def parse_manifest_index_arg(value: str) -> Path | None:
    if str(value).strip().lower() in {"", "none", "null", "off", "false"}:
        return None
    return Path(value)


def main():
    ap = argparse.ArgumentParser()
    for name in ("sifter", "sieve", "tracepicker", "trastrainer", "gleaner", "unisage"):
        ap.add_argument(f"--{name}", type=str, default=None, help=f"{name} v0.2 run dir")
    ap.add_argument("--add-uniform", action="store_true", help="add Uniform sampler")
    ap.add_argument(
        "--uniform-seed",
        type=int,
        default=None,
        help="seed for RQ1-compatible Uniform random scores; omit for legacy TraceID-only hash",
    )
    ap.add_argument("--retention", choices=["native", "topn", "both"], default="both",
                    help="native=each sampler's Decision subset; topn=equal-N top-N by score; both")
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.01, 0.025, 0.05, 0.1, 0.2, 1.0])
    ap.add_argument("--data-root", type=str, default=".")
    ap.add_argument(
        "--trace-manifest-index",
        type=parse_manifest_index_arg,
        default=Path("results/label_protocol_v02/trace_manifest_index.csv"),
        help="v0.2 canonical trace manifest index; set to none/off to disable filtering",
    )
    ap.add_argument("--dataset-filter", type=str, default=None, help="comma list, e.g. hipster,train_ticket")
    ap.add_argument("--cases", type=str, default=None, help="comma list of case names to keep (smoke)")
    ap.add_argument("--limit", type=int, default=None, help="cap number of cases (smoke)")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = out_dir / "_scratch"

    # run name -> {samplers: [...], sample_path per case_key}
    run_dirs = {n: getattr(args, n) for n in ("sifter", "sieve", "tracepicker", "trastrainer", "gleaner", "unisage")}
    run_dirs = {n: Path(p) for n, p in run_dirs.items() if p}
    if not run_dirs:
        ap.error("provide at least one run dir")

    # which logical samplers each run contributes
    run_samplers = {}
    for n in run_dirs:
        run_samplers[n] = (
            [
                "unisage_combined",
                "unisage_edge",
                "unisage_slot_gated_reserved",
                "unisage_dynamic_pattern_guard",
            ]
            if n == "unisage"
            else [n]
        )

    # index every run; build the case scope from the union of all runs (status==ok)
    indices = {n: load_run_index(d) for n, d in run_dirs.items()}
    case_meta: dict[str, dict] = {}  # case_key -> {dataset, batch, case, case_dir}
    sample_paths: dict[str, dict[str, Path]] = {}  # case_key -> {run_name: path}
    for n, idx in indices.items():
        for _, row in idx.iterrows():
            ck = row["case_key"]
            case_meta.setdefault(
                ck,
                dict(dataset=row["dataset"], batch=str(row["batch"]), case=str(row["case"]),
                     case_dir=Path(row["case_dir"])),
            )
            sample_paths.setdefault(ck, {})[n] = Path(row["sample_path"])

    case_keys = sorted(case_meta)
    if args.dataset_filter:
        keep = set(args.dataset_filter.split(","))
        case_keys = [ck for ck in case_keys if case_meta[ck]["dataset"] in keep]
    if args.cases:
        keep = set(args.cases.split(","))
        case_keys = [ck for ck in case_keys if case_meta[ck]["case"] in keep]
    if args.limit:
        case_keys = case_keys[: args.limit]

    manifest_ids_by_case = load_trace_manifest_ids(args.trace_manifest_index, data_root, set(case_keys))
    manifest_audit_rows: list[dict] = []

    modes = ["native", "topn"] if args.retention == "both" else [args.retention]
    gt_resolver = GroundTruth(data_root)
    per_case_rows = []

    for ci, ck in enumerate(case_keys, 1):
        meta = case_meta[ck]
        dataset, case, case_dir = meta["dataset"], meta["case"], data_root / meta["case_dir"]
        batch = meta["batch"]
        canonical_ids = manifest_ids_by_case.get(ck)
        gt = gt_resolver.inject_service(dataset, case, case_dir)
        if not gt:
            print(f"[{ci}/{len(case_keys)}] {ck}: NO_GT, skip")
            continue

        abn = load_spans(case_dir, dataset, normal=False)
        nrm = load_spans(case_dir, dataset, normal=True)
        if abn.empty or nrm.empty:
            print(f"[{ci}/{len(case_keys)}] {ck}: NO_TRACE_DATA, skip")
            continue
        normal_rate = TraceDurationCalculator._calculate_normal(nrm)

        # Distinct retained-ID sets often coincide across (sampler, mode) — e.g. Sieve's
        # decision == its top-N, and every sampler collapses to all-IDs at budget=1.0 —
        # so cache the RCA per ID set and compute the (costly) aggregate+rank once.
        rca_cache: dict[frozenset, list] = {}

        def rank_for(ids: set) -> list:
            key = frozenset(ids)
            if key not in rca_cache:
                sub = abn[abn["TraceID"].isin(ids)]
                if sub.empty:
                    rca_cache[key] = []
                else:
                    abnormal_rate = TraceDurationCalculator._calculate_abnormal(sub)
                    rca_cache[key] = rank_services(normal_rate, abnormal_rate, scratch)
            return rca_cache[key]

        print(f"[{ci}/{len(case_keys)}] {ck}  gt={gt}  spans_abn={len(abn)}")

        # (sampler, sample_df) tasks from the provided runs, plus uniform off any universe
        tasks: list[tuple[str, pd.DataFrame]] = []
        loaded: dict[str, pd.DataFrame] = {}
        for run_name, samplers in run_samplers.items():
            sp = sample_paths.get(ck, {}).get(run_name)
            if sp is None:
                continue
            sample = read_sample(data_root / sp)
            if canonical_ids is not None:
                sample, audit = filter_sample_to_manifest(
                    sample,
                    canonical_ids,
                    dataset=dataset,
                    batch=batch,
                    case=case,
                    run_name=run_name,
                )
                manifest_audit_rows.append(audit)
            loaded[run_name] = sample
            for sampler in samplers:
                tasks.append((sampler, sample))
        if args.add_uniform and loaded:
            uniform_sample = add_uniform_random_scores(
                next(iter(loaded.values())),
                seed=args.uniform_seed,
                dataset=dataset,
                batch=batch,
                case=case,
            )
            tasks.append(("uniform", uniform_sample))

        for sampler, sample in tasks:
            for mode in modes:
                for budget in args.budgets:
                    ids = retained_ids(sample, sampler, budget, mode)
                    if ids is None:
                        continue
                    ranked = rank_for(ids)
                    rank = (ranked.index(gt) + 1) if gt in ranked else 0
                    per_case_rows.append(dict(
                        dataset=dataset, batch=meta["batch"], case=case, sampler=sampler,
                        retention=mode, budget=budget, n_kept=len(ids), gt=gt, rank=rank,
                        top1=(ranked[0] if ranked else None),
                    ))

    if not per_case_rows:
        pd.DataFrame(manifest_audit_rows, columns=MANIFEST_AUDIT_COLUMNS).to_csv(
            out_dir / "fair_rca_manifest_filter_audit.csv", index=False
        )
        print("No rows produced.")
        return

    pd.DataFrame(manifest_audit_rows, columns=MANIFEST_AUDIT_COLUMNS).to_csv(
        out_dir / "fair_rca_manifest_filter_audit.csv", index=False
    )
    per_case = pd.DataFrame(per_case_rows)
    per_case.to_csv(out_dir / "fair_rca_per_case.csv", index=False)

    # aggregate AC@1 / AC@3 / MRR per (dataset, sampler, retention, budget)
    def agg(g):
        n = len(g)
        return pd.Series(dict(
            n_cases=n,
            n_kept_mean=g["n_kept"].mean(),
            ac1=(g["rank"] == 1).mean(),
            ac3=g["rank"].between(1, 3).mean(),
            mrr=g.loc[g["rank"] > 0, "rank"].rdiv(1.0).sum() / n if n else np.nan,
        ))

    summary = (
        per_case.groupby(["dataset", "sampler", "retention", "budget"], group_keys=False)[["rank", "n_kept"]]
        .apply(agg)
        .reset_index()
    )
    summary.to_csv(out_dir / "fair_rca_summary.csv", index=False)

    # plain-text tables (no tabulate dependency): one block per dataset/retention/metric
    lines = ["# Fair RCA on baseline-sampled traces (trace-only RCL held constant)\n"]
    for (dataset, retention), g in summary.groupby(["dataset", "retention"]):
        lines.append(f"\n## {dataset} — retention={retention}\n")
        for metric in ("ac1", "ac3", "mrr"):
            lines.append(f"\n### {metric.upper()} by budget (rows=sampler)\n")
            pivot = g.pivot_table(index="sampler", columns="budget", values=metric).round(3)
            lines.append("```\n" + pivot.to_string() + "\n```")
        lines.append("")
    (out_dir / "fair_rca_summary.md").write_text("\n".join(lines))

    print(f"\nWrote {out_dir}/fair_rca_per_case.csv, fair_rca_summary.csv, fair_rca_summary.md")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
