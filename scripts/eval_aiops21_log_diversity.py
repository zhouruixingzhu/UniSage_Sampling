"""Label-free log diversity for AIOps21 across sampling methods.

AIOps21 has NO line-level log fault labels (its original annotation procedure is
lost). Therefore we
report ONLY the three label-free template-diversity metrics that need nothing but
``template_hash``:

  * template_coverage = kept.template_hash.nunique() / base.template_hash.nunique()
  * template_entropy  = entropy(kept.template_hash)            (Shannon, bits)
  * template_jsd      = jensen_shannon_divergence(base, kept)  (template_hash dist.)

No problem-related metrics (A_exposed / strict-fault / unseen / rare recall) are
computed for AIOps21 -- those require fault labels that do not exist here.

Protocol mirrors the Hipster+Train-Ticket multi-budget linked-log fairness setup
(``scripts/build_linked_log_fairness_table.py``) and reuses
``log_protocol_eval.entropy`` / ``jensen_shannon_divergence`` verbatim and the
``log_protocol_materialize.stable_hash`` template-hash definition.

Methods
-------
Direct-log (rank logs directly, no trace join needed):
  * UniSage-native-product : kept = top-k logs by ``LogSamplingProbProduct``
  * UniSage-native-max     : kept = top-k logs by ``LogSamplingProbMax``
  * Uniform-log            : kept = top-k logs by a deterministic per-LogID hash

Trace-linked (retained TraceIDs -> kept logs by TraceID join):
  * UniSage-trace-linked    (CombinedSamplingProb, top-k traces)
  * Sieve/Gleaner/Sifter/TraStrainer/TracePicker-linked (per-budget decision cols)
  * Uniform-trace-linked    (random traces -> linked logs)

  >>> IMPORTANT AIOps21 DATA FACT <<<
  AIOps21 logs are NOT trace-tagged: in every case the log base
  ``log_sampling_result.csv`` has an entirely empty ``TraceID`` (and ``TraceId``)
  column. A log<->trace join therefore retains ZERO logs for any trace-linked
  method, so trace-linked log diversity is not defined on AIOps21. The script
  still *attempts* each trace-linked join and records the (empty) result with a
  ``status`` flag rather than emitting misleading zeros, so the asymmetry is
  fully traceable. The meaningful AIOps21 log-diversity numbers come from the
  three direct-log methods only.

Budgets: 0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2.

Scope is read from ``results/label_protocol_v02/trace_manifest_index.csv`` via the
``trace_label_scope`` column: ``primary_trace_label_scope`` (123 cases) and
``excluded_infrastructure_no_trace_exposure`` (36 cases) are evaluated and
reported separately. Coverage/JSD/entropy are label-free, so both scopes are
eligible.

Read-only on ``data/``. Writes only to ``results/aiops21_log_diversity/``.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import log_protocol_eval as log_eval
import log_protocol_materialize as lm


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DATA_ROOT = Path("data")
AIOPS21_ROOT = DATA_ROOT / "aiops21" / "all"
TRACE_MANIFEST_INDEX = Path("results/label_protocol_v02/trace_manifest_index.csv")
OUTPUT_DIR = Path("results/aiops21_log_diversity")

BUDGETS = [0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.10, 0.20]

PRIMARY_SCOPE = "primary_trace_label_scope"
EXCLUDED_SCOPE = "excluded_infrastructure_no_trace_exposure"

# UniSage native log ranking score columns in log_sampling_result.csv.
NATIVE_RANKINGS = {
    "UniSage-native-product": "LogSamplingProbProduct",
    "UniSage-native-max": "LogSamplingProbMax",
}

# Trace-sampling outputs feeding the (AIOps21-empty) trace-linked join.
# UniSage trace + baselines. Each baseline lists candidate seed-1 primary sample
# directories (first existing wins) and its per-budget decision-column prefix,
# mirroring MethodSpec in scripts/build_linked_log_fairness_table.py.
UNISAGE_TRACE_METHOD = "UniSage-trace-linked"
UNISAGE_TRACE_PROB_COL = "CombinedSamplingProb"

BASELINE_TRACE_METHODS = [
    {
        "method": "Gleaner-linked",
        "decision_prefix": "GleanerDecision",
        "sample_glob": "gleaner_v02_aiops21_primary_seed-1/samples/aiops21__all__{case}.gleaner.csv.gz",
        "alt_globs": [
            "gleaner_v02_aiops21_primary_seed-1_*/samples/aiops21__all__{case}.gleaner.csv.gz",
        ],
    },
    {
        "method": "Sieve-linked",
        "decision_prefix": "SieveDecision",
        # Prefer the completed run dir; fall back to PARTIAL/serial dirs.
        "sample_glob": "sieve_v02_aiops21_primary_seed-1_budgets-*/samples/aiops21__all__{case}.sieve.csv.gz",
        "alt_globs": [
            "sieve_v02_aiops21_primary_seed-1_serial/samples/aiops21__all__{case}.sieve.csv.gz",
            "sieve_v02_aiops21_primary_seed-1_PARTIAL_oom/samples/aiops21__all__{case}.sieve.csv.gz",
        ],
    },
    {
        "method": "Sifter-linked",
        "decision_prefix": "SifterDecision",
        "sample_glob": "sifter_v02_aiops21_primary_seed-1_*/samples/aiops21__all__{case}.sifter.csv.gz",
        "alt_globs": [],
    },
    {
        "method": "TraStrainer-linked",
        "decision_prefix": "TraStrainerDecision",
        "sample_glob": "trastrainer_v02_aiops21_primary_seed-1/samples/aiops21__all__{case}.trastrainer.csv.gz",
        "alt_globs": [
            "trastrainer_v02_aiops21_primary_seed-1_*/samples/aiops21__all__{case}.trastrainer.csv.gz",
        ],
    },
    {
        "method": "TracePicker-linked",
        "decision_prefix": "TracePickerDecision",
        "sample_glob": "tracepicker_v02_aiops21_primary_portable_seed1/samples/aiops21__all__{case}.tracepicker.csv.gz",
        "alt_globs": [
            "tracepicker_v02_aiops21_primary_portable_seed1_*/samples/aiops21__all__{case}.tracepicker.csv.gz",
        ],
    },
]
V02_RUNS_ROOT = Path("results/v02_sampling_runs")

UNIFORM_SEED_SALT = "aiops21_log_diversity::uniform"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def budget_token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def empty_id_series(series: pd.Series) -> pd.Series:
    """Boolean mask of empty / NaN-ish TraceID strings."""
    s = series.astype(str).str.strip()
    return s.isin({"", "nan", "none", "None", "NaN", "<NA>"})


def stable_uniform_score(value: object, salt: str = UNIFORM_SEED_SALT) -> float:
    return log_eval.stable_uniform_score(f"{salt}::{value}")


def load_log_base(case_dir: Path) -> Optional[pd.DataFrame]:
    """Load the per-case native log-sampling output (full-log base + ranking).

    Returns a frame with at least: LogID, template_hash, TraceID (normalized),
    plus the native score columns when present. ``base`` == every abnormal log
    row in ``log_sampling_result.csv``; ``kept`` is a subset chosen by a ranking.
    """
    path = case_dir / "log_sampling_result.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty or "log_temp" not in df.columns:
        return None
    out = pd.DataFrame()
    # template_hash via the exact stable_hash used by the materializer.
    out["template_hash"] = df["log_temp"].fillna("").astype(str).map(lm.stable_hash)
    # LogID: reuse if present, else deterministic positional id.
    if "LogID" in df.columns and df["LogID"].notna().any():
        out["LogID"] = df["LogID"].astype(str)
    else:
        out["LogID"] = [f"{case_dir.name}::row{i}" for i in range(len(df))]
    # Normalized TraceID (uppercased per protocol); empty -> "".
    if "TraceID" in df.columns:
        tid = df["TraceID"].astype("string").fillna("").astype(str).str.strip()
        tid = tid.mask(empty_id_series(tid), "").str.upper()
        out["TraceID"] = tid.to_numpy()
    else:
        out["TraceID"] = ""
    for col in NATIVE_RANKINGS.values():
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce").to_numpy()
    return out.reset_index(drop=True)


def diversity_row(base: pd.DataFrame, kept: pd.DataFrame) -> Dict[str, float]:
    """Three label-free metrics only. NaN-safe for empty ``kept``."""
    n_total = len(base)
    n_kept = len(kept)
    base_templates = int(base["template_hash"].nunique()) if n_total else 0
    if n_kept == 0 or base_templates == 0:
        return {
            "n_total_logs": n_total,
            "n_kept_logs": n_kept,
            "effective_log_fraction": (n_kept / n_total) if n_total else np.nan,
            "template_support": base_templates,
            "template_kept": int(kept["template_hash"].nunique()) if n_kept else 0,
            "template_coverage": np.nan if n_kept == 0 else 0.0,
            "template_entropy": np.nan,
            "template_jsd": np.nan,
        }
    return {
        "n_total_logs": n_total,
        "n_kept_logs": n_kept,
        "effective_log_fraction": n_kept / n_total,
        "template_support": base_templates,
        "template_kept": int(kept["template_hash"].nunique()),
        "template_coverage": kept["template_hash"].nunique() / base_templates,
        "template_entropy": log_eval.entropy(kept["template_hash"]),
        "template_jsd": log_eval.jensen_shannon_divergence(
            base["template_hash"], kept["template_hash"]
        ),
    }


def top_k(n_total: int, budget: float) -> int:
    if n_total <= 0:
        return 0
    return min(n_total, max(1, int(round(n_total * budget))))


# --------------------------------------------------------------------------- #
# Trace-sampling source resolution (for the AIOps21-empty trace-linked join)
# --------------------------------------------------------------------------- #
def find_baseline_sample(spec: Dict, case: str) -> Optional[Path]:
    patterns = [spec["sample_glob"], *spec.get("alt_globs", [])]
    for pattern in patterns:
        rel = pattern.format(case=case)
        matches = sorted(V02_RUNS_ROOT.glob(rel))
        if matches:
            return matches[0]
    return None


def load_trace_ids_unisage(case_dir: Path) -> Optional[pd.DataFrame]:
    """Return TraceID + CombinedSamplingProb for UniSage trace sampling."""
    path = case_dir / "trace_sampling_result.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "TraceID" not in df.columns or UNISAGE_TRACE_PROB_COL not in df.columns:
        return None
    out = pd.DataFrame()
    out["TraceID"] = df["TraceID"].astype(str).str.strip().str.upper()
    out[UNISAGE_TRACE_PROB_COL] = pd.to_numeric(df[UNISAGE_TRACE_PROB_COL], errors="coerce")
    return out.drop_duplicates("TraceID").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Per-case evaluation
# --------------------------------------------------------------------------- #
def eval_case(
    case: str,
    scope: str,
    skipped: List[dict],
    join_diag: Dict[str, Dict[str, int]],
) -> List[dict]:
    case_dir = AIOPS21_ROOT / case
    base = load_log_base(case_dir)
    rows: List[dict] = []
    if base is None:
        skipped.append({
            "scope": scope,
            "case": case,
            "method": "*",
            "reason": "missing_or_empty_log_sampling_result",
        })
        return rows

    n_total = len(base)
    log_link_present = int((~empty_id_series(base["TraceID"])).sum())
    base_traceids = set(base.loc[base["TraceID"].astype(bool), "TraceID"])

    def emit(method: str, budget: float, kept: pd.DataFrame, status: str, extra: Dict):
        row = {"scope": scope, "case": case, "method": method, "budget": budget, "status": status}
        row.update(diversity_row(base, kept))
        row.update(extra)
        rows.append(row)

    # ---- Direct-log methods -------------------------------------------------
    for method, col in NATIVE_RANKINGS.items():
        if col not in base.columns or base[col].notna().sum() == 0:
            skipped.append({"scope": scope, "case": case, "method": method,
                            "reason": f"missing_score_column_{col}"})
            continue
        ranked = base.dropna(subset=[col]).sort_values(
            [col, "LogID"], ascending=[False, True]
        ).reset_index(drop=True)
        for budget in BUDGETS:
            k = top_k(len(ranked), budget)
            emit(method, budget, ranked.head(k), "ok",
                 {"n_log_trace_link": log_link_present})

    # Uniform-log (deterministic per-LogID hash).
    ranked_u = base.copy()
    ranked_u["_u"] = ranked_u["LogID"].map(stable_uniform_score)
    ranked_u = ranked_u.sort_values(["_u", "LogID"], ascending=[False, True]).reset_index(drop=True)
    for budget in BUDGETS:
        k = top_k(len(ranked_u), budget)
        emit("Uniform-log", budget, ranked_u.head(k), "ok",
             {"n_log_trace_link": log_link_present})

    # ---- Trace-linked methods (AIOps21: log<->trace link is empty) ----------
    # We still attempt each join honestly. ``status`` flags the no-link case so
    # the empty kept-set is not mistaken for a real 0% diversity measurement.
    link_status = "ok" if log_link_present > 0 else "no_log_trace_link"

    def emit_trace_linked(method: str, decision: Dict[float, set], n_trace_total: int,
                          n_trace_ranked: int):
        for budget in BUDGETS:
            if budget not in decision:
                skipped.append({"scope": scope, "case": case, "method": method,
                                "reason": f"missing_decision_budget_{budget_token(budget)}"})
                continue
            kept_tids = decision[budget]
            kept = base[base["TraceID"].isin(kept_tids)] if kept_tids else base.iloc[0:0]
            jd = join_diag.setdefault(method, {"cases": 0, "nonempty_join_cases": 0})
            emit(method, budget, kept, link_status, {
                "n_log_trace_link": log_link_present,
                "n_trace_total": n_trace_total,
                "n_trace_ranked": n_trace_ranked,
                "n_trace_kept": len(kept_tids),
            })
        jd = join_diag.setdefault(method, {"cases": 0, "nonempty_join_cases": 0})
        jd["cases"] += 1
        # any budget producing a non-empty log join?
        if any(
            len(base[base["TraceID"].isin(decision.get(b, set()))]) > 0
            for b in BUDGETS if b in decision
        ):
            jd["nonempty_join_cases"] += 1

    # UniSage-trace-linked: top-k traces by CombinedSamplingProb.
    utr = load_trace_ids_unisage(case_dir)
    if utr is not None and not utr.empty:
        ranked_tr = utr.dropna(subset=[UNISAGE_TRACE_PROB_COL]).sort_values(
            [UNISAGE_TRACE_PROB_COL, "TraceID"], ascending=[False, True]
        ).reset_index(drop=True)
        n_tr = len(ranked_tr)
        decision = {b: set(ranked_tr["TraceID"].head(top_k(n_tr, b))) for b in BUDGETS}
        emit_trace_linked(UNISAGE_TRACE_METHOD, decision, n_tr, n_tr)
        # Uniform-trace-linked: random traces -> linked logs (deterministic).
        ranked_rand = utr.copy()
        ranked_rand["_u"] = ranked_rand["TraceID"].map(
            lambda v: stable_uniform_score(v, UNIFORM_SEED_SALT + "::trace")
        )
        ranked_rand = ranked_rand.sort_values(["_u", "TraceID"], ascending=[False, True]).reset_index(drop=True)
        decision_rand = {b: set(ranked_rand["TraceID"].head(top_k(len(ranked_rand), b))) for b in BUDGETS}
        emit_trace_linked("Uniform-trace-linked", decision_rand, len(ranked_rand), len(ranked_rand))
    else:
        skipped.append({"scope": scope, "case": case, "method": UNISAGE_TRACE_METHOD,
                        "reason": "missing_trace_sampling_result"})
        skipped.append({"scope": scope, "case": case, "method": "Uniform-trace-linked",
                        "reason": "missing_trace_sampling_result"})

    # Baseline trace samplers: per-budget decision columns.
    for spec in BASELINE_TRACE_METHODS:
        method = spec["method"]
        sample_path = find_baseline_sample(spec, case)
        if sample_path is None:
            skipped.append({"scope": scope, "case": case, "method": method,
                            "reason": "missing_baseline_sample"})
            continue
        try:
            sample = pd.read_csv(sample_path)
        except Exception as exc:  # noqa: BLE001
            skipped.append({"scope": scope, "case": case, "method": method,
                            "reason": f"sample_read_failed:{exc}"})
            continue
        if "TraceID" not in sample.columns:
            skipped.append({"scope": scope, "case": case, "method": method,
                            "reason": "sample_missing_TraceID"})
            continue
        sample = sample.drop_duplicates("TraceID").copy()
        sample["TraceID"] = sample["TraceID"].astype(str).str.strip().str.upper()
        prefix = spec["decision_prefix"]
        decision: Dict[float, set] = {}
        for budget in BUDGETS:
            col = f"{prefix}_{budget_token(budget)}"
            if col not in sample.columns:
                continue
            mask = sample[col]
            if not pd.api.types.is_bool_dtype(mask):
                if pd.api.types.is_numeric_dtype(mask):
                    mask = mask.fillna(0).astype(float) != 0
                else:
                    mask = mask.fillna("").astype(str).str.strip().str.lower().isin(
                        {"1", "true", "t", "yes", "y"}
                    )
            decision[budget] = set(sample.loc[mask, "TraceID"])
        if not decision:
            skipped.append({"scope": scope, "case": case, "method": method,
                            "reason": "no_decision_columns_present"})
            continue
        emit_trace_linked(method, decision, len(sample), len(sample))

    return rows


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
METRIC_COLS = ["template_coverage", "template_entropy", "template_jsd",
               "effective_log_fraction", "n_kept_logs", "n_total_logs",
               "template_support", "template_kept"]


def aggregate(per_case: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for (scope, method, budget), grp in per_case.groupby(["scope", "method", "budget"], sort=True):
        status = "ok" if (grp["status"] == "ok").any() else grp["status"].iloc[0]
        row = {
            "scope": scope,
            "method": method,
            "budget": budget,
            "status": status,
            "cases": int(grp["case"].nunique()),
        }
        for col in METRIC_COLS:
            s = pd.to_numeric(grp[col], errors="coerce").dropna()
            row[f"{col}_mean"] = float(s.mean()) if not s.empty else np.nan
            row[f"{col}_std"] = float(s.std(ddof=1)) if len(s) > 1 else np.nan
            row[f"{col}_n"] = int(s.count())
        rows.append(row)
    return pd.DataFrame(rows)


def render_summary(agg: pd.DataFrame, case_counts: Dict[str, int],
                   join_diag: Dict[str, Dict[str, int]]) -> str:
    lines = [
        "# AIOps21 label-free log diversity",
        "",
        "Label-free template-diversity metrics (coverage / JSD / entropy) for AIOps21",
        "across sampling methods. AIOps21 has NO line-level log fault labels, so NO",
        "problem-related metrics (A_exposed / strict-fault) are computed -- only the",
        "three diversity metrics, which need only `template_hash`.",
        "",
        "`base` = every abnormal log row in each case's `log_sampling_result.csv`.",
        "`kept` for direct-log methods = top-k logs by the method's score (k = round(",
        "n_logs * budget)); `template_hash = stable_hash(log_temp)`.",
        "",
        "**AIOps21 data fact:** AIOps21 logs are not trace-tagged (the `TraceID`/`TraceId`",
        "columns in `log_sampling_result.csv` are empty in every case), so a log<->trace",
        "join retains zero logs. Trace-linked methods are therefore reported as",
        "`no_log_trace_link` (not a real 0% measurement); the meaningful numbers are the",
        "three direct-log methods.",
        "",
        f"Cases evaluated: primary={case_counts.get(PRIMARY_SCOPE, 0)}, "
        f"excluded={case_counts.get(EXCLUDED_SCOPE, 0)}.",
        "",
    ]

    # Primary table at the budget whose mean effective-log fraction is nearest 5%.
    for scope, label in [(PRIMARY_SCOPE, "Primary (123-case scope)"),
                         (EXCLUDED_SCOPE, "Excluded infrastructure (36-case scope)")]:
        sub = agg[(agg["scope"] == scope) & (agg["status"] == "ok")].copy()
        lines += [f"## {label} -- direct-log methods at effective-log ~= 5%", ""]
        if sub.empty:
            lines += ["_No computable (ok-status) rows for this scope._", ""]
            continue
        lines += [
            "| method | budget | eff. log frac | template coverage | template JSD | template entropy | cases |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for method in sorted(sub["method"].unique()):
            ms = sub[sub["method"] == method]
            # pick row with effective_log_fraction_mean closest to 0.05
            ms = ms.assign(_d=(ms["effective_log_fraction_mean"] - 0.05).abs())
            best = ms.sort_values("_d").iloc[0]
            def f(c):
                v = best.get(c, np.nan)
                return "nan" if pd.isna(v) else f"{v:.3f}"
            lines.append(
                f"| {method} | {best['budget']:.3f} | {f('effective_log_fraction_mean')} | "
                f"{f('template_coverage_mean')} | {f('template_jsd_mean')} | "
                f"{f('template_entropy_mean')} | {int(best['cases'])} |"
            )
        lines.append("")

    # Full per-budget primary table (direct-log only).
    lines += ["## Primary: full per-budget (direct-log methods)", "",
              "| method | budget | eff. frac | coverage | JSD | entropy | cases |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    sub = agg[(agg["scope"] == PRIMARY_SCOPE) & (agg["status"] == "ok")].sort_values(
        ["method", "budget"]
    )
    for r in sub.itertuples(index=False):
        def f(v):
            return "nan" if pd.isna(v) else f"{v:.3f}"
        lines.append(
            f"| {r.method} | {r.budget:.3f} | {f(r.effective_log_fraction_mean)} | "
            f"{f(r.template_coverage_mean)} | {f(r.template_jsd_mean)} | "
            f"{f(r.template_entropy_mean)} | {int(r.cases)} |"
        )
    lines.append("")

    # Trace-linked status note.
    lines += ["## Trace-linked methods: join coverage (AIOps21)", "",
              "| method | cases joined | cases with non-empty log join |",
              "|---|---:|---:|"]
    for method in sorted(join_diag):
        d = join_diag[method]
        lines.append(f"| {method} | {d['cases']} | {d['nonempty_join_cases']} |")
    lines += ["",
              "All trace-linked methods join 0 logs because AIOps21 logs carry no TraceID.",
              ""]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    idx = pd.read_csv(TRACE_MANIFEST_INDEX)
    idx = idx[idx["dataset"] == "aiops21"].copy()
    scope_cases: Dict[str, List[str]] = {
        PRIMARY_SCOPE: sorted(idx.loc[idx["trace_label_scope"] == PRIMARY_SCOPE, "case"].astype(str)),
        EXCLUDED_SCOPE: sorted(idx.loc[idx["trace_label_scope"] == EXCLUDED_SCOPE, "case"].astype(str)),
    }

    per_case_rows: List[dict] = []
    skipped: List[dict] = []
    join_diag: Dict[str, Dict[str, int]] = {}
    evaluated_counts: Dict[str, int] = {}

    for scope, cases in scope_cases.items():
        evaluated = set()
        for i, case in enumerate(cases, start=1):
            rows = eval_case(case, scope, skipped, join_diag)
            if rows:
                evaluated.add(case)
            per_case_rows.extend(rows)
            if i % 25 == 0 or i == len(cases):
                print(f"[{scope}] {i}/{len(cases)} cases", flush=True)
        evaluated_counts[scope] = len(evaluated)

    per_case = pd.DataFrame(per_case_rows)
    agg = aggregate(per_case) if not per_case.empty else pd.DataFrame()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    per_case.to_csv(OUTPUT_DIR / "aiops21_log_diversity_per_case.csv", index=False)
    agg.to_csv(OUTPUT_DIR / "aiops21_log_diversity_aggregate.csv", index=False)
    pd.DataFrame(skipped, columns=["scope", "case", "method", "reason"]).to_csv(
        OUTPUT_DIR / "aiops21_log_diversity_skipped.csv", index=False
    )
    (OUTPUT_DIR / "aiops21_log_diversity_summary.md").write_text(
        render_summary(agg, evaluated_counts, join_diag)
    )

    # Provenance.
    skip_by_reason: Dict[str, int] = {}
    for s in skipped:
        skip_by_reason[s["reason"]] = skip_by_reason.get(s["reason"], 0) + 1
    provenance = {
        "task": "aiops21_label_free_log_diversity",
        "metrics": ["template_coverage", "template_entropy", "template_jsd"],
        "metrics_excluded": "problem-related (A_exposed/strict-fault) -- AIOps21 has no log fault labels",
        "budgets": BUDGETS,
        "scope_source": TRACE_MANIFEST_INDEX.as_posix(),
        "scope_case_counts": {k: len(v) for k, v in scope_cases.items()},
        "scope_cases_with_log_base": evaluated_counts,
        "template_hash": "log_protocol_materialize.stable_hash(log_temp)",
        "reused_metric_functions": [
            "log_protocol_eval.entropy",
            "log_protocol_eval.jensen_shannon_divergence",
            "log_protocol_eval.stable_uniform_score",
        ],
        "direct_log_methods": list(NATIVE_RANKINGS) + ["Uniform-log"],
        "trace_linked_methods": [UNISAGE_TRACE_METHOD, "Uniform-trace-linked"]
        + [s["method"] for s in BASELINE_TRACE_METHODS],
        "aiops21_log_trace_link": "ABSENT -- log_sampling_result.csv TraceID/TraceId empty in all cases; "
        "trace-linked log diversity is not defined on AIOps21 (status=no_log_trace_link).",
        "trace_linked_join_diagnostics": join_diag,
        "uniform_seed_salt": UNIFORM_SEED_SALT,
        "skipped_total": len(skipped),
        "skipped_by_reason": skip_by_reason,
        "outputs": {
            "per_case": "aiops21_log_diversity_per_case.csv",
            "aggregate": "aiops21_log_diversity_aggregate.csv",
            "skipped": "aiops21_log_diversity_skipped.csv",
            "summary": "aiops21_log_diversity_summary.md",
        },
    }
    (OUTPUT_DIR / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    print(f"\nWrote AIOps21 log-diversity outputs to {OUTPUT_DIR}")
    print(f"  per-case rows: {len(per_case)}")
    print(f"  primary cases with log base: {evaluated_counts.get(PRIMARY_SCOPE, 0)} / "
          f"{len(scope_cases[PRIMARY_SCOPE])}")
    print(f"  excluded cases with log base: {evaluated_counts.get(EXCLUDED_SCOPE, 0)} / "
          f"{len(scope_cases[EXCLUDED_SCOPE])}")
    if skipped:
        print(f"  skipped entries: {len(skipped)} (see aiops21_log_diversity_skipped.csv)")


if __name__ == "__main__":
    main()
