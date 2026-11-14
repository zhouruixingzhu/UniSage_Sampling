"""[UniSage] Single-case TraceRCA runner — emits ranked-service JSON to stdout.

Reads a scratch dir produced by ``UniSage/tracerca_adapter.py`` containing:
  invo_abnormal.pkl    -- invocation DataFrame for the (possibly sampled) abnormal window
  invo_historical.pkl  -- invocation DataFrame for the FULL fault-free reference
  case_meta.json       -- case metadata (echoed back for provenance)

Each invocation DataFrame has columns ``trace_id, source, target`` plus one
column per feature (default just ``latency``).

Pipeline (faithful to TraceRCA, IWQoS'21, the paper's main "Ours" path):
  1. reference mean/std per (source,target,feature)   [run_anomaly_detection_prepare_model.py]
  2. Fisher feature selection via ``stderr_criteria``  [run_selecting_features.py]
  3. invocation k-sigma anomaly detection              [run_anomaly_detection_invo.py]
  4. association-rule-mining localization (TraceRCA)   [run_localization_association_rule_mining_20210516.py]

Step 4 imports and calls the upstream ``TraceRCA`` class **verbatim** — it is the
paper's actual contribution. Steps 1-3 are faithful re-implementations of the
upstream formulas, hardened against the tiny per-edge sample sizes that
5%-budget sampling produces (upstream's vectorised ``.loc[(s,t), feats].values``
assumes >1 row per edge); numeric behaviour matches. See PROVENANCE.md.

Must live INSIDE baselines/tracerca/ so the upstream module imports by relative
path. Output: a single JSON line on stdout, ``{"ranked": [...], ...}``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from run_localization_association_rule_mining_20210516 import TraceRCA  # noqa: E402  verbatim core


# --- verbatim copy of run_selecting_features.stderr_criteria --------------------
# Copied (not imported) only so this runner does not pull in seaborn/tqdm, which
# upstream imports at module load purely for debug plots. Numeric logic unchanged.
def stderr_criteria(empirical, reference, threshold):
    empirical, reference = np.array(empirical), np.array(reference)
    historical_mean, historical_std = np.mean(reference), np.std(reference)
    historical_std = np.maximum(historical_std, historical_mean * 0.01 + 0.01)
    ref_ratio = np.mean(np.abs(reference - historical_mean)) / historical_std
    emp_ratio = np.mean(np.abs(empirical - historical_mean)) / historical_std
    return (emp_ratio - ref_ratio) > threshold * ref_ratio + 1.0


def build_reference(historical: pd.DataFrame, features: list[str]) -> dict:
    """Per (source,target,feature) -> (mean, std) with std>=0.1.

    Mirrors run_anomaly_detection_prepare_model.py's reference-stat tokens.
    """
    ref: dict[tuple, tuple[float, float]] = {}
    for (s, t), sub in historical.groupby(["source", "target"]):
        for f in features:
            vals = sub[f].to_numpy(dtype=float)
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                continue
            ref[(s, t, f)] = (float(np.mean(vals)), float(np.maximum(np.std(vals), 0.1)))
    return ref


def select_features(abnormal: pd.DataFrame, historical: pd.DataFrame,
                    features: list[str], fisher: float) -> dict:
    """Per (source,target) -> [useful features], mirroring run_selecting_features.py.

    Only (source,target) edges present in BOTH the abnormal window and the
    historical reference are considered (upstream uses np.intersect1d).
    """
    useful: dict[tuple, list[str]] = {}
    hist_groups = {k: v for k, v in historical.groupby(["source", "target"])}
    for (s, t), sub in abnormal.groupby(["source", "target"]):
        if (s, t) not in hist_groups:
            continue
        chosen = []
        for f in features:
            emp = np.sort(sub[f].to_numpy(dtype=float))
            ref = np.sort(hist_groups[(s, t)][f].to_numpy(dtype=float))
            emp = emp[~np.isnan(emp)]
            ref = ref[~np.isnan(ref)]
            if len(emp) == 0 or len(ref) == 0:
                continue
            if stderr_criteria(emp, ref, fisher):
                chosen.append(f)
        if chosen:
            useful[(s, t)] = chosen
    return useful


def detect(abnormal: pd.DataFrame, reference: dict, useful: dict,
           sigma: float) -> pd.DataFrame:
    """Per-invocation k-sigma anomaly flags in a new ``predict`` column.

    Faithful to run_anomaly_detection_invo.anomaly_detection_3sigma:
    ``predict = max_f ( |x_f - mean_f| > sigma*std_f )`` over useful features,
    with a missing reference token falling back to the empirical mean/std (also
    upstream behaviour). Robust to single-row (source,target) groups.
    """
    out = abnormal.copy()
    out["predict"] = 0
    for (s, t), idx in out.groupby(["source", "target"]).groups.items():
        feats = useful.get((s, t))
        if not feats:
            continue
        rows = out.loc[idx]
        flag = np.zeros(len(rows), dtype=bool)
        for f in feats:
            x = rows[f].to_numpy(dtype=float)
            ref = reference.get((s, t, f))
            if ref is None:
                mean = float(np.nanmean(x))
                std = float(np.maximum(np.nanstd(x), 0.1))
            else:
                mean, std = ref
            flag |= np.abs(x - mean) > sigma * std
        out.loc[idx, "predict"] = flag.astype(int)
    return out


def localize(abnormal_pred: pd.DataFrame, min_support: float, k: int,
             forbidden: frozenset) -> list[str]:
    """Run the upstream TraceRCA association-rule-mining localizer verbatim."""
    df = abnormal_pred[["trace_id", "source", "target", "predict"]].copy()
    if df.empty or int(df["predict"].sum()) == 0:
        # No abnormal invocation -> no abnormal itemsets -> upstream would crash
        # at TransactionEncoder([]) / xs(True). Honest empty ranking.
        return []
    tracerca = TraceRCA()
    ranked = tracerca(
        df,
        min_support_rate=min_support,
        k=k,
        predict_column="predict",
        source_column="source",
        target_column="target",
        trace_id_column="trace_id",
        forbidden_names=forbidden,
        enable_PRFL=False,
        quiet=True,
    )
    return [str(x) for x in ranked]


def run(scratch: Path, sigma: float, fisher: float, min_support: float,
        k: int, features: list[str], forbidden: frozenset) -> dict:
    abnormal = pd.read_pickle(scratch / "invo_abnormal.pkl")
    historical = pd.read_pickle(scratch / "invo_historical.pkl")
    feats = [f for f in features if f in abnormal.columns and f in historical.columns]
    diagnostics = {
        "n_invocations": int(len(abnormal)),
        "n_traces": int(abnormal["trace_id"].nunique()) if len(abnormal) else 0,
        "n_historical_invocations": int(len(historical)),
        "features_used": feats,
    }
    if not feats or abnormal.empty:
        return {"ranked": [], "diagnostics": diagnostics}

    reference = build_reference(historical, feats)
    useful = select_features(abnormal, historical, feats, fisher)
    abnormal_pred = detect(abnormal, reference, useful, sigma)
    n_abn_invo = int(abnormal_pred["predict"].sum())
    abn_trace_ids = set(abnormal_pred.loc[abnormal_pred["predict"] == 1, "trace_id"])
    diagnostics.update(
        n_useful_edges=len(useful),
        n_abnormal_invocations=n_abn_invo,
        n_abnormal_traces=len(abn_trace_ids),
    )
    ranked = localize(abnormal_pred, min_support, k, forbidden)
    return {"ranked": ranked, "diagnostics": diagnostics}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-dir", required=True, help="scratch dir with invo_*.pkl")
    ap.add_argument("--sigma", type=float, default=1.0, help="k-sigma AD threshold (upstream SIGMA=1)")
    ap.add_argument("--fisher", type=float, default=3.0, help="Fisher feature-selection threshold (upstream FISHER=3)")
    ap.add_argument("--min-support", type=float, default=0.1, help="FP-growth min support (upstream SUPPORT=0.1)")
    ap.add_argument("--k", type=int, default=100, help="top-k patterns (upstream K=100)")
    ap.add_argument("--features", default="latency", help="comma list of invocation features")
    ap.add_argument("--forbidden", default="", help="comma list of forbidden entry services")
    args = ap.parse_args()

    features = [f.strip() for f in args.features.split(",") if f.strip()]
    forbidden = frozenset(f.strip() for f in args.forbidden.split(",") if f.strip())
    try:
        result = run(Path(args.case_dir), args.sigma, args.fisher, args.min_support,
                     args.k, features, forbidden)
    except Exception as exc:  # surface as JSON so the harness records it, not a crash
        import traceback
        result = {"ranked": [], "error": f"{type(exc).__name__}: {exc}",
                  "traceback": traceback.format_exc()[-1500:]}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
