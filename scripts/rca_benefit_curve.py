
import sys as _sys
from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
"""RCA quality-benefit curve under v0.2 trace labels.

Degrade UniSage's internal RCA ranking by a factor eps in [0, 1] and measure, at
each eps:
  (a) RCA localization accuracy -- AC@1 / AC@3 / MRR of the degraded per-service
      anomaly ranking against the externally supplied root-cause service, and
  (b) downstream trace-sampling v0.2 A_exposed recall, retained in the top-B%,
      under several score compositions:
        - unisage_v1              = p_e * p_a, or p_e under no-analysis fallback,
        - product_only            = p_e * p_a without the fallback override,
        - max_composition         = max(p_e, p_a),
        - analysis_only           = p_a only,
        - edge_only_fallback      = p_e only (eps-independent floor).

The A_exposed labels are read from the frozen v0.2 trace manifests, not inferred
from sampler/RCA outputs. This replaces the earlier root-service-path proxy and
is the only metric from this script that is intended for paper-use screening.

Goal (strategic point S#4b; answers RevA "lightweight RCA" / RevC weak internal RCA):
show how sampling quality responds to RCA degradation. The decomposition lets us tell apart
"the analysis pillar tracks RCA quality" from "the composite is robust because the edge-case
pillar backstops it" -- a far more defensible story than a bare composite curve.

Method (fast + faithful): run the full pipeline ONCE per case to capture the real per-service
score map (final_ranking.csv) and the real per-trace sampling frame (p_e + service path + the
pipeline's p_a). Cache that to disk. Then sweep eps x seeds entirely in memory by perturbing
the score map and recomputing p_a / coverage with the EXACT pipeline formulas. No AD/RCL re-run.

NOTE: read the score map from final_ranking.csv (raw service_name, written by ranking()),
NOT from evaluate()'s returned combined_ranking -- log_sampling base-normalizes that object
in place, which would break raw-pod matching on the trace paths.

Degradation = partial random permutation of the score vector (matches S#4b "shuffle part of
top-k"): permute the scores of a random round(eps*N) subset of services among themselves.
  eps = 0 -> identity (sanity-checked against the pipeline's own p_a).
  eps = 1 -> full permutation: same score multiset, fully decorrelated from services
             (RCA accuracy -> chance), while preserving the score distribution.

Propagation is forced OFF to isolate the RCA -> trace-sampling pathway (cross-modal
propagation is a separate ablation); with it off, CombinedSamplingProb = p_e * p_a when
AD/RCL provides an analysis ranking, and CombinedSamplingProb = p_e under the explicit
edge-only fallback.

Run from UniSage/:
    python scripts/rca_benefit_curve.py [batch_path] [dataset_type]
Env: BENEFIT_SEEDS (default 20); FORCE_RERUN=1 to ignore the phase-1 cache.
Outputs: per-batch results/rq4_benefit_curve/rca_benefit_curve_<dataset>_<batch>_<attr>_<edge>_v02.{csv,md,png},
combined results/rq4_benefit_curve/rca_benefit_curves_<attr>_<edge>_v02.{csv,md}, and an ignored cache whose key
includes the result-producing pipeline fingerprint plus the v0.2 manifest index.
"""
import asyncio
import glob
import hashlib
import json
import os
import pickle
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from aiomultiprocess import Pool

# Isolate RCA -> trace sampling and make eps=0 reproducible (pipeline writes p_e * p_a).
os.environ["UNISAGE_PROPAGATE"] = "0"
os.environ.setdefault("UNISAGE_EDGE_LAT", "raw")
# UNISAGE_RCA_ATTR defaults to "target" (the shipped default = the structure-aware fix).

import UniSage  # noqa: E402
import sampling_labels as lp  # noqa: E402

BATCH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/hipster/batches/batch1")
DATASET = sys.argv[2] if len(sys.argv) > 2 else "hipster"
# Hipster and Train Ticket share the directory layout expected by the "hipster" adapter.
PIPE_DT = "aiops21" if DATASET == "aiops21" else "hipster"
EPS_GRID = [round(0.1 * i, 1) for i in range(11)]  # 0.0 .. 1.0
N_SEEDS = int(os.environ.get("BENEFIT_SEEDS", "20"))
BUDGETS = [0.01, 0.05, 0.10, 0.20]
ATTR = os.environ.get("UNISAGE_RCA_ATTR", "target")
EDGE = os.environ.get("UNISAGE_EDGE_LAT", "raw")  # pipeline default; affects cached p_e
TARGET = ATTR == "target"
LABEL_INDEX = Path(os.environ.get("LABEL_MANIFEST_INDEX", "results/label_protocol_v02/trace_manifest_index.csv"))
RES = Path("results/rq4_benefit_curve")


def pipeline_fingerprint():
    """Invalidate cached sampling frames whenever result-producing code changes."""
    digest = hashlib.sha256()
    paths = [Path(__file__), Path("UniSage.py"), Path("sampling.py"), Path("utils.py")]
    paths += sorted(Path("AD").rglob("*.py"))
    paths += sorted(Path("RCL").rglob("*.py"))
    for path in paths:
        digest.update(path.as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


FINGERPRINT = pipeline_fingerprint()
_manifest_token = hashlib.sha256(
    LABEL_INDEX.read_bytes() if LABEL_INDEX.exists() else str(LABEL_INDEX).encode()
).hexdigest()[:12]
_batch_token = str(BATCH).strip("/").replace("/", "_")
_cache_prefix = f"_benefit_cache_{_batch_token}_{ATTR}_{EDGE}_"
_cache_suffix = f"_v02_{_manifest_token}.pkl"
_tag = f"{_batch_token}_{ATTR}_{EDGE}_{FINGERPRINT}_v02_{_manifest_token}"
CACHE = RES / f"_benefit_cache_{_tag}.pkl"
OUT_STEM = RES / f"rca_benefit_curve_{DATASET}_{BATCH.name}_{ATTR}_{EDGE}_v02"
COMBINED_CSV = RES / f"rca_benefit_curves_{ATTR}_{EDGE}_v02.csv"
COMBINED_MD = RES / f"rca_benefit_curves_{ATTR}_{EDGE}_v02.md"


def _trace_id_series(values):
    return pd.Series(values).astype(str)


def manifest_lookup():
    """Frozen v0.2 primary-scope manifest paths for this dataset/batch."""
    if not LABEL_INDEX.exists():
        raise FileNotFoundError(f"Missing v0.2 manifest index: {LABEL_INDEX}")
    index = pd.read_csv(LABEL_INDEX)
    required = {"dataset", "batch", "case", "trace_label_scope", "manifest_path"}
    missing = required - set(index.columns)
    if missing:
        raise ValueError(f"Manifest index lacks columns: {sorted(missing)}")
    scoped = index[
        (index["dataset"] == DATASET)
        & (index["batch"].astype(str) == BATCH.name)
        & (index["trace_label_scope"] == lp.PRIMARY_SCOPE)
    ].copy()
    return {row.case: Path(row.manifest_path) for row in scoped.itertuples(index=False)}


def load_v02_manifest(case, lookup):
    """Read per-case frozen labels used for A_exposed recall."""
    path = lookup.get(case)
    if path is None:
        return None
    manifest = pd.read_csv(path, usecols=["TraceID", "A_exposed", "trace_label_scope"])
    if manifest.empty or manifest["trace_label_scope"].iloc[0] != lp.PRIMARY_SCOPE:
        return None
    manifest = manifest.drop_duplicates("TraceID").copy()
    trace_ids = set(_trace_id_series(manifest["TraceID"]))
    a_exposed_ids = set(_trace_id_series(manifest.loc[manifest["A_exposed"].astype(bool), "TraceID"]))
    return {
        "path": path.as_posix(),
        "trace_ids": trace_ids,
        "n_base": len(trace_ids),
        "a_exposed_ids": a_exposed_ids,
        "a_exposed_support": len(a_exposed_ids),
    }


def degrade_map(score_map, eps, rng):
    """Partial random permutation: shuffle the scores of round(eps*N) services among
    themselves. eps<=0 -> identity copy; eps=1 -> full permutation."""
    keys = list(score_map.keys())
    n = len(keys)
    if eps <= 0 or n < 2:
        return dict(score_map)
    vals = [score_map[k] for k in keys]
    k = int(round(eps * n))
    if k < 2:
        return dict(score_map)
    idx = rng.choice(n, size=k, replace=False)
    perm = rng.permutation(idx)
    new_vals = list(vals)
    for src, dst in zip(idx, perm):
        new_vals[dst] = vals[src]
    return {keys[i]: new_vals[i] for i in range(n)}


def rca_rank(score_map, gt):
    """1-based rank of gt service in score_map sorted descending; 0 if absent."""
    if not score_map or gt is None:
        return 0
    ranked = [k for k, _ in sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)]
    return (ranked.index(gt) + 1) if gt in ranked else 0


def norm_for_pa(score_map):
    """Per-case phi_s/max normalization before the p_a sigmoid (target mode), matching
    sampling.compute_anomaly_sampling_prob."""
    if TARGET and score_map:
        mx = max(score_map.values())
        if mx and mx > 0:
            return {k: v / mx for k, v in score_map.items()}
    return score_map


def trace_pa_vec(pods_list, norm_map):
    """p_a(trace) = tanh(max over services on the trace path present in the score map)."""
    out = np.empty(len(pods_list), dtype=float)
    for i, pods in enumerate(pods_list):
        scores = [norm_map[p] for p in pods if p in norm_map]
        out[i] = np.tanh(max(scores)) if scores else 0.0
    return out


def external_case_labels():
    """External root-cause metadata, without reading sampler/RCA outputs."""
    if DATASET == "aiops21":
        metadata_path = BATCH / "fault_injection.toml"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing AIOps21 metadata: {metadata_path}")
        labels = {}
        for record in lp.read_aiops_metadata(metadata_path):
            labels[record["case"]] = {
                "inject_pod": record.get("service"),
                "inject_type": record.get("chaos_type", ""),
            }
        return labels

    label_path = BATCH / "label.json"
    if not label_path.exists():
        raise FileNotFoundError(f"Missing benchmark label metadata: {label_path}")
    return json.load(open(label_path))


def a_exposed_recall_of_scores(case, scores, budgets, jitter=None):
    """Rank traces by score and return v0.2 A_exposed recall@B.

    The budget denominator matches scripts/evaluate_trace_sampling.py: k is computed from
    the frozen manifest's total trace count, then capped by the number of ranked
    traces. Optional jitter breaks ties neutrally for analysis-only rankings.
    """
    support = case["a_exposed_support"]
    if support <= 0:
        return None
    if len(scores) != len(case["tids"]):
        raise ValueError(f"score length mismatch for {case['case']}: {len(scores)} vs {len(case['tids'])}")

    prepare_fast_recall(case)
    score = np.asarray(scores, dtype=float)
    if jitter is not None:
        score = score + np.asarray(jitter, dtype=float)
    valid = case["_manifest_mask"] & np.isfinite(score)
    if not valid.any():
        return {B: 0.0 for B in budgets}

    idx = np.flatnonzero(valid)
    order = np.lexsort((case["_trace_tie_rank"][idx], -score[idx]))
    ranked_idx = idx[order]
    if case["_has_duplicate_tids"]:
        seen = set()
        unique_idx = []
        tids = case["tids"]
        for pos in ranked_idx:
            tid = tids[pos]
            if tid in seen:
                continue
            seen.add(tid)
            unique_idx.append(pos)
        ranked_idx = np.asarray(unique_idx, dtype=int)
    n_ranked = len(ranked_idx)
    hits = np.cumsum(case["_a_exposed_mask"][ranked_idx].astype(np.int64))

    out = {}
    for B in budgets:
        k = min(n_ranked, max(1, int(round(case["manifest_n"] * B)))) if n_ranked else 0
        out[B] = (float(hits[k - 1]) / support) if k else 0.0
    return out


def prepare_fast_recall(case):
    """Attach NumPy masks used by the hot recall loop."""
    if case.get("_fast_recall_ready"):
        return
    tids = np.asarray(case["tids"], dtype=str)
    manifest_ids = case["manifest_trace_ids"]
    a_exposed_ids = case["a_exposed_ids"]
    case["_manifest_mask"] = np.fromiter(
        (tid in manifest_ids for tid in tids),
        dtype=bool,
        count=len(tids),
    )
    case["_a_exposed_mask"] = np.fromiter(
        (tid in a_exposed_ids for tid in tids),
        dtype=bool,
        count=len(tids),
    )
    tie_rank = np.empty(len(tids), dtype=np.int64)
    tie_rank[np.argsort(tids, kind="mergesort")] = np.arange(len(tids), dtype=np.int64)
    case["_trace_tie_rank"] = tie_rank
    case["_has_duplicate_tids"] = len(set(tids.tolist())) != len(tids)
    case["_fast_recall_ready"] = True


async def build_cases():
    labels = external_case_labels()
    manifests = manifest_lookup()
    cases = []
    needed = {"TraceID", "EdgeCaseSamplingProb"}
    async with Pool() as pool:
        for case, info in labels.items():
            cdir = BATCH / case
            if not cdir.is_dir():
                continue
            manifest = load_v02_manifest(case, manifests)
            if manifest is None or manifest["a_exposed_support"] <= 0:
                continue
            gt = info.get("inject_pod")
            try:
                res = await UniSage.evaluate(cdir, pool, PIPE_DT)
            except Exception as e:
                print(f"  [skip {case}] evaluate error: {type(e).__name__}: {e}")
                continue
            if not res:
                continue
            _, _, _, tdf, _ = res
            if tdf is None or tdf.empty or not needed.issubset(tdf.columns):
                continue
            frank = cdir / "final_ranking.csv"
            if not frank.exists():
                continue
            cr = pd.read_csv(frank)
            if not {"service_name", "CombinedScore"}.issubset(cr.columns):
                continue
            score_map = {} if cr.empty else dict(zip(cr["service_name"], cr["CombinedScore"]))
            if score_map and "PodName" not in tdf.columns:
                print(f"  [skip {case}] missing PodName for non-empty RCA ranking")
                continue
            tids = _trace_id_series(tdf["TraceID"]).to_numpy()
            pa0 = (
                tdf["AnomalySamplingProb"].fillna(0.0).astype(float).values
                if "AnomalySamplingProb" in tdf.columns
                else np.zeros(len(tdf), dtype=float)
            )
            pods = [str(p).split(", ") for p in tdf["PodName"].values] if "PodName" in tdf.columns else [[] for _ in range(len(tdf))]
            cases.append({
                "case": case,
                "gt": gt,
                "itype": info.get("inject_type"),
                "score_map": score_map,
                "has_rca_ranking": bool(score_map),
                "tids": tids,
                "pe": tdf["EdgeCaseSamplingProb"].astype(float).values,
                "pods": pods,
                "pa0": pa0,
                "manifest_path": manifest["path"],
                "manifest_trace_ids": manifest["trace_ids"],
                "manifest_n": manifest["n_base"],
                "a_exposed_ids": manifest["a_exposed_ids"],
                "a_exposed_support": manifest["a_exposed_support"],
            })
    return cases


def compatible_cache():
    if CACHE.exists():
        return CACHE
    candidates = [
        path
        for path in RES.glob(f"{_cache_prefix}*{_cache_suffix}")
        if path != CACHE
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


async def run():
    cache_path = compatible_cache() if os.environ.get("FORCE_RERUN", "0") != "1" else None
    if cache_path is not None:
        cases = pickle.loads(cache_path.read_bytes())
        cache_kind = "exact" if cache_path == CACHE else "compatible"
        print(f"Loaded {len(cases)} {cache_kind} cached cases from {cache_path} (FORCE_RERUN=1 to rebuild).")
    else:
        RES.mkdir(exist_ok=True)
        try:
            cases = await build_cases()
        except Exception as exc:
            write_blocker_note(f"{type(exc).__name__}: {exc}")
            raise
        CACHE.write_bytes(pickle.dumps(cases))
        print(f"Built and cached {len(cases)} cases to {CACHE}.")

    n = len(cases)
    if n == 0:
        write_blocker_note("No usable cases after v0.2 primary-scope manifest filtering and pipeline evaluation.")
        print("No usable cases (no anomalies detected / missing columns / no v0.2 A_exposed support).")
        return
    n_labeled = sum(1 for c in cases if c["a_exposed_support"] > 0)
    for c in cases:
        c.setdefault("has_rca_ranking", bool(c.get("score_map")))
    n_rca = sum(1 for c in cases if c["has_rca_ranking"])
    print(f"Cases: {n} ({n_labeled} with v0.2 A_exposed support) from {BATCH}.")

    # --- Sanity: eps=0 recomputed p_a must reproduce the pipeline's AnomalySamplingProb ---
    max_pa_diff = 0.0
    for c in cases:
        pa = trace_pa_vec(c["pods"], norm_for_pa(c["score_map"]))
        if len(pa) == len(c["pa0"]):
            max_pa_diff = max(max_pa_diff, float(np.max(np.abs(pa - c["pa0"]))))
    print(f"Sanity: max |recomputed p_a - pipeline p_a| at eps=0 = {max_pa_diff:.2e}  "
          f"({'OK' if max_pa_diff < 1e-6 else 'WARNING: recompute not faithful'})")

    # --- Edge-only floor (p_e ranking, eps-independent), averaged over cases ---
    edge = {B: [] for B in BUDGETS}
    for c in cases:
        cv = a_exposed_recall_of_scores(c, c["pe"], BUDGETS)
        if cv:
            for B in BUDGETS:
                edge[B].append(cv[B])
    edge_mean = {B: (sum(edge[B]) / len(edge[B])) if edge[B] else float("nan") for B in BUDGETS}

    method_names = ["unisage_v1", "product_only", "max_composition", "analysis_only"]

    # --- Sweep eps x seeds: RCA accuracy + composition-specific A_exposed recall ---
    rows = []
    for eps in EPS_GRID:
        n_seeds = 1 if eps == 0.0 else N_SEEDS
        for seed in range(n_seeds):
            ranks = []
            recalls = {name: {B: [] for B in BUDGETS} for name in method_names}
            for c in cases:
                key = zlib.crc32(c["case"].encode()) ^ (seed * 1000003) ^ int(round(eps * 1e6))
                rng = np.random.default_rng(key & 0xFFFFFFFF)
                deg = degrade_map(c["score_map"], eps, rng)
                ranks.append(rca_rank(deg, c["gt"]))
                pa = trace_pa_vec(c["pods"], norm_for_pa(deg))
                jit = rng.uniform(0, 1e-9, size=len(pa))  # neutral tie-break for analysis-only
                product_score = c["pe"] * pa
                score_jobs = [
                    (["product_only"], product_score, None),
                    (["max_composition"], np.maximum(c["pe"], pa), None),
                    (["analysis_only"], pa, jit),
                ]
                if c["has_rca_ranking"]:
                    score_jobs[0][0].append("unisage_v1")
                else:
                    score_jobs.append((["unisage_v1"], c["pe"], None))
                for methods, scores, score_jitter in score_jobs:
                    cv = a_exposed_recall_of_scores(
                        c,
                        scores,
                        BUDGETS,
                        jitter=score_jitter,
                    )
                    if cv:
                        for method in methods:
                            for B in BUDGETS:
                                recalls[method][B].append(cv[B])
            m = len(ranks)
            row = {
                "eps": eps, "seed": seed,
                "AC1": sum(1 for r in ranks if r == 1) / m,
                "AC3": sum(1 for r in ranks if 1 <= r <= 3) / m,
                "MRR": sum(1.0 / r for r in ranks if r > 0) / m,
            }
            for B in BUDGETS:
                b = int(B * 100)
                for method in method_names:
                    values = recalls[method][B]
                    row[f"{method}_{b}"] = (sum(values) / len(values)) if values else float("nan")
            rows.append(row)

    df = pd.DataFrame(rows)
    df.insert(0, "dataset", DATASET)
    df.insert(1, "batch", BATCH.name)
    df.insert(2, "attr", ATTR)
    df.insert(3, "edge", EDGE)
    df.insert(4, "pipeline_fingerprint", FINGERPRINT)
    df.insert(5, "cases", n)
    df.insert(6, "cases_with_a_exposed_support", n_labeled)
    df.insert(7, "cases_with_rca_ranking", n_rca)
    df.insert(8, "cases_without_rca_ranking", n - n_rca)
    df.insert(9, "label_manifest_index", LABEL_INDEX.as_posix())
    df.insert(10, "label_manifest_index_token", _manifest_token)
    for B in BUDGETS:
        df[f"edge_only_fallback_{int(B*100)}"] = edge_mean[B]
    RES.mkdir(exist_ok=True)
    df.to_csv(OUT_STEM.with_suffix(".csv"), index=False)
    for suffix in [".blocker.csv", ".blocker.md"]:
        OUT_STEM.with_suffix(suffix).unlink(missing_ok=True)

    metric_cols = (["AC1", "AC3", "MRR"]
                   + [f"{method}_{int(B*100)}" for method in method_names for B in BUDGETS])
    g = df.groupby("eps")[metric_cols].agg(["mean", "std"])

    def cell(eps, col):
        mean = g.loc[eps, (col, "mean")]
        std = g.loc[eps, (col, "std")]
        return f"{mean:.3f}" if pd.isna(std) else f"{mean:.3f}±{std:.3f}"

    L = []
    L.append(f"# RQ4 RCA quality-benefit curve - {BATCH} ({DATASET})\n")
    L.append(f"- Cases: {n} ({n_labeled} with v0.2 A_exposed support; {n_rca} with RCA rankings, "
             f"{n - n_rca} edge-only fallbacks).  Seeds/eps: {N_SEEDS} "
             f"(eps=0 deterministic).  Attribution: {'target' if TARGET else 'pod'}.  "
             f"Edge reference: {EDGE}.  Propagation: OFF.")
    L.append(f"- Pipeline fingerprint: `{FINGERPRINT}`.")
    L.append(f"- v0.2 manifest index: `{LABEL_INDEX}` (`{_manifest_token}`).")
    L.append("- Metric scope: macro mean of frozen v0.2 `A_exposed` recall at each budget; "
             "labels are sampler/RCA-independent primary-scope trace manifests.")
    L.append(f"- eps=0 sanity (recomputed p_a vs pipeline): max diff = {max_pa_diff:.2e}.")
    L.append("- Degradation: partial random permutation of the per-service score vector "
             "(eps = fraction of services shuffled). eps=1 => RCA accuracy at chance.")
    L.append("- `unisage_v1` uses `p_e * p_a` when an RCA ranking exists and the actual "
             "edge-only fallback `p_e` when no analysis signal exists; `product_only` is "
             "the no-fallback product score.")
    L.append(f"- Edge-only fallback floor (p_e, eps-independent): " + "  ".join(
        f"@{int(B*100)}%={edge_mean[B]:.3f}" for B in BUDGETS) + ".")
    L.append(f"- Uniform-random floor = B itself ({', '.join(str(int(B*100))+'%' for B in BUDGETS)}).\n")

    L.append("## RCA accuracy + v0.2 A_exposed recall at 5% budget")
    L.append("| eps | AC@1 | AC@3 | MRR | UniSage-v1 | product-only | max composition | analysis-only | edge-only fallback |")
    L.append("|" + "---|" * 9)
    for eps in EPS_GRID:
        edge_cell = f"{edge_mean[0.05]:.3f}"
        L.append("| {:.1f} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            eps, cell(eps, "AC1"), cell(eps, "AC3"), cell(eps, "MRR"),
            cell(eps, "unisage_v1_5"), cell(eps, "product_only_5"),
            cell(eps, "max_composition_5"), cell(eps, "analysis_only_5"),
            edge_cell))

    L.append("\n## Budget sensitivity for UniSage-v1/product")
    L.append("| eps | @1% | @5% | @10% | @20% |")
    L.append("|" + "---|" * 5)
    for eps in EPS_GRID:
        L.append("| {:.1f} | {} | {} | {} | {} |".format(
            eps, cell(eps, "unisage_v1_1"), cell(eps, "unisage_v1_5"),
            cell(eps, "unisage_v1_10"), cell(eps, "unisage_v1_20")))

    md = "\n".join(L) + "\n"
    OUT_STEM.with_suffix(".md").write_text(md)
    figure_path = write_figure(df, OUT_STEM.with_suffix(".png"))
    print("\n" + md)
    print(f"Wrote {OUT_STEM.with_suffix('.csv')}, {OUT_STEM.with_suffix('.md')}, and {figure_path}")
    write_combined_summary()


def write_figure(df, path):
    series = [
        ("unisage_v1_5", "UniSage-v1/product", "#1f77b4"),
        ("max_composition_5", "max(p_e, p_a)", "#ff7f0e"),
        ("analysis_only_5", "analysis only", "#2ca02c"),
        ("edge_only_fallback_5", "edge-only fallback", "#d62728"),
    ]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        svg_path = path.with_suffix(".svg")
        write_svg_figure(df, svg_path, series, f"matplotlib unavailable: {type(exc).__name__}: {exc}")
        print(f"Wrote SVG figure fallback {svg_path}")
        return svg_path

    means = df.groupby("eps").mean(numeric_only=True)
    stds = df.groupby("eps").std(numeric_only=True)
    x = means.index.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for column, label, color in series:
        if column not in means:
            continue
        y = means[column].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", linewidth=2.0, label=label, color=color)
        if column in stds:
            err = stds[column].fillna(0.0).to_numpy(dtype=float)
            ax.fill_between(x, y - err, y + err, alpha=0.12, color=color)

    ax.set_xlabel("RCA degradation eps (fraction of services shuffled)")
    ax.set_ylabel("v0.2 A_exposed recall @ 5% budget")
    ax.set_title(f"{DATASET}/{BATCH.name}: RCA quality-benefit curve")
    ax.set_xlim(min(EPS_GRID), max(EPS_GRID))
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, axis="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_svg_figure(df, path, series, note):
    import html

    means = df.groupby("eps").mean(numeric_only=True)
    width, height = 900, 520
    left, right, top, bottom = 88, 230, 58, 78
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_min, x_max = min(EPS_GRID), max(EPS_GRID)
    y_min, y_max = 0.0, 1.02

    def sx(x):
        return left + ((float(x) - x_min) / (x_max - x_min)) * plot_w

    def sy(y):
        return top + (1.0 - ((float(y) - y_min) / (y_max - y_min))) * plot_h

    def text(x, y, body, size=13, anchor="middle", extra=""):
        return (
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" '
            f'font-size="{size}" text-anchor="{anchor}" fill="#222" {extra}>{html.escape(str(body))}</text>'
        )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        text(width / 2, 28, f"{DATASET}/{BATCH.name}: RCA quality-benefit curve", 18),
        text(width / 2, height - 24, "RCA degradation eps (fraction of services shuffled)", 14),
        text(
            20,
            top + plot_h / 2,
            "v0.2 A_exposed recall @ 5% budget",
            14,
            extra=f'transform="rotate(-90 20 {top + plot_h / 2:.1f})"',
        ),
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333" stroke-width="1.4"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333" stroke-width="1.4"/>',
    ]

    for tick in np.linspace(0.0, 1.0, 6):
        x = sx(tick)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#e6e6e6"/>')
        parts.append(text(x, top + plot_h + 22, f"{tick:.1f}", 12))
    for tick in np.linspace(0.0, 1.0, 6):
        y = sy(tick)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e6e6e6"/>')
        parts.append(text(left - 12, y + 4, f"{tick:.1f}", 12, anchor="end"))

    for idx, (column, label, color) in enumerate(series):
        if column not in means:
            continue
        points = []
        for eps, row in means.iterrows():
            value = row[column]
            if pd.isna(value):
                continue
            points.append((sx(eps), sy(value)))
        if not points:
            continue
        path_data = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        parts.append(f'<polyline points="{path_data}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for x, y in points:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" stroke="white" stroke-width="1"/>')
        lx = left + plot_w + 28
        ly = top + 30 + idx * 28
        parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 24}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        parts.append(text(lx + 32, ly + 4, label, 12, anchor="start"))

    parts.append(text(left + plot_w, height - 8, note, 10, anchor="end"))
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def write_blocker_note(reason):
    RES.mkdir(parents=True, exist_ok=True)
    text = "\n".join([
        f"# RQ4 RCA quality-benefit curve blocker - {BATCH} ({DATASET})",
        "",
        f"- Reason: {reason}",
        f"- Intended metric: frozen v0.2 `A_exposed` recall from `{LABEL_INDEX}`.",
        f"- Command context: dataset=`{DATASET}`, batch=`{BATCH}`, attr=`{ATTR}`, edge=`{EDGE}`, propagation=OFF.",
        "- No LaTeX or chapter-design files were modified by this script.",
        "",
    ])
    OUT_STEM.with_suffix(".blocker.md").write_text(text)
    pd.DataFrame(
        [{"dataset": DATASET, "batch": BATCH.name, "reason": reason, "label_manifest_index": LABEL_INDEX.as_posix()}]
    ).to_csv(OUT_STEM.with_suffix(".blocker.csv"), index=False)


def write_combined_summary():
    """Aggregate curves with matching attr/edge and frozen v0.2 manifest token."""
    pattern = f"rca_benefit_curve_*_{ATTR}_{EDGE}_v02.csv"
    frames = []
    for path in sorted(RES.glob(pattern)):
        frame = pd.read_csv(path)
        same_manifest = (
            "label_manifest_index_token" in frame
            and (frame["label_manifest_index_token"] == _manifest_token).all()
        )
        same_attr_edge = (
            "attr" in frame
            and "edge" in frame
            and (frame["attr"] == ATTR).all()
            and (frame["edge"] == EDGE).all()
        )
        if same_manifest and same_attr_edge:
            frames.append(frame)
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(COMBINED_CSV, index=False)
    fingerprints = ", ".join(
        f"`{fp}`" for fp in sorted(combined["pipeline_fingerprint"].dropna().astype(str).unique())
    )

    lines = [
        "# RQ4 RCA quality-benefit curves - cross-batch summary\n",
        f"- Pipeline fingerprints: {fingerprints}. Attribution: `{ATTR}`. "
        f"Edge reference: `{EDGE}`. Propagation: OFF.",
        f"- Metric scope: frozen v0.2 `A_exposed` recall from `{LABEL_INDEX}` (`{_manifest_token}`).",
        "- Values at degraded settings are mean +/- std across permutation seeds; "
        "the edge floor is independent of degradation.\n",
        "| dataset | batch | cases | eps | AC@1 | UniSage-v1@5% | max@5% | analysis@5% | edge@5% | UniSage-v1@10% | edge@10% |",
        "|" + "---|" * 11,
    ]
    for (dataset, batch), group in combined.groupby(["dataset", "batch"], sort=True):
        cases = int(group["cases"].iloc[0])
        for eps in [0.0, 0.5, 1.0]:
            rows = group[group["eps"] == eps]
            if rows.empty:
                continue

            def mean_std(col):
                mean = rows[col].mean()
                std = rows[col].std()
                return f"{mean:.3f}" if pd.isna(std) else f"{mean:.3f}+/-{std:.3f}"

            lines.append(
                f"| {dataset} | {batch} | {cases} | {eps:.1f} | {mean_std('AC1')} | "
                f"{mean_std('unisage_v1_5')} | {mean_std('max_composition_5')} | "
                f"{mean_std('analysis_only_5')} | {rows['edge_only_fallback_5'].iloc[0]:.3f} | "
                f"{mean_std('unisage_v1_10')} | {rows['edge_only_fallback_10'].iloc[0]:.3f} |"
            )
    COMBINED_MD.write_text("\n".join(lines) + "\n")
    print(f"Wrote {COMBINED_CSV} and {COMBINED_MD}")


if __name__ == "__main__":
    asyncio.run(run())
