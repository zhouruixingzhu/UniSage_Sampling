import hashlib
import os
import re
import pandas as pd
import numpy as np
from collections import defaultdict, deque
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from typing import List, Tuple, Dict, Set, Optional
from scipy.special import expit


TRACE_COMPOSITION_PRODUCT = "product"
TRACE_COMPOSITION_SLOT_GATED_RESERVED = "slot_gated_reserved_budget"
TRACE_COMPOSITION_DYNAMIC_PATTERN_GUARD = "dynamic_product_pattern_guard"
DEFAULT_TRACE_BUDGETS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.10, 0.20)
DEFAULT_TRACE_RESERVE_MIN_SLOTS = 50
DEFAULT_TRACE_PATTERN_GUARD_MIN_BUDGET = 0.05
LOG_COMPOSITION_MAX = "max"
LOG_COMPOSITION_PRODUCT = "product"
LOG_COMPOSITION_WEIGHTED = "weighted"
LOG_COMPOSITION_RESERVED_BUDGET = "reserved_budget"
LOG_COMPOSITION_EDGE = "edge"
LOG_COMPOSITION_ANALYSIS = "analysis"
DEFAULT_LOG_BUDGETS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.10, 0.20)
DEFAULT_LOG_PRIMARY_BUDGET = 0.05
DEFAULT_LOG_RESERVED_DIAGNOSTIC_PCT = 0.60
DEFAULT_LOG_RESERVED_RARITY_PCT = 0.25
FAULT_TEXT_RE = re.compile(
    r"\b(?:error|exception|fail(?:ed|ure)?|timeout|unavailable|5\d\d|fatal)\b",
    re.IGNORECASE,
)


def aggregate_trace_duration_stats(input_dir: Path, output_file: Path):
    input_dir = Path(input_dir)
    output_file = Path(output_file)
    trace_files = [input_dir / f for f in os.listdir(input_dir) if f.endswith("_trace.csv")]

    if not trace_files:
        print(f"No *_trace.csv files found in {input_dir}")
        return pd.DataFrame()

    df_list = []
    for file in trace_files:
        # print(f"Reading file: {file}")
        df = pd.read_csv(file, usecols=["TraceID", "Duration", "PodName", "OperationName", "SpanID", "ParentID"])
        df_list.append(df)

    combined_df = pd.concat(df_list, ignore_index=True)

    agg_df = combined_df.groupby(["TraceID", "PodName", "OperationName"])["Duration"].agg(
        MeanDuration="mean", StdDuration="std"
    ).reset_index()

    agg_df["StdDuration"] = agg_df["StdDuration"].fillna(0.0)
    agg_df.to_csv(output_file, index=False)
    return agg_df


def aggregate_trace(input_path: Path):
    input_path = Path(input_path)
    
    if input_path.is_file():
        # Single file case
        if input_path.suffix == '.csv':
            return pd.read_csv(input_path)
        else:
            print(f"File {input_path} is not a CSV file")
            return pd.DataFrame()
    elif input_path.is_dir():
        # Directory case - look for *_trace.csv files
        trace_files = [input_path / f for f in os.listdir(input_path) if f.endswith("_trace.csv")]
        
        if not trace_files:
            print(f"No *_trace.csv files found in {input_path}")
            return pd.DataFrame()
        
        df_list = []
        for file in trace_files:
            df = pd.read_csv(file)
            df_list.append(df)
        
        return pd.concat(df_list, ignore_index=True)
    else:
        print(f"Path {input_path} does not exist")
        return pd.DataFrame()


def extract_base_pod_name(pod_name: str) -> str:
    match = re.match(r"(.+)-[a-z0-9]{9,}-[a-z0-9]{5}$", pod_name)
    if match:
        return match.group(1)
    return pod_name


def compute_latency_sampling_prob(ref_df: pd.DataFrame, trace_stats_df: pd.DataFrame) -> pd.DataFrame:
    if trace_stats_df.empty:
        return pd.DataFrame(columns=['TraceID', 'LatencyMaxZScore', 'LatencySamplingProb'])
    
    ref_df['BasePodName'] = ref_df['PodName'].apply(extract_base_pod_name)
    trace_stats_df['BasePodName'] = trace_stats_df['PodName'].apply(extract_base_pod_name)

    ref = {
        (row.BasePodName, row.OperationName): (row.MeanDuration, max(row.StdDuration, 1e-6))
        for _, row in ref_df.iterrows()
    }

    trace_zscores = defaultdict(list)
    all_trace_ids = set(trace_stats_df['TraceID'].unique())
    # print("all_trace_ids:", len(all_trace_ids))

    for _, row in trace_stats_df.iterrows():
        key = (row.BasePodName, row.OperationName)
        duration = row.MeanDuration
        trace_id = row.TraceID

        if key in ref:
            mean_ref, std_ref = ref[key]
            z = abs((duration - mean_ref) / std_ref)
            trace_zscores[trace_id].append(z)

    trace_sampling_probs = {}

    for trace_id in all_trace_ids:
        zlist = trace_zscores.get(trace_id, [])
        if zlist:
            max_z = max(zlist)
            prob = expit(0.5 * (max_z - 5))  # sigmoid(center=5, slope=0.5)
        else:
            max_z = 0.0
            prob = 0.0
        trace_sampling_probs[trace_id] = (max_z, prob)

    output_df = pd.DataFrame([
        {'TraceID': tid, 'LatencyMaxZScore': max_z, 'LatencySamplingProb': prob}
        for tid, (max_z, prob) in trace_sampling_probs.items()
    ])

    return output_df


def build_trace_paths(df: pd.DataFrame) -> Dict[str, List[Tuple[str]]]:
    traces = defaultdict(list)
    for _, row in df.iterrows():
        traces[row['TraceID']].append((
            row['SpanID'], row['ParentID'], 
            f"{row['BasePodName']}::{row['OperationName']}"
        ))

    trace_paths = {}

    for trace_id, spans in traces.items():
        id_to_node = {s[0]: s[2] for s in spans}
        parent_map = defaultdict(list)
        for span_id, parent_id, _ in spans:
            parent_map[parent_id].append(span_id)

        roots = [sid for sid, parent, _ in spans if parent is None or parent not in id_to_node]
        paths = []

        for root in roots:
            stack = deque([[root]])
            while stack:
                path = stack.pop()
                current = path[-1]
                if current not in parent_map:
                    paths.append(tuple(id_to_node[sid] for sid in path))
                else:
                    for child in parent_map[current]:
                        stack.append(path + [child])

        trace_paths[trace_id] = paths

    return trace_paths


def jaccard_similarity(p1: Tuple[str], p2: Tuple[str]) -> float:
    set1, set2 = set(p1), set(p2)
    if not set1 and not set2:
        return 1.0
    return len(set1 & set2) / len(set1 | set2)


def compute_topology_sampling_prob(ref_df: pd.DataFrame, incoming_df: pd.DataFrame) -> pd.DataFrame:
    for df in [ref_df, incoming_df]:
        df['BasePodName'] = df['PodName'].apply(extract_base_pod_name)

    ref_paths = build_trace_paths(ref_df)
    incoming_paths = build_trace_paths(incoming_df)

    all_ref_path_set = set()
    for paths in ref_paths.values():
        for p in paths:
            all_ref_path_set.add(p)

    results = []
    jaccard_cache = {}

    for trace_id, paths in incoming_paths.items():
        if not paths:
            results.append(
                {
                    'TraceID': trace_id,
                    'TopologyScore': 1.0,
                    'TopologySamplingProb': np.tanh(1.0),
                    'CurrentPath': '',
                    'BestRefPath': '',
                }
            )
            continue

        if not all_ref_path_set:
            results.append(
                {
                    'TraceID': trace_id,
                    'TopologyScore': 1.0,
                    'TopologySamplingProb': np.tanh(1.0),
                    'CurrentPath': ' -> '.join(paths[0]),
                    'BestRefPath': '',
                }
            )
            continue

        max_sim = 0.0
        best_match = None
        best_current = None

        for p1 in paths:
            if p1 in all_ref_path_set:
                sim = 1.0
                match = p1
            else:
                sim = 0.0
                match = None
                for p2 in all_ref_path_set:
                    key = (p1, p2)
                    if key not in jaccard_cache:
                        jaccard_cache[key] = jaccard_similarity(p1, p2)
                    cur_sim = jaccard_cache[key]
                    if cur_sim > sim:
                        sim = cur_sim
                        match = p2

            if sim > max_sim:
                max_sim = sim
                best_current = p1
                best_match = match

        topology_score = 1 - max_sim
        topo_prob = np.tanh(topology_score)

        result = {
            'TraceID': trace_id,
            'TopologyScore': topology_score,
            'TopologySamplingProb': topo_prob,
        }

        # if topology_score > 0.0:
        result['CurrentPath'] = ' -> '.join(best_current) if best_current else ''
        result['BestRefPath'] = ' -> '.join(best_match) if best_match else ''

        results.append(result)

    return pd.DataFrame(results)


def compute_structural_pattern_keys(incoming_df: pd.DataFrame) -> pd.DataFrame:
    """Build label-independent trace structural signatures from parent-child span edges."""
    if incoming_df is None or incoming_df.empty:
        return pd.DataFrame(columns=["TraceID", "StructuralPatternKey"])
    required = {"TraceID", "SpanID", "ParentID"}
    if not required.issubset(incoming_df.columns):
        return pd.DataFrame(columns=["TraceID", "StructuralPatternKey"])

    df = incoming_df.copy()
    if "BasePodName" in df.columns:
        service = df["BasePodName"].fillna("").astype(str)
    elif "PodName" in df.columns:
        service = df["PodName"].apply(extract_base_pod_name).fillna("").astype(str)
    elif "Service" in df.columns:
        service = df["Service"].apply(extract_base_pod_name).fillna("").astype(str)
    elif "ServiceName" in df.columns:
        service = df["ServiceName"].apply(extract_base_pod_name).fillna("").astype(str)
    else:
        return pd.DataFrame(columns=["TraceID", "StructuralPatternKey"])

    if "OperationName" in df.columns:
        operation = df["OperationName"].fillna("").astype(str)
    elif "Operation" in df.columns:
        operation = df["Operation"].fillna("").astype(str)
    elif "SpanName" in df.columns:
        operation = df["SpanName"].fillna("").astype(str)
    else:
        operation = pd.Series("", index=df.index)

    df["TraceID"] = df["TraceID"].astype(str)
    df["SpanID"] = df["SpanID"].astype(str)
    df["ParentID"] = df["ParentID"].fillna("").astype(str)
    df["_struct_service"] = service
    df["_struct_operation"] = operation

    parent_services = df[["TraceID", "SpanID", "_struct_service"]].rename(
        columns={"SpanID": "ParentID", "_struct_service": "_matched_parent_service"}
    )
    spans = df.merge(parent_services, on=["TraceID", "ParentID"], how="left")
    parent_token = spans["ParentID"].str.lower()
    is_root = parent_token.isin({"root", "nan", "none", ""}) | spans["_matched_parent_service"].isna()
    spans["_struct_edge"] = np.where(
        is_root,
        "ROOT>" + spans["_struct_service"] + ":" + spans["_struct_operation"],
        spans["_matched_parent_service"] + ">" + spans["_struct_service"] + ":" + spans["_struct_operation"],
    )
    keys = (
        spans[["TraceID", "_struct_edge"]]
        .drop_duplicates()
        .sort_values(["TraceID", "_struct_edge"])
        .groupby("TraceID")["_struct_edge"]
        .agg("||".join)
        .rename("StructuralPatternKey")
        .reset_index()
    )
    return keys


def compute_edge_case_sampling_prob(ref_rate_df: pd.DataFrame, trace_stats_df: pd.DataFrame, ref_df: pd.DataFrame, incoming_df: pd.DataFrame) -> pd.DataFrame:
    lat_df = compute_latency_sampling_prob(ref_rate_df, trace_stats_df)
    topo_df = compute_topology_sampling_prob(ref_df, incoming_df)

    merged_df = pd.merge(lat_df, topo_df, on='TraceID', how='outer')
    merged_df['EdgeCaseSamplingProb'] = merged_df[['LatencySamplingProb', 'TopologySamplingProb']].max(axis=1)
    
    return merged_df


def compute_anomaly_sampling_prob(
    incoming_df: pd.DataFrame,
    combined_ranking: pd.DataFrame = None
) -> pd.DataFrame:

    raw_score_map = []
    df = combined_ranking
    if df is not None and not df.empty:
        raw_score_map = dict(zip(df['service_name'], df['CombinedScore']))
        # In target mode, normalize service scores per case (phi_s / max) before the sigmoid,
        # so p_a(t) discriminates the correctly-localized fault service from the rest instead of
        # saturating to ~1.0 (large phi_s -> tanh ~ 1 for every trace).
        if os.environ.get("UNISAGE_RCA_ATTR", "target") == "target" and raw_score_map:
            _mx = max(raw_score_map.values())
            if _mx and _mx > 0:
                raw_score_map = {k: v / _mx for k, v in raw_score_map.items()}

    trace_scores = {}
    trace_pod_names = {}
    grouped = incoming_df.groupby('TraceID')

    for trace_id, group in grouped:
        pod_names = group['PodName'].unique()
        all_scores = []

        for pod in pod_names:
            if pod in raw_score_map:
                all_scores.append(raw_score_map[pod])

        if all_scores:
            prob = np.tanh(max(all_scores))  # 或者用 np.tanh(max_score) 做非线性映射
        else:
            prob = 0.0

        trace_scores[trace_id] = prob
        trace_pod_names[trace_id] = ', '.join(pod_names)

    output_df = pd.DataFrame([
        {'TraceID': tid, 'PodName': trace_pod_names[tid], 'AnomalySamplingProb': prob}
        for tid, prob in trace_scores.items()
    ])

    return output_df


def budget_token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def parse_trace_budgets(value: Optional[str] = None) -> List[float]:
    raw = value if value is not None else os.environ.get("UNISAGE_TRACE_BUDGETS", "")
    if not raw:
        return list(DEFAULT_TRACE_BUDGETS)
    budgets: List[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        budget = float(item)
        if budget <= 0:
            raise ValueError(f"Trace budget must be positive, got {budget}")
        budgets.append(budget)
    return budgets or list(DEFAULT_TRACE_BUDGETS)


def _rank_percentile(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series(0.0, index=values.index)
    ranks = numeric.rank(method="average", pct=True, na_option="bottom")
    return ranks.fillna(0.0).astype(float)


def _top_by_score(frame: pd.DataFrame, score_column: str, k: int) -> pd.DataFrame:
    if k <= 0 or frame.empty:
        return frame.iloc[0:0].copy()
    ranked = frame.dropna(subset=[score_column]).copy()
    ranked["_sort_score"] = pd.to_numeric(ranked[score_column], errors="coerce")
    ranked = ranked.sort_values(["_sort_score", "TraceID"], ascending=[False, True], kind="mergesort")
    return ranked.head(min(k, len(ranked))).drop(columns=["_sort_score"])


def _take_new_by_score(frame: pd.DataFrame, score_column: str, k: int, used: set[str]) -> pd.DataFrame:
    if k <= 0:
        return frame.iloc[0:0].copy()
    available = frame[~frame["TraceID"].astype(str).isin(used)].copy()
    return _top_by_score(available, score_column, k)


def _trace_pattern_key(frame: pd.DataFrame) -> pd.Series:
    if "CurrentPath" in frame.columns:
        path = frame["CurrentPath"].fillna("").astype(str).str.strip()
    else:
        path = pd.Series("", index=frame.index)
    if "PodName" in frame.columns:
        pod = frame["PodName"].fillna("").astype(str).str.strip()
    else:
        pod = pd.Series("", index=frame.index)
    key = path.where(path != "", pod)
    return key.where(key != "", frame["TraceID"].astype(str))


def _take_pattern_novel(frame: pd.DataFrame, k: int, used: set[str]) -> pd.DataFrame:
    if k <= 0:
        return frame.iloc[0:0].copy()
    available = frame[~frame["TraceID"].astype(str).isin(used)].copy()
    if available.empty:
        return available
    available = available.sort_values(
        ["TracePatternCount", "EdgeRankPercentile", "AnalysisRankPercentile", "TraceID"],
        ascending=[True, False, False, True],
        kind="mergesort",
    )
    selected = []
    seen_patterns = set()
    for idx, row in available.iterrows():
        pattern = row["TracePatternKey"]
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


def _reserved_budget_selection(
    frame: pd.DataFrame,
    k: int,
    primary_pct: float = 0.70,
    edge_pct: float = 0.20,
    pattern_pct: float = 0.10,
    primary_score: str = "AnalysisProductScore",
) -> pd.DataFrame:
    """70% analysis/product, 20% edge, and 10% trace-pattern novelty."""
    if k <= 0:
        return frame.iloc[0:0].copy()
    primary_n = min(k, max(1, int(round(k * primary_pct))))
    edge_n = min(k - primary_n, int(round(k * edge_pct)))
    pattern_n = max(0, k - primary_n - edge_n)

    parts = []
    used: set[str] = set()

    primary = _take_new_by_score(frame, primary_score, primary_n, used)
    parts.append(primary)
    used.update(primary["TraceID"].astype(str))

    edge = _take_new_by_score(frame, "EdgeRankPercentile", edge_n, used)
    parts.append(edge)
    used.update(edge["TraceID"].astype(str))

    pattern = _take_pattern_novel(frame, pattern_n, used)
    parts.append(pattern)
    used.update(pattern["TraceID"].astype(str))

    if len(used) < k:
        fill = _take_new_by_score(frame, primary_score, k - len(used), used)
        parts.append(fill)
    return pd.concat(parts, ignore_index=False).drop_duplicates("TraceID").head(k)


def _product_context(frame: pd.DataFrame, k: int) -> Dict[str, float]:
    if k <= 0 or frame.empty:
        return {"structural_patterns": 0.0, "product_structural_coverage": np.nan, "path_like_fraction": np.nan}
    pattern_col = "StructuralPatternKey" if "StructuralPatternKey" in frame.columns else "TracePatternKey"
    all_patterns = frame[pattern_col].nunique(dropna=False)
    product = _top_by_score(frame, "CombinedSamplingProb", k)
    product_patterns = product[pattern_col].nunique(dropna=False) if not product.empty else 0
    path_like = frame["TracePatternKey"].astype(str).str.contains("->", regex=False).mean()
    return {
        "structural_patterns": float(all_patterns),
        "product_structural_coverage": product_patterns / all_patterns if all_patterns else np.nan,
        "path_like_fraction": float(path_like),
    }


def _dynamic_pattern_guard_selection(
    frame: pd.DataFrame,
    k: int,
    reserve_min_slots: int,
    budget: float,
    min_budget: float,
) -> Tuple[pd.DataFrame, str]:
    if budget < min_budget:
        return _top_by_score(frame, "CombinedSamplingProb", k), f"product_budget_floor{budget_token(min_budget)}"
    if k < reserve_min_slots:
        return _top_by_score(frame, "CombinedSamplingProb", k), "product_small_budget"
    ctx = _product_context(frame, k)
    patterns = ctx["structural_patterns"]
    cov = ctx["product_structural_coverage"]
    path_like = ctx["path_like_fraction"]
    if (
        not np.isnan(patterns)
        and patterns <= 20
        and not np.isnan(cov)
        and cov < 0.95
        and not np.isnan(path_like)
        and path_like >= 0.50
    ):
        kept = _reserved_budget_selection(
            frame,
            k,
            primary_pct=0.70,
            edge_pct=0.20,
            pattern_pct=0.10,
            primary_score="CombinedSamplingProb",
        )
        return kept, f"pattern_guard_open_patterns{patterns:.0f}_cov{cov:.3f}_pathlike{path_like:.3f}"
    return _top_by_score(frame, "CombinedSamplingProb", k), f"pattern_guard_product_patterns{patterns:.0f}_cov{cov:.3f}"


def _add_trace_composition_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["TraceID"] = out["TraceID"].astype(str)
    for column in ("CombinedSamplingProb", "EdgeCaseSamplingProb", "AnomalySamplingProb"):
        if column not in out.columns:
            out[column] = np.nan

    product_score = pd.to_numeric(out["CombinedSamplingProb"], errors="coerce").fillna(0.0)
    edge_score = pd.to_numeric(out["EdgeCaseSamplingProb"], errors="coerce").fillna(0.0)
    analysis_score = pd.to_numeric(out["AnomalySamplingProb"], errors="coerce").fillna(0.0)
    out["ProductRankPercentile"] = _rank_percentile(product_score)
    out["EdgeRankPercentile"] = _rank_percentile(edge_score)
    out["AnalysisRankPercentile"] = _rank_percentile(analysis_score)
    out["AnalysisProductScore"] = 0.5 * out["AnalysisRankPercentile"] + 0.5 * out["ProductRankPercentile"]

    out["TracePatternKey"] = _trace_pattern_key(out)
    pattern_counts = out["TracePatternKey"].value_counts(dropna=False)
    out["TracePatternCount"] = out["TracePatternKey"].map(pattern_counts).fillna(len(out)).astype(float)
    if "StructuralPatternKey" not in out.columns:
        out["StructuralPatternKey"] = out["TracePatternKey"]
    out["StructuralPatternKey"] = out["StructuralPatternKey"].fillna(out["TracePatternKey"]).astype(str)
    structural_counts = out["StructuralPatternKey"].value_counts(dropna=False)
    out["StructuralPatternCount"] = out["StructuralPatternKey"].map(structural_counts).fillna(len(out)).astype(float)
    return out


def apply_trace_composition_strategy(
    sampling_df: pd.DataFrame,
    method: Optional[str] = None,
    budgets: Optional[List[float]] = None,
    reserve_min_slots: Optional[int] = None,
    pattern_guard_min_budget: Optional[float] = None,
) -> pd.DataFrame:
    """Add optional budget-aware trace-retention decisions to a sampling frame.

    The default product composition remains the canonical `CombinedSamplingProb`.
    `slot_gated_reserved_budget` adds per-budget decision columns so evaluation
    and downstream storage can exercise the candidate without overwriting product
    scores during the validation phase.
    """
    if sampling_df is None or sampling_df.empty:
        return sampling_df

    method = (method or os.environ.get("UNISAGE_TRACE_COMPOSITION", TRACE_COMPOSITION_DYNAMIC_PATTERN_GUARD)).strip().lower()
    if method in {"", TRACE_COMPOSITION_PRODUCT, "current_product"}:
        return sampling_df
    supported_methods = {TRACE_COMPOSITION_SLOT_GATED_RESERVED, TRACE_COMPOSITION_DYNAMIC_PATTERN_GUARD}
    if method not in supported_methods:
        raise ValueError(f"Unsupported UNISAGE_TRACE_COMPOSITION={method}")

    budgets = budgets if budgets is not None else parse_trace_budgets()
    reserve_min_slots = (
        int(reserve_min_slots)
        if reserve_min_slots is not None
        else int(os.environ.get("UNISAGE_TRACE_RESERVE_MIN_SLOTS", DEFAULT_TRACE_RESERVE_MIN_SLOTS))
    )
    pattern_guard_min_budget = (
        float(pattern_guard_min_budget)
        if pattern_guard_min_budget is not None
        else float(os.environ.get("UNISAGE_TRACE_PATTERN_GUARD_MIN_BUDGET", DEFAULT_TRACE_PATTERN_GUARD_MIN_BUDGET))
    )

    out = _add_trace_composition_features(sampling_df)
    n_ranked = len(out)
    for budget in budgets:
        k = min(n_ranked, max(1, int(round(n_ranked * float(budget)))))
        token = budget_token(float(budget))
        if method == TRACE_COMPOSITION_DYNAMIC_PATTERN_GUARD:
            kept, mode = _dynamic_pattern_guard_selection(
                out,
                k,
                reserve_min_slots,
                budget=float(budget),
                min_budget=pattern_guard_min_budget,
            )
            prefix = "DynamicPatternGuard"
        elif k < reserve_min_slots:
            kept = _top_by_score(out, "CombinedSamplingProb", k)
            mode = "product_small_budget"
            prefix = "SlotGatedReserved"
        else:
            kept = _reserved_budget_selection(out, k)
            mode = "reserved_budget"
            prefix = "SlotGatedReserved"
        kept_ids = set(kept["TraceID"].astype(str))
        out[f"{prefix}Decision_{token}"] = out["TraceID"].astype(str).isin(kept_ids)
        out[f"{prefix}Mode_{token}"] = mode
        out[f"{prefix}TargetK_{token}"] = k

    out["TraceCompositionStrategy"] = method
    out["TraceReserveMinSlots"] = reserve_min_slots
    out["TracePatternGuardMinBudget"] = pattern_guard_min_budget
    return out


def combined_sampler(
    ref_rate_df: pd.DataFrame,
    trace_stats_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    incoming_df: pd.DataFrame,
    combined_ranking: pd.DataFrame = None
) -> pd.DataFrame:
    
    edge_case_df = compute_edge_case_sampling_prob(ref_rate_df, trace_stats_df, ref_df, incoming_df)
    anomaly_df = compute_anomaly_sampling_prob(incoming_df, combined_ranking)
    structural_df = compute_structural_pattern_keys(incoming_df)

    combined_df = pd.merge(edge_case_df, anomaly_df, on='TraceID', how='outer').fillna(0.0)
    if not structural_df.empty:
        combined_df = pd.merge(combined_df, structural_df, on='TraceID', how='left')
    
    # combined_df['CombinedSamplingProb'] = (combined_df['EdgeCaseSamplingProb'] + combined_df['AnomalySamplingProb']) / 2
    combined_df['CombinedSamplingProb'] = combined_df['EdgeCaseSamplingProb'] * combined_df['AnomalySamplingProb']
    combined_df = combined_df.sort_values(by='CombinedSamplingProb', ascending=False)
    return combined_df




def aggregate_log(input_dir: Path) -> pd.DataFrame:
    all_dfs = []
    for file in input_dir.glob("*.csv"):
        try:
            df = pd.read_csv(file)
            service_name = extract_base_pod_name(file.stem)
            df["Service"] = service_name
            all_dfs.append(df)
        except Exception as e:
            print(f"Error reading {file.name}: {e}")
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        return pd.DataFrame()  # return empty df if no valid files found


def _stable_hash(value: object, length: int = 24) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:length]


def _first_existing_column(frame: pd.DataFrame, candidates: List[str], default: str = "") -> pd.Series:
    for column in candidates:
        if column in frame.columns:
            return frame[column]
    return pd.Series(default, index=frame.index)


def _coalesce_columns(frame: pd.DataFrame, candidates: List[str], default: str = "") -> pd.Series:
    out = pd.Series(default, index=frame.index, dtype=object)
    missing = pd.Series(True, index=frame.index)
    for column in candidates:
        if column not in frame.columns:
            continue
        values = frame[column]
        usable = values.notna() & values.astype(str).str.len().gt(0) & ~values.astype(str).str.lower().isin({"nan", "none", "null"})
        take = missing & usable
        out.loc[take] = values.loc[take]
        missing = missing & ~take
    return out


def _severity_from_number(value: object) -> str:
    try:
        number = int(float(value))
    except Exception:
        return ""
    if number >= 17:
        return "FATAL"
    if number >= 9:
        return "ERROR"
    if number >= 5:
        return "WARN"
    if number > 0:
        return "INFO"
    return ""


def normalize_log_level(value: object) -> str:
    level = str(value or "").strip().upper()
    if not level or level in {"NAN", "NONE", "NULL"}:
        return ""
    if "FATAL" in level:
        return "FATAL"
    if "ERROR" in level or level == "ERR":
        return "ERROR"
    if "WARN" in level:
        return "WARN"
    if "INFO" in level:
        return "INFO"
    if "DEBUG" in level:
        return "DEBUG"
    if "TRACE" in level:
        return "TRACE"
    return level


def normalize_log_frame_for_sampling(df: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize log rows while preserving row count.

    The `LogID` mirrors the v0.2 manifest key for rows with parseable
    timestamps. Rows with invalid timestamps still receive a deterministic
    fallback key so the sampler output remains auditable.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "LogID", "Timestamp", "TraceID", "SpanID", "Service", "ServiceBase",
            "log_level", "log_temp", "Body", "ServiceTemplateKey", "_LogInputRow",
        ])

    frame = df.copy().reset_index(drop=True)
    frame["_LogInputRow"] = np.arange(len(frame))

    frame["TraceID"] = _coalesce_columns(frame, ["TraceID", "TraceId"], "").fillna("").astype(str)
    frame["SpanID"] = _coalesce_columns(frame, ["SpanID", "SpanId"], "").fillna("").astype(str)
    service = _coalesce_columns(
        frame,
        ["Service", "ServiceName", "PodName", "BasePodName", "k8s_container_name", "Container"],
        "",
    )
    frame["Service"] = service.fillna("").astype(str).map(extract_base_pod_name)
    frame["ServiceBase"] = frame["Service"].map(extract_base_pod_name)

    if "log_level" in frame.columns:
        level = frame["log_level"]
    elif "SeverityText" in frame.columns:
        level = frame["SeverityText"]
    elif "severity" in frame.columns:
        level = frame["severity"]
    elif "SeverityNumber" in frame.columns:
        level = frame["SeverityNumber"].map(_severity_from_number)
    else:
        level = pd.Series("", index=frame.index)
    frame["log_level"] = level.map(normalize_log_level)

    body = _first_existing_column(frame, ["Body", "message", "Log"], "").fillna("").astype(str)
    log_temp = _first_existing_column(frame, ["log_temp", "template_mined", "Body", "message", "Log"], "")
    frame["Body"] = body
    frame["log_temp"] = log_temp.fillna("").astype(str)
    empty_template = frame["log_temp"].str.len() == 0
    frame.loc[empty_template, "log_temp"] = frame.loc[empty_template, "Body"]

    if "Timestamp" not in frame.columns:
        frame["Timestamp"] = ""
    raw_ts = frame["Timestamp"].astype(object).where(frame["Timestamp"].notna(), "").astype(str)
    ts = pd.to_datetime(raw_ts, errors="coerce", utc=True)
    valid_ts = ts.notna()
    frame["_timestamp_ns"] = raw_ts
    if valid_ts.any():
        frame.loc[valid_ts, "_timestamp_ns"] = ts.loc[valid_ts].astype("int64").astype(str)
        frame.loc[valid_ts, "Timestamp"] = ts.loc[valid_ts].dt.tz_localize(None)
    frame["LogTimestampValid"] = valid_ts.astype(bool)

    key_columns = ["_timestamp_ns", "TraceID", "Service", "log_level", "log_temp", "Body"]
    key = frame[key_columns].fillna("").astype(str).agg("||".join, axis=1)
    occurrence = frame.groupby(key, sort=False).cumcount().astype(str)
    frame["LogID"] = (key + "||" + occurrence).map(_stable_hash)
    frame["ServiceTemplateKey"] = frame["ServiceBase"].astype(str) + "::" + frame["log_temp"].astype(str)
    return frame


def get_log_level_weight(log_level: str) -> float:
    level = normalize_log_level(log_level)
    if level in {"ERROR", "FATAL"}:
        return 1.0
    if level == "WARN":
        return 0.4
    if level == "INFO":
        return 0.2
    if level in {"DEBUG", "TRACE"}:
        return 0.1
    return 0.1


def count_log_templates(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "ServiceBase", "log_temp", "service_frequency", "service_total", "service_prob",
        "global_frequency", "global_total", "global_prob", "freq",
    ]
    if df is None or df.empty:
        return pd.DataFrame(columns=columns)

    frame = normalize_log_frame_for_sampling(df)
    if frame.empty:
        return pd.DataFrame(columns=columns)

    service_totals = frame.groupby("ServiceBase").size().rename("service_total").reset_index()
    service_counts = (
        frame.groupby(["ServiceBase", "log_temp"])
        .size()
        .rename("service_frequency")
        .reset_index()
        .merge(service_totals, on="ServiceBase", how="left", validate="many_to_one")
    )
    service_counts["service_prob"] = service_counts["service_frequency"] / service_counts["service_total"].clip(lower=1)

    global_total = len(frame)
    global_counts = frame.groupby("log_temp").size().rename("global_frequency").reset_index()
    global_counts["global_total"] = global_total
    global_counts["global_prob"] = global_counts["global_frequency"] / max(1, global_total)

    out = service_counts.merge(global_counts, on="log_temp", how="left", validate="many_to_one")
    out["freq"] = out["service_prob"]
    return out.sort_values(["ServiceBase", "service_prob", "log_temp"], ascending=[True, False, True]).reset_index(drop=True)


def compute_log_anomaly_sampling_prob(
    incoming_df: pd.DataFrame,
    combined_ranking: pd.DataFrame
) -> pd.DataFrame:

    incoming_df = normalize_log_frame_for_sampling(incoming_df)
    if combined_ranking is None or combined_ranking.empty or not {"service_name", "CombinedScore"}.issubset(combined_ranking.columns):
        score_map = {}
    else:
        ranking = combined_ranking.copy()
        ranking["service_name"] = ranking["service_name"].apply(extract_base_pod_name)
        score_map = dict(zip(ranking['service_name'], ranking['CombinedScore']))
    # In target mode, normalize service scores per case (phi_s / max) before the sigmoid so the
    # log anomaly probability discriminates the fault service instead of saturating.
    if os.environ.get("UNISAGE_RCA_ATTR", "target") == "target" and score_map:
        _mx = max(score_map.values())
        if _mx and _mx > 0:
            score_map = {k: v / _mx for k, v in score_map.items()}

    incoming_df['LogAnomalyScore'] = incoming_df['ServiceBase'].map(score_map).fillna(0.0)
    incoming_df['LogLevelWeight'] = incoming_df['log_level'].apply(get_log_level_weight)
    text = (incoming_df["Body"].fillna("").astype(str) + " " + incoming_df["log_temp"].fillna("").astype(str))
    incoming_df["LogFaultTextMatch"] = text.str.contains(FAULT_TEXT_RE)
    incoming_df["LogDiagnosticStrict"] = incoming_df["log_level"].isin({"ERROR", "FATAL"}) | incoming_df["LogFaultTextMatch"]
    incoming_df['LogAnomalySamplingProb'] = np.tanh(incoming_df['LogAnomalyScore']) * incoming_df['LogLevelWeight']

    return incoming_df


def compute_log_edgecase_sampling_prob(incoming_df, ref_df, epsilon=1e-6):
    incoming_df = normalize_log_frame_for_sampling(incoming_df)
    if ref_df is None or ref_df.empty:
        ref = count_log_templates(pd.DataFrame())
    elif {"ServiceBase", "log_temp", "service_prob", "global_prob"}.issubset(ref_df.columns):
        ref = ref_df.copy()
    else:
        ref = count_log_templates(ref_df)

    service_ref = ref[
        ["ServiceBase", "log_temp", "service_frequency", "service_prob"]
    ].drop_duplicates(["ServiceBase", "log_temp"]) if not ref.empty else pd.DataFrame(
        columns=["ServiceBase", "log_temp", "service_frequency", "service_prob"]
    )
    global_ref = ref[
        ["log_temp", "global_frequency", "global_prob"]
    ].drop_duplicates("log_temp") if not ref.empty else pd.DataFrame(
        columns=["log_temp", "global_frequency", "global_prob"]
    )

    n_before = len(incoming_df)
    incoming_df = incoming_df.merge(
        service_ref,
        on=["ServiceBase", "log_temp"],
        how="left",
        validate="many_to_one",
    )
    incoming_df = incoming_df.merge(
        global_ref,
        on="log_temp",
        how="left",
        validate="many_to_one",
    )
    if len(incoming_df) != n_before:
        raise ValueError(f"Log rarity reference join expanded rows: before={n_before} after={len(incoming_df)}")

    has_service = incoming_df["ServiceBase"].fillna("").astype(str).str.len() > 0
    service_prob = pd.to_numeric(incoming_df["service_prob"], errors="coerce").fillna(0.0)
    global_prob = pd.to_numeric(incoming_df["global_prob"], errors="coerce").fillna(0.0)
    service_count = pd.to_numeric(incoming_df["service_frequency"], errors="coerce").fillna(0).astype(int)
    global_count = pd.to_numeric(incoming_df["global_frequency"], errors="coerce").fillna(0).astype(int)

    incoming_df["ReferenceTemplateFrequency"] = np.where(has_service, service_prob, global_prob)
    incoming_df["ReferenceTemplateCount"] = np.where(has_service, service_count, global_count).astype(int)
    incoming_df["LogRarityReferenceScope"] = np.select(
        [
            has_service & (incoming_df["ReferenceTemplateCount"] > 0),
            has_service & (incoming_df["ReferenceTemplateCount"] == 0),
            ~has_service & (incoming_df["ReferenceTemplateCount"] > 0),
            ~has_service & (incoming_df["ReferenceTemplateCount"] == 0),
        ],
        [
            "service_template",
            "service_template_unseen",
            "global_template_fallback",
            "global_template_unseen",
        ],
        default="global_template_unseen",
    )
    incoming_df["LogRarityReferenceMissingService"] = ~has_service
    incoming_df["LogRarityReferenceMissingTemplate"] = incoming_df["ReferenceTemplateCount"] == 0
    incoming_df["prob"] = incoming_df["ReferenceTemplateFrequency"]

    incoming_df['rarity_score'] = -np.log(incoming_df['ReferenceTemplateFrequency'].astype(float) + epsilon)
    incoming_df['LogRaritySamplingProbRaw'] = 2 / (1 + np.exp(-2 * incoming_df['rarity_score'])) - 1

    incoming_df['LogLevelWeight'] = incoming_df['log_level'].apply(get_log_level_weight)
    incoming_df['LogRaritySamplingProb'] = incoming_df['LogRaritySamplingProbRaw'] * incoming_df['LogLevelWeight']
    incoming_df["LogServiceTemplateNovelty"] = incoming_df["ReferenceTemplateCount"] == 0
    incoming_df["ServiceTemplateCount"] = incoming_df["ServiceTemplateKey"].map(
        incoming_df["ServiceTemplateKey"].value_counts(dropna=False)
    ).fillna(len(incoming_df)).astype(float)

    return incoming_df


def parse_log_budgets(value: Optional[str] = None) -> List[float]:
    raw = value if value is not None else os.environ.get("UNISAGE_LOG_BUDGETS", "")
    if not raw:
        return list(DEFAULT_LOG_BUDGETS)
    budgets: List[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        budget = float(item)
        if budget <= 0:
            raise ValueError(f"Log budget must be positive, got {budget}")
        budgets.append(budget)
    return budgets or list(DEFAULT_LOG_BUDGETS)


def _score_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def _sort_logs_by_columns(frame: pd.DataFrame, score_column: str) -> pd.DataFrame:
    out = frame.copy()
    out["_primary_score"] = _score_column(out, score_column)
    out["_strict_diag"] = out.get("LogDiagnosticStrict", False)
    out["_strict_diag"] = out["_strict_diag"].fillna(False).astype(bool).astype(int)
    out["_rarity_raw"] = _score_column(out, "LogRaritySamplingProbRaw")
    out["_service_novel"] = out.get("LogServiceTemplateNovelty", False)
    out["_service_novel"] = out["_service_novel"].fillna(False).astype(bool).astype(int)
    return out.sort_values(
        ["_primary_score", "_strict_diag", "_rarity_raw", "_service_novel", "LogID"],
        ascending=[False, False, False, False, True],
        kind="mergesort",
    ).drop(columns=["_primary_score", "_strict_diag", "_rarity_raw", "_service_novel"])


def sort_log_sampling_frame(frame: pd.DataFrame, score_column: str = "LogSamplingProb") -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame
    return _sort_logs_by_columns(frame, score_column).reset_index(drop=True)


def _take_new_logs(frame: pd.DataFrame, score_column: str, k: int, used: Set[str]) -> pd.DataFrame:
    if k <= 0:
        return frame.iloc[0:0].copy()
    available = frame[~frame["LogID"].astype(str).isin(used)].copy()
    if available.empty:
        return available
    return _sort_logs_by_columns(available, score_column).head(k)


def _take_diverse_logs(frame: pd.DataFrame, k: int, used: Set[str]) -> pd.DataFrame:
    if k <= 0:
        return frame.iloc[0:0].copy()
    available = frame[~frame["LogID"].astype(str).isin(used)].copy()
    if available.empty:
        return available
    available = available.sort_values(
        ["ServiceTemplateCount", "LogRaritySamplingProbRaw", "LogAnomalySamplingProb", "LogID"],
        ascending=[True, False, False, True],
        kind="mergesort",
    )
    selected = []
    seen_templates = set()
    for idx, row in available.iterrows():
        key = row["ServiceTemplateKey"]
        if key in seen_templates:
            continue
        selected.append(idx)
        seen_templates.add(key)
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


def _reserved_log_selection(
    frame: pd.DataFrame,
    k: int,
    diagnostic_pct: float = DEFAULT_LOG_RESERVED_DIAGNOSTIC_PCT,
    rarity_pct: float = DEFAULT_LOG_RESERVED_RARITY_PCT,
) -> pd.DataFrame:
    if k <= 0 or frame.empty:
        return frame.iloc[0:0].copy()

    diagnostic_n = min(k, max(1, int(round(k * diagnostic_pct))))
    rarity_n = min(k - diagnostic_n, int(round(k * rarity_pct)))
    diversity_n = max(0, k - diagnostic_n - rarity_n)

    parts = []
    used: Set[str] = set()
    diagnostic = _take_new_logs(frame, "LogAnomalySamplingProb", diagnostic_n, used)
    parts.append(diagnostic)
    used.update(diagnostic["LogID"].astype(str))

    rarity = _take_new_logs(frame, "LogRaritySamplingProbRaw", rarity_n, used)
    parts.append(rarity)
    used.update(rarity["LogID"].astype(str))

    diverse = _take_diverse_logs(frame, diversity_n, used)
    parts.append(diverse)
    used.update(diverse["LogID"].astype(str))

    if len(used) < k:
        fill = _take_new_logs(frame, "LogSamplingProbMax", k - len(used), used)
        parts.append(fill)

    return pd.concat(parts, ignore_index=False).drop_duplicates("LogID").head(k)


def apply_log_reserved_budget_strategy(
    frame: pd.DataFrame,
    budgets: Optional[List[float]] = None,
    diagnostic_pct: Optional[float] = None,
    rarity_pct: Optional[float] = None,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame
    budgets = budgets if budgets is not None else parse_log_budgets()
    diagnostic_pct = (
        float(diagnostic_pct)
        if diagnostic_pct is not None
        else float(os.environ.get("UNISAGE_LOG_RESERVED_DIAGNOSTIC_PCT", DEFAULT_LOG_RESERVED_DIAGNOSTIC_PCT))
    )
    rarity_pct = (
        float(rarity_pct)
        if rarity_pct is not None
        else float(os.environ.get("UNISAGE_LOG_RESERVED_RARITY_PCT", DEFAULT_LOG_RESERVED_RARITY_PCT))
    )

    out = frame.copy()
    n_ranked = len(out)
    for budget in budgets:
        token = budget_token(float(budget))
        k = min(n_ranked, max(1, int(round(n_ranked * float(budget)))))
        kept = _reserved_log_selection(out, k, diagnostic_pct=diagnostic_pct, rarity_pct=rarity_pct)
        order = {log_id: rank for rank, log_id in enumerate(kept["LogID"].astype(str), start=1)}
        rank_col = f"LogReservedBudgetRank_{token}"
        decision_col = f"LogReservedBudgetDecision_{token}"
        score_col = f"LogReservedBudgetScore_{token}"
        out[rank_col] = out["LogID"].astype(str).map(order)
        out[decision_col] = out[rank_col].notna()
        out[score_col] = np.where(
            out[decision_col],
            1.0 - ((out[rank_col].fillna(k + 1) - 1) / max(k, 1)),
            0.0,
        )
        out[f"LogReservedBudgetTargetK_{token}"] = k

    out["LogReservedDiagnosticPct"] = diagnostic_pct
    out["LogReservedRarityPct"] = rarity_pct
    return out


def combine_log_sampling_probabilities(
    incoming_df: pd.DataFrame,
    method: str = "max"
) -> pd.DataFrame:
    incoming_df = normalize_log_frame_for_sampling(incoming_df)
    method = (method or os.environ.get("UNISAGE_LOG_COMPOSITION", LOG_COMPOSITION_PRODUCT)).strip().lower()
    if method == "combined":
        method = LOG_COMPOSITION_PRODUCT

    for column in ("LogAnomalySamplingProb", "LogRaritySamplingProb", "LogRaritySamplingProbRaw"):
        if column not in incoming_df.columns:
            incoming_df[column] = 0.0
        incoming_df[column] = pd.to_numeric(incoming_df[column], errors="coerce").fillna(0.0)

    incoming_df["LogSamplingProbMax"] = incoming_df[["LogAnomalySamplingProb", "LogRaritySamplingProb"]].max(axis=1)
    incoming_df["LogSamplingProbProduct"] = incoming_df["LogAnomalySamplingProb"] * incoming_df["LogRaritySamplingProb"]
    anomaly_weight = float(os.environ.get("UNISAGE_LOG_WEIGHT_ANALYSIS", "0.7"))
    edge_weight = float(os.environ.get("UNISAGE_LOG_WEIGHT_EDGE", "0.3"))
    total_weight = anomaly_weight + edge_weight
    if total_weight <= 0:
        anomaly_weight, edge_weight, total_weight = 0.7, 0.3, 1.0
    incoming_df["LogSamplingProbWeighted"] = (
        anomaly_weight * incoming_df["LogAnomalySamplingProb"]
        + edge_weight * incoming_df["LogRaritySamplingProb"]
    ) / total_weight
    incoming_df["LogSamplingProbMean"] = incoming_df[["LogAnomalySamplingProb", "LogRaritySamplingProb"]].mean(axis=1)

    incoming_df = apply_log_reserved_budget_strategy(incoming_df)

    if method == LOG_COMPOSITION_MAX:
        incoming_df['LogSamplingProb'] = incoming_df["LogSamplingProbMax"]
    elif method == LOG_COMPOSITION_PRODUCT:
        incoming_df['LogSamplingProb'] = incoming_df["LogSamplingProbProduct"]
    elif method == "mean":
        incoming_df['LogSamplingProb'] = incoming_df["LogSamplingProbMean"]
    elif method == LOG_COMPOSITION_WEIGHTED:
        incoming_df['LogSamplingProb'] = incoming_df["LogSamplingProbWeighted"]
    elif method == LOG_COMPOSITION_EDGE:
        incoming_df['LogSamplingProb'] = incoming_df["LogRaritySamplingProb"]
    elif method == LOG_COMPOSITION_ANALYSIS:
        incoming_df['LogSamplingProb'] = incoming_df["LogAnomalySamplingProb"]
    elif method == LOG_COMPOSITION_RESERVED_BUDGET:
        primary_budget = float(os.environ.get("UNISAGE_LOG_PRIMARY_BUDGET", DEFAULT_LOG_PRIMARY_BUDGET))
        score_col = f"LogReservedBudgetScore_{budget_token(primary_budget)}"
        if score_col not in incoming_df.columns:
            primary_budget = DEFAULT_LOG_PRIMARY_BUDGET
            score_col = f"LogReservedBudgetScore_{budget_token(primary_budget)}"
        incoming_df['LogSamplingProb'] = incoming_df[score_col] if score_col in incoming_df.columns else incoming_df["LogSamplingProbMax"]
    else:
        raise ValueError(f"Unsupported log combination method: {method}")

    incoming_df["LogCompositionStrategy"] = method
    incoming_df["LogWeightAnalysis"] = anomaly_weight
    incoming_df["LogWeightEdge"] = edge_weight
    return incoming_df


def cross_modal_propagation(
    trace_df: pd.DataFrame,
    log_df: pd.DataFrame,
    w_t: float = 0.5,
    w_l: float = 0.5,
    log_method: Optional[str] = None,
    guard: Optional[bool] = None,
    log_cap: Optional[float] = None,
    trace_percentile: Optional[float] = None,
    diag_top_k: Optional[int] = None,
    return_diagnostics: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-Modal Signal Propagation Bias (paper Sec. 3.4.2).

    Propagate anomaly evidence across modalities via the shared traceID, then
    recompute the composite probabilities. Both updates use the ORIGINAL p_a
    values so the two directions are order-independent:

        trace->log:  p_a'(l) = max(p_a(l), 1 - (1 - p_a(t_l)) ** w_t)
        log->trace:  p_a'(t) = max(p_a(t), 1 - (1 - max_{l in L_t} p_a(l)) ** w_l)

    p_a(t) = AnomalySamplingProb (per trace), p_a(l) = LogAnomalySamplingProb (per log).
    If the traceID linkage or the anomaly columns are unavailable, the inputs are
    returned unchanged (the paper's graceful-degradation clause). Guarded mode
    limits trace->log propagation to high-confidence traces and caps the
    propagated contribution, preventing the pre-normalization flooding observed
    in earlier smoke checks.
    """
    t_key = "TraceID" if "TraceID" in trace_df.columns else None
    l_key = "TraceID" if "TraceID" in log_df.columns else ("TraceId" if "TraceId" in log_df.columns else None)
    diagnostics = {
        "applied": False,
        "skip_reason": None,
        "guarded": None,
        "log_cap": None,
        "trace_percentile": None,
        "trace_threshold": None,
        "n_traces": int(len(trace_df)) if trace_df is not None else 0,
        "n_logs": int(len(log_df)) if log_df is not None else 0,
        "n_trace_anomaly_changed": 0,
        "n_log_anomaly_changed": 0,
        "trace_anomaly_changed_pct": 0.0,
        "log_anomaly_changed_pct": 0.0,
        "trace_topk_overlap": None,
        "log_topk_overlap": None,
    }
    if (t_key is None or l_key is None
            or "AnomalySamplingProb" not in trace_df.columns
            or "LogAnomalySamplingProb" not in log_df.columns):
        diagnostics["skip_reason"] = "missing_traceid_or_anomaly_columns"
        if return_diagnostics:
            return trace_df, log_df, diagnostics
        return trace_df, log_df  # no cross-modal linkage -> propagation disabled

    trace_df = trace_df.copy()
    log_df = normalize_log_frame_for_sampling(log_df)

    guard = (
        os.environ.get("UNISAGE_PROPAGATE_GUARD", "1") != "0"
        if guard is None
        else bool(guard)
    )
    log_cap = (
        float(os.environ.get("UNISAGE_PROPAGATE_LOG_CAP", "0.95"))
        if log_cap is None
        else float(log_cap)
    )
    trace_percentile = (
        float(os.environ.get("UNISAGE_PROPAGATE_TRACE_PERCENTILE", "0.80"))
        if trace_percentile is None
        else float(trace_percentile)
    )
    diag_top_k = (
        int(os.environ.get("UNISAGE_PROPAGATE_DIAG_TOPK", "100"))
        if diag_top_k is None
        else int(diag_top_k)
    )

    # Original p_a values, used for BOTH directions.
    trace_df["AnomalySamplingProb"] = pd.to_numeric(trace_df["AnomalySamplingProb"], errors="coerce").fillna(0.0)
    log_df["LogAnomalySamplingProb"] = pd.to_numeric(log_df["LogAnomalySamplingProb"], errors="coerce").fillna(0.0)
    original_trace_pa = trace_df["AnomalySamplingProb"].copy()
    original_log_pa = log_df["LogAnomalySamplingProb"].copy()

    def _top_ids(frame: pd.DataFrame, score_column: str, id_column: str, k: int) -> Set[str]:
        if frame.empty or k <= 0 or score_column not in frame.columns:
            return set()
        ranked = frame.assign(_score=pd.to_numeric(frame[score_column], errors="coerce").fillna(0.0))
        return set(ranked.sort_values(["_score", id_column], ascending=[False, True], kind="mergesort")[id_column].astype(str).head(k))

    trace_ids_before = _top_ids(trace_df, "AnomalySamplingProb", t_key, diag_top_k)
    log_id_col = "LogID" if "LogID" in log_df.columns else l_key
    log_ids_before = _top_ids(log_df, "LogAnomalySamplingProb", log_id_col, diag_top_k)

    pa_t = trace_df.drop_duplicates(subset=[t_key]).set_index(t_key)["AnomalySamplingProb"].clip(0, 1)
    max_log_pa = log_df.groupby(l_key)["LogAnomalySamplingProb"].max().clip(0, 1)
    if guard and not pa_t.empty:
        threshold = float(pa_t.quantile(trace_percentile))
        eligible_trace_pa = pa_t.where(pa_t >= threshold, 0.0)
    else:
        threshold = 0.0
        eligible_trace_pa = pa_t

    # trace -> log
    mapped_pa_t = log_df[l_key].map(eligible_trace_pa).fillna(0.0).clip(0, 1)
    propagated_log_pa = 1 - (1 - mapped_pa_t) ** w_t
    if guard and log_cap < 1.0:
        propagated_log_pa = propagated_log_pa.clip(upper=log_cap)
    log_df["LogAnomalySamplingProbBeforePropagation"] = original_log_pa
    log_df["LogAnomalySamplingProb"] = np.maximum(log_df["LogAnomalySamplingProb"], propagated_log_pa)
    log_df["LogPropagationDelta"] = log_df["LogAnomalySamplingProb"] - original_log_pa

    # log -> trace
    mapped_max_log = trace_df[t_key].map(max_log_pa).fillna(0.0).clip(0, 1)
    trace_df["AnomalySamplingProbBeforePropagation"] = original_trace_pa
    trace_df["AnomalySamplingProb"] = np.maximum(
        trace_df["AnomalySamplingProb"], 1 - (1 - mapped_max_log) ** w_l
    )
    trace_df["TracePropagationDelta"] = trace_df["AnomalySamplingProb"] - original_trace_pa

    # Recompute composites with updated p_a.
    if "EdgeCaseSamplingProb" in trace_df.columns:
        trace_df["CombinedSamplingProb"] = (
            trace_df["EdgeCaseSamplingProb"] * trace_df["AnomalySamplingProb"]
        )
    if "LogRaritySamplingProb" in log_df.columns:
        log_df = combine_log_sampling_probabilities(
            log_df,
            method=log_method or os.environ.get("UNISAGE_LOG_COMPOSITION", LOG_COMPOSITION_PRODUCT),
        )

    trace_ids_after = _top_ids(trace_df, "AnomalySamplingProb", t_key, diag_top_k)
    log_ids_after = _top_ids(log_df, "LogAnomalySamplingProb", log_id_col, diag_top_k)
    trace_changed = trace_df["TracePropagationDelta"].abs() > 1e-12
    log_changed = log_df["LogPropagationDelta"].abs() > 1e-12
    diagnostics.update({
        "applied": True,
        "guarded": guard,
        "log_cap": log_cap,
        "trace_percentile": trace_percentile,
        "trace_threshold": threshold,
        "n_trace_anomaly_changed": int(trace_changed.sum()),
        "n_log_anomaly_changed": int(log_changed.sum()),
        "trace_anomaly_changed_pct": float(trace_changed.mean()) if len(trace_changed) else 0.0,
        "log_anomaly_changed_pct": float(log_changed.mean()) if len(log_changed) else 0.0,
        "trace_topk_overlap": (
            len(trace_ids_before & trace_ids_after) / len(trace_ids_before | trace_ids_after)
            if trace_ids_before or trace_ids_after else None
        ),
        "log_topk_overlap": (
            len(log_ids_before & log_ids_after) / len(log_ids_before | log_ids_after)
            if log_ids_before or log_ids_after else None
        ),
    })
    if return_diagnostics:
        return trace_df, log_df, diagnostics
    return trace_df, log_df
