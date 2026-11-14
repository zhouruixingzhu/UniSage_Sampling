import asyncio
from pathlib import Path
from aiomultiprocess import Pool
import asyncio
import os
import csv
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Sequence, List, Tuple, Union, Dict
from datetime import datetime, timedelta
import time
import numpy as np
import pandas as pd
import re
from aiomultiprocess import Pool
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans

from AD.abstract import Detector, Parser
from AD.log import LogDetector, LogParser
from AD.metric import MetricDetector, MetricParser
from AD.trace import TraceDetector, TraceParser
from AD.utils.data_utils import _apply_self_latency, _apply_responsible_service
from RCL.log_rcl import log_rcl
from RCL.metric_rcl import metric_rcl
from RCL.trace_rcl import (
    calculate_changes,
    get_top_spans,
    read_and_merge_data,
    sort_and_filter_data,
)
from utils import (
    get_overall_time_range,
    extract_log_trace_time_ranges,
    extract_metric_time_ranges,
)
from sampling import (
    combined_sampler,
    aggregate_trace,
    aggregate_trace_duration_stats,
    compute_edge_case_sampling_prob,
    compute_log_edgecase_sampling_prob,
    count_log_templates,
    compute_log_anomaly_sampling_prob,
    aggregate_log,
    combine_log_sampling_probabilities,
    cross_modal_propagation,
    apply_trace_composition_strategy,
    sort_log_sampling_frame,
)

# Cross-modal signal propagation (route 1, paper Sec. 3.4.2). Env-toggled for ablation:
#   UNISAGE_PROPAGATE=0 disables it; UNISAGE_W_T / UNISAGE_W_L override the weights.
PROPAGATE = os.environ.get("UNISAGE_PROPAGATE", "1") != "0"
PROP_W_T = float(os.environ.get("UNISAGE_W_T", "0.5"))
PROP_W_L = float(os.environ.get("UNISAGE_W_L", "0.5"))


@contextmanager
def pushd(path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


async def trace_rcl(base_dir, service_list):
    base_dir = Path(base_dir)
    with pushd(base_dir):
        # Try both paths: trace_ad_output (aiops21) or trace/trace_ad_output (hipster)
        if Path("trace_ad_output/normal_rate.csv").exists():
            normal_path = "trace_ad_output/normal_rate.csv"
            abnormal_path = "trace_ad_output/abnormal_rate.csv"
        else:
            normal_path = "trace/trace_ad_output/normal_rate.csv"
            abnormal_path = "trace/trace_ad_output/abnormal_rate.csv"
        
        merged_df = read_and_merge_data(normal_path, abnormal_path)
        merged_df = calculate_changes(merged_df)
        weighted_change_result = sort_and_filter_data(merged_df)
        return get_top_spans("rca_result/trace_rca_results.csv"), weighted_change_result



def log_sampling(system_anomalous, base_dir, combined_ranking, dataset_type="hipster"):
    """Sample logs with support for both hipster and aiops21 formats."""
    base_dir = Path(base_dir).absolute()

    def _empty_log_frame(reason: str) -> pd.DataFrame:
        (base_dir / "log_sampling_provenance.json").write_text(json.dumps({
            "dataset_type": dataset_type,
            "system_anomalous": bool(system_anomalous),
            "status": "skipped",
            "reason": reason,
        }, indent=2) + "\n")
        return pd.DataFrame()

    def _score_and_write_logs(incoming_raw: pd.DataFrame, ref_raw: pd.DataFrame, source: str) -> pd.DataFrame:
        input_rows = int(len(incoming_raw)) if incoming_raw is not None else 0
        ref_rows = int(len(ref_raw)) if ref_raw is not None else 0
        if incoming_raw is None or incoming_raw.empty:
            return _empty_log_frame("empty_incoming_logs")

        ref_freq = count_log_templates(ref_raw) if ref_raw is not None and not ref_raw.empty else count_log_templates(pd.DataFrame())
        scored = compute_log_edgecase_sampling_prob(incoming_raw, ref_freq)
        if system_anomalous:
            scored = compute_log_anomaly_sampling_prob(scored, combined_ranking)
            composition = os.environ.get("UNISAGE_LOG_COMPOSITION", "product")
        else:
            scored = compute_log_anomaly_sampling_prob(scored, pd.DataFrame(columns=["service_name", "CombinedScore"]))
            scored["LogSamplingFallback"] = "edge_only_no_analysis_signal"
            composition = "edge"

        scored = combine_log_sampling_probabilities(scored, method=composition)
        scored = sort_log_sampling_frame(scored)
        scored.to_csv("log_sampling_result.csv", index=False)

        if len(scored) != input_rows:
            print(f"Warning: Log sampling row-count mismatch for {base_dir.name}: input={input_rows} output={len(scored)}")
        provenance = {
            "dataset_type": dataset_type,
            "source": source,
            "system_anomalous": bool(system_anomalous),
            "status": "ok",
            "input_rows": input_rows,
            "output_rows": int(len(scored)),
            "reference_rows": ref_rows,
            "row_count_expanded": bool(len(scored) > input_rows),
            "row_count_changed": bool(len(scored) != input_rows),
            "log_composition": composition,
            "log_budgets": os.environ.get("UNISAGE_LOG_BUDGETS", ""),
            "log_primary_budget": os.environ.get("UNISAGE_LOG_PRIMARY_BUDGET", ""),
            "propagation_enabled_env": os.environ.get("UNISAGE_PROPAGATE", "1"),
            "propagation_guard_env": os.environ.get("UNISAGE_PROPAGATE_GUARD", "1"),
            "propagation_log_cap_env": os.environ.get("UNISAGE_PROPAGATE_LOG_CAP", "0.95"),
            "propagation_trace_percentile_env": os.environ.get("UNISAGE_PROPAGATE_TRACE_PERCENTILE", "0.80"),
            "missing_service_rows": int(scored.get("LogRarityReferenceMissingService", pd.Series(dtype=bool)).fillna(False).sum()),
            "missing_template_rows": int(scored.get("LogRarityReferenceMissingTemplate", pd.Series(dtype=bool)).fillna(False).sum()),
            "reference_scopes": scored.get("LogRarityReferenceScope", pd.Series(dtype=str)).value_counts(dropna=False).to_dict(),
        }
        (base_dir / "log_sampling_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        return scored

    if dataset_type == "aiops21":
        # aiops21: single CSV files under normal/ and abnormal/
        abnormal_log_file = base_dir / "abnormal" / "logs.csv"
        normal_log_file = base_dir / "normal" / "logs.csv"

        # If abnormal logs are missing, skip sampling for this case
        if not abnormal_log_file.exists():
            return _empty_log_frame("missing_abnormal_logs")

        with pushd(base_dir):
            try:
                incoming_df = pd.read_csv(abnormal_log_file)
            except Exception as e:
                print(f"Warning: Failed to read abnormal logs: {e}")
                return _empty_log_frame("failed_read_abnormal_logs")

            # Reference normal logs for template frequencies; if missing, use empty ref
            if normal_log_file.exists():
                try:
                    ref_df = pd.read_csv(normal_log_file)
                except Exception as e:
                    print(f"Warning: Failed to read normal logs: {e}")
                    ref_df = pd.DataFrame()
            else:
                ref_df = pd.DataFrame()

            return _score_and_write_logs(incoming_df, ref_df, "aiops21_csv")

    else:
        # hipster/train_ticket: logs are under base_dir/log/parsed and parent/fault_free/log/parsed
        parent_dir = base_dir.parent
        input_path = base_dir / "log" / "parsed"
        normal_path = parent_dir / "fault_free" / "log" / "parsed"

        if not input_path.exists() or not normal_path.exists():
            return _empty_log_frame("missing_log_or_reference_path")

        with pushd(base_dir):
            incoming_df = aggregate_log(input_path)
            ref_df = aggregate_log(normal_path)
            return _score_and_write_logs(incoming_df, ref_df, "parsed_log_dir")


def trace_sampling(system_anomalous, base_dir, combined_ranking, dataset_type="hipster"):
    """Sample traces with support for both hipster and aiops21 formats."""
    base_dir = Path(base_dir)
    
    if dataset_type == "aiops21":
        # For aiops21: traces are in base_dir/trace_ad_output (not nested under trace/)
        normal_rate_path = base_dir / "trace_ad_output" / "normal_rate.csv"
        abnormal_rate_path = base_dir / "trace_ad_output" / "abnormal_rate.csv"
        normal_trace_path = base_dir / "normal" / "traces.csv"
    else:
        # For hipster/train_ticket: traces are under parent/fault_free/trace/
        normal_rate_path = base_dir / "trace" / "trace_ad_output" / "normal_rate.csv"
        abnormal_rate_path = base_dir / "trace" / "trace_ad_output" / "abnormal_rate.csv"
        normal_trace_path = base_dir.parent / "fault_free" / "trace" / "normal_trace.csv"
        
    # Check if paths exist
    if not normal_rate_path.exists():
        return pd.DataFrame()
    
    with pushd(base_dir):
        try:
            if dataset_type == "aiops21":
                incoming_df = aggregate_trace(str(base_dir / "abnormal" / "traces.csv"))
                # Normalize aiops21 column names to match sampling expectations
                incoming_df = incoming_df.rename(columns={
                    'TraceId': 'TraceID',
                    'ServiceName': 'PodName',
                    'SpanName': 'OperationName',
                    'SpanId': 'SpanID',
                    'ParentSpanId': 'ParentID'
                })
                # Compute per-trace span stats for aiops21
                trace_stats_df = incoming_df.groupby(['TraceID', 'PodName', 'OperationName'])['Duration'].agg(
                    MeanDuration='mean', StdDuration='std'
                ).reset_index()
                trace_stats_df['StdDuration'] = trace_stats_df['StdDuration'].fillna(0.0)
            else:
                incoming_df = aggregate_trace("trace")
                trace_stats_df = aggregate_trace_duration_stats("trace", "trace/trace_stats.csv")

            ref_df = pd.read_csv(normal_trace_path)
            if dataset_type == "aiops21":
                ref_df = ref_df.rename(columns={
                    'TraceId': 'TraceID',
                    'ServiceName': 'PodName',
                    'SpanName': 'OperationName',
                    'SpanId': 'SpanID',
                    'ParentSpanId': 'ParentID'
                })

            # Edge-case latency reference consistency. The self-latency/reattribution transform
            # (UNISAGE_RCA_ATTR=target) is for the RCL phi_s only; it must NOT silently change
            # the always-on, analysis-independent edge-case sampler. But that transform also
            # writes normal_rate.csv, and the incoming trace_stats above is RAW -> an asymmetric
            # duration/key comparison that spuriously inflated p_e in target mode.
            #   raw     : reference recomputed RAW from normal_trace -> both raw, attribution-
            #             independent edge-case pillar (DEFAULT = the fix).
            #   selflat : incoming also self-latency/reattributed -> both transformed (consistent).
            #   asym    : the old buggy behavior (reference transformed, incoming raw); kept only
            #             to reproduce the pre-fix coverage numbers.
            edge_lat = os.environ.get("UNISAGE_EDGE_LAT", "raw")
            if edge_lat == "selflat":
                _t = _apply_responsible_service(_apply_self_latency(incoming_df))
                trace_stats_df = _t.groupby(["TraceID", "PodName", "OperationName"])["Duration"].agg(
                    MeanDuration="mean", StdDuration="std").reset_index()
                trace_stats_df["StdDuration"] = trace_stats_df["StdDuration"].fillna(0.0)

            ref_rate_df = pd.read_csv(normal_rate_path)
            if edge_lat == "raw":
                ref_rate_df = ref_df.groupby(["PodName", "OperationName"])["Duration"].agg(
                    MeanDuration="mean", StdDuration="std").reset_index()
                ref_rate_df["StdDuration"] = ref_rate_df["StdDuration"].fillna(0.0)
            
            if system_anomalous:
                sampling_df = combined_sampler(ref_rate_df, trace_stats_df, ref_df, incoming_df, combined_ranking)
                sampling_df = sampling_df.sort_values(by=['CombinedSamplingProb', 'LatencyMaxZScore'],ascending=[False, False])
            else:
                sampling_df = compute_edge_case_sampling_prob(ref_rate_df, trace_stats_df, ref_df, incoming_df)
                if sampling_df.empty:
                    return sampling_df
                sampling_df['AnomalySamplingProb'] = np.nan
                sampling_df['CombinedSamplingProb'] = sampling_df['EdgeCaseSamplingProb']
                sampling_df['SamplingFallback'] = 'edge_only_no_analysis_signal'
                sampling_df = sampling_df.sort_values(by=['EdgeCaseSamplingProb', 'LatencyMaxZScore'],ascending=[False, False])

            sampling_df = apply_trace_composition_strategy(sampling_df)
            sampling_df.to_csv("trace_sampling_result.csv", index=False)
            return sampling_df
        except Exception as e:
            print(f"Warning: Trace sampling failed: {e}")
            return pd.DataFrame()

        return sampling_df


def ranking(base_dir: Path, anomalous: Sequence, score: Sequence[pd.DataFrame]):
    scores = []
    for i, (anomalous, score) in enumerate(zip(anomalous, score)):
        if anomalous and not score.empty:
            scores.append(score)
    
    if not scores:
        # No anomalies detected, return empty dataframe
        combined_scores = pd.DataFrame(columns=['service_name', 'CombinedScore', 'Index'])
    else:
        # print(scores)
        combined_scores = pd.concat(scores, ignore_index=True).groupby('service_name', as_index=False)['anomaly_score'].sum()
        combined_scores = combined_scores[combined_scores['anomaly_score'].abs() > 0]
        combined_scores = combined_scores.rename(columns={'anomaly_score': 'CombinedScore'})
        combined_scores = combined_scores.sort_values(by='CombinedScore', ascending=False)

        # Add an index column starting from 1
        combined_scores = combined_scores.reset_index(drop=True)
        combined_scores['Index'] = combined_scores.index + 1

    combined_scores.to_csv(base_dir / 'final_ranking.csv', index=False)
    # print(combined_scores)

    return combined_scores

def is_valid_case_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    # Match both hipster format (03_53) and aiops21 format (apache01-0309-0120)
    return (re.match(r"^\d+_\d+$", path.name) or  # hipster format
            re.match(r"^[a-zA-Z0-9]+-\d{4}-\d{4}$", path.name))  # aiops21 format

async def process_case(base_dir, pool: Pool, file_type="log", dataset_type="hipster"):
    system_anomalous = False
    metric_abnormal_time = None
    log_trace_abnormal_time = None
    if file_type == "metric":
        parser = MetricParser(base_dir, dataset_type=dataset_type)
        data = await parser.parse(pool)
        detector = MetricDetector(base_dir)
        system_anomalous, anomalies = await detector.detect(data, pool)
        metric_abnormal_time = extract_metric_time_ranges(anomalies)
        # print("metric_abnormal_time:",metric_abnormal_time)
        if system_anomalous:
            return system_anomalous, *metric_rcl(base_dir), metric_abnormal_time
    if file_type == "log":
        parser = LogParser((base_dir), dataset_type=dataset_type)
        data = await parser.parse(pool)
        detector = LogDetector(base_dir, data)
        system_anomalous, anomalies = await pool.apply(detector.detect)
        log_trace_abnormal_time = extract_log_trace_time_ranges(anomalies)
        if system_anomalous:
            return system_anomalous, *await pool.apply(log_rcl, args=(base_dir, anomalies)), log_trace_abnormal_time
    if file_type == 'trace':
        parser = TraceParser((base_dir), dataset_type=dataset_type)
        data = await parser.parse(pool)
        detector = TraceDetector(base_dir, data)
        system_anomalous, anomalies = await pool.apply(detector.detect)
        log_trace_abnormal_time = extract_log_trace_time_ranges(anomalies)
        if system_anomalous:
            return system_anomalous, *await pool.apply(trace_rcl, args=(base_dir, anomalies)), log_trace_abnormal_time
    return system_anomalous, [], pd.DataFrame(), None


async def evaluate(base_dir, pool, dataset_type="hipster"):
    base_dir = Path(base_dir)
    case = base_dir.name
    print(case)
    base_dir = base_dir.absolute()
    file_types = ["log", "trace", "metric"]
    # ablation experiments
    # file_types = ["metric", "trace"]

    start_rca = time.time()
    result = await asyncio.gather(*(process_case(base_dir, pool, file_type, dataset_type) for file_type in file_types))
    end_rca = time.time()
    print(f"[{case}] RCA (process_case) time: {end_rca - start_rca:.4f} seconds")

    result = list(zip(*result))
    system_anomalous = any(result[0])
    log_anomalous = result[0][0]
    event_list, score_list, time_list = list(result[1]), list(result[2]), list(result[3])

    predict_abnormal_time_range = get_overall_time_range(time_list)
    print(predict_abnormal_time_range)

    combined_ranking = ranking(base_dir, result[0], result[2])

    if system_anomalous:
        combined_events = {}
        for file_type, events in zip(file_types,result[1]):
            combined_events[f"{file_type}_events"] = events
    else:
        combined_events = {}
        print(f"No anomalies detected in case {base_dir}; falling back to edge-only sampling.")

    start_trace = time.time()
    trace_sampling_df = trace_sampling(system_anomalous, base_dir, combined_ranking, dataset_type)
    trace_fallback = not system_anomalous
    if system_anomalous and (trace_sampling_df is None or trace_sampling_df.empty):
        print(f"[{case}] Trace sampling produced no ranking; falling back to edge-only sampling.")
        trace_sampling_df = trace_sampling(False, base_dir, combined_ranking, dataset_type)
        trace_fallback = True
        if trace_sampling_df is not None and not trace_sampling_df.empty:
            trace_sampling_df['SamplingFallback'] = 'edge_only_after_trace_sampling_failure'
    end_trace = time.time()
    print(f"[{case}] Trace sampling time: {end_trace - start_trace:.4f} seconds")

    start_log = time.time()
    log_sampling_df = log_sampling(system_anomalous, base_dir, combined_ranking, dataset_type)
    end_log = time.time()
    print(f"[{case}] Log sampling time: {end_log - start_log:.4f} seconds")

    if (PROPAGATE and system_anomalous and not trace_fallback
            and trace_sampling_df is not None and log_sampling_df is not None
            and not trace_sampling_df.empty and not log_sampling_df.empty):
        trace_sampling_df, log_sampling_df, propagation_diag = cross_modal_propagation(
            trace_sampling_df, log_sampling_df, w_t=PROP_W_T, w_l=PROP_W_L, return_diagnostics=True
        )
        trace_sampling_df = apply_trace_composition_strategy(trace_sampling_df)
        trace_sampling_df = trace_sampling_df.sort_values(by="CombinedSamplingProb", ascending=False)
        log_sampling_df = sort_log_sampling_frame(log_sampling_df)
        trace_sampling_df.to_csv(base_dir / "trace_sampling_result.csv", index=False)
        log_sampling_df.to_csv(base_dir / "log_sampling_result.csv", index=False)
        (base_dir / "cross_modal_propagation_diagnostics.json").write_text(
            json.dumps(propagation_diag, indent=2) + "\n"
        )
        print(f"[{case}] Cross-modal propagation applied (w_t={PROP_W_T}, w_l={PROP_W_L}, guard={propagation_diag.get('guarded')})")

    return case, combined_ranking, predict_abnormal_time_range, trace_sampling_df, log_sampling_df




######## run one case ########
# async def main():    
#     # Define paths with their dataset types
#     base_dirs_with_types = [
#         # (R"data/hipster/batches/batch1/03_53", "hipster"),
#         # (R"data/train_ticket/batches/batch1/08_43", "hipster"),
#         (R"data/aiops21/test/apache01-0309-0120", "aiops21"),
#     ]
#     async with Pool() as pool:    
#         prediction_data = await asyncio.gather(*(evaluate(base_dir, pool, dataset_type) for base_dir, dataset_type in base_dirs_with_types))
#         return prediction_data


# root_base_dir = Path(r"data/train_ticket/batches/batch2")
root_base_dir = Path(r"data/aiops21/all/")
dataset_type = "aiops21"
async def main():
    case_dirs = [p for p in root_base_dir.iterdir() if is_valid_case_dir(p)]
    prediction_data = []
    async with Pool() as pool:
        for case_dir in case_dirs:
            print(f"\n========== Running case: {case_dir.name} ==========")
            result = await evaluate(case_dir, pool, dataset_type)
            prediction_data.append(result)
    return prediction_data


@contextmanager
def timer():
    start_time = time.time()
    yield
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.4f} seconds")
if __name__ == "__main__":
    with timer():
        asyncio.run(main())
