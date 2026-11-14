import os
import re
import numpy as np
import pandas as pd
from pathlib import Path

def read_and_merge_data(normal_path, abnormal_path):
    normal_df = pd.read_csv(normal_path)
    abnormal_df = pd.read_csv(abnormal_path)
    merged_df = pd.merge(normal_df, abnormal_df, on=["PodName", "OperationName"], suffixes=('_normal', '_abnormal'))
    return merged_df


def calculate_changes(merged_df):
    merged_df['ChangePercentage'] = (
        (merged_df['MeanDuration_abnormal'] - merged_df['MeanDuration_normal']) / merged_df['MeanDuration_normal']
    ) * 100
    # merged_df['ChangeDescription'] = merged_df['ChangePercentage'].apply(
    #     lambda x: f"increase {abs(x):.2f}%" if x > 0 else f"decrease {abs(x):.2f}%"
    # )
    return merged_df


def _build_pod_map(pod_series):
    """base service name (lowercased) -> full PodName (with hash); one pod per service."""
    m = {}
    for pod in pod_series.astype(str).unique():
        mm = re.match(r"(.+)-[a-z0-9]{9,}-[a-z0-9]{5}$", pod)
        base = (mm.group(1) if mm else pod).lower()
        m.setdefault(base, pod)
    return m


def _target_service(operation_name, emitter_pod, pod_map):
    """Attribute a span to the service that PERFORMS the operation (the callee for a
    client span), parsed from OperationName; fall back to the emitter pod when there is
    no recognizable service token (e.g. internal `sql.*` spans). This credits a slow
    caller->callee client span to the callee, fixing latency absorption (e.g.
    network_delay) where the delay otherwise lands on the waiting caller."""
    mtok = re.search(r"([A-Za-z][A-Za-z0-9]*Service|Frontend)", str(operation_name))
    if mtok:
        base = mtok.group(1).lower()
        if base in pod_map:
            return pod_map[base]
    return emitter_pod


def sort_and_filter_data(merged_df):
    os.makedirs("rca_result", exist_ok=True)
    merged_df = merged_df[merged_df['ChangePercentage'] >= 0]
    
    # Grouping the data based on the specified columns
    aggregated_df = merged_df.groupby(
        ['PodName', 'OperationName', 'ParentID_abnormal']
    ).agg(
        MeanDuration_normal=('MeanDuration_normal', 'mean'),
        MeanDuration_abnormal=('MeanDuration_abnormal', 'mean'),
        ChangePercentage=('ChangePercentage', 'mean'),
        TraceId_abnormal=('TraceID_abnormal', lambda x: list(x)),
        SpanNum=('TraceID_abnormal', 'size')
    ).reset_index()

    aggregated_df = aggregated_df.sort_values(by='ChangePercentage', key=abs, ascending=False)

    # Adding the ChangeDescription column
    aggregated_df['ChangeDescription'] = aggregated_df['ChangePercentage'].apply(
        lambda x: f"increase {abs(x):.2f}%" if x > 0 else f"decrease {abs(x):.2f}%"
    )

    filtered_df = aggregated_df[aggregated_df['ChangePercentage'].abs() > 5].copy()

    # Dropping the ChangePercentage column
    # filtered_df = filtered_df.drop(columns=['ChangePercentage'])

    filtered_df.loc[:, 'WeightedChange'] = filtered_df['ChangePercentage'] * (1 / (filtered_df.index + 1))

    # In target mode the responsible (callee) service is assigned upstream, naming-agnostic,
    # in data_utils._apply_responsible_service (via the trace tree), so PodName here already
    # denotes the attributed service. In pod mode PodName is the span emitter (current method).
    filtered_df.loc[:, 'AttrService'] = filtered_df['PodName']

    if os.environ.get("UNISAGE_RCA_ATTR", "target") == "target":
        # De-biased, traffic-invariant: each service's worst relative degradation.
        # Drops the span_count/total weighting that over-amplified high-traffic services.
        # Exclude non-finite ChangePercentage: self-latency can drive a span's normal
        # baseline to ~0, making Delta/0 -> inf, which would otherwise dominate the max
        # (e.g. a fast leaf op). Keep only finite, real degradations.
        finite = filtered_df[np.isfinite(filtered_df['ChangePercentage'])]
        filtered_df.loc[:, 'anomaly_score'] = filtered_df['ChangePercentage']
        result = finite.groupby('AttrService')['ChangePercentage'].max().reset_index()
        result = result.rename(columns={'ChangePercentage': 'anomaly_score'})
    else:
        service_span_count = (filtered_df.groupby('AttrService')['SpanNum'].sum()).to_dict()
        total_spans = filtered_df['SpanNum'].sum()
        filtered_df.loc[:, 'anomaly_score'] = filtered_df.apply(
            lambda row: row['WeightedChange'] * (service_span_count[row['AttrService']] / total_spans), axis=1
        )
        result = filtered_df.groupby('AttrService')['anomaly_score'].sum().reset_index()

    result = result[result['AttrService'] != 'front_service']
    result = result.sort_values(by='anomaly_score', ascending=False)
    result = result.rename(columns={'AttrService': 'service_name'})
    # result.to_csv("rcl_output/trace_service_scores.csv", index=False)

    filtered_df.to_csv("rca_result/trace_rca_results.csv", index=False)

    return result


def get_top_spans(file_path):
    df = pd.read_csv(file_path).round(3)
    trace_events = []
    for index, row in df.iterrows():
        trace_events.append(
            {
                "top": index + 1,
                "span_name": row['OperationName'],
                "service_name": row['PodName'],
                "parent_service_name": row['ParentID_abnormal'],
                "normal_duration_per_min": round(row['MeanDuration_normal'] / 60000000, 3),
                "observed_duration": round(row['MeanDuration_abnormal'] / 60000000, 3),
                "pattern": row['ChangeDescription'],
                "trace_id": row['TraceId_abnormal']
            }
        )
    return trace_events


def main():
    merged_df = read_and_merge_data("mean_std/normal_rate.csv", "mean_std/abnormal_rate.csv")
    merged_df = calculate_changes(merged_df)
    filtered_df = sort_and_filter_data(merged_df)
    get_top_spans("rca_result/trace_rca_results.csv")



