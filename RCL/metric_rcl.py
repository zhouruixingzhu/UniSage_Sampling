import csv
import os
import shutil
from datetime import datetime

import pandas as pd
from pathlib import Path


def abnormal_time_range(file_path):
    earliest_start = None
    latest_end = None

    try:
        with open(file_path, mode='r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                third_dash_index = row['TimeRanges'].find(
                    '-', row['TimeRanges'].find('-', row['TimeRanges'].find('-') + 1) + 1
                )
                if third_dash_index == -1:
                    # print("Skipping invalid time range:", row['TimeRanges'])
                    continue

                start_time_str = row['TimeRanges'][:third_dash_index]
                end_time_str = row['TimeRanges'][third_dash_index + 1 :]

                start_time = datetime.fromisoformat(start_time_str)
                end_time = datetime.fromisoformat(end_time_str)

                if earliest_start is None or start_time < earliest_start:
                    earliest_start = start_time
                if latest_end is None or end_time > latest_end:
                    latest_end = end_time

        return earliest_start, latest_end

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
        return None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None


def load_and_filter_metrics(folder, earliest=None, latest=None, lower_percentile=0, upper_percentile=1):
    """Load metrics from either hipster format (per-metric CSV files) or aiops21 format (single metrics.csv)."""
    metrics = {}

    # Convert earliest and latest to datetime if provided
    if earliest is not None:
        earliest = pd.to_datetime(earliest)
    if latest is not None:
        latest = pd.to_datetime(latest)

    metrics_file = os.path.join(folder, "metrics.csv")
    if os.path.exists(metrics_file):
        # aiops21 format: single metrics.csv file
        try:
            df = pd.read_csv(metrics_file)
            
            # For aiops21, rename TimeUnix to Time for compatibility
            if 'TimeUnix' in df.columns:
                df = df.copy()
                df['Time'] = pd.to_datetime(df['TimeUnix'], errors='coerce')
                df = df.drop(columns=['TimeUnix'])
            elif 'Time' not in df.columns:
                # If neither TimeUnix nor Time exists, skip
                return metrics
            
            # Group by service/pod and filter by time
            if 'k8s_pod_name' in df.columns:
                for service_name, group_df in df.groupby('k8s_pod_name'):
                    # Apply time filter
                    if earliest is not None and latest is not None:
                        filtered_df = group_df[(group_df['Time'] >= earliest) & (group_df['Time'] <= latest)]
                        if filtered_df.empty:
                            continue
                    else:
                        filtered_df = group_df
                    
                    # Pivot to get metrics as columns
                    if 'MetricName' in filtered_df.columns and 'Value' in filtered_df.columns:
                        pivot_df = filtered_df.pivot_table(
                            index='Time', 
                            columns='MetricName', 
                            values='Value', 
                            aggfunc='mean'
                        )
                        if not pivot_df.empty:
                            pivot_df = pivot_df.reset_index()
                            metrics[service_name] = pivot_df
        except Exception as e:
            print(f"Error loading aiops21 metrics: {e}")
            return metrics
    
    # Hipster/train_ticket format: individual metric CSV files
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            path = os.path.join(folder, file)
            try:
                df = pd.read_csv(path)
                df = df.drop(columns=['TimeStamp', 'PodName', 'PodServerLatencyP99(s)', 'PodServerLatencyP95(s)', 'PodServerLatencyP90(s)', 'PodClientLatencyP99(s)', 'PodClientLatencyP95(s)', 'PodClientLatencyP90(s)'], errors='ignore')

                if 'Time' not in df.columns:
                    # print(f"'Time' column missing in {file}, skipping.")
                    continue

                # Parse Time column to datetime
                df['Time'] = pd.to_datetime(df['Time'], errors='coerce')

                # Apply time filter only if both bounds are provided
                if earliest is not None and latest is not None:
                    df = df[(df['Time'] >= earliest) & (df['Time'] <= latest)]
                    if df.empty:
                        print(f"No data in the specified time range for file {file}, skipping.")
                        continue

                filtered_df = df[['Time']].copy()

                for column in df.columns:
                    if column != 'Time':
                        df[column] = pd.to_numeric(df[column], errors='coerce')
                        lower_bound = df[column].quantile(lower_percentile)
                        upper_bound = df[column].quantile(upper_percentile)
                        filtered_col = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
                        filtered_df[column] = filtered_col[column]

                metrics[file] = filtered_df

            except Exception as e:
                print(f"Error processing file {file}: {str(e)}")

    return metrics


def compute_statistics(metrics_dict):
    stats = {}
    for filename, df in metrics_dict.items():
        df = df.iloc[:, 1:]
        summary_df = pd.DataFrame({
            'Metric': df.columns,
            'Max': df.max().round(3),
            'Min': df.min().round(3),
            'Mean': df.mean().round(3),
        })
        stats[filename] = summary_df
    return stats


def calculate_overlap(normal_stats, abnormal_stats, k=0.8):
    all_results = []

    for filename in normal_stats:
        if filename not in abnormal_stats:
            continue

        normal_df = normal_stats[filename]
        abnormal_df = abnormal_stats[filename]

        for i, row in normal_df.iterrows():
            metric = row['Metric']
            if (row['Min'] == 0 and row['Max'] == 0 and row['Mean'] == 0):
                continue

            normal_range = (row['Min'], row['Max'])
            normal_avg = row['Mean']
            abnormal_row = abnormal_df[abnormal_df['Metric'] == metric]
            if abnormal_row.empty:
                continue

            abnormal_range = (abnormal_row.iloc[0]['Min'], abnormal_row.iloc[0]['Max'])
            abnormal_avg = abnormal_row.iloc[0]['Mean']

            if abnormal_range == normal_range:
                # print("overlap is 100%")
                overlap_pct = 100
            else:
                max_lower = max(normal_range[0], abnormal_range[0])
                min_upper = min(normal_range[1], abnormal_range[1])
                overlap = max(0, min_upper - max_lower)
                total_range = normal_range[1] - normal_range[0]
                overlap_pct = (overlap / total_range) * 100 if total_range > 0 else 0

            avg_change = abs(abnormal_avg - normal_avg) / normal_avg * 100 if normal_avg != 0 else 0
            change_score = k * overlap_pct + (1 - k) * avg_change

            # service_name = '-'.join(filename.split('.')[0].split('-')[:-2])
            service_name = filename.split('.')[0]
            result = {
                'service_name': service_name,
                'metric_name': metric,
                'change_score': change_score,
                'overlap_percentage': overlap_pct,
                'avg_change': avg_change,
                'abnormal_max': abnormal_range[1],
                'abnormal_min': abnormal_range[0],
                'abnormal_average': abnormal_avg,
                'normal_max': normal_range[1],
                'normal_min': normal_range[0],
                'normal_average': normal_avg,
            }
            all_results.append(result)
            

    df_results = pd.DataFrame(all_results)
    # print(df_results)
    if len(df_results) == 0:
        # Return empty dataframe with expected columns
        return pd.DataFrame(columns=['service_name', 'metric_name', 'change_score', 'overlap_percentage',
                                    'avg_change', 'abnormal_max', 'abnormal_min', 'abnormal_average',
                                    'normal_max', 'normal_min', 'normal_average'])
    
    df_results = df_results[df_results['change_score'].abs() > 0]
    if len(df_results) == 0:
        # Return empty dataframe if no significant changes
        return pd.DataFrame(columns=['service_name', 'metric_name', 'change_score', 'overlap_percentage',
                                    'avg_change', 'abnormal_max', 'abnormal_min', 'abnormal_average',
                                    'normal_max', 'normal_min', 'normal_average'])
    
    df_results = df_results.sort_values(by='change_score', ascending=False)
    # print(df_results)
    return df_results


def rank_services(df):
    df = df.copy()
    
    # Handle empty dataframe
    if len(df) == 0:
        return pd.DataFrame(columns=['service_name', 'anomaly_score'])
    
    df = df.sort_values(by='change_score', ascending=False).reset_index(drop=True)

    df['weighted_change'] = df['change_score'] * (1 / (df.index + 1))
    service_counts = df['service_name'].value_counts()
    total_metrics = len(df)

    df['anomaly_score'] = df.apply(
        lambda row: row['weighted_change'] * (service_counts[row['service_name']] / total_metrics),
        axis=1
    )
    service_scores = df.groupby('service_name')['anomaly_score'].sum().reset_index()
    return service_scores.sort_values(by='anomaly_score', ascending=False)



def metric_rcl(base_path):
    base_path = Path(base_path)
    parent_dir = base_path.parent

    # Determine if this is aiops21 or hipster/train_ticket format
    # aiops21 has normal/ and abnormal/ subdirectories with metrics.csv
    # hipster has parent_dir/fault_free/metric and parent_dir/metric
    if (base_path / 'normal').exists() and (base_path / 'abnormal').exists():
        # aiops21 format
        normal_metric_path = base_path / 'normal'
        abnormal_metric_path = base_path / 'abnormal'
        service_list_path = base_path / 'metric' / 'metric_ad_output' / 'service_ranking.csv'
    else:
        # hipster/train_ticket format
        normal_metric_path = parent_dir / 'fault_free' / 'metric'
        abnormal_metric_path = parent_dir / 'metric'
        service_list_path = base_path / 'metric' / 'metric_ad_output' / 'service_ranking.csv'
    
    output_path = base_path / 'rca_result'

    normal_metric_path.mkdir(parents=True, exist_ok=True)
    abnormal_metric_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    # print("Loading metrics...")
    normal_metrics = load_and_filter_metrics(str(normal_metric_path))
    earliest, latest = abnormal_time_range(service_list_path)
    abnormal_metrics = load_and_filter_metrics(str(abnormal_metric_path), earliest, latest)
    

    # print("Computing statistics...")
    normal_stats = compute_statistics(normal_metrics)
    abnormal_stats = compute_statistics(abnormal_metrics)
    # print(abnormal_stats)

    # print("Calculating overlaps...")
    rcl_df = calculate_overlap(normal_stats, abnormal_stats)
    rcl_df.to_csv(output_path / "metric_rca_results.csv", index=False)

    # print("Ranking services...")
    service_ranking = rank_services(rcl_df)
    # service_ranking.to_csv(output_path / "metric_service_scores.csv", index=False)
    # print("metric rcl done")

    return rcl_df, service_ranking
