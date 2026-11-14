import asyncio
from pathlib import Path
from typing import List
import re

import pandas as pd
from aiomultiprocess import Pool
from pandas import DataFrame

from ..abstract import Parser
import datetime
import datetime
from pathlib import Path
import pandas as pd
from aiomultiprocess import Pool
from pandas import DataFrame
from ..abstract import Parser

class MetricParser(Parser):
    def __init__(self, base_dir: Path, dataset_type: str = "hipster"):
        super().__init__(base_dir)
        self.parent_dir = base_dir.parent
        self.dataset_type = dataset_type
        
        if dataset_type == "hipster":
            # hipster/train_ticket format
            self.normal_dir = self.parent_dir / "fault_free" / "metric"
            self.abnormal_dir = self.parent_dir / "metric"
            self.output_dir = self.base_dir / "metric" / "metric_ad_output"
        else:
            # aiops21 format - metrics.csv is directly in normal/abnormal directories
            self.normal_dir = self.base_dir / "normal"
            self.abnormal_dir = self.base_dir / "abnormal"
            self.output_dir = self.base_dir / "metric_ad_output"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.abnormal_start_time = self._get_abnormal_start_time(base_dir)


    def _get_abnormal_start_time(self, base_dir: Path) -> datetime.datetime:
        """Get abnormal start time from metadata or directory name."""
        # For aiops21, try to read from metadata.json first
        metadata_file = base_dir / "metadata.json"
        if self.dataset_type == "aiops21" and metadata_file.exists():
            try:
                import json
                with open(metadata_file) as f:
                    metadata = json.load(f)
                    fault_time = metadata.get('fault_start_time', '')
                    # Parse ISO format: "2021-03-09T01:20:00"
                    if 'T' in fault_time:
                        time_part = fault_time.split('T')[1]
                        hour, minute = time_part.split(':')[:2]
                        return datetime.datetime.strptime(f"{hour}_{minute}", "%H_%M")
            except (json.JSONDecodeError, ValueError, KeyError):
                pass
        
        # Fallback: try to extract from directory name
        time_str = base_dir.name
        try:
            return datetime.datetime.strptime(time_str, "%H_%M")
        except ValueError:
            # For datasets like aiops21 where the directory name doesn't follow HH_MM format,
            # try to extract HH_MM from the directory name
            match = re.search(r'(\d{2})(\d{2})', time_str)
            if match:
                try:
                    hour, minute = match.groups()
                    return datetime.datetime.strptime(f"{hour}_{minute}", "%H_%M")
                except ValueError:
                    pass
            # Default: return a time at midnight
            return datetime.datetime.strptime("00_00", "%H_%M")

    async def parse(self, pool: Pool):
        if self.dataset_type == "hipster":
            # hipster/train_ticket format: directory with multiple CSV files
            tasks = [
                self._load_processed_metrics(self.normal_dir, is_abnormal=False),
                self._load_processed_metrics(self.abnormal_dir, is_abnormal=True),
            ]
            results = await asyncio.gather(*tasks)
            return results
        else:
            # aiops21 format: single metrics.csv file
            normal_results = await self._load_aiops21_metrics(self.normal_dir, is_abnormal=False)
            abnormal_results = await self._load_aiops21_metrics(self.abnormal_dir, is_abnormal=True)
            
            # Align features between normal and abnormal data for each service
            # This ensures both have the same columns
            normal_dict = {name: df for df, name in normal_results}
            abnormal_dict = {name: df for df, name in abnormal_results}
            
            aligned_normal = []
            aligned_abnormal = []
            
            for service_name in set(normal_dict.keys()) & set(abnormal_dict.keys()):
                normal_df = normal_dict[service_name]
                abnormal_df = abnormal_dict[service_name]
                
                # Get all columns that are in both
                time_col = 'Time' if 'Time' in normal_df.columns else normal_df.columns[0]
                numeric_cols_normal = set(c for c in normal_df.columns if c != time_col and c != 'PodName')
                numeric_cols_abnormal = set(c for c in abnormal_df.columns if c != time_col and c != 'PodName')
                
                # Common columns
                common_cols = numeric_cols_normal & numeric_cols_abnormal
                
                # If no common columns, try to align by metric name patterns
                if not common_cols:
                    common_cols = numeric_cols_normal
                
                # Keep only common columns and fill missing with 0
                select_cols = [time_col] + list(common_cols) + ([c for c in ['PodName'] if c in normal_df.columns])
                normal_df = normal_df[[c for c in select_cols if c in normal_df.columns]]
                abnormal_df = abnormal_df[[c for c in select_cols if c in abnormal_df.columns]]
                
                # Ensure both have the same columns
                all_cols = set(normal_df.columns) | set(abnormal_df.columns)
                for col in all_cols:
                    if col not in normal_df.columns:
                        normal_df[col] = 0
                    if col not in abnormal_df.columns:
                        abnormal_df[col] = 0
                
                aligned_normal.append((normal_df, service_name))
                aligned_abnormal.append((abnormal_df, service_name))
            
            return [aligned_normal, aligned_abnormal]

    async def _load_processed_metrics(self, dir_path: Path, is_abnormal: bool):
        results = []
        pass_file = ["source_50", "dependency", "destination_50", "front_service"]

        for file in dir_path.glob("*.csv"):
            if any(skip_str in file.name for skip_str in pass_file):
                # print(f"Skipped {file.name}")
                continue

            pod_name = file.stem
            df = await asyncio.to_thread(pd.read_csv, file)
            if "PodWorkload(Ops)" in df.columns:
                df = df.drop(columns=["PodWorkload(Ops)"])

            if is_abnormal:
                df = await asyncio.to_thread(
                    self._filter_abnormal_window, df, file
                )

            results.append((df, pod_name))
        return results

    def _filter_abnormal_window(self, df: pd.DataFrame, file_path: Path) -> pd.DataFrame:
        if 'Time' not in df.columns:
            print(f"Skipped {file_path.name}: No 'Time' column found.")
            return df

        df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
        df = df.dropna(subset=['Time'])


        start = self.abnormal_start_time
        end = start + datetime.timedelta(minutes=4)
        mask = (df['Time'].dt.time >= start.time()) & (df['Time'].dt.time <= end.time())

        return df[mask]

    async def _load_aiops21_metrics(self, dir_path: Path, is_abnormal: bool):
        """Load metrics from aiops21 format (single metrics.csv file).
        
        Converts from long format (one row per metric per timestamp) to wide format
        where each column is a metric for a specific pod.
        """
        results = []
        # aiops21 has metrics.csv directly in the normal/abnormal directory
        metrics_file = dir_path / "metrics.csv"
        
        if not metrics_file.exists():
            print(f"Metrics file not found at {metrics_file}")
            return results
        
        # Read the metrics file
        df = await asyncio.to_thread(pd.read_csv, metrics_file)
        
        # Group by pod/service and create one dataframe per service
        if 'k8s_pod_name' in df.columns:
            for service_name, group_df in df.groupby('k8s_pod_name'):
                # Convert TimeUnix to Time column for compatibility
                group_df = group_df.copy()
                group_df['Time'] = pd.to_datetime(group_df['TimeUnix'], errors='coerce')
                
                # Pivot the data: rows are timestamps, columns are metrics
                # Create a unique metric identifier combining MetricName
                if 'MetricName' in group_df.columns:
                    # For each metric, pivot by timestamp
                    pivoted_list = []
                    for metric_name, metric_group in group_df.groupby('MetricName'):
                        # Pivot Time vs Value for this metric
                        metric_df = metric_group[['Time', 'Value']].drop_duplicates(subset=['Time'])
                        metric_df = metric_df.set_index('Time').rename(columns={'Value': metric_name})
                        pivoted_list.append(metric_df)
                    
                    if pivoted_list:
                        # Merge all metrics for this pod
                        service_df = pd.concat(pivoted_list, axis=1)
                        service_df = service_df.reset_index()
                        
                        # Add PodName column for compatibility
                        service_df['PodName'] = service_name
                        
                        # Filter abnormal window if needed
                        if is_abnormal:
                            service_df = await asyncio.to_thread(
                                self._filter_abnormal_window_aiops21, service_df
                            )
                        
                        results.append((service_df, service_name))
        
        return results

    def _filter_abnormal_window_aiops21(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter metrics for the abnormal time window in aiops21 format."""
        if 'TimeUnix' not in df.columns:
            return df
        
        # Convert datetime string to datetime (aiops21 uses ISO format strings, not Unix timestamps)
        df = df.copy()
        df['Time'] = pd.to_datetime(df['TimeUnix'], errors='coerce')
        df = df.dropna(subset=['Time'])
        
        start = self.abnormal_start_time
        end = start + datetime.timedelta(minutes=4)
        mask = (df['Time'].dt.time >= start.time()) & (df['Time'].dt.time <= end.time())
        
        return df[mask]
