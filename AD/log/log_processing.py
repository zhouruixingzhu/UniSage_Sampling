import asyncio
from pathlib import Path
from typing import Sequence, Tuple
import os
import re
import json
import pandas as pd
from aiomultiprocess import Pool
from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig

from ..abstract import Parser
from ..utils.data_utils import LogLevelCalculator
from ..utils.parse_utils import extract_log


class LogParser(Parser):
    def __init__(self, base_dir: Path, dataset_type: str = "hipster"):
        super().__init__(base_dir)
        self.parent_dir = base_dir.parent
        self.dataset_type = dataset_type
        
        # Support both dataset formats:
        # Format 1 (hipster/train_ticket): fault_free/log/normal_log.csv at parent level
        # Format 2 (aiops21): normal/logs.csv and abnormal/logs.csv at case level
        
        if dataset_type == "hipster":
            # hipster/train_ticket format
            self.normal_group_log_path = self.parent_dir / "fault_free" / "log"
            self.normal_parsed_dir = self.parent_dir / "fault_free" / "log" / "parsed"
            self.abnormal_group_log_path = self.base_dir / "log"
            self.log_ad_output_dir = self.base_dir / "log" / "log_ad_output"
            self.parsed_dir = self.base_dir / "log" / "parsed"
        else:
            # Use aiops21 format: base_dir is the case directory (e.g., apache01-0304-1150)
            self.normal_group_log_path = self.base_dir / "normal" / "logs.csv"
            self.normal_parsed_dir = self.base_dir / "normal" / "parsed"
            self.abnormal_group_log_path = self.base_dir / "abnormal" / "logs.csv"
            self.log_ad_output_dir = self.base_dir / "log_ad_output"
            self.parsed_dir = self.base_dir / "parsed"
        
        self.normal_parsed_dir.mkdir(parents=True, exist_ok=True)
        self.log_ad_output_dir.mkdir(parents=True, exist_ok=True)
        self.parsed_dir.mkdir(parents=True, exist_ok=True)



    async def parse(self, pool: Pool):
        """Process log data and save results to CSV file for each pod."""
        normal_groups, abnormal_groups = await asyncio.gather(
            *[self.read_and_group(normal) for normal in [True, False]]
        )
        # print(normal_groups)
        all_services = set(normal_groups.keys()).union(set(abnormal_groups.keys()))
        # print(all_services)
        
        # Handle case where there are no services
        if not all_services:
            return pd.DataFrame(), pd.DataFrame()
        
        result = await asyncio.gather(
            *[
                self._process_pod(
                    service,
                    normal_groups.get(service, pd.DataFrame()),
                    abnormal_groups.get(service, pd.DataFrame()),
                    pool,
                )
                for service in all_services
            ]
        )
        
        # Handle case where result is empty
        if not result:
            return pd.DataFrame(), pd.DataFrame()
            
        normal_result, abnormal_result = zip(
            *[((normal_group, pod), (abnormal_group, pod)) for normal_group, abnormal_group, pod in result]
        )
        # print("normal_result:",normal_result)
        # print("abnormal_result:", abnormal_result)
        normal_rate, abnormal_rate = await asyncio.gather(
            self._calculate_rate(normal_result, True, pool), self._calculate_rate(abnormal_result, False, pool)
        )
        return normal_rate, abnormal_rate

    async def read_and_group(self, normal):
        """Read and group log data by service/pod name."""
        if normal:
            log_path = self.normal_group_log_path
        else:
            log_path = self.abnormal_group_log_path
        
        # For aiops21: logs.csv is directly in normal/ or abnormal/ directory
        if self.dataset_type == "aiops21":
            if log_path.is_file():
                # Direct logs.csv file
                if not log_path.exists():
                    print(f"Log file not found at {log_path}")
                    return {}
                
                df = await asyncio.to_thread(pd.read_csv, log_path)
                
                # Group by ServiceName (aiops21 uses ServiceName for services)
                if "ServiceName" in df.columns:
                    return {name: group for name, group in df.groupby("ServiceName")}
                else:
                    # Treat entire dataset as one group if no service column
                    return {"default": df}
            else:
                # logs.csv file in normal/ or abnormal/ directory
                if not log_path.exists():
                    print(f"Log file not found at {log_path}")
                    return {}
                df = await asyncio.to_thread(pd.read_csv, log_path)
                if "ServiceName" in df.columns:
                    return {name: group for name, group in df.groupby("ServiceName")}
                else:
                    return {"default": df}
        
        # For hipster/train_ticket: process multiple log files from directory
        else:
            if not log_path.exists():
                print(f"Log directory not found at {log_path}")
                return {}
            
            all_files = [
                os.path.join(log_path, f) for f in os.listdir(log_path)
                if f.endswith("_log.csv")
            ]
            
            if not all_files:
                print("No log files found.")
                return {}

            df_list = []
            for file in all_files:
                df = await asyncio.to_thread(pd.read_csv, file, on_bad_lines="skip")
                df_list.append(df)
            
            merged_df = pd.concat(df_list, ignore_index=True)
        
            
        def extract_message(log_entry):
            try:
                log_dict = json.loads(log_entry)
                log_content = log_dict.get("log", "")

                try:
                    log_inner = json.loads(log_content)
                    return log_inner.get("message", "")
                except json.JSONDecodeError:
                    return log_content.strip()
            except json.JSONDecodeError:
                return ""

        def extract_severity(log_entry):
            try:
                log_dict = json.loads(log_entry)
                log_str = log_dict.get("log", "")
                try:
                    log_inner = json.loads(log_str)
                    if "severity" in log_inner:
                        return log_inner["severity"]
                except json.JSONDecodeError:
                    pass

                match = re.search(r'\b(INFO|ERROR|WARN|DEBUG|TRACE|FATAL)\b', log_str, re.IGNORECASE)
                if match:
                    return match.group(1).upper()

                return ""
            except Exception:
                return ""
        
        # Abnormal raw logs carry a single "Log" column to be parsed; the
        # fault-free normal_log.csv is already processed (has Body/SeverityText).
        if "Log" in merged_df.columns:
            merged_df["Body"] = merged_df["Log"].apply(extract_message)
            merged_df["SeverityText"] = merged_df["Log"].apply(extract_severity)
            merged_df.drop(columns=["Log"], inplace=True)
        
        df = merged_df.dropna(subset=["Body"])
        name_col = "PodName" if "PodName" in df.columns else "k8s_container_name"
        return {name: group for name, group in df.groupby(name_col)}


    async def _process_pod(self, pod, normal_group: pd.DataFrame, abnormal_group: pd.DataFrame, pool: Pool):
        """Process logs for a pod and save results to a CSV file."""
        template_miner = get_template_miner()
        # print("here:",normal_group)
        if not normal_group.empty:
            normal_group, template_miner = await pool.apply(
                batch_process, args=(normal_group, pod, template_miner, self.normal_parsed_dir)
            )
        if not abnormal_group.empty:
            abnormal_group, _ = await pool.apply(
                batch_process, args=(abnormal_group, pod, template_miner, self.parsed_dir )
            )
        return (normal_group, abnormal_group, pod)

    async def _calculate_rate(self, result: Sequence[Tuple[pd.DataFrame, str]], normal: bool, pool: Pool):
        """Calculate log level rate for each pod."""
        # filter out empty dataframes
        result = [r for r in result if not r[0].empty]
        rate = await pool.apply(LogLevelCalculator.calculate, args=(result, normal))
        await asyncio.to_thread(
            rate.to_csv, self.log_ad_output_dir / ("normal_rate.csv" if normal else "abnormal_rate.csv"), index=False, mode="w"
        )
        return rate


def get_template_miner(use_persistence_handler="None", config=TemplateMinerConfig()):

    if use_persistence_handler == "redis":
        from drain3.redis_persistence import RedisPersistence

        template_miner = TemplateMiner(
            persistence_handler=RedisPersistence(
                redis_host="localhost", redis_port=6379, redis_db=1, redis_key="drain", redis_pass=None, is_ssl=False
            ),
            config=config,
        )
    elif use_persistence_handler == "file":
        template_miner = TemplateMiner(
            persistence_handler=FilePersistence("./drain_state"),
            config=config,
        )
    else:
        template_miner = TemplateMiner(config=config)
    return template_miner


def parse_with_drain(log_df: pd.DataFrame, pod, template_miner: TemplateMiner) -> pd.DataFrame:
    """parse logs with drain and add template id and template mined columns to the dataframe."""
    log_df["temp_id"] = pd.Series(dtype=str)
    log_df["log_temp"] = pd.Series(dtype=str)

    if pod == "ts-ui-dashboard":
        template_miner.config.drain_sim_th = 0.8
    for index, row in log_df.iterrows():
        result = template_miner.add_log_message(row["message"])
        log_df.at[index, "temp_id"] = result["cluster_id"]
        log_df.at[index, "log_temp"] = result["template_mined"]
    return log_df, template_miner


async def batch_process(
    batch: pd.DataFrame, pod, template_miner: TemplateMiner, output_dir: Path
) -> Tuple[pd.DataFrame, TemplateMiner]:
    """Process each batch and return the processed DataFrame."""
    batch[["Body", "message", "log_level"]] = batch.apply(extract_log, axis="columns", result_type="expand")
    batch.dropna(subset=["log_level"], inplace=True)  # Drop rows with no log level
    batch, template_miner = parse_with_drain(batch, pod, template_miner)
    final_df = batch[[c for c in ["Timestamp", "TraceID", "log_level", "temp_id", "log_temp", "Body"] if c in batch.columns]].copy()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{pod}.csv"
    if final_df.size == 0:
        # print(f"No logs for {pod}")
        return pd.DataFrame(), template_miner
    # Train Ticket log files contain a mixture of timezone-qualified and naive ISO
    # timestamps; pandas 2.x needs explicit mixed-format parsing for the same column.
    final_df.loc[:, "Timestamp"] = pd.to_datetime(final_df["Timestamp"], utc=True, format="mixed")

    final_df = final_df.sort_values(by="Timestamp")  # sort otherwise the rolling window will not work

    await asyncio.to_thread(final_df.to_csv, output_file, index=False, mode="w")
    # print(f"Logs for {pod} saved to {output_file}")
    return final_df, template_miner
