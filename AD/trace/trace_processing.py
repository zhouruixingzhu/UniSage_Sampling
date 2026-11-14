import asyncio
import base64
import json
from pathlib import Path
import os
import pandas as pd
from aiomultiprocess import Pool
from ..abstract import Parser
from ..utils.data_utils import TraceDurationCalculator


class TraceParser(Parser):
    def __init__(self, base_dir, dataset_type: str = "hipster"):
        super().__init__(base_dir)
        # self.file_name = "traces.csv"
        self.parent_dir = base_dir.parent
        self.dataset_type = dataset_type
              
        if dataset_type == "hipster":
            # hipster/train_ticket format
            self.normal_group_trace_path = self.parent_dir / "fault_free" / "trace" / "normal_trace.csv"
            self.abnormal_group_trace_path = self.base_dir / "trace"
            self.output_dir = self.base_dir / "trace" / "trace_ad_output"
        else:
            # Use aiops21 format: base_dir is the case directory
            self.normal_group_trace_path = self.base_dir / "normal" / "traces.csv"
            self.abnormal_group_trace_path = self.base_dir / "abnormal" / "traces.csv"
            self.output_dir = self.base_dir / "trace_ad_output"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def parse(self, pool: Pool):
        """Parse trace data and calculate mean and standard deviation of service spans."""
        tasks = [self._process_func(pool, normal=normal) for normal in [True, False]]
        return await asyncio.gather(*tasks)

    async def _process_func(self, pool: Pool, normal=True):
        """Process trace data and calculate mean and standard deviation of service spans."""
        if normal:
            trace_group_path = self.normal_group_trace_path
            # Check if the normal trace file exists
            if not trace_group_path.exists():
                print(f"Normal trace file not found at {trace_group_path}")
                return pd.DataFrame()
            parsed_csv_path = trace_group_path
            trace_df = await parse_and_save_csv(trace_group_path, parsed_csv_path, pool, batch_size=50000)
        else:
            # Check if abnormal trace path exists
            if not self.abnormal_group_trace_path.exists():
                print(f"Abnormal trace path not found at {self.abnormal_group_trace_path}")
                return pd.DataFrame()
            
            # Check if it's a file (aiops21 format) or directory
            if self.abnormal_group_trace_path.is_file():
                # aiops21 format: direct traces.csv file
                trace_df = pd.read_csv(self.abnormal_group_trace_path)
            else:
                # hipster/train_ticket format: directory with multiple trace files
                all_files = [
                    os.path.join(self.abnormal_group_trace_path, f) for f in os.listdir(self.abnormal_group_trace_path)
                    if f.endswith("_trace.csv")
                ]
            
                if not all_files:
                    print("No trace files found.")
                    return pd.DataFrame()
                
                # df_list = [pd.read_csv(file) for file in all_files]
                df_list = []
                for file in all_files:
                    # print(f"Reading file: {file}")
                    df = pd.read_csv(file)
                    df_list.append(df)

                trace_df = pd.concat(df_list, ignore_index=True)

        # Normalize column names to support both aiops21 and hipster/train_ticket formats
        trace_df = normalize_trace_columns(trace_df)
        
        if "ts" in self.base_dir.name:
            trace_df = await pool.apply(ui_span_name_parser, args=(trace_df,))
        data = await pool.apply(TraceDurationCalculator.calculate, args=(trace_df, normal))
        output_file = self.output_dir / ("normal_rate.csv" if normal else "abnormal_rate.csv")
        await asyncio.to_thread(data.to_csv, output_file, index=False, mode="w")
        return data



def normalize_trace_columns(trace_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize trace column names to support both aiops21 and hipster/train_ticket formats.
    
    Maps:
    - ServiceName -> PodName
    - SpanName -> OperationName
    - TraceId -> TraceID
    - SpanId -> SpanID
    - ParentSpanId -> ParentID
    """
    rename_dict = {}
    
    # Map ServiceName to PodName if it exists
    if "ServiceName" in trace_df.columns and "PodName" not in trace_df.columns:
        rename_dict["ServiceName"] = "PodName"
    
    # Map SpanName to OperationName if it exists
    if "SpanName" in trace_df.columns and "OperationName" not in trace_df.columns:
        rename_dict["SpanName"] = "OperationName"
    
    # Map TraceId to TraceID if it exists
    if "TraceId" in trace_df.columns and "TraceID" not in trace_df.columns:
        rename_dict["TraceId"] = "TraceID"
    
    # Map SpanId to SpanID if it exists
    if "SpanId" in trace_df.columns and "SpanID" not in trace_df.columns:
        rename_dict["SpanId"] = "SpanID"
    
    # Map ParentSpanId to ParentID if it exists
    if "ParentSpanId" in trace_df.columns and "ParentID" not in trace_df.columns:
        rename_dict["ParentSpanId"] = "ParentID"
    
    if rename_dict:
        trace_df = trace_df.rename(columns=rename_dict)
    
    # Ensure required columns exist, fill with defaults if not
    if "TraceID" not in trace_df.columns:
        trace_df["TraceID"] = None
    if "ParentID" not in trace_df.columns:
        trace_df["ParentID"] = None
    if "ParentServiceName" not in trace_df.columns:
        trace_df["ParentServiceName"] = None
    
    return trace_df


def resolve_parent_service_name(parsed_df: pd.DataFrame) -> pd.DataFrame:
    """Resolve ParentServiceName by looking up ServiceName for matching ParentSpanId."""
    span_id_to_service = parsed_df.set_index("SpanID")["PodName"].to_dict()
    parsed_df["ParentServiceName"] = parsed_df["ParentID"].map(span_id_to_service)
    mask = parsed_df["ParentID"].notna() & parsed_df["ParentServiceName"].isna()
    parsed_df.loc[mask, "ParentID"] = pd.NA
    return parsed_df


async def batch_process(batch: pd.DataFrame):
    parsed_data = []
    for _, row in batch.iterrows():
        parsed_data.append(
            {
                "StartTime": row["StartTimeUnixNano"],
                "TraceID": row["TraceID"],
                "SpanID": row["SpanID"],
                "OperationName": row["OperationName"],
                "PodName": row["PodName"],
                "Duration": row["Duration"],
                "ParentID": row["ParentID"],
            }
        )
    return pd.DataFrame(parsed_data)


#ts用的
async def ui_span_name_parser(df: pd.DataFrame) -> pd.DataFrame:
    child_df = df[['ParentSpanId', 'SpanName']].rename(columns={'ParentSpanId': 'SpanId', 'SpanName': 'ChildSpanName'})
    merged_df = df.merge(child_df, on='SpanId', how='left')
    merged_df.loc[merged_df['ServiceName'] == 'ts-ui-dashboard', 'SpanName'] = merged_df['ChildSpanName']
    merged_df.drop(columns='ChildSpanName', inplace=True)
    return merged_df


async def parse_and_save_csv(input_file, output_parsed_csv, pool: Pool, batch_size=100000) -> pd.DataFrame:
    """Parse the CSV file, handle both old and new formats, and save the parsed data to a new CSV."""

    trace_df = pd.read_csv(input_file, engine="pyarrow")

    if "model" in trace_df.columns:

        batches = [trace_df.iloc[start : start + batch_size] for start in range(0, len(trace_df), batch_size)]

        # Use aiomultiprocessing to process batches in parallel

        results = await pool.map(batch_process, batches)
        parsed_df = pd.concat(results, ignore_index=True)
        # parsed_df = resolve_parent_service_name(parsed_df)
        parsed_df["StartTime"] = pd.to_datetime(parsed_df["StartTime"])
        parsed_df = parsed_df.sort_values(by="StartTime")

        await asyncio.to_thread(parsed_df.to_csv, output_parsed_csv, index=False, mode="w")
        return parsed_df
    else:
        return trace_df
