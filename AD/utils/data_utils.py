import os
from itertools import starmap
import pandas as pd
from ..abstract import Calculator


def _apply_self_latency(trace_df):
    """If UNISAGE_RCA_ATTR=target, replace each span's Duration with its EXCLUSIVE
    (self) duration = duration - sum of its direct children's durations, computed per
    trace. Isolates a service/edge's own latency from absorbed downstream latency."""
    if os.environ.get("UNISAGE_RCA_ATTR", "target") != "target":
        return trace_df
    if not {"TraceID", "SpanID", "ParentID", "Duration"}.issubset(trace_df.columns):
        return trace_df
    child = (trace_df.groupby(["TraceID", "ParentID"])["Duration"].sum()
             .reset_index().rename(columns={"ParentID": "SpanID", "Duration": "_child"}))
    trace_df = trace_df.merge(child, on=["TraceID", "SpanID"], how="left")
    trace_df["_child"] = trace_df["_child"].fillna(0.0)
    trace_df["Duration"] = (trace_df["Duration"] - trace_df["_child"]).clip(lower=0)
    return trace_df.drop(columns=["_child"])


def _apply_responsible_service(trace_df):
    """In target mode, reattribute each span to the service it CALLS: if a span has exactly
    one cross-service child (a clean client->server call), credit it to that callee's PodName;
    otherwise keep its own PodName (own work / orchestrator / leaf). Naming-agnostic: uses the
    trace tree (SpanID/ParentID/PodName), not OperationName, so it works across systems."""
    if os.environ.get("UNISAGE_RCA_ATTR", "target") != "target":
        return trace_df
    if not {"TraceID", "SpanID", "ParentID", "PodName"}.issubset(trace_df.columns):
        return trace_df
    parent = trace_df[["TraceID", "SpanID", "PodName"]].rename(
        columns={"SpanID": "ParentID", "PodName": "_ppod"})
    ch = trace_df[["TraceID", "ParentID", "PodName"]].merge(
        parent, on=["TraceID", "ParentID"], how="inner")
    ch = ch[ch["PodName"] != ch["_ppod"]]  # cross-service children only
    if ch.empty:
        return trace_df
    g = ch.groupby(["TraceID", "ParentID"])["PodName"].agg(["nunique", "first"])
    g = g[g["nunique"] == 1]  # parents that call exactly one other service
    if g.empty:
        return trace_df
    resp = {key: pod for key, pod in g["first"].items()}
    trace_df = trace_df.copy()
    trace_df["PodName"] = [
        resp.get((t, s), pod)
        for t, s, pod in zip(trace_df["TraceID"], trace_df["SpanID"], trace_df["PodName"])
    ]
    return trace_df


class TraceDurationCalculator(Calculator):
    @staticmethod
    async def calculate(data, normal):
        """wrapper function to calculate mean and standard deviation of service spans."""
        return (
            TraceDurationCalculator._calculate_normal(data)
            if normal
            else TraceDurationCalculator._calculate_abnormal(data)
        )

    @staticmethod
    def _process_window(window_slice: pd.DataFrame, current_start):
        """Process a time window of trace data."""
        span_groups = window_slice.groupby(["PodName", "OperationName"])
        mean_duration = span_groups["Duration"].mean()
        std_duration = span_groups["Duration"].std()
        parent_service_name = span_groups["ParentID"].first()
        trace_id = span_groups["TraceID"].first()

        window_result = pd.DataFrame(
            {
                "MeanDuration": mean_duration,
                "StdDuration": std_duration,
                "ParentID": parent_service_name,
                "TraceID": trace_id,
                "StartTime": current_start,
            }
        )

        window_result["PodName"] = window_result.index.get_level_values("PodName")
        window_result["OperationName"] = window_result.index.get_level_values("OperationName")

        return window_result

    @staticmethod
    def _calculate_abnormal(trace_df: pd.DataFrame):
        """Process abnormal trace data and calculate mean and standard deviation of service spans within a rolling window."""
        trace_df = _apply_self_latency(trace_df)
        trace_df = _apply_responsible_service(trace_df)
        # Handle both aiops21 format (Timestamp) and hipster/train_ticket format (StartTimeUnixNano)
        if "StartTimeUnixNano" in trace_df.columns:
            trace_df["StartTime"] = pd.to_datetime(trace_df["StartTimeUnixNano"])
        elif "Timestamp" in trace_df.columns:
            trace_df["StartTime"] = pd.to_datetime(trace_df["Timestamp"])
        else:
            raise ValueError("No timestamp column found in trace data. Expected 'StartTimeUnixNano' or 'Timestamp'")
        
        trace_df = trace_df.set_index("StartTime").sort_values(by="StartTime")

        start_time = trace_df.index.min()
        end_time = trace_df.index.max()

        window_size = pd.Timedelta(minutes=2)
        step_size = pd.Timedelta(minutes=1)

        args = []
        for current_start in pd.date_range(start=start_time, end=end_time, freq=step_size):
            current_end = current_start + window_size
            if current_end > end_time:
                break

            window_slice = trace_df.loc[current_start:current_end]
            args.append([window_slice, current_start])

        if not args:
            # Return empty DataFrame with expected columns if no windows generated
            return pd.DataFrame(columns=["PodName", "OperationName", "MeanDuration", "StdDuration", "ParentID", "TraceID", "StartTime"])

        results = pd.concat(list(starmap(TraceDurationCalculator._process_window, args)))

        return results[
            ["PodName", "OperationName", "MeanDuration", "StdDuration", "ParentID", "TraceID", "StartTime"]
        ]

    @staticmethod
    def _calculate_normal(trace_df: pd.DataFrame):
        """Process normal trace data and calculate mean and standard deviation of service spans."""
        # Ensure Duration column exists and is numeric
        if "Duration" not in trace_df.columns:
            raise ValueError("Duration column not found in trace data")

        trace_df = _apply_self_latency(trace_df)
        trace_df = _apply_responsible_service(trace_df)
        span_groups = trace_df.groupby(["PodName", "OperationName"])
        mean_duration = span_groups["Duration"].mean()
        std_duration = span_groups["Duration"].std()
        parent_service_name = span_groups["ParentID"].first()
        trace_id = span_groups["TraceID"].first()
        data = []
        for (service, span_name), mean in mean_duration.items():
            data.append(
                {
                    "PodName": service,
                    "OperationName": span_name,
                    "MeanDuration": mean,
                    "StdDuration": std_duration[(service, span_name)],
                    "ParentID": parent_service_name[(service, span_name)],
                    "TraceID": trace_id[(service, span_name)],
                }
            )
        return pd.DataFrame(data)


class LogLevelCalculator(Calculator):
    @staticmethod
    async def calculate(data, normal=True):
        """wrapper function to calculate error and warn rates."""
        # print("Data size:", len(data))
        calculator = LogLevelCalculator._calculate_normal if normal else LogLevelCalculator._calculate_abnormal
        # print("Data size:", len(data))
        return pd.concat(list(starmap(calculator, data))).round(3)

    @staticmethod
    def _process_window(window_slice: pd.DataFrame, pod, current_start):
        """Process a time window of log data."""

        proportion = window_slice["log_level"].value_counts(normalize=True)

        window_result = pd.DataFrame(
            {
                "ServiceName": [pod],
                "ErrorRate": [proportion.get("ERROR", 0)],
                "WarnRate": [proportion.get("WARN", 0)],
                "StartTime": [current_start],
            },
            index=[0],  # let pandas know it's a single row
        )
        return window_result

    @staticmethod
    def _calculate_abnormal(log_df: pd.DataFrame, pod):
        """Process abnormal log data and calculate error and warn rates within a rolling window."""
        log_df["Timestamp"] = pd.to_datetime(log_df["Timestamp"]).dt.tz_localize(None)
        log_df.set_index("Timestamp", inplace=True)

        start_time = log_df.index.min()
        end_time = log_df.index.max()

        window_size = pd.Timedelta(minutes=2)
        step_size = pd.Timedelta(minutes=1)

        args = []
        for current_start in pd.date_range(start=start_time, end=end_time, freq=step_size):

            current_end = current_start + window_size
            window_slice = log_df.loc[current_start:current_end]

            args.append((window_slice, pod, current_start))
        try:
            results = pd.concat(list(starmap(LogLevelCalculator._process_window, args)))
        except ValueError:
            results = pd.DataFrame()

        return results

    @staticmethod
    def _compute_statistics(df: pd.DataFrame):
        """Compute mean and standard deviation of error and warn rates."""
        return pd.DataFrame(
            {
                "ServiceName": [df["ServiceName"].iloc[0]],
                "ErrorRateMean": [df["ErrorRate"].mean()],
                "ErrorRateStd": [df["ErrorRate"].std()],
                "WarnRateMean": [df["WarnRate"].mean()],
                "WarnRateStd": [df["WarnRate"].std()],
            },
            index=[0],
        )

    @staticmethod
    def _calculate_normal(log_df: pd.DataFrame, pod):
        """Calculate error and warn rates for normal log data."""
        # print(log_df.columns.tolist())
        log_df["Timestamp"] = pd.to_datetime(log_df["Timestamp"]).dt.tz_localize(None)
        log_df.set_index("Timestamp", inplace=True)

        start_time = log_df.index.min()
        end_time = log_df.index.max()

        window_size = pd.Timedelta(seconds=30)
        args = []

        for current_start in pd.date_range(start=start_time, end=end_time, freq=window_size):
            current_end = current_start + window_size

            window_slice = log_df.loc[current_start:current_end]
            args.append((window_slice, pod, current_start))

        try:
            results = pd.concat(list(starmap(LogLevelCalculator._process_window, args)))
        except ValueError:
            results = pd.DataFrame()

        return LogLevelCalculator._compute_statistics(results)

