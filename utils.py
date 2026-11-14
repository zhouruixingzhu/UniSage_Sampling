from datetime import datetime
import pandas as pd
import re


def parse_datetime_with_nanoseconds(time_str):
    """
    Parse a datetime string that may include nanoseconds.
    """
    # If it's already a pandas Timestamp, convert to datetime
    if isinstance(time_str, pd.Timestamp):
        return time_str.to_pydatetime()
    
    # Handle string input
    time_str = str(time_str).strip()
    
    # Try parsing with pandas which handles nanoseconds
    try:
        return pd.Timestamp(time_str).to_pydatetime()
    except Exception:
        # Fallback to datetime.fromisoformat for standard ISO format
        try:
            return datetime.fromisoformat(time_str)
        except Exception as e:
            raise ValueError(f"Unable to parse datetime: {time_str}") from e


def get_overall_time_range(time_ranges):
    # print("time_ranges:",time_ranges)
    earliest_start_time = None
    latest_end_time = None

    for time_range in time_ranges:
        if time_range is None or time_range == (None, None):
            continue

        start_time_str, end_time_str = time_range
        
        # Convert strings to datetime objects
        try:
            start_time = datetime.fromisoformat(start_time_str)
            end_time = datetime.fromisoformat(end_time_str)
        except ValueError:
            continue  # Skip invalid datetime format entries
        
        # Update the earliest start time
        if earliest_start_time is None or start_time < earliest_start_time:
            earliest_start_time = start_time
        
        # Update the latest end time
        if latest_end_time is None or end_time > latest_end_time:
            latest_end_time = end_time

    # If no valid time ranges found, return None, None
    if earliest_start_time is None or latest_end_time is None:
        return (None, None)
    
    # Return the times in ISO 8601 format as strings
    return (earliest_start_time.isoformat(), latest_end_time.isoformat())

def extract_log_trace_time_ranges(log_trace_anomalies):

    if not log_trace_anomalies:
        return None, None

    earliest_start_time = None
    latest_end_time = None

    for service, anomaly_data in log_trace_anomalies.items():
        if 'TimeRanges' in anomaly_data:
            start_time, end_time = anomaly_data['TimeRanges']
            
            # Parse times using the provided parse_datetime_with_nanoseconds function
            start_time_parsed = parse_datetime_with_nanoseconds(str(start_time))
            end_time_parsed = parse_datetime_with_nanoseconds(str(end_time))
            
            # Update the earliest start time and latest end time
            if earliest_start_time is None or start_time_parsed < earliest_start_time:
                earliest_start_time = start_time_parsed
            if latest_end_time is None or end_time_parsed > latest_end_time:
                latest_end_time = end_time_parsed

    # Convert times to ISO 8601 strings if valid
    earliest_start_time_str = earliest_start_time.strftime("%Y-%m-%dT%H:%M:%S") if earliest_start_time else None
    latest_end_time_str = latest_end_time.strftime("%Y-%m-%dT%H:%M:%S") if latest_end_time else None

    return (earliest_start_time_str, latest_end_time_str)


def extract_metric_time_ranges(metric_anomalies):

    if not metric_anomalies:
        return None, None

    earliest_start_time = None
    latest_end_time = None

    for anomaly in metric_anomalies:
        time_range = anomaly[2].strip()
        
        try:
            pattern = r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})-(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})'
            match = re.match(pattern, time_range)

            if match:
                start_time_str, end_time_str = match.groups()

            start_time = pd.Timestamp(start_time_str).strftime("%Y-%m-%dT%H:%M:%S")
            end_time = pd.Timestamp(end_time_str).strftime("%Y-%m-%dT%H:%M:%S")

            # Update the earliest start time and the latest end time
            if earliest_start_time is None or start_time < earliest_start_time:
                earliest_start_time = start_time
            if latest_end_time is None or end_time > latest_end_time:
                latest_end_time = end_time

        except Exception as e:
            print(f"Error processing time range '{time_range}': {e}")
            continue

    return (earliest_start_time, latest_end_time)
