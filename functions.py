import os
import math
import json
import time
import traceback
import numpy as np
import pandas as pd
from json.decoder import JSONDecodeError
from influxdb import client as influxdb
from datetime import timedelta, datetime
from multiprocessing.pool import ThreadPool
from flask import current_app
from config import db_name, tb_name, db_hostname,db_port,parameters,time_var,fill_method,all_tags,tb_name_target
from config import parameter_ranges,cvs_1,units_1,units_2,targets_new, timestamp, resample_dataframe,threshold,threshold_past,all_tags
from config import previous_path,similar_plus_cap,data_24hrs,min_max_path,previous_Fingerprint_path,timestamp_1,target_vars,change_data
from config import control_parameters,control_parameters_unit,process_parameters,process_parameters_unit,min_max_ranges
#import pythoncom
#import win32com.client


def get_end_datetime():
    return datetime(2025,4,21,15,00,00)

def change_json(f_Json):
    try:
        # Ensure 'deviation' exists in the JSON
        if "deviation" not in f_Json or not isinstance(f_Json["deviation"], dict):
            raise ValueError("Invalid JSON structure: 'deviation' key missing or not a dictionary")

        # Sort the 'deviation' dictionary based on the 'order' key
        f_Json["deviation"]= dict(sorted(f_Json["deviation"].items(), key= lambda x: x[1].get('order', float('inf'))))
        deviation_range= f_Json["deviation"]

        # Scale 'Higher' and 'Lower' values for each key
        for key, values in deviation_range.items():
            if "Higher" in values and "Lower" in values:
                values["Higher"] /= 100
                values["Lower"] /= 100

        return deviation_range

    except Exception as e:
        print(f"Error in change_json: {e}")
        return {}  # Return an empty dictionary in case of failure


def read_data(start_time, end_time):
    """
    Args:
        start_time (datetime): The start time for the query.
        end_time (datetime): The end time for the query.
    Returns:
        pd.DataFrame: Dataframe containing the query results.
    """
    try:
        # Initialize InfluxDB client with connection pooling
        db = influxdb.InfluxDBClient(db_hostname, db_port, pool_size=10)
        db.switch_database(db_name)

        #field_str = ", ".join(all_tags)

        # Construct query to fetch data within the given time range
        query = f"""SELECT timestamp,k5027tekilnchanneltemperaturesh1,k5028tekilnchanneltemperarutesh2,k5026te1kilntemperaturepyrometer1,k5026te2kilntemperaturepyrometer2,k5001accombustionairbloweractualspeed,coalfuelcounternominal,k5314ficcontrolleroutput,k5351acrotaryvalvespeed,k5352acrotaryvalvespeed,k5353acrotaryvalvespeed,k5354acrotaryvalvespeed,k5355acrotaryvalvespeed,k5356acrotaryvalvespeed,k5357acrotaryvalvespeed,k5358acrotaryvalvespeed,k5359acrotaryvalvespeed,k5360acrotaryvalvespeed,k5301cobincoaldust,k5053telimedischargetempsh11,k5054telimedischargetempsh21,k5029tekilnlancetemperaturesh1,k5030tekilnlancetemperaturesh2 FROM {tb_name} WHERE time >= '{start_time.isoformat()}Z' AND time <= '{end_time.isoformat()}Z'"""

        # Execute query asynchronously using ThreadPool to avoid blocking
        with ThreadPool(processes=1) as pool:
            datapoints = pool.apply_async(db.query, (query,))
            result = datapoints.get()  # Wait for query execution

        # Convert results to DataFrame
        dataframe = pd.DataFrame(result.get_points()) if result else pd.DataFrame()

        if not dataframe.empty:
            # Convert timestamps and format them properly
            dataframe[timestamp] = pd.to_datetime(dataframe[timestamp])
            dataframe[timestamp] = dataframe[timestamp].dt.strftime('%Y-%m-%d %H:%M:%S')
            dataframe[timestamp] = pd.to_datetime(dataframe[timestamp])  # Ensure datetime type

            # Round numerical values and drop NaN values for better data integrity
            dataframe = dataframe.round(6).dropna()
            dataframe1 = dataframe.rename(columns=change_data)

            return dataframe1
        else:
            return pd.DataFrame()  # Return empty DataFrame if no data
    except Exception as e:
        print(f"Error in read_data: {e}")
        return pd.DataFrame()
    
# Reads data from an InfluxDB database between the specified time range.
def read_data_store(start_time, end_time):
    """
    Args:
        start_time (datetime): The start time for the query.
        end_time (datetime): The end time for the query.
    Returns:
        pd.DataFrame: Dataframe containing the query results.
    """
    try:
        # Initialize InfluxDB client with connection pooling
        db= influxdb.InfluxDBClient(db_hostname, db_port, pool_size= 10)
        db.switch_database(db_name)
        
        #field_str = ", ".join(all_tags)

        # Construct query to fetch data within the given time range
        query= f"""SELECT * FROM {tb_name_target} WHERE time >= '{start_time.isoformat()}Z' AND time <= '{end_time.isoformat()}Z'"""

        # Execute query asynchronously using ThreadPool to avoid blocking
        with ThreadPool(processes= 1) as pool:
            datapoints= pool.apply_async(db.query, (query,))
            result= datapoints.get()  # Wait for query execution

        # Convert results to DataFrame
        dataframe= pd.DataFrame(result.get_points()) if result else pd.DataFrame()

        if not dataframe.empty:
            # Convert timestamps and format them properly
            dataframe[timestamp_1]= pd.to_datetime(dataframe[timestamp_1])
            dataframe[timestamp_1]= dataframe[timestamp_1].dt.strftime('%Y-%m-%d %H:%M:%S')
            dataframe[timestamp_1]= pd.to_datetime(dataframe[timestamp_1]) # Ensure datetime type
            dataframe=dataframe.set_index(timestamp_1).resample(resample_dataframe).first().reset_index()

            # Round numerical values and drop NaN values for better data integrity
            dataframe= dataframe.round(6).dropna()
        return dataframe

    except Exception as e:
        print(f"Error in read_data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of failure

def deviation(df_dev_filtered, real_data, deviation_ranges):
    """
    Vectorized filtering using cumulative boolean masks.
    
    Args:
        df_dev_filtered (pd.DataFrame): Historical data to be filtered.
        real_data (pd.DataFrame): Real-time data containing the latest values.
        deviation_ranges (dict): Dictionary containing deviation factors for each parameter.
            For example: 
                { 'param1': {'Lower': 0.95, 'Higher': 1.05},
                  'param2': {'Lower': 0.90, 'Higher': 1.10}, ... }
        threshold (int): Minimum number of records required after filtering (default is 300).
        
    Returns:
        tuple: (Filtered DataFrame, count of filters successfully applied)
    """
    count_dev = 0  # Counter for the number of filters applied
    df_filtered = df_dev_filtered.copy()
    
    # Start with a mask of all True values for the entire DataFrame.
    mask = np.ones(len(df_filtered), dtype=bool)
    
    try:
        deviation_ranges['Coal fuel counter actual [kg]'] = {'Lower': 0.95,'Higher': 1.05,'Min': 0,'Max': 5000,'order': 16,'target': -1}
        # Iterate over each parameter and its deviation factors.
        deviation_ranges = {
            'Coal fuel counter actual [kg]': deviation_ranges['Coal fuel counter actual [kg]'],
            **{k: v for k, v in deviation_ranges.items() if k != 'Coal fuel counter actual [kg]'}
        }
        for parameter, ranges in deviation_ranges.items():
            # Calculate lower and upper threshold limits using the last real-time value.
            last_value = real_data[parameter].iat[-1]
            lower = ranges['Lower'] * last_value
            higher = ranges['Higher'] * last_value
            
            # Build the boolean condition for the entire DataFrame for this parameter.
            condition = df_filtered[parameter].between(lower, higher).to_numpy()
            
            # Update the cumulative mask.
            new_mask = mask & condition
            
            # If applying this filter would drop the count below the threshold,
            # then return the DataFrame filtered by the previous mask and the count so far.
            if new_mask.sum() < threshold:
                return df_filtered.loc[mask], count_dev
            else:
                mask = new_mask
                count_dev += 1
                
        # Return the fully filtered DataFrame and the count of applied filters.
        return df_filtered.loc[mask], count_dev

    except KeyError as e:
        print(f"Error: Missing parameter in data - {e}")
    except Exception as e:
        print(f"Unexpected error in deviation_fast: {e}")
    
    return df_filtered, count_dev

# Filters past data based on deviation ranges for each parameter compared to real-time data.
def deviation_past(df_dev_filtered, real_data, deviation_ranges):
    """
    Vectorized filtering using cumulative boolean masks.
    
    Args:
        df_dev_filtered (pd.DataFrame): Historical data to be filtered.
        real_data (pd.DataFrame): Real-time data containing the latest values.
        deviation_ranges (dict): Dictionary containing deviation factors for each parameter.
            For example: 
                { 'param1': {'Lower': 0.95, 'Higher': 1.05},
                  'param2': {'Lower': 0.90, 'Higher': 1.10}, ... }
        threshold (int): Minimum number of records required after filtering (default is 300).
        
    Returns:
        tuple: (Filtered DataFrame, count of filters successfully applied)
    """
    count_dev = 0  # Counter for the number of filters applied
    df_filtered = df_dev_filtered.copy()
    
    # Start with a mask of all True values for the entire DataFrame.
    mask = np.ones(len(df_filtered), dtype=bool)
    
    try:
        deviation_ranges['Coal fuel counter actual [kg]'] = {'Lower': 0.95,'Higher': 1.05,'Min': 0,'Max': 5000,'order': 16,'target': -1}
        # Iterate over each parameter and its deviation factors.
        deviation_ranges = {
            'Coal fuel counter actual [kg]': deviation_ranges['Coal fuel counter actual [kg]'],
            **{k: v for k, v in deviation_ranges.items() if k != 'Coal fuel counter actual [kg]'}
        }
        for parameter, ranges in deviation_ranges.items():
            # Calculate lower and upper threshold limits using the last real-time value.
            last_value = real_data[parameter].iat[-1]
            lower = ranges['Lower'] * last_value
            higher = ranges['Higher'] * last_value
            
            # Build the boolean condition for the entire DataFrame for this parameter.
            condition = df_filtered[parameter].between(lower, higher).to_numpy()
            
            # Update the cumulative mask.
            new_mask = mask & condition
            
            # If applying this filter would drop the count below the threshold,
            # then return the DataFrame filtered by the previous mask and the count so far.
            if new_mask.sum() < threshold_past:
                return df_filtered.loc[mask], count_dev
            else:
                mask = new_mask
                count_dev += 1
                
        # Return the fully filtered DataFrame and the count of applied filters.
        return df_filtered.loc[mask], count_dev

    except KeyError as e:
        print(f"Error: Missing parameter in data - {e}")
    except Exception as e:
        print(f"Unexpected error in deviation_fast: {e}")
    
    return df_filtered, count_dev

# def check_trend(series, timestamps, a, b):
#     # Convert the series and timestamps to numpy arrays for faster operations
#     values = np.array(series)
    
#     trends = []
    
#     # If we're looking for an increasing trend (b < a)
#     if b < a:
#         start_idx = None  # Track the start of the trend
#         for i in range(len(values)):
#             if start_idx is None and values[i] == b:
#                 start_idx = i  # Start of a potential trend
#             elif start_idx is not None:
#                 # Check if it's still increasing and valid
#                 if values[i] > values[i-1] and values[i] <= a:
#                     # Update start_idx only if the current value is not equal to b
#                     if values[i] != b:
#                         last_valid_idx = i  # Keep track of the last valid index
#                 elif values[i] == a:
#                     # If we reach 'a', save the trend
#                     trends.append((timestamps[start_idx], timestamps[last_valid_idx]))
#                     start_idx = None  # Reset to look for more trends
#                     last_valid_idx = None  # Reset last valid index
#                 else:
#                     # If the trend breaks, reset
#                     start_idx = None

#     # If we're looking for a decreasing trend (b > a)
#     elif b > a:
#         start_idx = None  # Track the start of the trend
#         for i in range(len(values)):
#             if start_idx is None and values[i] == b:
#                 start_idx = i  # Start of a potential trend
#             elif start_idx is not None:
#                 # Check if it's still decreasing and valid
#                 if values[i] < values[i-1] and values[i] >= a:
#                     # Update start_idx only if the current value is not equal to b
#                     if values[i] != b:
#                         last_valid_idx = i  # Keep track of the last valid index
#                 elif values[i] == a:
#                     # If we reach 'a', save the trend
#                     trends.append((timestamps[start_idx], timestamps[last_valid_idx]))
#                     start_idx = None  # Reset to look for more trends
#                     last_valid_idx = None  # Reset last valid index
#                 else:
#                     # If the trend breaks, reset
#                     start_idx = None

#     return trends
def check_trend(series, timestamps, a, b):
    """
    Vectorized-ish implementation for detecting trends.
    
    For an increasing trend (b < a):
      - A trend starts when the value equals b.
      - The trend continues if the subarray from the candidate start to the first occurrence of a
        is strictly increasing and all values are <= a.
      - The trend is recorded as (start_timestamp, timestamp just before a occurs).
    
    For a decreasing trend (b > a), analogous logic applies.
    
    Args:
        series (iterable): Sequence of numeric values.
        timestamps (iterable): Corresponding timestamps.
        a (numeric): Target end value for the trend.
        b (numeric): Starting value for the trend.
        
    Returns:
        list of tuples: Each tuple is (start_timestamp, end_timestamp) for a detected trend.
    """
    values = np.array(series)
    ts = np.array(timestamps)
    trends = []
    
    # Find all indices where values equal the start value b
    candidate_starts = np.where(values == b)[0]
    # Precompute indices where values equal a (the desired end marker)
    a_indices = np.where(values == a)[0]
    
    # Loop only over candidate start indices
    for start_idx in candidate_starts:
        # Use searchsorted to quickly find the first index in a_indices that is > start_idx
        pos = np.searchsorted(a_indices, start_idx + 1)
        if pos >= len(a_indices):
            continue  # No valid end found for this candidate
        end_candidate = a_indices[pos]
        
        # Extract the segment from candidate start up to and including the first occurrence of a
        segment = values[start_idx:end_candidate + 1]
        
        if b < a:
            # For an increasing trend, the segment must be strictly increasing and all values <= a
            if np.all(np.diff(segment) > 0) and np.all(segment <= a):
                # Record trend from the candidate start timestamp to the last valid index (before a)
                if end_candidate - 1 >= start_idx:
                    trends.append((ts[start_idx], ts[end_candidate - 1]))
        elif b > a:
            # For a decreasing trend, the segment must be strictly decreasing and all values >= a
            if np.all(np.diff(segment) < 0) and np.all(segment >= a):
                if end_candidate - 1 >= start_idx:
                    trends.append((ts[start_idx], ts[end_candidate - 1]))
                    
    return trends

#Compares real-time data with historical fingerprint data to find similar timestamps within a deviation range.
def similarity_minus(real_t, fingerprint_t_r, real_data, df_similar_previous, previous_time, deviation_ranges):
    """
    A vectorized version of similarity_minus that compares a fixed real‑time window with many historical windows 
    (one per fingerprint timestamp) using a rolling window view.
    
    Assumptions:
      - Both real_data and df_similar_previous are resampled to a uniform grid 
        (default is '1T' meaning one-minute intervals).
      - fingerprint_t_r are timestamps that appear in the resampled historical data.
    
    Args:
      real_t (datetime): The current real‑time timestamp.
      fingerprint_t_r (list): List of fingerprint timestamps (as datetime objects).
      real_data (pd.DataFrame): Real‑time data containing a timestamp column.
      df_similar_previous (pd.DataFrame): Historical fingerprint data containing a timestamp column.
      previous_time (int): Time difference (in minutes) to use as the window length.
      deviation_ranges (dict): Deviation limits for each parameter. For example,
             { 'col1': {'Lower': 0.9, 'Higher': 1.1}, 'col2': {'Lower': 0.95, 'Higher': 1.05}, ... }.
    
    Returns:
      list: List of fingerprint timestamps (from fingerprint_t_r) for which at least one row in the window 
            meets the deviation condition.
    """
    # 1. Prepare the real‑time window.
    real_t_minus = real_t - timedelta(minutes=previous_time)
    real_minus = real_data[real_data[timestamp].between(real_t_minus, real_t)]
    real_minus = real_minus.set_index(timestamp).resample(resample_dataframe).first().reset_index()
    
    # 2. Resample and sort the historical fingerprint data.
    df_prev = df_similar_previous.copy()
    df_prev[timestamp] = pd.to_datetime(df_prev[timestamp])
    df_prev = df_prev.set_index(timestamp).resample(resample_dataframe).first().reset_index()
    df_prev.sort_values(timestamp, inplace=True)
    
    deviation_ranges['Coal fuel counter actual [kg]'] = {'Lower': 0.9,'Higher': 1.1,'Min': 0,'Max': 5000,'order': 16,'target': -1}
    # 3. Select the columns to compare.
    dev_cols = list(deviation_ranges.keys())
    
    # Convert the real-time window to a NumPy array.
    real_arr = real_minus[dev_cols].to_numpy()  # shape: (L, num_params)
    L = real_arr.shape[0]
    if L == 0:
        return []
    
    # Precompute the multipliers for the lower and upper deviation bounds.
    lower_bounds = np.array([deviation_ranges[col]['Lower'] for col in dev_cols])
    upper_bounds = np.array([deviation_ranges[col]['Higher'] for col in dev_cols])
    
    # 4. Convert the historical data and timestamps to NumPy arrays.
    prev_arr = df_prev[dev_cols].to_numpy()  # shape: (N, num_params)
    ts_arr = df_prev[timestamp].to_numpy()  # sorted array of timestamps
    
    # 5. For each fingerprint timestamp, determine its location in the historical data.
    fingerprint_arr = np.array(fingerprint_t_r, dtype='datetime64[ns]')
    # Using searchsorted gives the insertion index; subtract one to get the index of the last value <= fingerprint.
    f_index = np.searchsorted(ts_arr, fingerprint_arr, side='right') - 1

    # Only consider those fingerprint timestamps for which there is enough history (i.e. index >= L-1)
    valid_mask = f_index >= (L - 1)
    valid_f_indices = f_index[valid_mask]
    valid_fingerprint = fingerprint_arr[valid_mask]
    
    if len(valid_f_indices) == 0:
        return []
    
    # 6. Build a rolling window view of the historical data.
    # The idea is to create an array of shape (num_windows, L, num_params) where each window is a slice of length L.
    N = prev_arr.shape[0]
    num_windows = N - L + 1
    if num_windows <= 0:
        return []
    
    # Create the rolling window using np.lib.stride_tricks.
    s0, s1 = prev_arr.strides
    windows = np.lib.stride_tricks.as_strided(prev_arr, 
                   shape=(num_windows, L, prev_arr.shape[1]), 
                   strides=(s0, s0, s1))
    
    # For each valid fingerprint timestamp, the corresponding window is taken from the rolling array.
    # If the fingerprint timestamp is at index f in df_prev, then the window covering [f - L + 1, f] is at index (f - L + 1)
    window_indices = valid_f_indices - L + 1  # indices into the rolling windows array
    selected_windows = windows[window_indices]  # shape: (num_valid, L, num_params)
    
    # 7. For each window, compute the deviation bounds.
    # These are computed elementwise: lower bound = historical value * lower_multiplier, etc.
    window_lower = selected_windows * lower_bounds   # broadcasts over the window shape
    window_upper = selected_windows * upper_bounds
    
    # 8. Compare the real‑time window (real_arr) with each historical window.
    # We expand real_arr to shape (1, L, num_params) and compare to get a boolean array.
    cond = (real_arr[None, :, :] >= window_lower) & (real_arr[None, :, :] <= window_upper)
    # For each window (i.e. for each fingerprint), check if there is any row where all parameters match.
    matches = np.any(np.all(cond, axis=2), axis=1)
    
    # 9. Collect and return the fingerprint timestamps that matched.
    final_similar = [pd.Timestamp(ts) for ts, m in zip(valid_fingerprint, matches) if m]
    return final_similar

# Identifies stable fingerprints based on similarity conditions.
def similarity_plus(similar_minus, df_similar_future, future_time):
    """
    Vectorized implementation to identify stable fingerprints from future data.
    
    Args:
        similar_minus (list): List of fingerprint timestamps (datetime objects) from the previous step.
        df_similar_future (pd.DataFrame): Future data containing a timestamp column.
        future_time (int): Future time window (in minutes) to check for stability.
        parameter_ranges (dict): Dict mapping parameter names to their allowed bounds, e.g.:
            { 'col1': {'Lower': 10, 'Higher': 20}, 'col2': {'Lower': 5, 'Higher': 15}, ... }
        similar_plus_cap (float): Percentage threshold (0-100) for matching data points.
        timestamp_col (str): Name of the timestamp column.
        resample_rule (str): Frequency rule for resampling (default '1T' for one-minute intervals).
    
    Returns:
        list: List of fingerprint timestamps (as pd.Timestamp) that are stable.
    """
    # Copy and resample the future data.
    df_future = df_similar_future.copy()
    df_future[timestamp] = pd.to_datetime(df_future[timestamp])
    df_future = df_future.set_index(timestamp).resample(resample_dataframe).first().reset_index()
    
    # Select only the columns to check.
    param_cols = list(parameter_ranges.keys())
    
    # Convert the future data (for the selected parameters) and timestamps to NumPy arrays.
    future_arr = df_future[param_cols].to_numpy()  # shape: (N, num_params)
    ts_future = df_future[timestamp].to_numpy()  # sorted timestamps
    
    # Determine the expected number of samples in the future window.
    # Since we use <= to include both endpoints, a window of "future_time" minutes yields (future_time + 1) samples.
    window_length = future_time + 1
    N = future_arr.shape[0]
    num_possible_windows = N - window_length + 1
    if num_possible_windows <= 0:
        return []
    
    # Build a rolling window view of future_arr:
    # windows will be a 3D array of shape (num_possible_windows, window_length, num_params)
    s0, s1 = future_arr.strides
    windows = np.lib.stride_tricks.as_strided(
        future_arr,
        shape=(num_possible_windows, window_length, future_arr.shape[1]),
        strides=(s0, s0, s1)
    )
    
    # For each fingerprint timestamp in similar_minus, determine its start index in the future data.
    # We use searchsorted to locate the first index in ts_future that is >= the fingerprint timestamp.
    similar_minus_arr = np.array(similar_minus, dtype='datetime64[ns]')
    start_indices = np.searchsorted(ts_future, similar_minus_arr, side='left')
    
    # Only keep those fingerprints for which a full window exists.
    valid_mask = start_indices <= num_possible_windows - 1
    valid_indices = start_indices[valid_mask]
    valid_fingerprints = similar_minus_arr[valid_mask]
    if len(valid_indices) == 0:
        return []
    
    # Extract the corresponding future windows.
    # Each window has shape (window_length, num_params) and there are len(valid_indices) windows.
    selected_windows = windows[valid_indices]  # shape: (num_valid, window_length, num_params)
    
    # Precompute the scalar bounds for each parameter.
    lower_bounds = np.array([parameter_ranges[col]['Lower'] for col in param_cols])
    upper_bounds = np.array([parameter_ranges[col]['Higher'] for col in param_cols])
    
    # Compare each element in the window against its corresponding lower and upper bound.
    # The lower_bounds and upper_bounds are broadcasted over the window dimensions.
    condition = (selected_windows >= lower_bounds) & (selected_windows <= upper_bounds)
    
    # Compute the percentage of matching data points for each window.
    total_points = selected_windows.shape[1] * selected_windows.shape[2]
    matching_points = np.sum(condition, axis=(1, 2))
    matching_percentage = (matching_points / total_points) * 100
    
    # Identify stable fingerprints where the matching percentage exceeds the threshold.
    stable_mask = matching_percentage > similar_plus_cap
    stable_fingerprints = [pd.Timestamp(ts) for ts, flag in zip(valid_fingerprints, stable_mask) if flag]
    
    return stable_fingerprints

# Filters timestamps based on deviation range conditions.
def min_max_calculation(similar_plus, data_df, deviation_ranges):
    """
    Vectorized filtering of timestamps based on min-max deviation conditions.
    
    Args:
        similar_plus (list): List of timestamps to check (datetime objects).
        data_df (pd.DataFrame): DataFrame containing historical data.
        deviation_ranges (dict): Dictionary with min-max ranges for each parameter.
            Example: { 'col1': {'Min': 10, 'Max': 20}, 'col2': {'Min': 5, 'Max': 15}, ... }
        timestamp_col (str): Name of the timestamp column in data_df.
        
    Returns:
        list: List of timestamps (as pd.Timestamp) that satisfy all deviation conditions.
    """
    try:
        # Ensure the timestamp column is in datetime format.
        data_df[timestamp] = pd.to_datetime(data_df[timestamp])
        
        # Start with a mask that selects only the rows with timestamps in similar_plus.
        mask = data_df[timestamp].isin(similar_plus)
        
        # Combine conditions for each parameter using vectorized filtering.
        for col, bounds in deviation_ranges.items():
            mask &= data_df[col].between(bounds['Min'], bounds['Max'])
        
        # Extract unique timestamps that satisfy all conditions.
        valid_timestamps = data_df.loc[mask, timestamp].unique()
        return list(valid_timestamps)
    
    except KeyError as e:
        print(f"Error: Missing column in data - {e}")
        return []
    except Exception as e:
        print(f"Unexpected error in min_max_calculation: {e}")
        return []

# Extracts and resamples data for a given time range.
def get_complete_data(start, previous_time, future_time, df_all):
    """
    Args:
        start (str/datetime): Start timestamp.
        previous_time (int): Minutes to look back.
        future_time (int): Minutes to look ahead.
        df_all (pd.DataFrame): DataFrame containing all timestamps and data.

    Returns:
        pd.DataFrame: Filtered and resampled DataFrame.
    """
    try:
        end_time1= get_end_datetime()  # The end timestamp.
        end_time= end_time1 - timedelta(minutes= time_var)  # Time difference (in minutes) to adjust for GMT time zone.
        start_time= end_time - timedelta(minutes= 15)

        real_data= read_data(start_time= start_time, end_time= end_time)
        real_data= real_data.set_index(timestamp).resample(resample_dataframe).first().reset_index()  # Set timestamp as index and resample the data
        real_data= real_data.fillna(method= fill_method)

        # Convert start time to datetime
        time_t = pd.to_datetime(start)
        time_t_minus = time_t - timedelta(minutes=previous_time)
        time_t_plus = time_t + timedelta(minutes=future_time)
        df_all[timestamp] = pd.to_datetime(df_all[timestamp])
        df_filtered= df_all[(df_all[timestamp] >= time_t_minus) & (df_all[timestamp] <= time_t_plus)].drop_duplicates(subset= [timestamp])  # Remove duplicate timestamps

        # Resample to 1-minute intervals
        df_filtered= (
            df_filtered
            .set_index(timestamp)  # Set timestamp as index for resampling
            .resample(resample_dataframe)  # Resample to 1-minute intervals
            .first()  # Keep the first value in each interval
        )
        df_filtered.reset_index(inplace=True) # Reset the index to make 'timestamp' a regular column
        now = datetime.now()
        new_time_index = pd.date_range(start=now - timedelta(minutes=15), periods=31, freq='1T')
        df_filtered['timestamp'] = new_time_index[:len(df_filtered)] 

        df_filtered.loc[:14, all_tags] = real_data.loc[:14, all_tags].values

        return df_filtered

    except Exception as e:
        print(f"Error in get_complete_data: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error
    

# Stores real-time and recommended values into a CSV file.
def store_data(df_store, df_real, recommendation_path, future_time):
    """
    Args:
        df_store (pd.DataFrame): DataFrame containing recommended values.
        df_real (pd.DataFrame): DataFrame containing real-time values.
        recommendation_path (str): Path to save the CSV file.
    Returns:
        None
    """
    try:
        # Construct recommendation dictionary
        recommendation = {
            "Time(t)": df_real[timestamp].iat[-1],
            "recommendation_Time(t)": pd.to_datetime(df_real[timestamp].iat[-1]) + timedelta(minutes= future_time),
            "fingerprint_Time(t)": df_store[timestamp].iat[0],
            **{f"{column}_real": df_real[column].iat[0] for column in cvs_1},
            **{f"{column}_recommended": df_store[column].iat[0] for column in cvs_1}
        }
        # Convert to DataFrame
        df_recommended = pd.DataFrame([recommendation])
        # Append data to CSV, handling file write exceptions
        df_recommended.to_csv(recommendation_path, mode="a", header=not os.path.exists(recommendation_path), index=False)
    except Exception as e:
        print(f"Error in store_data: {e}")

# Calculates the total specific heat consumption based on real-time data.
def heat_consumption_real(real_df):
    """
    Args:
        real_df (pd.DataFrame): DataFrame containing real-time data.
    Returns:
        float: Total specific heat consumption.
    """
    try:
        # Extract last values safely
        get_last_value = lambda col: real_df[col].iat[-1] if col in real_df.columns else 0

        # Main Burner Heat Contributions
        mbc_heat = (get_last_value("Coal fuel counter actual [kg]") / 1000) * 33.5 if get_last_value("Coal fuel counter actual [kg]") > 0 else 0
        main_burner_total = mbc_heat

        # Kiln Feed & Clinker Production
        kiln_feed_true = get_last_value("Stone per cycle setpoint [kg]") if get_last_value("Stone per cycle setpoint [kg]") > 0 else 0
        clinker_production = kiln_feed_true * 0.63

        # Specific Heat for Main Burner
        specific_heat_main_burner = main_burner_total / clinker_production if clinker_production > 0 else 0

        # Total Specific Heat Consumption
        total_specific_heat = specific_heat_main_burner

        return total_specific_heat

    except Exception as e:
        print(f"Error in heat_consumption_real: {e}")
        return 0  # Return 0 in case of an error
    
#  Generates a fingerprint response trend by comparing real-time and recommended values,calculating parameter changes, and structuring the output in a JSON-like format.
def create_fingerprint_response(df_real,df_recommend,df_complete,real_heat,t_minus_30,t_pLus_30,min_max_total):
    """
    Args:
        df_real (pd.DataFrame): Real-time data DataFrame.
        df_recommend (pd.DataFrame): Recommended setpoints DataFrame.
        df_complete (pd.DataFrame): Complete historical dataset.
        real_heat (float): Total heat consumption.
        kiln_o2_avg (float): Kiln inlet O2 average.
        alt_fuel (float): Alternative fuel rate percentage.
        cal_fuel (float): Calciner fuel rate percentage.
        t_minus_30 (float): Trend value for -30 minutes.
        t_plus_30 (float): Trend value for +30 minutes.
        min_max_total (float): Min-max total value.
    Returns:
        dict: JSON-like dictionary containing fingerprint response trend data.
    """
    try:
        #for past graph
        end_time_local = df_complete.iloc[14]['timestamp']  # 15th row
        end_time = end_time_local + timedelta(minutes=420)
        start_time = end_time - timedelta(minutes=15)
        influx_df = read_data_store(start_time, end_time)

        #Prediction
        end_time_local = get_end_datetime()
        end_time = end_time_local - timedelta(minutes=time_var)
        start_time = end_time - timedelta(minutes=720)
        Predict_df = read_data(start_time, end_time)
        Predict_df=Predict_df.set_index(timestamp).resample(resample_dataframe).first().reset_index()
        cao_value=predict(Predict_df)

        # Load previous JSON data
        with open(previous_path, "r") as read_file:
            previous_jsons = json.load(read_file)

        # Prepare the future fingerprint data
        df_fingerprint_future = df_complete[list(parameter_ranges.keys()) + [timestamp]]

        # get the middle value
        middle_index = len(df_fingerprint_future) // 2
        # List of target row indices
        target_indices = [middle_index+5, middle_index+10, middle_index+15]
        base_row = df_fingerprint_future.iloc[middle_index]

        all_changes = {}
        all_values = {}  # For storing actual values

        for index in target_indices:
            changes = {}
            values = {}
            target_row = df_fingerprint_future.iloc[index]

            for col in list(parameter_ranges.keys()):
                diff_value = target_row[col] - base_row[col]
                diff_value_rounded = round(diff_value, 2)

                threshold, threshold_type = parameters.get(col, parameters['other'])

                if threshold_type == 'absolute':
                    changes[col] = (
                        'no changes' if abs(diff_value) <= threshold or math.isnan(diff_value) 
                        else f'Increase by {diff_value_rounded}' if diff_value > 0 
                        else f'Decrease by {abs(diff_value_rounded)}'
                    )
                elif threshold_type == 'percentage':
                    change = (diff_value / base_row[col]) * 100 if base_row[col] != 0 else float('nan')
                    changes[col] = (
                        'no changes' if abs(change) < threshold or math.isnan(change) 
                        else f'Increase by {diff_value_rounded}' if diff_value > 0 
                        else f'Decrease by {abs(diff_value_rounded)}'
                    )

                # Store actual value for each col
                values[col] = target_row[col]

            base_row = target_row
            all_changes[index] = pd.DataFrame({'timestamp': [target_row[timestamp]], **changes})
            all_values[index] = pd.DataFrame({'timestamp': [target_row[timestamp]], **values})

        # Convert DataFrames to dictionaries with text changes
        fingerprint_future = [
            {
                "timestamp": df['timestamp'].dt.strftime("%Y-%m-%d %H:%M:%S").tolist(),
                **{col: df[col].tolist() for col in list(parameter_ranges.keys())}
            }
            for df in all_changes.values()
        ]

        # Convert DataFrames to dictionaries with actual values
        fingerprint_future_values = [
            {
                "timestamp": df['timestamp'].dt.strftime("%Y-%m-%d %H:%M:%S").tolist(),
                **{col: df[col].tolist() for col in list(parameter_ranges.keys())}
            }
            for df in all_values.values()
        ]

        fingerprint_response = {
                "actions": {
                    "control": [
                        dict(zip(
                            ('current_setpoint', 'fingerprint_set_point', 'var_name','units','percentage'),
                            (
                                str(df_real[var].iat[-1]),
                                df_recommend[var][0],
                                f'{var}',
                                f'{unit}',
                                (
                                    0 if int(df_recommend[var][0] == 0) or int(df_real[var].iat[-1]) == 0 
                                    else 100 - abs((df_real[var].iat[-1] - df_recommend[var][0]) / df_real[var].iat[-1]) * 100
                                )
                            )
                        ))
                        for var, unit in zip(control_parameters, control_parameters_unit)
                    ],

                    "process": [
                        dict(zip(
                            ('current_setpoint', 'fingerprint_set_point', 'var_name','units','percentage'),
                            (
                                str(df_real[var].iat[-1]),
                                df_recommend[var][0],
                                f'{var}',
                                f'{unit}',
                                (
                                    0 if int(df_recommend[var][0] == 0) or int(df_real[var].iat[-1]) == 0 
                                    else 100 - abs((df_real[var].iat[-1] - df_recommend[var][0]) / df_real[var].iat[-1]) * 100
                                )
                            )
                        ))
                        for var, unit in zip(process_parameters, process_parameters_unit)
                    ],

                    "summary": {
                        "total_heat_consumption": real_heat,
                        **{
                            f"{target} ({unit})": float(df_real[target].iat[-1])
                            for target, unit in zip(targets_new, units_2)
                        }
                    }
                },

                "complete_Data": [
                    {
                        "timestamp": df_complete[timestamp].dt.strftime("%Y-%m-%d %H:%M:%S").to_list(),
                        **{column: df_complete[column].to_list() for column in list(parameter_ranges.keys())}
                    }
                ],

                "past_Data": [
                    {
                        **{column: influx_df[column].to_list() for column in cvs_1}
                    }
                ],

                "fingerprint_future": fingerprint_future_values,
                "actionsCount": len(cvs_1),
                "current_timestamp": str(df_real[timestamp].iat[-1]),
                "fingerprint_timestamp": str(df_recommend[timestamp][0]),
                "fingerprint_Found": "True",
                "t_Minus_30_Total": t_minus_30,
                "t_PLus_30_Total": t_pLus_30,
                "min_Max_total": min_max_total,
                "parameter_changes": previous_jsons,
                "CaO (%)": cao_value,
                "min_max_ranges":min_max_ranges
            }

        return fingerprint_response  
    except Exception as e:
        print(f"Error in create_fingerprint_response: {e}")
        traceback.print_exc()
        return {}


# Generate a fingerprint guidance response with real-time, recommended, and complete dataset information.
def create_fingerprint_guidance_response(df_real, df_recommend, df_complete, real_heat, t_minus_30, t_pLus_30, min_max_total):
    """
    Parameters:
    - df_real (pd.DataFrame): Real-time data with timestamps.
    - df_recommend (pd.DataFrame): Recommended fingerprint data.
    - df_complete (pd.DataFrame): Complete dataset.
    - real_heat (float): Total heat consumption.
    - kiln_o2_avg (float): Average kiln oxygen level.
    - alt_fuel (float): Alternative fuel rate.
    - cal_fuel (float): Calciner fuel rate.
    - t_Minus_30 (list): Data from t-30 minutes.
    - t_PLus_30 (list): Data from t+30 minutes.
    - min_max_total (list): Min-max calculations.

    Returns:
    - dict: Fingerprint guidance response.
    """
    try:
        #for past graph
        end_time_local = df_complete.iloc[14]['timestamp']  # 15th row
        end_time = end_time_local + timedelta(minutes=420)
        start_time = end_time - timedelta(minutes=15)
        influx_df = read_data_store(start_time, end_time)

        #Prediction
        end_time_local = get_end_datetime()
        end_time = end_time_local - timedelta(minutes=time_var)
        start_time = end_time - timedelta(minutes=720)
        Predict_df = read_data(start_time, end_time)
        Predict_df=Predict_df.set_index(timestamp).resample(resample_dataframe).first().reset_index()
        cao_value=predict(Predict_df)

        # Extract the last timestamp and generate 15 future timestamps at 1-minute intervals
        last_timestamp = df_real[timestamp].max()
        future_timestamp = pd.date_range(start=last_timestamp + pd.Timedelta(minutes= 1), periods= 15, freq= resample_dataframe).to_list()
        new_future_timestamp = [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in future_timestamp]

        fingerprint_guidance_response = {
                'actions': [
                    dict(zip(('current_setpoint', 'fingerprint_set_point', 'var_name','units','percentage'), (str(
                        df_real[var].iat[-1]), df_recommend[var][0], f'{var}',f'{unit}', (0 if int(df_recommend[var][0]==0) or int(df_real[var].iat[-1])==0 else 100-abs((df_real[var].iat[-1]-df_recommend[var][0])/df_real[var].iat[-1])*100))))
                    for var, unit in zip(cvs_1, units_1)
                ] + [
                    {
                    "total_heat_consumption": real_heat,
                    **{
                        f'{target} ({unit})': float(df_real[target].iat[-1])
                        for target, unit in zip(targets_new,units_2) 
                    }  
                    } 
                ],
                "complete_Data":[
                    {"timestamp":df_complete[timestamp].dt.strftime("%Y-%m-%d %H:%M:%S").to_list(),
                    **
                    {column :df_complete[column].to_list() for column in list(parameter_ranges.keys())}
                    }
                    ],
                "parallel_Data":[
                    {"1_timestamp":df_real[timestamp].dt.strftime("%Y-%m-%d %H:%M:%S").to_list(),
                    **
                    {column :df_real[column].to_list() for column in list(parameter_ranges.keys())}
                    }
                    ],
                "past_Data": [
                    {
                        **{column: influx_df[column].to_list() for column in cvs_1}
                    }
                ],
                "future_Data":new_future_timestamp,
                "actionsCount": len(cvs_1),
                "current_timestamp": str(df_real[timestamp].iat[-1]),
                "fingerprint_timestamp": str(df_recommend[timestamp][0]),
                "fingerprint_Found":"True",
                "t_Minus_30_Total":t_minus_30,
                "t_PLus_30_Total":t_pLus_30,
                "min_Max_total":min_max_total,
                "CaO (%)":cao_value
            }
        
        return fingerprint_guidance_response
    
    except KeyError as ke:
        print(f"KeyError: Missing column {ke}")
    except IndexError as ie:
        print(f"IndexError: Dataframe index out of bounds {ie}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return {}  # Return an empty dictionary in case of an error

#  Generates a fingerprint response trend by comparing real-time and recommended values,c
def create_fingerprint_response_auto(df_real,df_recommend,df_complete,real_heat,t_minus_30,t_pLus_30,min_max_total):
    """
    Args:
        df_real (pd.DataFrame): Real-time data DataFrame.
        df_recommend (pd.DataFrame): Recommended setpoints DataFrame.
        df_complete (pd.DataFrame): Complete historical dataset.
        real_heat (float): Total heat consumption.
        kiln_o2_avg (float): Kiln inlet O2 average.
        alt_fuel (float): Alternative fuel rate percentage.
        cal_fuel (float): Calciner fuel rate percentage.
        t_minus_30 (float): Trend value for -30 minutes.
        t_plus_30 (float): Trend value for +30 minutes.
        min_max_total (float): Min-max total value.
    Returns:
        dict: JSON-like dictionary containing fingerprint response trend data.
    """
    try:
        #for past graph
        end_time_local = df_complete.iloc[14]['timestamp']  # 15th row
        end_time = end_time_local + timedelta(minutes=420)
        start_time = end_time - timedelta(minutes=15)
        influx_df = read_data_store(start_time, end_time)

        #Prediction
        end_time_local = get_end_datetime()
        end_time = end_time_local - timedelta(minutes=time_var)
        start_time = end_time - timedelta(minutes=720)
        Predict_df = read_data(start_time, end_time)
        Predict_df=Predict_df.set_index(timestamp).resample(resample_dataframe).first().reset_index()
        cao_value=predict(Predict_df)

        # Load previous JSON data
        with open(previous_path,"r") as read:
            previous_jsons=json.load(read)
        # Prepare the future fingerprint data
        # Prepare the future fingerprint data
        df_fingerprint_future = df_complete[list(parameter_ranges.keys()) + [timestamp]]

        # get the middle value
        middle_index = len(df_fingerprint_future) // 2
        # List of target row indices
        target_indices = [middle_index+5, middle_index+10, middle_index+15]
        base_row = df_fingerprint_future.iloc[middle_index]

        all_changes = {}
        all_values = {}  # For storing actual values

        for index in target_indices:
            changes = {}
            values = {}
            target_row = df_fingerprint_future.iloc[index]

            for col in list(parameter_ranges.keys()):
                diff_value = target_row[col] - base_row[col]
                diff_value_rounded = round(diff_value, 2)

                threshold, threshold_type = parameters.get(col, parameters['other'])

                if threshold_type == 'absolute':
                    changes[col] = (
                        'no changes' if abs(diff_value) <= threshold or math.isnan(diff_value) 
                        else f'Increase by {diff_value_rounded}' if diff_value > 0 
                        else f'Decrease by {abs(diff_value_rounded)}'
                    )
                elif threshold_type == 'percentage':
                    change = (diff_value / base_row[col]) * 100 if base_row[col] != 0 else float('nan')
                    changes[col] = (
                        'no changes' if abs(change) < threshold or math.isnan(change) 
                        else f'Increase by {diff_value_rounded}' if diff_value > 0 
                        else f'Decrease by {abs(diff_value_rounded)}'
                    )

                # Store actual value for each col
                values[col] = target_row[col]

            base_row = target_row
            all_changes[index] = pd.DataFrame({'timestamp': [target_row[timestamp]], **changes})
            all_values[index] = pd.DataFrame({'timestamp': [target_row[timestamp]], **values})

        # Convert DataFrames to dictionaries with text changes
        fingerprint_future = [
            {
                "timestamp": df['timestamp'].dt.strftime("%Y-%m-%d %H:%M:%S").tolist(),
                **{col: df[col].tolist() for col in list(parameter_ranges.keys())}
            }
            for df in all_changes.values()
        ]

        # Convert DataFrames to dictionaries with actual values
        fingerprint_future_values = [
            {
                "timestamp": df['timestamp'].dt.strftime("%Y-%m-%d %H:%M:%S").tolist(),
                **{col: df[col].tolist() for col in list(parameter_ranges.keys())}
            }
            for df in all_values.values()
        ]

        fingerprint_response = {
                "actions": {
                    "control": [
                        dict(zip(
                            ('current_setpoint', 'fingerprint_set_point', 'var_name','units','percentage'),
                            (
                                str(df_real[var].iat[-1]),
                                df_recommend[var][0],
                                f'{var}',
                                f'{unit}',
                                (
                                    0 if int(df_recommend[var][0] == 0) or int(df_real[var].iat[-1]) == 0 
                                    else 100 - abs((df_real[var].iat[-1] - df_recommend[var][0]) / df_real[var].iat[-1]) * 100
                                )
                            )
                        ))
                        for var, unit in zip(control_parameters, control_parameters_unit)
                    ],

                    "process": [
                        dict(zip(
                            ('current_setpoint', 'fingerprint_set_point', 'var_name','units','percentage'),
                            (
                                str(df_real[var].iat[-1]),
                                df_recommend[var][0],
                                f'{var}',
                                f'{unit}',
                                (
                                    0 if int(df_recommend[var][0] == 0) or int(df_real[var].iat[-1]) == 0 
                                    else 100 - abs((df_real[var].iat[-1] - df_recommend[var][0]) / df_real[var].iat[-1]) * 100
                                )
                            )
                        ))
                        for var, unit in zip(process_parameters, process_parameters_unit)
                    ],

                    "summary": {
                        "total_heat_consumption": real_heat,
                        **{
                            f"{target} ({unit})": float(df_real[target].iat[-1])
                            for target, unit in zip(targets_new, units_2)
                        }
                    }
                },
                "complete_Data":[
                    {"timestamp":df_complete[timestamp].dt.strftime("%Y-%m-%d %H:%M:%S").to_list(),
                    **
                    {column :df_complete[column].to_list() for column in list(parameter_ranges.keys())}
                    }
                    ],
                "past_Data": [
                    {
                        **{column: influx_df[column].to_list() for column in cvs_1}
                    }
                ],
                "fingerprint_future":fingerprint_future_values,
                "actionsCount": len(cvs_1),
                "current_timestamp": str(df_real[timestamp].iat[-1]),
                "fingerprint_timestamp": str(df_recommend[timestamp][0]),
                "fingerprint_Found":"True",
                "t_Minus_30_Total":t_minus_30,
                "t_PLus_30_Total":t_pLus_30,
                "min_Max_total":min_max_total,
                "parameter_changes": previous_jsons,
                "CaO (%)":cao_value,
                "min_max_ranges":min_max_ranges
            }
        
        return fingerprint_response
    
    except Exception as e:
        print(f"Error in create_fingerprint_response_auto: {e}")
        return {}

#  Generates a fingerprint response trend by comparing real-time and recommended values,calculating parameter changes, and structuring the output in a JSON-like format.
def create_fingerprint_response_trend(df_real, df_recommend, df_complete, real_heat, t_minus_30, t_plus_30, min_max_total, trend_end_time):
    """
    Args:
        df_real (pd.DataFrame): Real-time data DataFrame.
        df_recommend (pd.DataFrame): Recommended setpoints DataFrame.
        df_complete (pd.DataFrame): Complete historical dataset.
        real_heat (float): Total heat consumption.
        kiln_o2_avg (float): Kiln inlet O2 average.
        alt_fuel (float): Alternative fuel rate percentage.
        cal_fuel (float): Calciner fuel rate percentage.
        t_minus_30 (float): Trend value for -30 minutes.
        t_plus_30 (float): Trend value for +30 minutes.
        min_max_total (float): Min-max total value.
        trend_end_time (str): End time of the detected trend.
    Returns:
        dict: JSON-like dictionary containing fingerprint response trend data.
    """
    try:
        # Load previous JSON data
        with open(previous_path, "r") as read_file:
            previous_jsons = json.load(read_file)

        # Prepare the future fingerprint data
        df_fingerprint_future = df_complete[list(parameter_ranges.keys()) + [timestamp]]

        # get the middle value
        middle_index = len(df_fingerprint_future) // 2
        # List of target row indices
        target_indices = [middle_index+5, middle_index+10, middle_index+15]
        base_row = df_fingerprint_future.iloc[middle_index]

        all_changes = {}
        all_values = {}  # For storing actual values

        for index in target_indices:
            changes = {}
            values = {}
            target_row = df_fingerprint_future.iloc[index]

            for col in list(parameter_ranges.keys()):
                diff_value = target_row[col] - base_row[col]
                diff_value_rounded = round(diff_value, 2)

                threshold, threshold_type = parameters.get(col, parameters['other'])

                if threshold_type == 'absolute':
                    changes[col] = (
                        'no changes' if abs(diff_value) <= threshold or math.isnan(diff_value) 
                        else f'Increase by {diff_value_rounded}' if diff_value > 0 
                        else f'Decrease by {abs(diff_value_rounded)}'
                    )
                elif threshold_type == 'percentage':
                    change = (diff_value / base_row[col]) * 100 if base_row[col] != 0 else float('nan')
                    changes[col] = (
                        'no changes' if abs(change) < threshold or math.isnan(change) 
                        else f'Increase by {diff_value_rounded}' if diff_value > 0 
                        else f'Decrease by {abs(diff_value_rounded)}'
                    )

                # Store actual value for each col
                values[col] = target_row[col]

            base_row = target_row
            all_changes[index] = pd.DataFrame({'timestamp': [target_row[timestamp]], **changes})
            all_values[index] = pd.DataFrame({'timestamp': [target_row[timestamp]], **values})

        # Convert DataFrames to dictionaries with text changes
        fingerprint_future = [
            {
                "timestamp": df['timestamp'].dt.strftime("%Y-%m-%d %H:%M:%S").tolist(),
                **{col: df[col].tolist() for col in list(parameter_ranges.keys())}
            }
            for df in all_changes.values()
        ]

        # Convert DataFrames to dictionaries with actual values
        fingerprint_future_values = [
            {
                "timestamp": df['timestamp'].dt.strftime("%Y-%m-%d %H:%M:%S").tolist(),
                **{col: df[col].tolist() for col in list(parameter_ranges.keys())}
            }
            for df in all_values.values()
        ]

        # Define fingerprint_Response dictionary
        fingerprint_response_trend = {
                'actions': [
                    dict(zip(('current_setpoint', 'fingerprint_set_point', 'var_name','units','percentage'), (str(
                        df_real[var].iat[-1]), df_recommend[var][0], f'{var}',f'{unit}', (0 if int(df_recommend[var][0]==0) or int(df_real[var].iat[-1])==0 else 100-abs((df_real[var].iat[-1]-df_recommend[var][0])/df_real[var].iat[-1])*100))))
                    for var, unit in zip(cvs_1, units_1)
                ] + [
                    {
                    "total_heat_consumption": real_heat,
                    **{
                        f'{target} ({unit})': float(df_real[target].iat[-1])
                        for target, unit in zip(targets_new,units_2) 
                    }  
                    } 
                ],
                "complete_Data":[
                    {"1_timestamp":df_complete[timestamp].dt.strftime("%Y-%m-%d %H:%M:%S").to_list(),
                    **
                    {column :df_complete[column].to_list() for column in list(parameter_ranges.keys())}
                    }
                    ],
                "fingerprint_future":fingerprint_future_values,
                "actionsCount": len(cvs_1),
                "current_timestamp": str(df_real[timestamp].iat[-1]),
                "fingerprint_timestamp": str(df_recommend[timestamp][0]),
                "fingerprint_Found":"True",
                "t_Minus_30_Total":t_minus_30,
                "t_PLus_30_Total":t_plus_30,
                "min_Max_total":min_max_total,
                "parameter_changes": previous_jsons,
                "trend_end":trend_end_time,
                "trend_found":"True"
            }
        
        return fingerprint_response_trend
    except Exception as e:
        print(f"Error in create_fingerprint_response_trend: {e}")
        return {}
    
def create_response(df_4_months, real_data, df_complete, min_max_r, previous_time, future_time, guidance_value,reccommendation_path, t_minus_30, t_plus_30, min_max_total):
    """
    Args:
    df_4_months (DataFrame): DataFrame containing historical subset of data.
    real_data (DataFrame): Real-time data to compare against historical data.
    df_complete (DataFrame): Complete dataset including timestamps before and after the given time.
    min_max_r (list): List of timestamps representing min-max points.
    previous_time (str): Start time of the time window.
    future_time (str): End time of the time window.
    kiln_o2_avg (list): Average kiln O2 values.
    reccommendation_path (str): Path to store recommendations.
    t_minus_30 (str): Timestamp 30 minutes before the current time.
    t_plus_30 (str): Timestamp 30 minutes after the current time.
    min_max_total (list): Complete min-max timestamp list.
    trends_list (list of tuples): List containing (start_time, end_time) pairs.
    Returns:
    list: A list of fingerprint responses based on the trends.
    """
    try:
        end_time=get_end_datetime()-timedelta(minutes=time_var)
        start_time_new=end_time-timedelta(minutes=data_24hrs)
        real_data_guidance = read_data(start_time=start_time_new, end_time=end_time)
        real_data_guidance = real_data_guidance.set_index(timestamp).resample(resample_dataframe).first().reset_index()

        fingerprint_response_list=[]

        # Convert timestamps to string format for consistency
        min_max_time=[str(time) for time in min_max_r]
        df_min_max=df_4_months[df_4_months[timestamp].isin(min_max_time)]
        
        if(df_min_max.shape[0]==0):
            df_min_max=real_data_guidance[real_data_guidance[timestamp].isin(min_max_time)]
        
        min_max_r=df_min_max[timestamp].to_list()
        # Selecting specific timestamps from min_max_r
        if len(min_max_r)>5:
            min_max= min_max_r[:2]+ min_max_r[len(min_max_r)//2:(len(min_max_r)//2)+1] + min_max_r[-1:]
        else:
            min_max=min_max_r

        # Iterate through selected timestamps
        try:
            for i in min_max:
                # Filter data for the selected timestamp
                data=df_4_months[df_4_months[timestamp]==str(i)].reset_index(drop=True)
    
                if data.shape[0]==0:
                    data=real_data_guidance[real_data_guidance[timestamp]==str(i)]
                data.reset_index(inplace=True,drop=True)
                # Store the processed data
                store_data(df_store= data, df_real= real_data, recommendation_path= reccommendation_path, future_time= future_time)

                # Retrieve complete dataset with previous and future time
                complete_data= get_complete_data(start= str(i),previous_time= previous_time,future_time= future_time,df_all=df_complete)
                if complete_data.shape[0]==0:
                    complete_data=get_complete_data(start=str(i),previous_time=previous_time,future_time=future_time,df_all=real_data_guidance)

                complete_data=complete_data.fillna(method='bfill')
                # calculating heat consumption  real
                real_heat=heat_consumption_real(real_df=real_data)
                #create fingerprint response

                if guidance_value:
                    fingerprint_response_final=create_fingerprint_guidance_response(df_real=real_data,df_recommend=data,df_complete=complete_data,real_heat=real_heat,
                                                                                    t_minus_30=t_minus_30,
                                                                                    t_pLus_30=t_plus_30,min_max_total=min_max_total)
                else:
                    fingerprint_response_final=create_fingerprint_response(df_real=real_data,df_recommend=data,df_complete=complete_data,real_heat=real_heat,
                                                                           t_minus_30=t_minus_30,
                                                                           t_pLus_30=t_plus_30,min_max_total=min_max_total)
                fingerprint_response_list.append(fingerprint_response_final)
        
        except Exception as e:
            print(f"Error processing timestamp {i}: {e}")
    
    except Exception as e:
        print(f"Critical error in create_response: {e}")
    
    return fingerprint_response_list
    

def create_response_auto(df_4_months, real_data, df_complete, min_max_r, previous_time, future_time, recommendation_path, t_minus_30, t_plus_30, min_max_total):
    try:
        
        end_time=get_end_datetime()-timedelta(minutes=time_var)
        start_time_new=end_time-timedelta(minutes=data_24hrs)
        real_data_guidance1 = read_data(start_time=start_time_new, end_time=end_time)
        real_data_guidance1= real_data_guidance1.set_index(timestamp).resample(resample_dataframe).first().reset_index()

        fingerprint_response_list_auto = [] # Initialize the response list
        min_max_time = [str(time) for time in min_max_r] # Convert timestamps to strings

        df_min_max = df_4_months[df_4_months[timestamp].isin(min_max_time)] # Filter df for min/max times

        min_max_r=df_min_max[timestamp].to_list()
        # Create min_max_auto based on a specific pattern of timestamps
        min_max_auto = min_max_r[:2] + min_max_r[len(min_max_r)//2:(len(min_max_r)//2)+1] + min_max_r[-1:] if len(min_max_r) > 5 else min_max_r
        # Store the last timestamp for future processing
        min_max_auto_str=[min_max_auto[-1]]
        similar_plus_fingerprint_str=[timestamp_new.isoformat() if isinstance(timestamp_new,datetime) else datetime.strptime(timestamp_new,"%Y-%m-%d %H:%M:%S").isoformat() for timestamp_new in min_max_auto_str]
        
        # Write the new min/max timestamp to min_max.json
        with open(min_max_path, "w") as data:
            json.dump(similar_plus_fingerprint_str, data)
        
        # Extract data for the last timestamp from min_max_auto and modify the 'coalMainBurner' value if necessary
        data = df_4_months[df_4_months[timestamp] == str(min_max_auto[-1])].reset_index(drop=True)
        if data.shape[0]==0:
                    data=real_data_guidance1[real_data_guidance1[timestamp]==str(i)]
        
        # Store data in the provided path
        store_data(df_store=data, df_real=real_data, recommendation_path=recommendation_path,future_time=future_time)
        # Retrieve the complete data for further analysis
        complete_data = get_complete_data(start=str(min_max_auto[-1]), previous_time=previous_time, future_time=future_time, df_all=df_complete)
        if complete_data.shape[0]==0:
                    complete_data=get_complete_data(start=str(i),previous_time=previous_time,future_time=future_time,df_all=real_data_guidance1)

        complete_data=complete_data.fillna(method='bfill')
        real_heat = heat_consumption_real(real_df=real_data)

        # Only proceed if there's sufficient complete data
        if complete_data.shape[0] >= 30:
            fingerprint_Response_au = create_fingerprint_response(df_real=real_data, df_recommend=data, df_complete=complete_data, real_heat=real_heat, 
                                                                 t_minus_30=t_minus_30, 
                                                                t_pLus_30=t_plus_30, min_max_total=min_max_total)
            with open(previous_Fingerprint_path, "w") as k_data:
                    fingerprint_Response_Auto = {"data": [fingerprint_Response_au]}
                    json.dump(fingerprint_Response_Auto, k_data, indent=4)
        
        # Initialize min_max_count and process remaining timestamps in min_Max
        min_max_count=0
        
        min_Max=[min_max_auto[-1]]  # Ensure min_Max is not empty
        for i in min_Max:
            data = df_4_months[df_4_months[timestamp] == str(i)].reset_index(drop=True)
            if data.shape[0]==0:
                    data=real_data_guidance1[real_data_guidance1[timestamp]==str(i)]
            # Store the updated data
            store_data(df_store=data, df_real=real_data, recommendation_path=recommendation_path,future_time=future_time)
            
            # Retrieve and process complete data for the timestamp
            complete_data = get_complete_data(start=str(i), previous_time=previous_time, future_time=future_time, df_all=df_complete)
            if complete_data.shape[0]==0:
                    complete_data=get_complete_data(start=str(i),previous_time=previous_time,future_time=future_time,df_all=real_data_guidance1)
                    
            complete_data=complete_data.fillna(method='bfill')
            
            if complete_data.shape[0]>=30:
                min_max_count=min_max_count+1
                real_heat = heat_consumption_real(real_df=real_data)
                fingerprint_response_auto = create_fingerprint_response_auto(df_real=real_data, df_recommend=data, df_complete=complete_data, real_heat=real_heat, 
                                                                             t_minus_30=t_minus_30, 
                                                                            t_pLus_30=t_plus_30, min_max_total=min_max_total)
                fingerprint_response_list_auto.append(fingerprint_response_auto)
            min_max_count=min_max_count+1
        return fingerprint_response_list_auto, min_max_count
    
    except Exception as e:
        # Handle any exceptions that might occur during execution
        print(f"Error in create_response_auto: {e}")
        return [], 0  # Return empty results in case of an error
    
# Generates a fingerprint response list based on identified trends in the data.
def create_response_trend(df_4_months, real_data, df_complete, min_max_r, previous_time, future_time,reccommendation_path, t_minus_30, t_plus_30, min_max_total, trends_list):
    """
    Args:
    df_4_months (DataFrame): DataFrame containing historical subset of data.
    real_data (DataFrame): Real-time data to compare against historical data.
    df_complete (DataFrame): Complete dataset including timestamps before and after the given time.
    min_max_r (list): List of timestamps representing min-max points.
    previous_time (str): Start time of the time window.
    future_time (str): End time of the time window.
    kiln_o2_avg (list): Average kiln O2 values.
    reccommendation_path (str): Path to store recommendations.
    t_minus_30 (str): Timestamp 30 minutes before the current time.
    t_plus_30 (str): Timestamp 30 minutes after the current time.
    min_max_total (list): Complete min-max timestamp list.
    trends_list (list of tuples): List containing (start_time, end_time) pairs.
    Returns:
    list: A list of fingerprint responses based on the trends.
    """
    fingerprint_response_list = []
    try:
        # Convert timestamps to string format for consistency
        min_max_time = [str(time) for time in min_max_r]
        df_min_max = df_4_months[df_4_months[timestamp].isin(min_max_time)].copy()
        min_max_r = df_min_max[timestamp].tolist()
        
        # Selecting specific timestamps from min_max_r
        if len(min_max_r) > 5:
            min_max = min_max_r[:2] + min_max_r[len(min_max_r)//2:(len(min_max_r)//2) + 1] + min_max_r[-1:]
        else:
            min_max = min_max_r
        
        # Iterate through selected timestamps
        for i in min_max:
            try:
                # Extract end_time from trends_list
                end_time = next((end_time for start_time, end_time in trends_list if str(start_time) == str(i)), None)
                
                # Filter data for the selected timestamp
                data = df_4_months[df_4_months[timestamp] == str(i)].reset_index(drop=True)
                
                # Store the processed data
                store_data(df_store= data, df_real= real_data, recommendation_path= reccommendation_path, future_time= future_time)
            
                # Retrieve complete dataset with previous and future time
                complete_data = get_complete_data(start= str(i), previous_time= previous_time, future_time= future_time, df_all= df_complete)
                complete_data.fillna(method='bfill', inplace=True)
                
                # Compute heat consumption and fuel ratios
                real_heat = heat_consumption_real(real_df= real_data)
                
                # Generate fingerprint response 
                fingerprint_response_final = create_fingerprint_response_trend(df_real= real_data, df_recommend= data, df_complete= complete_data, 
                                                                                real_heat= real_heat,
                                                                                t_minus_30= t_minus_30, t_plus_30= t_plus_30, 
                                                                                min_max_total= min_max_total, trend_end_time= str(end_time))
                fingerprint_response_list.append(fingerprint_response_final)
            except Exception as e:
                print(f"Error processing timestamp {i}: {e}")
                continue
    except Exception as e:
        print(f"Critical error in create_response_trend: {e}")
    
    return fingerprint_response_list

# Identifies past similar trends based on deviation calculations and similarity analysis.
def find_past(real_data_new, deviation_ranges, previous_time, future_time, end_time_new):
    """
    Parameters:
    real_data_new (DataFrame): Real-time data.
    deviation_ranges (dict): Deviation thresholds for key parameters.
    previous_time (str): Start time for comparison.
    future_time (str): End time for comparison.
    end_time_new (str): The reference end timestamp.
    Returns:
    list: A list of min-max timestamps representing similar past trends.
    """
    try:
        # Trim the dataset by removing the first and last 60 rows
        real_data1 = real_data_new[60:-60].copy()
                        
        # Identify deviations from recent data
        df_deviation, count_n = deviation_past(df_dev_filtered= real_data1, real_data= real_data_new, deviation_ranges= deviation_ranges)
        
        if df_deviation.empty:
            print("Warning: No deviations found in past data.")
            return []
        
        # Convert timestamps to datetime format
        fingerprint_t = pd.to_datetime(df_deviation[timestamp].values.tolist())
        
        # Identify similar past timestamps based on deviation ranges
        similar_minus = similarity_minus(real_t= end_time_new, fingerprint_t_r= fingerprint_t, real_data= real_data_new, df_similar_previous= real_data1, 
                                         previous_time= previous_time, deviation_ranges= deviation_ranges)
        # Find forward similarities for the identified timestamps
        similar_plus = similarity_plus(similar_minus= similar_minus, df_similar_future= real_data1, future_time= future_time)
        
        # Check for min-max trends only if sufficient deviations are found
        if count_n >= 3:
            min_max = min_max_calculation(similar_plus, real_data1, deviation_ranges)
        else:
            min_max = []
        
        # Ensure a valid number of min-max points
        if count_n < 3 or len(min_max) < 2:
            min_max = []
        
        return min_max
    
    except Exception as e:
        print(f"Error in find_past: {e}")
        return []


def update_socket_after_delay():
    time.sleep(3)  # Wait for 3 seconds
    with open('../../../files/json/socket.json', 'r+') as f:
        socket = json.load(f)
        socket["socket_stop"] = False
        f.seek(0)
        json.dump(socket, f, indent=4)
        f.truncate()

def get_excel_workbook(workbook_name, max_attempts=10, delay=2):
    attempts = 0
    while attempts < max_attempts:
        try:
            #pythoncom.CoInitialize()  # Initialize COM
            #excel = win32com.client.GetObject(None, "Excel.Application")
            time.sleep(delay)  # Allow time for Excel to initialize

            # Try to find the specified workbook
            # for wb in excel.Workbooks:
            #     if wb.Name == workbook_name:
            #         return wb  # Return the workbook if found
            
            print("Workbook not found, retrying...")
            attempts += 1
            time.sleep(delay)  # Wait before retrying
        except Exception as e:
            print(f"Error occurred: {e}, retrying...")
            attempts += 1
            time.sleep(delay)  # Wait before retrying
        finally:
            #pythoncom.CoUninitialize()
            # Uninitialize COM for this attempt
            print("uninitialize.")  

    print("Max attempts reached. Unable to access the workbook.")
    return None

def load_json_with_retries(path, retries=2, delay=1):
    for attempt in range(retries):
        try:
            with open(path, "r") as file:
                return json.load(file)
        except (JSONDecodeError, FileNotFoundError):
            time.sleep(delay)
    raise Exception(f"Failed to load JSON from {path} after {retries} retries.")

def predict(df):
    """
    Given a DataFrame with 12 hours of 1-minute resampled sensor data,
    return the predicted value of parameter_x using a trained model.
    """

    # Check data sufficiency
    if df.shape[0] < 30:
        raise ValueError("Insufficient data: at least 30 rows expected after resampling.")

    # Feature engineering
    feature_row = []
    for col in target_vars:
        y = df[col].values
        x = np.arange(len(y))

        feature_row += [
            np.mean(y), np.std(y), np.min(y), np.max(y)
        ]

        try:
            slope = np.polyfit(x, y, 1)[0]
        except:
            slope = 0
        feature_row.append(slope)

        feature_row.append(y[0])
        feature_row.append(y[-1])
        feature_row.append(y[-1] - y[0])

    # Prepare for prediction
    X_real = np.array([feature_row])

    # Load model
    model = current_app.config['model']

    # Predict and return
    prediction = model.predict(X_real)[0]
    return prediction
