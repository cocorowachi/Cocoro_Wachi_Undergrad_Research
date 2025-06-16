import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
import math

from datetime import datetime, date, timedelta
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_datetime64_any_dtype as is_datetime

def insert_mirrored_rows(df, num_rows=30):
    """
    Insert chronologically mirrored data point at head and tail of df
    """
    df = df.copy()
    mirrored_rows_head = df.iloc[:num_rows].copy()
    mirrored_rows_head = mirrored_rows_head.iloc[::-1].reset_index(drop=True)

    mirrored_rows_tail = df.iloc[-num_rows:].copy()
    mirrored_rows_tail = mirrored_rows_tail.iloc[::-1].reset_index(drop=True)

    df_extended = pd.concat([mirrored_rows_head, df, mirrored_rows_tail], ignore_index=True)
    
    return df_extended

def moving_avg(df, length=24):
    """ 
    Finds moving average for past 24 hour
    """
    df = insert_mirrored_rows(df.copy())
    # columns_to_edit = [col for col in df.columns if col != 'Date Time']
    # out_df = df.copy()
    # for col in columns_to_edit:
    #     out_df[col] = df[col].rolling(window=length).mean()
    out_df = df.set_index('Date Time').rolling(window=length).mean().reset_index()
    return out_df.iloc[30:-30].reset_index(drop=True)
    
def moving_avg_mid(df, lengths=[3]):
    """ 
    Finds centered moving averages for given window lengths.
    Overwrites the same columns on each iteration.
    """
    df = insert_mirrored_rows(df.copy())
    columns_to_edit = [col for col in df.columns if col != 'Date Time']
    numeric_df = df[columns_to_edit]
    for length in lengths:
        numeric_df[columns_to_edit] = numeric_df[columns_to_edit].rolling(window=length, center=True).mean()

    return df.iloc[30:-30].reset_index(drop=True)

def get_event(flow_df, precip_df):
    precip_df = precip_df.copy()
    flow_df = flow_df.copy()
    
    precip_df = moving_avg_mid(precip_df, [24])
    flow_df = moving_avg_mid(flow_df, [24])
    
    precip_df.set_index("Date Time", inplace=True)
    flow_df.set_index("Date Time", inplace=True)

    flow_col_name = flow_df.columns[0]
    precip_col_name = precip_df.columns[0]
    
    events = []
    in_event = False
    original_flow = flow_df[flow_col_name].min() * 15
    event_start = None
    
    for date in precip_df.index:
        if not in_event and precip_df.loc[date, precip_col_name] > 0.0:
            event_start = date
            in_event = True  
            
        if in_event:
            flow_value = flow_df.loc[date, flow_col_name] 
            
            if pd.isna(flow_value) or (flow_value <= original_flow):
                event_end = date
                events.append((event_start, event_end))
                in_event = False  
    
    merged_events = []
    for i in range(len(events)):
        if i == 0:
            merged_events.append(events[i])
        else:
            prev_start, prev_end = merged_events[-1]
            curr_start, curr_end = events[i]
    
            if (curr_start - prev_end) <= timedelta(hours=4):
                merged_events[-1] = (prev_start, curr_end)
            else:
                merged_events.append(events[i])

    filtered_events = []
    for start, end in merged_events:
        if (end - start) >= timedelta(hours=18):
            max_flow = flow_df.loc[start:end, flow_col_name].max()
            precip_event_df = precip_df.loc[start:end, precip_col_name]
            avg_precip = precip_event_df.sum()/precip_event_df.count()
            
            if max_flow > 0.7 and avg_precip > 0.002 and not precip_event_df.isna().any():
                filtered_events.append((start, end - pd.Timedelta(hours=1)))

    return filtered_events


def top_n_events(flow_df, events, n):    
    # Ensure the 'date' column is in datetime format
    
    # Calculate max value in each range
    range_max_values = []
    for start, end in events:
        mask = (flow_df["Date Time"] >= start) & (flow_df["Date Time"] <= end)
        max_val = flow_df.loc[mask].iloc[:, 1].max()
        range_max_values.append(((start, end), max_val))
    
    # Sort by max value descending and get top 10
    top_n_ranges_only = [r[0] for r in sorted(range_max_values, key=lambda x: x[1], reverse=True)[:n]]
    return top_n_ranges_only