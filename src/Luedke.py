import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time

from scipy.signal import find_peaks
from scipy.stats import pearsonr
from datetime import datetime, date, timedelta
from scipy.optimize import differential_evolution
from sklearn.metrics import r2_score, mean_squared_error
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_datetime64_any_dtype as is_datetime

def trim(df, start, end):
    return df[(df['Date Time'] >= start) & (df['Date Time'] <= end)].reset_index(drop = True)
    

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
    df = insert_mirrored_rows(df.copy()).drop(columns=['Date Time'], inplace=False)
    out_df = df.rolling(window=length).mean()
    return out_df.iloc[30:-30]

def moving_avg_mid(df, lengths=[3]):
    """ 
    Finds centered moving averages for given window lengths.
    Overwrites the same columns on each iteration.
    """
    df = insert_mirrored_rows(df.copy())
    for length in lengths:
        df = df.rolling(window=length, center=True).mean()

    return df.iloc[30:-30].reset_index(drop=True)

def normalize(df):
    df_normalized = df.copy()
    df_normalized = (df - df.min()) / (df.max() - df.min())
    return df_normalized
    
def luedke_2_level(streamflow_df, precip_df, catch_area, RD, RW_max, HHL, precip_avg_time_step=1, stream_flow_avg_time_step=1):
    streamflow_df = streamflow_df.copy()
    precip_df = precip_df.copy()
    
    # Level 2
    MASF_t_df = moving_avg(streamflow_df, stream_flow_avg_time_step)  # (13)
    MASF_scaled_df = normalize(MASF_t_df)  # (12)
    RW_t_df = RW_max * MASF_scaled_df  # (11)
    
    # Level 1
    MAP_t = moving_avg(precip_df, precip_avg_time_step)  # (3)
    shape_factor = 0.5 ** (1 / HHL)  # (2)
    RW_nd = RW_t_df.to_numpy().reshape(-1, 1)
    Q_t_nd = np.copy(RW_nd)
    MAP_nd = MAP_t.to_numpy().reshape(-1, 1)
    
    for row_i in range(1, len(RW_nd)): # (1)
        Q_t_nd[row_i] = catch_area * (
            RD + ((RW_nd[row_i] + RW_nd[row_i - 1]) / 2)
        ) * MAP_nd[row_i] * (1 - shape_factor) + (shape_factor * Q_t_nd[row_i - 1])
    
    return np.squeeze(pd.DataFrame(Q_t_nd))

def luedke_3_level(streamflow_df, precip_df, catch_area, RD, HHL, AMHL, cold_shcf, hot_shcf, precip_avg_time_step=1):
    streamflow_df = streamflow_df.copy()
    precip_df = precip_df.copy()
    
    # Level 3
    L = 1.2 * (cold_shcf - hot_shcf)  # (7)
    k = 4.7964 / (30 - 70)  # (8)
    x_0 = (70 + 30) / 2  # (9)
    MASF_t_df = moving_avg(streamflow_df, precip_avg_time_step)  # (13)
    MASF_scaled_df = 70 - (normalize(MASF_t_df) * 40)  # (15)
    SHCF_t = (L / (1 + np.exp(-k * (MASF_scaled_df - x_0)))) + cold_shcf - (11/12) * L  # (14)
    
    # Level 2
    MAP_t = moving_avg(precip_df, precip_avg_time_step)  # (3)
    AMRF = 0.5 ** (1 / AMHL)  # (5)
    SHCF_nd = SHCF_t.to_numpy().reshape(-1, 1)
    RW_nd = np.copy(SHCF_nd)
    MAP_nd = MAP_t.to_numpy().reshape(-1, 1)
    
    for row_i in range(1, len(RW_nd)):
        RW_nd[row_i] = (AMRF - 1) / np.log(AMRF) * SHCF_nd[row_i] * MAP_nd[row_i] + (AMRF * RW_nd[row_i - 1])  # (4)
        if np.log(AMRF) == 0:
            print(np.log(AMRF))
    # Level 1
    shape_factor = 0.5 ** (1 / HHL)  # (2)
    Q_t_nd = np.copy(RW_nd)
    
    for row_i in range(1, len(Q_t_nd)): # (1)
        Q_t_nd[row_i] = catch_area * (
            RD + ((RW_nd[row_i] + RW_nd[row_i - 1]) / 2)
        ) * MAP_nd[row_i] * (1 - shape_factor) + (shape_factor * Q_t_nd[row_i - 1])
    
    return np.squeeze(pd.DataFrame(Q_t_nd))

def trim(df, event):
    return df[(df['Date Time'] >= event[0]) & (df['Date Time'] <= event[1] - pd.Timedelta(hours=1))].reset_index(drop = True)
    
def optimize_l3(events, flow_meter_df, precip_meter_df, catchment_area, time_step_hours):
    g_event = 0
    flow_event_df = 0
    observed_volume = 0
    observed_peak_flow = 0 
    stdev_obs = 0
    date_time_df = 0


    def objective(params):
        # print(g_event)
        precip_event_df = trim(precip_meter_df, g_event).iloc[:, 1]
        # unpack the 6 parameters
        p1, p2, p3, p4, p5 = params
        # run your predictive model
        predictions = luedke_3_level(flow_event_df, precip_event_df, catchment_area, p1, p2, p3, p4, p5, time_step_hours)

        r2 = r2_score(flow_event_df, predictions)
        r2 = (-r2 + 1)/2 # 1
            
        simulated_volume = np.sum(predictions) * time_step_hours
                
        volume_error_percent = abs((simulated_volume - observed_volume) / observed_volume) # 2
        
        simulated_peak_flow = predictions.max()
        peak_flow_perc = abs((simulated_peak_flow - observed_peak_flow) / observed_peak_flow) # 3
            
        r, p_value = pearsonr(flow_event_df, predictions) # 4
    
        stdev_sim = predictions.std()
    
        C_b = (2*stdev_obs*stdev_sim)/((stdev_obs**2)+(stdev_sim**2)+(stdev_obs-stdev_sim)**2) # 5
        #############################
        flow_event_df.index = date_time_df['Date Time'].values
        predictions.index = date_time_df['Date Time'].values

        total_flow = flow_event_df.sum()
        avg_flow = total_flow/precip_event_df.count()*1.25
        peaks, properties = find_peaks(moving_avg_mid(flow_event_df, [5,5]), height=avg_flow)
        peak_times = flow_event_df.index[peaks]
        event_windows = [(peak - pd.Timedelta(hours=18), peak + pd.Timedelta(hours=24)) for peak in peak_times]
        
        mask = pd.Series(False, index=flow_event_df.index)
        for start, end in event_windows:
            mask |= flow_event_df.index.to_series().between(start, end)

        fil_flow_event_df = flow_event_df[mask]
        fil_predictions = predictions[mask]

        fil_r, p_value = pearsonr(fil_flow_event_df, fil_predictions) # 4
    
        fil_stdev_sim = fil_predictions.std()
        fil_stdev_obs = fil_flow_event_df.std()
    
        fil_C_b = (2*fil_stdev_obs*fil_stdev_sim)/((fil_stdev_obs**2)+(fil_stdev_sim**2)+(fil_stdev_obs-fil_stdev_sim)**2) # 5

        m = (1-r*C_b) + (1-fil_r*fil_C_b)
        # print(m)
        # return m
        mod = (m)+(volume_error_percent+peak_flow_perc)
        return mod

    bounds = [(0, 1), (10, 40), (20, 75), (0, 1), (0, 1)]

    optimized_params = []

    for event in events:
        print(event)
        g_event = event
        flow_event_df = trim(flow_meter_df, g_event).iloc[:, 1]
        observed_volume = np.sum(flow_event_df) * time_step_hours
        observed_peak_flow = flow_event_df.max()
        stdev_obs = flow_event_df.std()
        date_time_df = pd.DataFrame({'Date Time': pd.date_range(start=event[0], end=event[1] - pd.Timedelta(hours=1), freq='h')})
        result = differential_evolution(objective, bounds)
        # Best parameters
        best_params = result.x
        
        # Best meteric
        best_meter = result.fun

        optimized_params.append(best_params)
    average_param = np.mean(optimized_params, axis=0)
    return optimized_params
    return tuple(average_param)