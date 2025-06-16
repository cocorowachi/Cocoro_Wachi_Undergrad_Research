import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time

from datetime import datetime, date, timedelta
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_datetime64_any_dtype as is_datetime

def read_coord(df):
    stations = {}
    for row in df.itertuples():
        stations[row.name] = (row.longitude, row.latitude)
    return stations
    

def euclidean_distance(lat1, lon1, lat2, lon2):
    return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)
    
# Convert dictionary values (coordinates) into NumPy arrays for calculations
# precipitation_points = np.array(list(precipitation_stations.values()))
# flow_stations = np.array(list(flow_stations.values()))

def weighted_average(precip_dict, flow_dict, n_nearest=3):
    weighted_averages = {}

    for flow_name, flow in flow_dict.items():
        distances = np.array([
            euclidean_distance(flow[0], flow[1], precip[0], precip[1]) 
            for precip in precip_dict.values()
        ])
        
        # Prevent division by zero by adding a small threshold
        distances = np.where(distances == 0, 1e-6, distances) 
        
        # Get the indices of the 3 nearest precipitation points
        nearest_indices = np.argsort(distances)[:n_nearest]
        
        # Compute weights (inverse of distance)
        weights = 1 / distances[nearest_indices]
        
        # Normalize weights so they sum to 1
        weights /= np.sum(weights)
        
        # Create weight dictionary with precipitation station names
        weight_dict = {name: 0 for name in precip_dict.keys()}  
        for idx, weight in zip(nearest_indices, weights):
            precip_name = list(precip_dict.keys())[idx]
            weight_dict[precip_name] = weight 
        
        # Store weights for the sewerage station
        weighted_averages[flow_name] = weight_dict

    return pd.DataFrame(weighted_averages)

def calculate(precipitation_stations, flow_stations):
    weights_dict = weighted_average(precipitation_stations, flow_stations)

    df_weights = pd.DataFrame(weights_dict)

    precipitation_points = np.array(list(precipitation_stations.values()))
    flow_stations = np.array(list(flow_stations.values()))
    plt.figure(figsize=(8, 6))
    plt.scatter(precipitation_points[:, 0], precipitation_points[:, 1], c='blue', label='Precipitation Points')
    plt.scatter(flow_stations[:, 0], flow_stations[:, 1], c='red', label='Sewerage Points')
    
    for i, flow in enumerate(flow_stations):
        for j in np.argsort([euclidean_distance(flow[0], flow[1], precip[0], precip[1]) for precip in precipitation_points])[:3]:
            plt.plot([flow[0], precipitation_points[j, 0]], [flow[1], precipitation_points[j, 1]], 'k--', alpha=0.5)
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.grid(True)
    plt.show()
    # print(df_weights)
    # print(df_weights.T)
    return df_weights

def calculate_weighted_precip(precip_df, weights_df):
    weights_T_df = weights_df.T
    weighted_precip_dict = {'Date Time': precip_df['Date Time']}
    cols = [col for col in precip_df.columns if col != 'Date Time']
    precip_num_df = precip_df[cols]
    precip_num_df.columns = precip_num_df.columns.str.split(' ').str[0]

    for stream_col_name, weights in weights_df.items():
        weighted_avg = (precip_num_df * weights).sum(axis=1) / sum(weights)
        weighted_precip_dict[stream_col_name] = weighted_avg

    weighted_precip_df = pd.DataFrame(weighted_precip_dict)
    return weighted_precip_df