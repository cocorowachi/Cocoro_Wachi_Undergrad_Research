import pandas as pd
import numpy as np
import matplotlib.pyplot as pl


from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta, SU
from functools import reduce
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_datetime64_any_dtype as is_datetime

def set_na(edit_df, with_na_df):
    """
    re-sets any nan values from
    """
    edit_df = edit_df.copy()
    with_na_df = with_na_df.copy()
    edit_df[with_na_df.isna()] = np.nan
    return edit_df

def get_CDT_days(start_year, end_year):
    """
    Returns a list of of dates which CST->CDT happen, which is the second Sunday of March
    """
    dates = []
    for year in range(start_year, end_year + 1):
        # Get the first day of March
        first_march = date(year, 3, 1)
        # Find the second Sunday
        second_sunday = first_march + relativedelta(weekday=SU(+2))
        datetime_sunday = datetime(year, 3, second_sunday.day, 3, 0, 0)
        dates.append(datetime_sunday)
    return dates

def to_hourly_cum(df):
    df = df.copy()
    columns_to_subtract = [col for col in df.columns if col != 'Date Time']
    copy_df = df[columns_to_subtract]
    copy_df = copy_df.diff()
    copy_df.loc[df['Date Time'].dt.hour == 0, columns_to_subtract] = df.loc[df['Date Time'].dt.hour == 0, columns_to_subtract]
    out_df = pd.concat([df['Date Time'], copy_df], axis=1,ignore_index=True)
    out_df.columns = df.columns
    return out_df

def fill_time(df):
    df = df.copy()

    # Set date_time as the index
    df.set_index('Date Time', inplace=True)
    
    # Create a complete datetime range
    start = df.index.min()
    end = df.index.max()
    full_range = pd.date_range(start=start, end=end, freq='H')  # Hourly frequency
    
    # Reindex the DataFrame with the full range, filling missing values with NaN
    df = df.reindex(full_range)
    
    # Reset the index to make date_time a column again if needed
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Date Time'}, inplace=True)
    return df

def separate_time(df):
    """
    Extract only datapoint at each hour marks.
    """
    df = df.copy()
    df['Date Time'] = pd.to_datetime(df['Date Time'])
    df = df[df['Date Time'].dt.minute == 00]
    df = df.reset_index(drop=True)
    return df

def clean_USGS_df(df):
    """
    Unifies data from USGS.
    Adjusts for transition from CST->CDT by adding a 2am with the same values as 1am.
    """
    out_df = df.copy()
    
    # Preprocessing
    out_df.replace('Ice', 0, inplace=True)
    out_df['Stream Flow'] = pd.to_numeric(out_df['Stream Flow'], errors='coerce')
    out_df = separate_time(out_df)
    years = set(pd.to_datetime(get_CDT_days(2010, 2024)))  # Convert years to set for O(1) lookup

    new_rows = []  # Collect new rows here
    
    # Add missing 2:00 AM entries for CDT transition
    for i in range(len(out_df) - 1, 0, -1):
        if out_df.iloc[i]['Date Time'] in years:
            new_row = out_df.iloc[[i - 1]].copy()  # Copy to avoid reference issues
            new_row.loc[:, 'Date Time'] += timedelta(hours=1)
            new_rows.append((i, new_row))  # Store new row with insertion index

    # Insert all new rows efficiently
    if new_rows:
        new_dataframes = []
        last_idx = 0

        for i, new_row in sorted(new_rows):  # Ensure order is maintained
            new_dataframes.append(out_df.iloc[last_idx:i])  # Add previous chunk
            new_dataframes.append(new_row)  # Add new row
            last_idx = i
        
        new_dataframes.append(out_df.iloc[last_idx:])  # Add remaining data
        out_df = pd.concat(new_dataframes, ignore_index=True)

    # Clean numeric columns (faster method)
    num_columns = out_df.select_dtypes(include='number').columns
    out_df[num_columns] = out_df[num_columns].mask(out_df[num_columns] < 0).ffill()
    
    # Apply additional cleaning
    out_df = set_na(out_df, df)
    
    return fill_time(out_df[['Date Time', 'Stream Flow']])

def clean_MMSD_df(df):
    """
    Set of instructions to unify data from USGS
    MMSD data is already adjusted for CST->CDT
    """
    out_df = df.copy()
    out_df.rename(columns={'Unnamed: 0':'Date Time'}, inplace=True)
    cols = out_df.columns.drop('Date Time')
    out_df[cols] = out_df[cols].apply(pd.to_numeric)
    
    out_df = separate_time(out_df)
    num_columns = out_df.select_dtypes(include='number').columns
    
    # Clean numeric columns (faster method)
    num_columns = out_df.select_dtypes(include='number').columns
    out_df[num_columns] = out_df[num_columns].mask(out_df[num_columns] < 0).ffill()
    
    # Apply additional cleaning
    out_df = set_na(out_df, df)
    return fill_time(out_df)