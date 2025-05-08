# module_2e: Read data from InfluxDB
# version: 2.1
# function:
#   This module fetches stock, prices, market indicators, advanced features,
#   and market open indicators data from InfluxDB.
#   It provides a consistent, production-grade interface to retrieve
#   and preprocess data for downstream modules.

import os
import datetime
import numpy as np
import pandas as pd
from influxdb_client import InfluxDBClient
from dotenv import load_dotenv

load_dotenv()

# Initialize InfluxDB configuration from environment variables
url = os.getenv("INFLUXDB_URL")
token = os.getenv("INFLUXDB_TOKEN")
org = os.getenv("INFLUXDB_ORG")
bucket = os.getenv("INFLUXDB_BUCKET", "fintrade")

def _to_utc_index(df: pd.DataFrame, time_col: str = '_time') -> pd.DataFrame:
    """
    Ensures the time column is converted to a UTC-based datetime index without a timezone.
    """
    if time_col not in df.columns:
        return df

    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors='coerce')
    df = df.dropna(subset=[time_col])
    df.set_index(time_col, inplace=True)
    if hasattr(df.index, 'tz_convert'):
        df.index = df.index.tz_convert(None)
    df.sort_index(inplace=True)
    return df

def get_influxdb_client() -> InfluxDBClient:
    return InfluxDBClient(url=url, token=token, org=org)

def read_available_stocks(query_api):
    """
    Reads all distinct stock_ids from 'stocks' measurement.
    """
    flux_query = f'''
    from(bucket: "{bucket}")
      |> range(start: 0)
      |> filter(fn: (r) => r._measurement == "stocks")
      |> keep(columns: ["stock_id"])
      |> distinct(column: "stock_id")
    '''
    df = query_api.query_data_frame(org=org, query=flux_query)
    if df.empty:
        return []
    df = df.drop(columns=['result', 'table'], errors='ignore')
    if 'stock_id' not in df.columns:
        return []
    return df['stock_id'].dropna().unique().tolist()

def read_prices_data(query_api, stock_id: str, interval: str, start_date: str, end_date: str = None) -> pd.DataFrame:
    """
    Reads OHLCV + 'transactions' from 'prices' measurement for a given stock_id and date range.
    Applies aggregateWindow as per the interval.
    """
    if end_date is None:
        end_date = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')

    flux_query = f'''
    from(bucket: "{bucket}")
        |> range(start: time(v: "{start_date}"), stop: time(v: "{end_date}"))
        |> filter(fn: (r) => r._measurement == "prices" and r["stock_id"] == "{stock_id}")
        |> aggregateWindow(every: {interval}, fn: last, createEmpty: false)
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        |> keep(columns: ["_time", "open_price", "close_price", "high_price", "low_price", "volume", "transactions", "stock_id"])
        |> sort(columns: ["_time"])
    '''
    df = query_api.query_data_frame(org=org, query=flux_query)
    if df.empty:
        return pd.DataFrame(columns=["open_price", "close_price", "high_price",
                                     "low_price", "volume", "transactions", "stock_id"])

    df = df.drop(columns=['result', 'table'], errors='ignore').dropna(subset=['_time'])
    df = _to_utc_index(df)
    return df

def read_market_open_indicator_data(query_api, stock_id: str, start_date: str, end_date: str = None) -> pd.DataFrame:
    """
    Reads the is_open indicator from the 'market_open_indicator' measurement.
    """
    if end_date is None:
        end_date = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')

    flux_query = f'''
    from(bucket: "{bucket}")
        |> range(start: time(v: "{start_date}"), stop: time(v: "{end_date}"))
        |> filter(fn: (r) => r._measurement == "market_open_indicator" and r["stock_id"] == "{stock_id}")
        |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
        |> sort(columns: ["_time"])
    '''
    df = query_api.query_data_frame(org=org, query=flux_query)
    if df.empty:
        return pd.DataFrame(columns=['stock_id', 'is_open'])

    df = df.drop(columns=['result', 'table', '_start', '_stop', '_measurement'], errors='ignore')
    if '_time' not in df.columns or 'stock_id' not in df.columns or 'is_open' not in df.columns:
        return pd.DataFrame(columns=['stock_id', 'is_open'])
    df = _to_utc_index(df)
    # Normalize indices to midnight
    df.index = df.index.normalize()
    return df[['stock_id', 'is_open']]

def read_market_indicators_data(query_api, stock_id: str, start_date: str, end_date: str = None) -> pd.DataFrame:
    """
    Reads market indicators (log returns, etc.) and merges them with is_open from market_open_indicator.
    """
    if end_date is None:
        end_date = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')

    flux_query = f'''
    from(bucket: "{bucket}")
        |> range(start: time(v: "{start_date}"), stop: time(v: "{end_date}"))
        |> filter(fn: (r) => r._measurement == "market_indicators" and r["stock_id"] == "{stock_id}")
        |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
        |> sort(columns: ["_time"])
    '''
    market_df = query_api.query_data_frame(org=org, query=flux_query)
    if market_df.empty:
        return pd.DataFrame(columns=['stock_id', 'log_return_prevclose_to_open', 'log_return_prevclose_to_close',
                                     'log_return_open_to_close', 'log_return_open_to_high', 'log_return_open_to_low', 'is_open'])

    market_df = market_df.drop(columns=['result', 'table', '_start', '_stop', '_measurement'], errors='ignore')
    if '_time' not in market_df.columns or 'stock_id' not in market_df.columns:
        return pd.DataFrame()

    market_df = _to_utc_index(market_df)
    # Normalize indices to midnight
    market_df.index = market_df.index.normalize()

    required_cols = [
        'log_return_prevclose_to_open',
        'log_return_prevclose_to_close',
        'log_return_open_to_close',
        'log_return_open_to_high',
        'log_return_open_to_low'
    ]
    for c in required_cols:
        if c not in market_df.columns:
            market_df[c] = np.nan

    open_df = read_market_open_indicator_data(query_api, stock_id, start_date, end_date)
    print(f"open_df: {open_df.head(10)}")
    if open_df.empty:
        # if no is_open found return value error
        raise ValueError(f"Market open indicator not found for {stock_id}.")

    merged = market_df.join(open_df[['is_open']], how='left')
    print(f"merged: {merged.head(10)}")
    return merged[['stock_id'] + required_cols + ['is_open']]

def read_stock_data_from_influx(query_api, stock_ids: list, start_date: str, end_date: str = None, interval="1d") -> pd.DataFrame:
    """
    Reads prices and market indicators for multiple stocks and merges them.
    Returns a combined DataFrame.
    """
    if end_date is None:
        end_date = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')

    all_data = []
    for stock_id in stock_ids:
        try:
            prices_data = read_prices_data(query_api, stock_id, interval, start_date, end_date)
            market_data = read_market_indicators_data(query_api, stock_id, start_date, end_date)
            if prices_data.empty or market_data.empty:
                continue
            merged = prices_data.join(market_data.drop(columns=['stock_id'], errors='ignore'), how='outer')
            merged['stock_id'] = stock_id
            all_data.append(merged)
        except Exception as e:
            print(f"Error reading data for {stock_id}: {e}")

    if not all_data:
        return pd.DataFrame()

    final_df = pd.concat(all_data)
    final_df.sort_index(inplace=True)
    return final_df

#############################################################
# NEW FUNCTION: read_all_features_data
#############################################################
def read_all_features_data(query_api,
                           stock_id: str,
                           start_date: str,
                           end_date: str = None) -> pd.DataFrame:
    """
    Reads both:
      - Market indicators data (from 'market_indicators' measurement),
      - Advanced features (from 'advanced_features_v2' measurement),
    merges them into one DataFrame.

    Return:
      A DataFrame indexed by time, containing all columns from both sets:
       (log_return_prevclose_to_close, ... ) + (figarch_vol, trend_factor, distribution params, etc.)

    If either query returns empty, an empty DataFrame is returned.
    """
    if end_date is None:
        end_date = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')

    # 1) Read market indicators (including log returns, is_open, etc.)
    market_df = read_market_indicators_data(query_api, stock_id, start_date, end_date)
    # We'll forcibly drop 'stock_id' after merging to keep clean
    # or we can keep it if we prefer. We'll keep for demonstration.

    # 2) Read advanced features from 'advanced_features'
    flux_query = f'''
    from(bucket: "{bucket}")
        |> range(start: time(v: "{start_date}"), stop: time(v: "{end_date}"))
        |> filter(fn: (r) => r._measurement == "advanced_features" and r["stock_id"] == "{stock_id}")
        |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
        |> sort(columns: ["_time"])
    '''
    adv_df = query_api.query_data_frame(org=org, query=flux_query)
    if not adv_df.empty:
        adv_df = adv_df.drop(columns=['result', 'table', '_start', '_stop', '_measurement'], errors='ignore')
        if '_time' in adv_df.columns:
            adv_df = _to_utc_index(adv_df)
    else:
        adv_df = pd.DataFrame()

    # remove the 1st 60 rows since window for some advanced features is 60
    adv_df = adv_df.iloc[60:]
    market_df = market_df.iloc[60:]

    # Normalize indices to midnight
    adv_df.index = adv_df.index.normalize()

    print(f'adv_df head: {adv_df.head(10)}')
    print(f'market_df head: {market_df.head(10)}')
    if market_df.empty and adv_df.empty:
        return pd.DataFrame()

    # If advanced features are empty, we just return the market_df
    if adv_df.empty:
        return market_df

    # Merge them on index
    merged_df = market_df.join(adv_df.drop(columns=['stock_id'], errors='ignore'), how='outer')
    # Re-inject stock_id if needed
    merged_df['stock_id'] = stock_id

    # Sort by index
    merged_df.sort_index(inplace=True)
    return merged_df
