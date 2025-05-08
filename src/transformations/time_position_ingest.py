"""
time_position_ingest.py

Purpose:
1) Given a market DataFrame with a datetime index, compute:
   - normalized_hour_of_day = hour/24
   - normalized_day_of_week = (day_index + 1)/7  (where Monday=0 in Python, so we shift by +1)
   - normalized_month_of_year = month/12
2) Round to 2 decimals.
3) Write results to InfluxDB in the measurement "time_position".

Production-Grade Aspects:
- Minimal overhead via vectorized pandas operations
- Batch writing to Influx
- Logging for maintainability
- Handling timezone normalization and a 5-hour offset as per instructions
"""

import os
import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime
from influxdb_client import Point
from src.db_management.module_2a import write_batch  # or your custom batch writing function

def ingest_time_position_data(write_api, market_df: pd.DataFrame, batch_size: int = 1000):
    """
    Ingest time-position data into InfluxDB:
    For each row in market_df, compute:
      - normalized_hour_of_day
      - normalized_day_of_week
      - normalized_month_of_year
    Store them as fields in measurement="time_position".

    Args:
      write_api: InfluxDB write API
      market_df: A DataFrame with at least a DateTimeIndex 
      batch_size: size of each batch chunk

    Returns:
      None, writes data to InfluxDB
    """

    if market_df.empty:
        print("No data in market_df, skipping ingestion of time-position.")
        return

    # Step 1: Prepare the index
    # If the user wants to match a schedule, e.g. adding 5 hours, do so:
    market_df = market_df.copy()
    # Remove any timezone, then normalize to midnight and shift by +5 hours
    market_df.index = market_df.index.tz_localize(None)  # ensure no timezone
    market_df.index = market_df.index.normalize() + pd.Timedelta(hours=5)  # if needed

    # Step 2: Extract date/time components in a vectorized manner
    # Using .index directly for year, month, day, hour, dayofweek, etc.
    # dayofweek in pandas: Monday=0, Sunday=6
    hours = market_df.index.hour
    day_of_week = market_df.index.dayofweek  # 0=Monday, 6=Sunday
    month = market_df.index.month  # 1=Jan, 12=Dec

    # Step 3: Compute normalized values
    # normalized_hour_of_day = hour_of_day / 24
    # normalized_day_of_week = (day_of_week + 1)/7 => Monday=1 => 1/7 => 0.14, ...
    # normalized_month_of_year = month/12 => 1 => 0.08, 2 => 0.16, ...
    # Round to 2 decimals
    normalized_hour_of_day = np.round(hours / 24, 2)
    normalized_day_of_week = np.round((day_of_week + 1) / 7, 2)
    normalized_month_of_year = np.round(month / 12, 2)

    # Step 4: Construct Points for Influx (time_position measurement)
    # We'll store date/time as the index
    points = []
    for ts, hour_val, dow_val, moy_val in zip(market_df.index, normalized_hour_of_day, normalized_day_of_week, normalized_month_of_year):
        # We can store the date/time as the timestamp (ts).
        # The fields are the normalized features.
        # For device name or other tags, use "time_position" as measurement name, no specific tags if not needed.
        pt = {
            "measurement": "time_position",
            "tags": {},  # no stock_id needed if purely time-based, else we can add
            "time": ts,
            "fields": {
                "normalized_hour_of_day": float(hour_val),
                "normalized_day_of_week": float(dow_val),
                "normalized_month_of_year": float(moy_val)
            }
        }
        points.append(pt)

    # Step 5: Batch Write
    for i in range(0, len(points), batch_size):
        batch = points[i : i+batch_size]
        write_batch(write_api, batch)

    print(f"Finished ingest_time_position_data for {len(points)} points.")
