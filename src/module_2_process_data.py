"""
module_2_process_data.py
-----------------------------------------
Function:
Reads raw Parquet files from data/raw/, merges them with a custom 
'trading-slot' calendar with 30-minute slots from 9:30 to 16:00 each calendar day,
forward-fills missing OHLC prices, sets Volume=0 and Transactions=0 for any missing slot,
and adds an 'IsOpen' flag indicating a trading slot (1) or not (0).
Saves the cleaned data to data/processed/.
"""

import os
import logging
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

def build_trading_slots_index(start_date: str, end_date: str) -> pd.DatetimeIndex:
    """
    Build a custom index from start_date to end_date that includes 30‑minute slots
    from 9:30 to 16:00 each calendar day.

    For each day the generated timestamps are:
      9:30, 10:00, 10:30, 11:00, 11:30, 12:00, 12:30,
      13:00, 13:30, 14:00, 14:30, 15:00, 15:30, 16:00

    Note: This function does not skip weekends or holidays.
    """
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # Generate daily range
    daily_range = pd.date_range(start=start_dt, end=end_dt, freq='D')
    
    # Define the 30-minute trading slots in decimal hours (9.5 corresponds to 9:30, etc.)
    TRADING_SLOTS = [9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5,
                     13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0]
    
    all_datetimes = []
    for day in daily_range:
        for slot in TRADING_SLOTS:
            dt = day + pd.Timedelta(hours=slot)
            all_datetimes.append(dt)
            
    return pd.DatetimeIndex(all_datetimes, name='timestamp').sort_values()

def preprocess_stock_data(
    df: pd.DataFrame,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Reindex DataFrame to only the designated 30‑minute trading slots (9:30–16:00 each day).
    Forward-fill OHLC values from the last known slot.
    For newly introduced slots, set Volume=0, Transactions=0, and IsOpen=0.

    :param df: DataFrame with a DateTimeIndex and columns [Open, High, Low, Close, Volume, Transactions]
    :param start_date: start date (YYYY-MM-DD)
    :param end_date: end date (YYYY-MM-DD)
    :return: Preprocessed DataFrame adjusted to half‑hour intervals.
    """
    if df.empty:
        logger.warning("Received empty DataFrame. Returning empty.")
        return df

    # Remove timezone information, sort the index, and drop duplicate indices.
    df.index = df.index.tz_localize(None)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep='last')]

    # Generate the trading slots index based on 30-minute intervals.
    trading_index = build_trading_slots_index(start_date, end_date)

    # Reindex the DataFrame to include all trading slots; missing slots become NaN.
    df = df.reindex(trading_index)

    # Forward-fill OHLC values from the last known non-NaN value.
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = df[col].ffill()

    # For slots that were originally missing, set Volume and Transactions to 0.
    originally_missing = df['Volume'].isna()
    df.loc[originally_missing, 'Volume'] = 0
    df.loc[originally_missing, 'Transactions'] = 0

    # Drop rows that still have missing critical values, if any.
    df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume', 'Transactions'], inplace=True)

    # Create 'IsOpen': 1 if the slot was originally present (trading occurred), else 0.
    df['IsOpen'] = (~originally_missing.loc[df.index]).astype(int)

    return df

def preprocess_and_save_all_raw_files(start_date: str, end_date: str) -> None:
    """
    Reads all Parquet files from data/raw/, preprocesses them to contain only the 30‑minute
    trading slots (9:30–16:00), and saves the processed files to data/processed/.
    """
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    raw_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.parquet')]
    if not raw_files:
        logger.warning("No raw Parquet files found in %s", RAW_DATA_DIR)
        return

    for raw_file in raw_files:
        file_path = os.path.join(RAW_DATA_DIR, raw_file)
        try:
            df = pd.read_parquet(file_path)
            logger.info("Preprocessing %s with shape=%s", raw_file, df.shape)

            processed_df = preprocess_stock_data(df, start_date, end_date)
            if processed_df.empty:
                logger.warning("%s resulted in empty DataFrame after preprocessing.", raw_file)
                continue

            # Save processed file with an updated name indicating preprocessing.
            out_file = raw_file.replace('.parquet', '_processed.parquet')
            out_path = os.path.join(PROCESSED_DATA_DIR, out_file)
            processed_df.to_parquet(out_path)
            logger.info("Saved processed data to %s", out_path)

        except (pd.errors.EmptyDataError, FileNotFoundError, OSError) as e:
            logger.error("Error reading/preprocessing %s: %s", raw_file, e)

def main():
    """
    Example usage:
      python module_1b.py
    """
    start_date = "2020-01-01"
    end_date = "2024-12-31"
    preprocess_and_save_all_raw_files(start_date, end_date)

if __name__ == "__main__":
    main()