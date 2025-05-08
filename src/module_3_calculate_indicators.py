"""
module_3_calculate_indicators.py
-----------------------------------------
Function:
Reads processed OHLCV data from data/processed/, calculates log returns,
standardizes them, computes additional indicators (S_t, L_t), 
and saves the updated DataFrame (including the new columns)
to a designated indicators folder.
"""

import os
import logging
from typing import List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Define base directories and ensure the output directory exists
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
INDICATORS_DATA_DIR = os.path.join(BASE_DIR, "data", "indicators")
os.makedirs(INDICATORS_DATA_DIR, exist_ok=True)

###############################################################################
# 1) Load Processed Data
###############################################################################
def load_processed_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Loads the processed Parquet file for a given ticker from data/processed/.
    Naming convention: <TICKER>_<start_date>_<end_date>_30minute_processed.parquet
    
    :param ticker: Ticker symbol.
    :param start_date: Start date (YYYY-MM-DD).
    :param end_date: End date (YYYY-MM-DD).
    :return: DataFrame if file exists, otherwise an empty DataFrame.
    """
    filename_pattern = f"{ticker}_{start_date}_{end_date}_30minute_processed.parquet"
    file_path = os.path.join(PROCESSED_DATA_DIR, filename_pattern)

    if not os.path.exists(file_path):
        logger.warning("Processed file not found for %s. Path=%s", ticker, file_path)
        return pd.DataFrame()

    try:
        df = pd.read_parquet(file_path)
        logger.info("Loaded processed data for %s with shape=%s", ticker, df.shape)
        return df
    except (pd.errors.EmptyDataError, FileNotFoundError, OSError) as e:
        logger.exception("Failed to load processed file for %s: %s", ticker, e)
        return pd.DataFrame()

###############################################################################
# 2) Compute Log Returns (Existing)
###############################################################################
def calculate_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates log-return columns then standardizes them.
    Expects input DataFrame to have columns: [Open, High, Low, Close].
    
    Log returns are scaled by a factor (default: 1e3).
    
    :param df: Input DataFrame with required price columns.
    :return: Updated DataFrame with new log-return columns.
    """
    required_cols = {'Open', 'High', 'Low', 'Close'}
    if not required_cols.issubset(df.columns):
        logger.warning("DataFrame missing required columns for log-return calculation. Found: %s", df.columns)
        return df

    scaling_factor = 1e3
    # Calculate log returns efficiently in a vectorized form
    df['log_return_prevclose_to_close'] = np.log(df['Close'] / df['Close'].shift(1)) * scaling_factor
    df['log_return_prevclose_to_open']  = np.log(df['Open']  / df['Close'].shift(1)) * scaling_factor
    df['log_return_open_to_close']      = np.log(df['Close'] / df['Open']) * scaling_factor
    df['log_return_open_to_high']       = np.log(df['High']  / df['Open']) * scaling_factor
    df['log_return_open_to_low']        = np.log(df['Low']   / df['Open']) * scaling_factor

    # Replace infinities with NaN, then fill missing values with 0.
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    log_return_cols = [
        'log_return_prevclose_to_close',
        'log_return_prevclose_to_open',
        'log_return_open_to_close',
        'log_return_open_to_high',
        'log_return_open_to_low'
    ]

    # Standardize the log-return columns
    scaler = StandardScaler()
    df[log_return_cols] = scaler.fit_transform(df[log_return_cols])
    # Convert to float32 to reduce memory usage.
    df[log_return_cols] = df[log_return_cols].astype('float32')

    return df

###############################################################################
# 3) Compute Additional Indicators (S_t, L_t)
###############################################################################
def compute_liquidity_indicator(df: pd.DataFrame, volume_window: int = 20) -> pd.DataFrame:
    """
    Compute a composite liquidity indicator S_t from Volume and Transactions.
    Steps:
      1) Compute rolling mean/std for Volume & Transactions
      2) Convert Volume, Transactions to z-scores
      3) Combine them as S_t = alpha * zVolume + beta * zTransactions
    
    :param df: DataFrame with columns ['Volume', 'Transactions'].
    :param volume_window: Rolling window size for mean/std.
    :return: Updated DataFrame with new column 'S_t'.
    """
    if 'Volume' not in df.columns or 'Transactions' not in df.columns:
        logger.warning("DataFrame missing 'Volume' or 'Transactions'. Cannot compute S_t.")
        return df

    # Rolling stats for volume
    df['vol_mean'] = df['Volume'].rolling(volume_window, min_periods=1).mean()
    df['vol_std']  = df['Volume'].rolling(volume_window, min_periods=1).std(ddof=1)
    df['vol_std'].replace(0, np.nan, inplace=True)  # avoid division by zero
    df['vol_std'].fillna(method='bfill', inplace=True)

    # Rolling stats for transactions
    df['tx_mean'] = df['Transactions'].rolling(volume_window, min_periods=1).mean()
    df['tx_std']  = df['Transactions'].rolling(volume_window, min_periods=1).std(ddof=1)
    df['tx_std'].replace(0, np.nan, inplace=True)
    df['tx_std'].fillna(method='bfill', inplace=True)

    # z-scores
    df['zVolume'] = (df['Volume'] - df['vol_mean']) / df['vol_std']
    df['zTransactions'] = (df['Transactions'] - df['tx_mean']) / df['tx_std']

    # Combine, e.g., alpha=1, beta=1 for simplicity
    df['S_t'] = df['zVolume'] + df['zTransactions']
    df.drop(['vol_mean','vol_std','tx_mean','tx_std','zVolume','zTransactions'], axis=1, inplace=True)

    # Fill any remaining NaNs
    df['S_t'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['S_t'].fillna(0, inplace=True)
    df['S_t'] = df['S_t'].astype('float32')

    return df

def compute_normalized_price_level(df: pd.DataFrame, ma_window: int = 50) -> pd.DataFrame:
    """
    Compute L_t = ln(Close / MA), a normalized price-level measure.
    
    :param df: DataFrame with column 'Close'
    :param ma_window: rolling window for the moving average
    :return: df with new column 'L_t'
    """
    if 'Close' not in df.columns:
        logger.warning("DataFrame missing 'Close' column. Cannot compute L_t.")
        return df

    df['ma_Close'] = df['Close'].rolling(ma_window, min_periods=1).mean()
    # L_t = ln(Close / ma_Close)
    df['L_t'] = np.log(df['Close'] / df['ma_Close'])
    df.drop(['ma_Close'], axis=1, inplace=True)

    df['L_t'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['L_t'].fillna(0, inplace=True)
    df['L_t'] = df['L_t'].astype('float32')

    return df

###############################################################################
# 4) Save Final Indicators
###############################################################################
def save_with_indicators(ticker: str, start_date: str, end_date: str, df: pd.DataFrame) -> None:
    """
    Save the DataFrame (with new indicator columns) as a Parquet file.
    The file is stored in the INDICATORS_DATA_DIR folder.
    
    :param ticker: Ticker symbol.
    :param start_date: Start date.
    :param end_date: End date.
    :param df: DataFrame to save.
    """
    out_file = f"{ticker}_{start_date}_{end_date}_processed_with_indicators.parquet"
    out_path = os.path.join(INDICATORS_DATA_DIR, out_file)
    try:
        df.to_parquet(out_path)
        logger.info("Saved updated data with log returns & advanced indicators for %s to %s", ticker, out_path)
    except (pd.errors.EmptyDataError, FileNotFoundError, OSError) as e:
        logger.exception("Failed to save file for %s: %s", ticker, e)

###############################################################################
# 5) Main Routine: compute_log_returns_for_ticker
###############################################################################
def compute_log_returns_for_ticker(ticker: str, start_date: str, end_date: str) -> None:
    """
    End-to-end routine for processing a single ticker:
      1. Load processed data.
      2. Calculate log returns.
      3. Compute composite liquidity S_t and normalized level L_t
      4. Save updated DataFrame with new indicator columns.
    
    :param ticker: Ticker symbol.
    :param start_date: Start date.
    :param end_date: End date.
    """
    df = load_processed_data(ticker, start_date, end_date)
    if df.empty:
        logger.warning("No data to process for %s. Skipping log-return computation.", ticker)
        return

    # Step A: Basic log returns
    df = calculate_log_returns(df)

    # Step B: Composite liquidity indicator
    df = compute_liquidity_indicator(df, volume_window=20)

    # Step C: Normalized price level
    df = compute_normalized_price_level(df, ma_window=50)

    # Step D: Save final
    save_with_indicators(ticker, start_date, end_date, df)

###############################################################################
# Example usage
###############################################################################
def main() -> None:
    """
    Example usage:
      Process a list of tickers by loading the processed files,
      computing log returns, S_t, L_t, and saving the updated DataFrame.
    """
    tickers: List[str] = ["MSFT", "AMZN"]
    start_date = "2020-01-01"
    end_date = "2024-12-31"

    for ticker in tickers:
        compute_log_returns_for_ticker(ticker, start_date, end_date)

if __name__ == "__main__":
    main()
