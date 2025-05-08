"""
module_4b.py
-----------------------------------------
Reads the data (with log returns) from data/processed/ (or the file saved by module_4a).
Computes advanced features:
  - FIGARCH volatility on log_return_prevclose_to_close
  - Local-level trend
  - Volume/transactions rolling metrics
  - Distribution parameters (skew, kurt)
  - Factor+ARFIMA placeholders
Saves the final result with advanced features appended.
"""

import os
import logging
import shutil
import numpy as np
import pandas as pd

from arch import arch_model
from statsmodels.tsa.statespace.structural import UnobservedComponents
from scipy.stats import skew, kurtosis

# joblib cache for repeated computations
from joblib import Memory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
# Optionally, you could store new files in a separate folder like:
INDICATORS_DATA_DIR = os.path.join(BASE_DIR, "data", "indicators")

CACHE_DIR = './data/cache_advanced_features'
memory = Memory(location=CACHE_DIR, verbose=0)


def load_data_with_indicators(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load the Parquet file that includes log-return columns
    (produced by module_4a).
    """
    filename = f"{ticker}_{start_date}_{end_date}_processed_with_indicators.parquet"
    file_path = os.path.join(INDICATORS_DATA_DIR, filename)

    if not os.path.exists(file_path):
        logger.warning("No data with indicators found for %s at %s", ticker, file_path)
        return pd.DataFrame()

    df = pd.read_parquet(file_path)
    logger.info("Loaded data w/ indicators for %s with shape=%s", ticker, df.shape)
    return df


@memory.cache
def fit_figarch_volatility(returns: pd.Series, p: int = 1, q: int = 1, dist: str = 'normal') -> pd.Series:
    """
    Fit a FIGARCH(p,q) model for long-memory volatility, returns the conditional vol series.
    """
    clean_ret = returns.dropna()
    if len(clean_ret) < 200:
        return pd.Series(index=returns.index, data=np.nan, name='figarch_vol')

    try:
        model = arch_model(clean_ret, mean='Constant', vol='FIGARCH', p=p, q=q, dist=dist)
        res = model.fit(update_freq=0, disp='off')
        vol = res.conditional_volatility
        vol.name = 'figarch_vol'
        vol.index = clean_ret.index
        return vol.reindex(returns.index)
    except (ValueError, np.linalg.LinAlgError) as e:
        logger.error("FIGARCH fitting error: %s", e)
        return pd.Series(index=returns.index, data=np.nan, name='figarch_vol')


@memory.cache
def extract_trend_locallevel(close_prices: pd.Series) -> pd.Series:
    """
    Local-level model (Kalman filter) approach for trend extraction.
    """
    cp = close_prices.dropna()
    if len(cp) < 50:
        return pd.Series(index=close_prices.index, data=np.nan, name='trend_factor')

    mod = UnobservedComponents(cp, level='local level')
    res = mod.fit(disp=False)
    trend_vals = res.smoothed_state[0, :]
    return pd.Series(trend_vals, index=cp.index, name='trend_factor')


@memory.cache
def extract_volume_order_flow(volume: pd.Series, transactions: pd.Series, rolling_window: int=20) -> pd.DataFrame:
    """
    Example: rolling means over a 20-slot window (which might be 20 x 30 min = 10 hours).
    """
    df = pd.DataFrame(index=volume.index)
    if len(volume) < rolling_window:
        df['roll_mean_volume'] = np.nan
        df['roll_mean_transactions'] = np.nan
        return df

    df['roll_mean_volume'] = volume.rolling(rolling_window).mean()
    df['roll_mean_transactions'] = transactions.rolling(rolling_window).mean()
    return df


@memory.cache
def fit_distribution_params(returns: pd.Series, window: int=20) -> pd.DataFrame:
    """
    Rolling skew/kurt as placeholders for advanced distribution fitting.
    """
    df = pd.DataFrame(index=returns.index)
    r_clean = returns.dropna()
    if len(r_clean) < window:
        df['dist_skew'] = np.nan
        df['dist_kurt'] = np.nan
        return df

    def rolling_skew_func(x):
        return skew(x.dropna())

    def rolling_kurt_func(x):
        return kurtosis(x.dropna())

    df['dist_skew'] = r_clean.rolling(window).apply(rolling_skew_func, raw=False)
    df['dist_kurt'] = r_clean.rolling(window).apply(rolling_kurt_func, raw=False)
    return df.reindex(returns.index)


@memory.cache
def factor_and_arfima_adjusted_returns(returns: pd.Series) -> pd.Series:
    """
    Placeholder for factor-model + ARFIMA residual.
    Returns the same series for demonstration.
    """
    return returns


def compute_advanced_features_for_ticker(ticker: str, start_date: str, end_date: str) -> None:
    """
    1. Load data with log returns from module_4a.
    2. Fit FIGARCH on log_return_prevclose_to_close
    3. Extract local-level trend
    4. Rolling volume/trade metrics
    5. Rolling distribution params
    6. Factor + ARFIMA (placeholder)
    7. Save final DF to data/processed/ (or separate location).
    """
    df = load_data_with_indicators(ticker, start_date, end_date)
    if df.empty:
        logger.warning("No data to compute advanced features for %s. Skipping.", ticker)
        return

    required = {'Close', 'Volume', 'Transactions', 'log_return_prevclose_to_close'}
    missing = required - set(df.columns)
    if missing:
        logger.warning("Missing columns %s for advanced feats in %s. Skipping.", missing, ticker)
        return

    # 2) FIGARCH
    ret_for_vol = df['log_return_prevclose_to_close']
    figarch_vol = fit_figarch_volatility(ret_for_vol)

    # 3) Trend Factor
    trend_factor = extract_trend_locallevel(df['Close'])

    # 4) Rolling Volume/Transactions
    vol_order_df = extract_volume_order_flow(df['Volume'], df['Transactions'], rolling_window=20)

    # 5) Distribution
    dist_df = fit_distribution_params(ret_for_vol, window=20)

    # 6) Factor + ARFIMA
    factor_arfima_resid = factor_and_arfima_adjusted_returns(ret_for_vol)

    # Consolidate
    feats_df = pd.DataFrame(index=df.index)
    feats_df['figarch_vol'] = figarch_vol
    feats_df['trend_factor'] = trend_factor
    feats_df = feats_df.join(vol_order_df, how='left')
    feats_df = feats_df.join(dist_df, how='left')
    feats_df['factor_arfima_resid'] = factor_arfima_resid

    # Clean up
    feats_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    feats_df.dropna(how='all', inplace=True)

    if feats_df.empty:
        logger.info("No advanced feats after cleaning for %s. Skipping save.", ticker)
        return

    # Merge back into original DF (so we keep all columns in one place)
    final_df = df.join(feats_df, how='left')

    # Save final DataFrame
    out_file = f"{ticker}_{start_date}_{end_date}_processed_w_advanced.parquet"
    out_path = os.path.join(INDICATORS_DATA_DIR, out_file)
    final_df.to_parquet(out_path)

    logger.info("Saved advanced features for %s to %s (shape=%s)", ticker, out_path, final_df.shape)


def main():
    """
    Example usage:
      python module_4b.py
    """
    tickers = ["MSFT", "AMZN"]
    start_date = "2020-01-01"
    end_date   = "2024-12-31"

    for ticker in tickers:
        compute_advanced_features_for_ticker(ticker, start_date, end_date)

    # Optionally clear cache
    shutil.rmtree(CACHE_DIR, ignore_errors=True)

if __name__ == "__main__":
    main()