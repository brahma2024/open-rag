"""
module_7a.py
-----------------------------------------
Orchestrates the multi-step workflow using Prefect:
1. Fetch 30-min raw data via module_1a (Polygon API).
2. Preprocess data to uniform 30-min slots via module_1b.
3. Compute basic log returns (module_4a).
4. Compute advanced features (module_4b).

Usage Example:
  python module_7a.py
"""

import asyncio
import logging
import sys
from pathlib import Path
from prefect import flow, task, get_run_logger

# Ensure the src directory is in the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Local imports
from src.data_ingestion.module_1a import fetch_and_save_raw_data_async
from src.data_ingestion.module_1b import preprocess_and_save_all_raw_files
from src.indicators.module_4a import compute_log_returns_for_ticker
from src.indicators.module_4b import compute_advanced_features_for_ticker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

###################################################################
# Task 1: Fetch Raw Data for Tickers
###################################################################
@task
def fetch_raw_data_task(tickers: list[str], start_date: str, end_date: str,
                        interval: str = "30", timespan: str = "minute"):
    """
    Runs the async fetch for multiple tickers in parallel. 
    Orchestrates module_1a.fetch_and_save_raw_data_async.
    """
    logger = get_run_logger()

    if not tickers:
        logger.warning("No tickers provided to fetch_raw_data_task.")
        return
    
    logger.info("Starting raw data fetch for tickers: %s", tickers)

    # We run an async function from a sync context by calling asyncio.run()
    async def run_fetch():
        await fetch_and_save_raw_data_async(tickers, start_date, end_date, interval, timespan)

    asyncio.run(run_fetch())
    logger.info("Finished fetching raw data for %s", tickers)


###################################################################
# Task 2: Preprocess & Save All Raw Files
###################################################################
@task
def preprocess_data_task(start_date: str, end_date: str):
    """
    Calls module_1b to read data/raw/*.parquet, 
    reindex to 30-min custom slots, fill missing, 
    then save to data/processed/*.parquet.
    """
    logger = get_run_logger()
    logger.info("Starting data preprocessing (module_1b).")
    preprocess_and_save_all_raw_files(start_date, end_date)
    logger.info("Finished data preprocessing.")


###################################################################
# Task 3: Compute Log Returns
###################################################################
@task
def compute_log_returns_task(tickers: list[str], start_date: str, end_date: str):
    """
    For each ticker, calls module_4a.compute_log_returns_for_ticker, which:
      - Loads the *_processed.parquet
      - Adds log return columns
      - Saves to *_processed_with_indicators.parquet
    """
    logger = get_run_logger()
    logger.info("Starting log-return computations for tickers: %s", tickers)

    for ticker in tickers:
        compute_log_returns_for_ticker(ticker, start_date, end_date)

    logger.info("Finished computing log returns.")


###################################################################
# Task 4: Compute Advanced Features
###################################################################
@task
def compute_advanced_features_task(tickers: list[str], start_date: str, end_date: str):
    """
    For each ticker, calls module_4b.compute_advanced_features_for_ticker, which:
      - Loads *_processed_with_indicators.parquet
      - Computes FIGARCH vol, local-level trend, distribution params, etc.
      - Saves *_processed_w_advanced.parquet
    """
    logger = get_run_logger()
    logger.info("Starting advanced feature calculations for tickers: %s", tickers)

    for ticker in tickers:
        compute_advanced_features_for_ticker(ticker, start_date, end_date)

    logger.info("Finished computing advanced features.")


###################################################################
# Main Flow: orchestrate the entire pipeline
###################################################################
@flow
def stock_market_workflow(
    tickers: list[str],
    start_date: str,
    end_date: str,
    interval: str = "30",     # 30-min bars
    timespan: str = "minute"  # "minute" for Polygon
):
    """
    Orchestration Flow using Prefect.

    Steps:
      1. Fetch raw data (module_1a).
      2. Preprocess to uniform 30-min slots, save to data/processed (module_1b).
      3. Compute log returns, save to data/processed (module_4a).
      4. Compute advanced features (FIGARCH, distribution, etc.) (module_4b).
    """
    logger = get_run_logger()
    logger.info("----- Starting stock_market_workflow -----")

    # Step 1: Fetch & save raw data
    fetch_raw_data_task(tickers, start_date, end_date, interval, timespan)

    # Step 2: Preprocess all raw => processed
    preprocess_data_task(start_date, end_date)

    # Step 3: Compute log returns for each ticker
    compute_log_returns_task(tickers, start_date, end_date)

    # Step 4: Compute advanced features
    compute_advanced_features_task(tickers, start_date, end_date)

    logger.info("----- Completed stock_market_workflow -----")


#################################
# Command-line usage example
#################################
if __name__ == "__main__":
    # Example user inputs
    tickers = ["MSFT", "AMZN"]
    start_date = "2020-01-01"
    end_date   = "2024-12-31"

    # Run the flow
    stock_market_workflow(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        interval="30",
        timespan="minute"
    )
