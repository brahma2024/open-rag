"""
module_1_fetch_raw_data.py
-----------------------------------------
Function:
Fetches 30-minute OHLCV data from Polygon.io for multiple tickers (async).
Saves each ticker's raw data as Parquet in data/raw/.
Handles 'None' parameter issues by validating inputs.
"""

import os
import datetime
import logging
import asyncio
import aiohttp
import pandas as pd
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt

# Load environment variables (POLYGON_API_KEY, etc.)
load_dotenv()

# Configure logging for production-level clarity
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Global settings
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
SEMAPHORE_LIMIT = 10  # concurrency limit (tweak to avoid rate limits)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")


def _validate_input_params(ticker: str, start_date: str, end_date: str, interval: str, timespan: str) -> None:
    """
    Raise ValueError if any critical input is None or empty.
    Ensures we never pass 'None' to the API.
    """
    if not ticker:
        raise ValueError("Ticker must be a non-empty string.")
    if not POLYGON_API_KEY:
        raise ValueError("POLYGON_API_KEY is missing. Please set it in your environment.")
    if not start_date:
        raise ValueError("start_date must not be None or empty.")
    if not end_date:
        raise ValueError("end_date must not be None or empty.")
    if not interval:
        raise ValueError("interval must not be None or empty.")
    if not timespan:
        raise ValueError("timespan must not be None or empty.")


def build_url(ticker: str, interval: str, timespan: str, from_date: str, end_date: str) -> str:
    """
    Build the Polygon.io API URL to fetch data.
    """
    return (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker.upper()}/range/"
        f"{interval}/{timespan}/{from_date}/{end_date}"
    )


@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(5))
async def fetch_polygon_ohlcv_async(
    session: aiohttp.ClientSession,
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "30",      # 30-minute aggregates
    timespan: str = "minute"     # Use "minute" (singular) as supported by the API
) -> pd.DataFrame:
    """
    Fetch 30-minute OHLCV data for a single ticker from the Polygon API.
    Retries up to 5 times with exponential backoff on errors.
    Raises ValueError if inputs are None or empty.
    Returns a DataFrame with columns [Open, High, Low, Close, Volume, Transactions].
    """
    _validate_input_params(ticker, start_date, end_date, interval, timespan)

    # Convert start/end dates to YYYY-MM-DD
    start_date_str = datetime.datetime.fromisoformat(start_date.replace('Z', '')).strftime('%Y-%m-%d')
    end_date_str = datetime.datetime.fromisoformat(end_date.replace('Z', '')).strftime('%Y-%m-%d')

    base_url = build_url(ticker, interval, timespan, start_date_str, end_date_str)

    # These query params do NOT include start_date or end_date (they are in the URL path already)
    params = {
        'apiKey': POLYGON_API_KEY,
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 50000
    }

    all_data = []
    max_iterations = 1000  # safeguard against infinite loops
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        logger.info("Fetching data from URL: %s", base_url)
        async with session.get(base_url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                results = data.get('results', [])
                if results:
                    all_data.extend(results)
                    if len(results) < params['limit']:
                        # We've fetched all the available data
                        break

                    # Increment the start date by one day using the last data point's timestamp
                    last_timestamp = pd.to_datetime(results[-1]['t'], unit='ms')
                    new_from_str = (last_timestamp + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                    base_url = build_url(ticker, interval, timespan, new_from_str, end_date_str)
                else:
                    logger.warning("No results found for %s in the given range.", ticker)
                    break
            else:
                err_text = await response.text()
                logger.error("Error %s fetching %s: %s", response.status, ticker, err_text)
                break

    if iteration == max_iterations:
        logger.error("Max iteration limit reached for %s. Breaking out to avoid infinite loop.", ticker)

    if not all_data:
        logger.warning("No data returned for %s. Returning empty DataFrame.", ticker)
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.rename(columns={
        'o': 'Open', 'h': 'High', 'l': 'Low',
        'c': 'Close', 'v': 'Volume', 'n': 'Transactions'
    }, inplace=True)

    # Keep only relevant columns
    return df[['Open', 'High', 'Low', 'Close', 'Volume', 'Transactions']]


async def fetch_ticker_data_and_save(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "30",       # 30-minute aggregates
    timespan: str = "minute"      # Use "minute" (singular)
) -> None:
    """
    Asynchronously fetch data for a single ticker and save to Parquet in data/raw/.
    Raises ValueError if inputs are invalid.
    """
    _validate_input_params(ticker, start_date, end_date, interval, timespan)

    async with aiohttp.ClientSession() as session:
        df = await fetch_polygon_ohlcv_async(session, ticker, start_date, end_date, interval, timespan)

    if df.empty:
        logger.info("[%s] No data to save. DataFrame is empty.", ticker)
        return

    # Build file path
    file_name = f"{ticker}_{start_date}_{end_date}_{interval}{timespan}.parquet"
    file_path = os.path.join(RAW_DATA_DIR, file_name)

    # Ensure data/raw/ exists
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    # Save to Parquet
    df.to_parquet(file_path)
    logger.info("[%s] Saved raw data to %s", ticker, file_path)


async def fetch_and_save_raw_data_async(
    tickers: list[str],
    start_date: str,
    end_date: str,
    interval: str = "30",       # 30-minute aggregates
    timespan: str = "minute"      # Use "minute" (singular)
) -> None:
    """
    Fetch data for multiple tickers in parallel (async) and save results to data/raw/.
    Raises ValueError if inputs are invalid.
    """
    if not tickers:
        raise ValueError("No tickers provided.")

    sem = asyncio.Semaphore(SEMAPHORE_LIMIT)

    async def semaphore_wrapper(ticker: str) -> None:
        async with sem:
            await fetch_ticker_data_and_save(ticker, start_date, end_date, interval, timespan)

    tasks = [semaphore_wrapper(t) for t in tickers]
    await asyncio.gather(*tasks)


def main():
    """
    Example command-line entry point or direct call.
    Modify tickers, date range, etc. as needed.
    """
    tickers = ["MSFT", "AMZN"]
    start_date = "2020-01-01"
    end_date = "2024-12-31"
    interval = "30"       # 30-minute aggregates
    timespan = "minute"   # use "minute" as supported by Polygon
    
    # Run async gather
    asyncio.run(fetch_and_save_raw_data_async(tickers, start_date, end_date, interval, timespan))


if __name__ == "__main__":
    main()