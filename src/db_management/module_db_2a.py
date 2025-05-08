# module 2a: fintrade influxdb schema management, writes data to influxdb
# version: 1.0
# function: create a production-grade set of functions that handle data type validation and avoid schema collisions in InfluxDB, 
# we need to implement robust data type checks and conversions for each field entry. 
# This will ensure that data types are consistent and avoid potential schema conflicts.

from influxdb_client import InfluxDBClient, Point, WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Initialize the InfluxDB client
url = os.getenv("INFLUXDB_URL")
token = os.getenv("INFLUXDB_TOKEN")
org = os.getenv("INFLUXDB_ORG")
bucket = os.getenv("INFLUXDB_BUCKET")

# client = InfluxDBClient(url=url, token=token, org=org)
# write_api = client.write_api(write_options=SYNCHRONOUS)

# Helper function to validate and convert data to float
def safe_float(value, field_name):
    try:
        return float(value)
    except (ValueError, TypeError):
        print(f"Warning: Field '{field_name}' has an invalid value '{value}' and will not be written.")
        return None

# Helper function to validate and convert data to int
def safe_int(value, field_name):
    try:
        return int(value)
    except (ValueError, TypeError):
        print(f"Warning: Field '{field_name}' has an invalid value '{value}' and will not be written.")
        return None

# Function to write in batches
def write_batch(write_api, points):
    if points:
        write_api.write(bucket=bucket, org=org, record=points)
        print(f"Batch of {len(points)} points written to InfluxDB.")
 
# Function to write stock metadata
def write_stock_metadata(write_api, stock_id, ticker, name, exchange, sector, currency, is_active=True):
    if isinstance(is_active, bool):  # Validate is_active as a boolean
        point = (
            Point("stocks")
            .tag("stock_id", stock_id)
            .field("ticker", str(ticker))
            .field("name", str(name))
            .field("exchange", str(exchange))
            .field("sector", str(sector))
            .field("currency", str(currency))
            .field("is_active", is_active)
        )
        write_api.write(bucket=bucket, org=org, record=point)
    else:
        print(f"Warning: 'is_active' must be a boolean value. Skipping write for stock_id '{stock_id}'.")

# Function to write stock prices
def write_stock_price(stock_id, timestamp, open_price, close_price, high_price, low_price, volume, transactions, adjusted_close=None):
    open_price = safe_float(open_price, "open_price")
    close_price = safe_float(close_price, "close_price")
    high_price = safe_float(high_price, "high_price")
    low_price = safe_float(low_price, "low_price")
    volume = safe_int(volume, "volume")
    transactions = safe_int(transactions, "transactions")
    adjusted_close = safe_float(adjusted_close if adjusted_close is not None else close_price, "adjusted_close")

    if None not in [open_price, close_price, high_price, low_price, volume, transactions, adjusted_close]:
        point = (
            Point("prices")
            .tag("stock_id", stock_id)
            .field("open_price", open_price)
            .field("close_price", close_price)
            .field("high_price", high_price)
            .field("low_price", low_price)
            .field("volume", volume)
            .field("transactions", transactions)
            .field("adjusted_close", adjusted_close)
            .time(timestamp)
        )
        return point
    else:
        print(f"Warning: Invalid data for {stock_id} at {timestamp}. Skipping this point.")
        return None
    
# Function to write market indicators
def write_market_indicator(stock_id, indicator_name, value, timestamp, data_source):
    # Handle cases where the value may not be defined (e.g., during the initial period)
    if value is None:
        value = float('nan')  # Ensure NaN is stored as a float

    # Validate the value to ensure only floats are written to the database
    value = safe_float(value, "value")

    if value is not None:  # Only write if the value is valid and numeric
        point = (
            Point("market_indicators")
            .tag("stock_id", stock_id)
            .tag("name", str(indicator_name))
            .field("value", value)
            .field("data_source", str(data_source))
            .time(timestamp)
        )
        return point
    else:
        print(f"Warning: Skipping write for '{indicator_name}' for stock '{stock_id}' at {timestamp} due to invalid value.")
        return None

# Writing to Influx with write_batch
def write_advanced_features(write_api, stock_id: str, features_df: pd.DataFrame,
                           batch_size: int, bucket: str):
    """
    Writes advanced features to InfluxDB in measurement "advanced_features_v2"
    using write_batch from module_2a.
    """
    from module_2a import write_batch
    points = []
    for ts, row in features_df.iterrows():
        for col, val in row.items():
            if pd.isna(val):
                continue
            pt = {
                "measurement": "advanced_features",
                "tags": {"stock_id": stock_id},
                "time": ts,
                "fields": {col: float(val)}
            }
            points.append(pt)

    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        write_batch(write_api, batch)

# Function to write financial metrics
def write_financial_metric(write_api, stock_id, metric_type, value, timestamp, frequency):
    value = safe_float(value, "value")

    if value is not None:
        point = (
            Point("financial_metrics")
            .tag("stock_id", stock_id)
            .tag("metric_type", str(metric_type))
            .field("value", value)
            .field("frequency", str(frequency))
            .time(timestamp)
        )
        write_api.write(bucket=bucket, org=org, record=point)

# Function to write corporate actions
def write_corporate_action(write_api, stock_id, action_type, timestamp, details, value=None):
    value = safe_float(value if value is not None else 0, "value")

    if value is not None:
        point = (
            Point("corporate_actions")
            .tag("stock_id", stock_id)
            .field("action_type", str(action_type))
            .field("details", str(details))
            .field("value", value)
            .time(timestamp)
        )
        write_api.write(bucket=bucket, org=org, record=point)

# Function to write market open-close indicator | helps identify which days the market was open or close
def write_market_indicator(stock_id, timestamp, is_open):
    point = Point("market_open_indicator") \
        .tag("stock_id", stock_id) \
        .field("is_open", int(is_open)) \
        .time(timestamp)
    return point

# Function to write options and derivatives data
def write_options_data(write_api, underlying_stock_id, expiry_date, strike_price, option_type, timestamp, price):
    strike_price = safe_float(strike_price, "strike_price")
    price = safe_float(price, "price")

    if strike_price is not None and price is not None:
        point = (
            Point("options_derivatives")
            .tag("underlying_stock_id", underlying_stock_id)
            .tag("option_type", str(option_type))  # "CALL" or "PUT"
            .field("expiry_date", str(expiry_date.isoformat()))
            .field("strike_price", strike_price)
            .field("price", price)
            .time(timestamp)
        )
        write_api.write(bucket=bucket, org=org, record=point)

# Function to write news and sentiment analysis data
def write_news_sentiment_data(write_api, news_id, stock_id, headline, source, sentiment_score, article_text, timestamp):
    sentiment_score = safe_float(sentiment_score, "sentiment_score")

    if sentiment_score is not None:
        point = (
            Point("news_sentiment")
            .tag("news_id", news_id)
            .tag("stock_id", stock_id)
            .tag("source", str(source))
            .field("headline", str(headline))
            .field("sentiment_score", sentiment_score)
            .field("article_text", str(article_text))
            .time(timestamp)
        )
        write_api.write(bucket=bucket, org=org, record=point)

# Function to write trading strategy results data
def write_trading_strategy_results(write_api, strategy_id, stock_id, entry_timestamp, exit_timestamp, entry_price, exit_price, profit_loss):
    entry_price = safe_float(entry_price, "entry_price")
    exit_price = safe_float(exit_price, "exit_price")
    profit_loss = safe_float(profit_loss, "profit_loss")

    if None not in [entry_price, exit_price, profit_loss]:
        point = (
            Point("trading_strategy_results")
            .tag("strategy_id", strategy_id)
            .tag("stock_id", stock_id)
            .field("entry_price", entry_price)
            .field("exit_price", exit_price)
            .field("profit_loss", profit_loss)
            .field("entry_timestamp", str(entry_timestamp.isoformat()))
            .field("exit_timestamp", str(exit_timestamp.isoformat()))
            .time(exit_timestamp)  # Use exit time as the time reference for the point
        )
        write_api.write(bucket=bucket, org=org, record=point)



# # Example Usage
# if __name__ == "__main__":
#     # Insert a stock metadata entry
#     write_stock_metadata(
#         stock_id="1",
#         ticker="AAPL",
#         name="Apple Inc.",
#         exchange="NASDAQ",
#         sector="Technology",
#         currency="USD",
#         is_active=True
#     )

#     # Insert a stock price entry
#     write_stock_price(
#         stock_id="1",
#         timestamp=datetime.datetime.utcnow(),
#         open_price=145.67,
#         close_price=146.34,
#         high_price=147.10,
#         low_price=144.89,
#         volume=78000000,
#         adjusted_close=146.20
#     )

#     # Insert a financial metric entry
#     write_financial_metric(
#         stock_id="1",
#         metric_type="EPS",
#         value=3.67,
#         timestamp=datetime.datetime.utcnow(),
#         frequency="quarterly"
#     )

#     # Insert a corporate action entry
#     write_corporate_action(
#         stock_id="1",
#         action_type="Dividend",
#         timestamp=datetime.datetime.utcnow(),
#         details="Quarterly dividend payment of $0.22 per share",
#         value=0.22
#     )

#     # Insert a market indicator entry
#     write_market_indicator(
#         indicator_name="S&P 500 Index",
#         value=4321.67,
#         timestamp=datetime.datetime.utcnow(),
#         data_source="Bloomberg"
#     )

#     print("Data successfully written to InfluxDB!")

# # Close the client when done
# client.close()
