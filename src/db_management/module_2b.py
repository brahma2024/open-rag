# module 2b: delete data based on conditions from influxdb
# version: 1.0
# Function: You need to use the delete_api from the influxdb-client library to delete data programmatically based on conditions.

from influxdb_client import InfluxDBClient, DeleteApi
import datetime
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Initialize the InfluxDB client
url = os.getenv("INFLUXDB_URL")
token = os.getenv("INFLUXDB_TOKEN")
org = os.getenv("INFLUXDB_ORG")
bucket = os.getenv("INFLUXDB_BUCKET")

client = InfluxDBClient(url=url, token=token, org=org)
delete_api = client.delete_api()

# Initialize the InfluxDB client
client = InfluxDBClient(url=url, token=token, org=org)
delete_api = client.delete_api()

# Function to delete entries based on tags
def delete_entries(measurement, tag_key, tag_value, start_date="1970-01-01T00:00:00Z", end_date=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')):
    # Create a predicate for tags only
    predicate = f'_measurement="{measurement}" AND "{tag_key}"=\'{tag_value}\''
    try:
        delete_api.delete(start=start_date, stop=end_date, predicate=predicate, bucket=bucket, org=org)
        print(f"Deleted entries for tag '{tag_key}' with value '{tag_value}' in measurement '{measurement}'.")
    except Exception as e:
        print(f"Error deleting entries: {e}")

# Example call to delete entries based on a specific tag key and value
# Make sure 'tag_key' and 'tag_value' correspond to tags in your schema, not fields
delete_entries("market_indicators", "name", "angle_100_50")

# Close the client after usage
client.close()