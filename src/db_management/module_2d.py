# module 2d: delete influxdb data at measurement level
# version: 1.0
# function: deletes the influxdb data at measurement level, you can mention specific measurement names for which data needs to be deleted


from influxdb_client import InfluxDBClient
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

# Initialize the InfluxDB client
client = InfluxDBClient(url=url, token=token, org=org)
delete_api = client.delete_api()

# Function to delete a specific measurement
def delete_measurement(measurement, start_date="1970-01-01T00:00:00Z", end_date=datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')):
    """
    Deletes all data from a specific measurement in the bucket within the specified time range.
    Parameters:
    - measurement: The name of the measurement to delete.
    - start_date: The start date for deletion (default is epoch start).
    - end_date: The end date for deletion (default is the current time).
    """
    predicate = f'_measurement="{measurement}"'
    try:
        delete_api.delete(start=start_date, stop=end_date, predicate=predicate, bucket=bucket, org=org)
        print(f"All data for measurement '{measurement}' has been deleted.")
    except Exception as e:
        print(f"Error deleting measurement '{measurement}': {e}")

# Specify the measurement to delete
# measurement_to_delete = ["prices", "stocks", "market_indicators"]
measurement_to_delete = ["prices", "stocks", "market_indicators", "market_open_indicator", "advanced_features"]
# measurement_to_delete = ["advanced_features_v2"]
for measurement in measurement_to_delete:
    # Call the function to delete the measurement
    delete_measurement(measurement)

# Close the client after usage
client.close()
