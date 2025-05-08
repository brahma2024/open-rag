# module 2c: Deletes all data including bucket. Then recreates the bucket
# version: 1.0
# To completely clean all data, including measurements, tag keys, and fields, from InfluxDB, you need to remove data at the bucket level. 
# This ensures that all traces of data, including the schema associated with the measurements (like stock names and field keys), are deleted.
# then recreate the bucket

from influxdb_client import InfluxDBClient
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
buckets_api = client.buckets_api()

# Delete the bucket
bucket_list = buckets_api.find_bucket_by_name(bucket)
if bucket_list:
    buckets_api.delete_bucket(bucket_list)
    print(f"Bucket '{bucket}' has been deleted.")
else:
    print(f"Bucket '{bucket}' does not exist.")

# Re-create the bucket after deletion
buckets_api.create_bucket(bucket_name=bucket, org=org)
print(f"Bucket '{bucket}' has been recreated.")

# Close the client after the operation
client.close()
