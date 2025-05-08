# module 7b: Monitor python function execution and save the stats in influxdb
# version: 1.0
# function:
# 1. Use cProfile to profile Python function execution and store the metrics in InfluxDB

import cProfile
import pstats
import io
from influxdb_client import InfluxDBClient, Point, WriteOptions
import datetime
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize InfluxDB client
url = os.getenv("INFLUXDB_URL")
token = os.getenv("INFLUXDB_TOKEN")
org = os.getenv("INFLUXDB_ORG")
bucket = os.getenv("INFLUXDB_BUCKET")
client = InfluxDBClient(url=url, token=token, org=org)
write_api = client.write_api(write_options=WriteOptions(batch_size=1))

# Function to profile and push Python function metrics to InfluxDB
def profile_function(func, *args, **kwargs):
    profiler = cProfile.Profile()
    profiler.enable()
    func(*args, **kwargs)  # Execute the function
    profiler.disable()

    # Capture profiler output
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats()

    # Parse the output and push metrics to InfluxDB
    lines = s.getvalue().splitlines()[5:]  # Skip the header lines
    for line in lines:
        if line.strip() == '' or line.startswith('ncalls'):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue

        # Extract metrics (customize as needed)
        ncalls = int(parts[0])  # Number of calls
        tottime = float(parts[1])  # Total time in the function
        percall = float(parts[2])  # Time per call
        cumtime = float(parts[3])  # Cumulative time
        function_name = parts[-1]  # Function name

        # Create a point to write to InfluxDB
        point = (
            Point("python_function_profile")
            .tag("function_name", function_name)
            .field("ncalls", ncalls)
            .field("total_time", tottime)
            .field("per_call_time", percall)
            .field("cumulative_time", cumtime)
            .time(datetime.datetime.utcnow())
        )
        write_api.write(bucket=bucket, org=org, record=point)

# # Example usage
# def example_function():
#     # Simulate a task
#     for i in range(1000000):
#         _ = i ** 2

# profile_function(example_function)

# # Close the InfluxDB client
# write_api.close()
# client.close()
