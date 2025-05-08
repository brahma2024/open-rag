# module 5d: Backtesting the model with saving predictions
# version: 3.3
# function: 
#   - inference engine
#   - backtesting on historical data
#   - save predictions to historical_data.csv

import pandas as pd
import torch
from influxdb_client import InfluxDBClient
from dotenv import load_dotenv
import os
from torch.optim import Adam

from src.ml_models.module_5b import (
    prepare_market_data, 
    prepare_data_for_training, 
    chronological_split
)
from src.ml_models.module_5a import CTRNN

# Load environment variables from the .env file
load_dotenv()

# Initialize the InfluxDB client
url = os.getenv("INFLUXDB_URL")
token = os.getenv("INFLUXDB_TOKEN")
org = os.getenv("INFLUXDB_ORG")
bucket = os.getenv("INFLUXDB_BUCKET")

client = InfluxDBClient(url=url, token=token, org=org)
query_api = client.query_api()

# File paths
model_path = "ctrnn_model_checkpoint.pth"
csv_file_path = "C:\\Users\\naman\\Desktop\\projects\\market_data_tempcsv\\historical_data.csv"

# Function to write data to CSV
def write_to_csv(data, filename):
    try:
        data.to_csv(filename, index=False)
        print(f"Data successfully written to {filename}")
    except Exception as e:
        print(f"Error writing to CSV: {e}")

# Function to perform inference
def perform_inference(ct_rnn, test_data):
    ct_rnn.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        initial_state = test_data[:, 0, :]  # Use the initial state for each sequence
        t = torch.linspace(0, 1, test_data.shape[1], device=test_data.device)
        predictions = ct_rnn(initial_state, t).squeeze()
    return predictions

# Main function
def main():
    stock_id = "MSFT"  # Example stock ID
    start_date = "2020-01-01T00:00:00Z"  # Adjust the start date as needed
    end_date = "2024-10-30T23:59:59Z"  # Adjust the end date as needed

    # Fetch and prepare the data
    numeric_df, timestamps = prepare_market_data(query_api, stock_id, start_date, end_date)

    # Preprocess the data to obtain test sequences
    batched_data, batched_labels, batched_timestamps = prepare_data_for_training(numeric_df, timestamps, batch_size=32, sequence_length=200)
    
    # Split into training, validation, and test sets
    _, _, _, _, _, _, test_data, _, _ = chronological_split(batched_data, batched_labels, batched_timestamps)

    # Convert test_data to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_data = test_data.to(device)

    # Load the trained model
    input_dim = 6
    hidden_dim = 128
    output_dim = 1
    ct_rnn = CTRNN(input_dim, hidden_dim, output_dim).to(device)

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        ct_rnn.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully.")
    else:
        raise FileNotFoundError(f"No model checkpoint found at {model_path}")

    # Perform inference
    predictions = perform_inference(ct_rnn, test_data)

    # Extract the required values for comparison
    test_data_cpu = test_data.cpu()  # Move test_data to CPU for DataFrame compatibility
    data_last_timestep = test_data_cpu[:, -1, :]  # Extract the 200th timestep of each sequence
    
    # Extract the 'log_return_prevclose_to_close' feature value
    log_return_index = numeric_df.columns.get_loc('log_return_prevclose_to_close')
    log_return_values = data_last_timestep[:, log_return_index].numpy()

    # Extract the timestamp for each sequence at the 200th timestep
    timestamps = numeric_df.index[-test_data_cpu.size(0):]

    # Create a DataFrame with log_return values, model predictions, and timestamps
    results_df = pd.DataFrame({
        'Timestamp': timestamps,
        'Log_Return_PrevClose_to_Close_200th': log_return_values,
        'Prediction': predictions.cpu().numpy()
    })

    # Print the values
    print(results_df)

    # Save the results to CSV
    write_to_csv(results_df, csv_file_path)

if __name__ == "__main__":
    main()