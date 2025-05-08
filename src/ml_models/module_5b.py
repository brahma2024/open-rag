# module_5b.py: data_processing.py
# version: 3.1
# function:
#   - Fetch and prepare market data
#   - Chronologically split raw data into train/val/test sets
#   - Define a PyTorch Dataset for sequence creation using a sliding window
#   - Provide a function to create DataLoaders for training, validation, testing

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from src.ml_models.module_2e import read_all_features_data

def prepare_market_data(query_api, stock_id, start_date, end_date):
    """
    Reads data from read_all_features_data, ensuring that '_time' is a column
    so we can create a timestamps array. Returns numeric_df and timestamps.
    """
    df = read_all_features_data(query_api, stock_id, start_date, end_date)

    if df is None or df.empty:
        raise ValueError(f"No data found for the given stock ID ({stock_id}) and date range ({start_date} to {end_date}).")

    # If '_time' is the index but not a column, re-inject it:
    if '_time' not in df.columns and df.index.name == '_time':
        df['_time'] = df.index

    if '_time' not in df.columns:
        raise ValueError("No '_time' column found or set as index for the dataset.")

    # Optional: Print debug info (in production, you might remove or reduce verbosity)
    print(f"Data shape for {stock_id}: {df.shape}")
    print(f"Columns: {df.columns}")
    print(f"Sample data:\n{df['roll_mean_volume'].head(10)}")

    # Sort by index if needed, but don't reassign df to None
    df.sort_index(inplace=True)

    # Extract timestamps as a numpy array
    timestamps = df['_time'].to_numpy()

    # Drop unwanted columns to keep only numeric features
    # 'stock_id' might not always exist, so errors='ignore'
    numeric_df = df.drop(columns=['_time', 'stock_id'], errors='ignore')
    # Keep only numeric types
    numeric_df = numeric_df.select_dtypes(include=[np.number])

    if 'log_return_prevclose_to_close' not in numeric_df.columns:
        raise ValueError("Required column 'log_return_prevclose_to_close' not found in data.")

    return numeric_df, timestamps

def chronological_split_raw(numeric_df, timestamps, train_ratio=0.7, val_ratio=0.15):
    total_length = len(numeric_df)
    train_size = int(total_length * train_ratio)
    val_size = int(total_length * val_ratio)
    test_size = total_length - train_size - val_size

    train_df = numeric_df.iloc[:train_size]
    train_times = timestamps[:train_size]

    val_df = numeric_df.iloc[train_size:train_size+val_size]
    val_times = timestamps[train_size:train_size+val_size]

    test_df = numeric_df.iloc[train_size+val_size:]
    test_times = timestamps[train_size+val_size:]

    return (train_df, train_times, val_df, val_times, test_df, test_times)

class MarketDataset(Dataset):
    def __init__(self, numeric_df, timestamps, sequence_length=30, window_step=None):
        self.sequence_length = sequence_length
        self.window_step = window_step if window_step is not None else sequence_length // 2
        self.numeric_data = torch.tensor(numeric_df.values, dtype=torch.float32)
        self.timestamps = timestamps
        self.log_return_index = numeric_df.columns.get_loc('log_return_prevclose_to_close')
        self.indices = self._compute_indices()

    def _compute_indices(self):
        max_start = len(self.numeric_data) - self.sequence_length
        return [i for i in range(0, max_start+1, self.window_step)]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        end_idx = start_idx + self.sequence_length
        seq = self.numeric_data[start_idx:end_idx]
        label = (seq[-1, self.log_return_index] > 0).float()
        input_seq = seq[:-1, :]  # [sequence_length-1, features]

        # Remove timestamps from item to avoid collation errors, but could be re-added as needed
        return input_seq, label

def create_dataloaders(train_df, train_times, val_df, val_times, test_df, test_times,
                       sequence_length=30, window_step=None, batch_size=32, num_workers=0):
    train_dataset = MarketDataset(train_df, train_times, sequence_length, window_step)
    val_dataset = MarketDataset(val_df, val_times, sequence_length, window_step)
    test_dataset = MarketDataset(test_df, test_times, sequence_length, window_step)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=False, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            drop_last=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             drop_last=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
