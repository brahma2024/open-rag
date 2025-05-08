"""
module_4_sequence_dataset.py
-----------------------------------------
Function:
A custom PyTorch Dataset to:
    1) Load the final indicator-enriched DataFrame from /data/indicators/.
    2) Convert each row to a token vector.
    3) Create sequences over a sliding window for the Neural SDE input.
    4) Handle multi-stock logic (stock_id, sector_id) and compute support/resistance label.
"""

import os
from typing import Optional
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import yaml


def load_nasdaq_gics_config(gics_config="nasdaq100_gics.yaml"):
    with open(gics_config, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    return config_data


class MarketSDEDataset(Dataset):
    """
    A custom PyTorch Dataset to:
      1) Load the final indicator-enriched DataFrame from /data/indicators/.
      2) Convert each row to a token vector.
      3) Create sequences over a sliding window for the Neural SDE input.
      4) Handle multi-stock logic (stock_id, sector_id).
    """
    def __init__(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        sequence_length: int = 30,
        window_step: Optional[int] = None,
        data_dir: str = "data/indicators",
        gics_config: str = "nasdaq100_gics.yaml"
    ):
        """
        :param tickers: List of ticker symbols to include.
        :param start_date: Start date string.
        :param end_date: End date string.
        :param sequence_length: Number of time steps in each sequence.
        :param window_step: Step size for sliding window, default = sequence_length // 2 or so.
        :param data_dir: Path to the folder containing the *with_indicators.parquet files.
        """
        self.tickers = tickers
        self.sequence_length = sequence_length
        self.window_step = window_step if window_step is not None else max(1, sequence_length // 2)
        self.data_dir = data_dir

        self.all_sequences = []  # will store tuples of (token_seq, stock_id, sector_id, label_vec)
        self.gics_config = load_nasdaq_gics_config(gics_config)  # load sector info
        self._build_dataset(start_date, end_date)

    def _build_dataset(self, start_date: str, end_date: str):
        """
        Loads each ticker's file from data_dir, tokenizes it, and appends sequences.
        Maps tickers to sector info using the loaded configuration.
        """
        for tkr in self.tickers:
            file_name = f"{tkr}_{start_date}_{end_date}_processed_with_indicators.parquet"
            file_path = os.path.join(self.data_dir, file_name)
            if not os.path.exists(file_path):
                print(f"Warning: {file_path} not found. Skipping {tkr}.")
                continue
            
            df = pd.read_parquet(file_path)
            df.sort_index(inplace=True)

            # Extract token columns and close prices
            token_cols = [
                "log_return_prevclose_to_close",
                "log_return_prevclose_to_open",
                "log_return_open_to_close",
                "log_return_open_to_high",
                "log_return_open_to_low",
                "S_t",
                "L_t",
                "IsOpen"
            ]
            token_data = df[token_cols].astype(np.float32).values  # shape [num_bars, token_dim]
            close_arr = df["Close"].astype(np.float32).values

            # Define a helper to check support breach based on future prices.
            def check_support_breach(prices, t, horizon5=5, sr_level=100.0):
                """
                Returns 1 if the price crosses sr_level in the next 5 bars, else 0.
                """
                window = prices[t+1 : t+1 + horizon5]
                return 1.0 if np.any(window < sr_level) else 0.0
    
            # Define a helper to compute labels.
            def make_labels(close_prices, idx):
                # next_1 bar return
                next_1 = close_prices[idx+1] / close_prices[idx] - 1.0
                # next_5 bar return
                next_5 = close_prices[idx+5] / close_prices[idx] - 1.0
                # next_10 bar return
                next_10 = close_prices[idx+10] / close_prices[idx] - 1.0
                # realized volatility over next 5 bars
                window = close_prices[idx+1 : idx+6]
                logrets = np.diff(np.log(window))
                rv5 = np.std(logrets)
                # support/resistance breach label
                sr_label = check_support_breach(close_prices, idx, horizon5=5, sr_level=100.0)
                label_vec = np.array([next_1, next_5, next_10, rv5, sr_label], dtype=np.float32)
                return label_vec

            # Map a ticker to its sector info using the loaded config.
            config = load_nasdaq_gics_config()
            stock_info = config["stocks"].get(tkr)
            if stock_info is None:
                print(f"Warning: stock info for {tkr} not found in config. Using default IDs.")
                stock_id = -1
                sector_id = -1
            else:
                stock_id = stock_info["gics_sector"].get("code", -1)
                sector_id = stock_info.get("sector_id", -1)

            # Ensure the token_data array only covers indices with full future information.
            max_future = 10
            max_valid_index = len(close_arr) - max_future
            token_data = token_data[:max_valid_index, :]
            
            max_start = len(token_data) - self.sequence_length
            for start_idx in range(0, max_start + 1, self.window_step):
                end_idx = start_idx + self.sequence_length
                label_t = end_idx - 1  # last bar of the sequence as label index
                label_vec = make_labels(close_arr, label_t)
                seq_tokens = token_data[start_idx:end_idx, :]
                self.all_sequences.append((seq_tokens, stock_id, sector_id, label_vec))

    def __len__(self):
        return len(self.all_sequences)
    
    def __getitem__(self, idx):
        token_seq, stock_id, sector_id, label_vec = self.all_sequences[idx]
        token_seq_t = torch.from_numpy(token_seq)
        stock_id_t = torch.tensor(stock_id, dtype=torch.long)
        sector_id_t = torch.tensor(sector_id, dtype=torch.long)
        label_t = torch.from_numpy(label_vec).float()  # shape e.g. [5]
        return (token_seq_t, stock_id_t, sector_id_t, label_t)