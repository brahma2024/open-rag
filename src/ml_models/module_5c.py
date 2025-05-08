# module_5c.py: train.py
# version: 4.0
# function:
#   - Parse arguments or load config
#   - Set seeds for reproducibility
#   - Fetch and combine data from multiple tickers
#   - Chronologically split data and create PyTorch Datasets/DataLoaders using sliding windows
#   - Initialize and train CTRNN model
#   - Implement early stopping, model checkpointing, and logging

import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from dotenv import load_dotenv
import logging
import time

from influxdb_client import InfluxDBClient
from torchdiffeq import odeint

# Local imports
from src.ml_models.module_5b import (
    prepare_market_data, 
    chronological_split_raw, 
    create_dataloaders
)
from src.ml_models.module_5a import CTRNN  # CTRNN model definition assumed available


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(ct_rnn, optimizer, criterion, train_loader, device):
    # ct_rnn.train()
    total_loss = 0.0
    for input_seq, labels in train_loader:
        input_seq, labels = input_seq.to(device), labels.to(device)
        optimizer.zero_grad()

        # input_seq: [batch_size, seq_length-1, features]
        # integrate from t=0 to t=1 with seq_length steps
        t = torch.linspace(0, 1, input_seq.size(1) + 1, device=device)
        
        initial_state = input_seq[:, 0, :]  # [batch_size, features]
        predicted = ct_rnn(initial_state, t)  # predicted: [batch_size, seq_length]

        loss = criterion(predicted[:, -1].squeeze(), labels.squeeze())
        loss.backward()
        nn.utils.clip_grad_norm_(ct_rnn.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * input_seq.size(0)
    return total_loss / len(train_loader.dataset)


def evaluate(ct_rnn, criterion, data_loader, device):
    ct_rnn.eval()
    total_loss = 0.0
    with torch.no_grad():
        for input_seq, labels in data_loader:
            input_seq, labels = input_seq.to(device), labels.to(device)
            t = torch.linspace(0, 1, input_seq.size(1) + 1, device=device)
            initial_state = input_seq[:, 0, :]
            predicted = ct_rnn(initial_state, t)
            loss = criterion(predicted[:, -1].squeeze(), labels.squeeze())
            total_loss += loss.item() * input_seq.size(0)
    return total_loss / len(data_loader.dataset)


def fetch_and_combine_data(query_api, stock_ids, start_date, end_date):
    """
    Fetch data for multiple stock_ids, concatenate them, and return a combined numeric_df and timestamps.
    Data from each ticker is appended one after another (vertically).
    """
    all_dfs = []
    all_times = []

    for sid in stock_ids:
        numeric_df, timestamps = prepare_market_data(query_api, sid, start_date, end_date)
        all_dfs.append(numeric_df)
        all_times.append(timestamps)

    # Combine data from all tickers
    combined_df = pd.concat(all_dfs, axis=0)
    combined_times = np.concatenate(all_times)

    # Resort by time if not already sorted after concatenation
    # This ensures chronological order across all tickers combined
    sort_idx = np.argsort(combined_times)
    combined_df = combined_df.iloc[sort_idx]
    combined_times = combined_times[sort_idx]

    return combined_df, combined_times

def main(args):
    load_dotenv()

    # Logging setup
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    # Set seed for reproducibility
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Initialize InfluxDB client
    url = os.getenv("INFLUXDB_URL")
    token = os.getenv("INFLUXDB_TOKEN")
    org = os.getenv("INFLUXDB_ORG")
    bucket = os.getenv("INFLUXDB_BUCKET")

    client = InfluxDBClient(url=url, token=token, org=org)
    query_api = client.query_api()

    # Parse multiple tickers
    stock_ids = [sid.strip() for sid in args.stock_ids.split(",") if sid.strip()]
    if len(stock_ids) == 0:
        raise ValueError("No valid stock_ids provided. Please provide at least one ticker.")

    # Fetch and combine data for all tickers
    numeric_df, timestamps = fetch_and_combine_data(query_api, stock_ids, args.start_date, args.end_date)
    logging.info(f"Combined dataset shape: {numeric_df.shape}, covering {len(timestamps)} timestamps.")

    # Chronologically split raw data into train/val/test
    (train_df, train_times,
     val_df, val_times,
     test_df, test_times) = chronological_split_raw(
         numeric_df, timestamps,
         train_ratio=args.train_ratio,
         val_ratio=args.val_ratio
     )

    # Create DataLoaders with sliding windows
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, train_times,
        val_df, val_times,
        test_df, test_times,
        sequence_length=args.sequence_length,
        window_step=args.window_step,
        batch_size=args.batch_size
    )

    input_dim = numeric_df.shape[1]  # number of features
    ct_rnn = CTRNN(input_dim, args.hidden_dim, args.output_dim).to(device)
    optimizer = Adam(ct_rnn.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    best_val_loss = float('inf')
    start_epoch = 0
    if args.resume and os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        ct_rnn.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['loss']
        logging.info(f"Resuming training from epoch {start_epoch} with best val loss {best_val_loss:.4f}")

    no_improvement_count = 0

    for epoch in range(start_epoch, args.num_epochs):
        start_time = time.time()

        train_loss = train_one_epoch(ct_rnn, optimizer, criterion, train_loader, device)
        val_loss = evaluate(ct_rnn, criterion, val_loader, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_count = 0
            torch.save({
                'model_state_dict': ct_rnn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': best_val_loss
            }, args.model_path)
        else:
            no_improvement_count += 1

        epoch_duration = time.time() - start_time
        logging.info(f"Epoch {epoch}/{args.num_epochs} - Train Loss: {train_loss:.4f}, "
                     f"Val Loss: {val_loss:.4f}, Time: {epoch_duration:.2f}s, "
                     f"No Improvement: {no_improvement_count}")

        if no_improvement_count >= args.patience:
            logging.info("Early stopping triggered.")
            break

    # Evaluate on test set
    test_loss = evaluate(ct_rnn, criterion, test_loader, device)
    logging.info(f"Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    # stock_ids = "ADBE, AMD, ABNB, GOOGL, GOOG, AMZN, AEP, AMGN, ADI, ANSS, AAPL, AMAT, APP, ARM, ASML, AZN, TEAM, ADSK, ADP, BKR, BIIB, BKNG, MSFT"
    parser = argparse.ArgumentParser(description='Train CTRNN model on multiple time-series data (multiple tickers).')
    parser.add_argument('--stock_ids', type=str, default="ADBE", help='Comma-separated list of stock IDs, e.g. "AAPL,MSFT"')
    # parser.add_argument('--stock_ids', type=str, default="ADBE,AMD,ABNB,GOOGL,GOOG,AMZN,AEP,AMGN,ADI,ANSS,AAPL,AMAT,APP,ARM,ASML,AZN,TEAM,ADSK,ADP,BKR,BIIB,BKNG,MSFT",
    #                    help='Comma-separated list of stock IDs, e.g. "AAPL,MSFT"')
    parser.add_argument('--start_date', type=str, default="2020-01-01T00:00:00Z", help='Start date for data')
    parser.add_argument('--end_date', type=str, default="2024-10-30T23:59:59Z", help='End date for data')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--sequence_length', type=int, default=30, help='Sequence length')
    parser.add_argument('--window_step', type=int, default=15, help='Window step for sliding sequences')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension size for the model')
    parser.add_argument('--output_dim', type=int, default=1, help='Output dimension for the model')
    parser.add_argument('--model_path', type=str, default='ctrnn_model_checkpoint.pth', help='Path to save the model')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Train split ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation split ratio')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint if available')

    args = parser.parse_args()
    main(args)