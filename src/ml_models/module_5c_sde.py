# module_5c_sde.py
# version: 5.1
# purpose:
#   - End-to-end training of a Neural SDE model with an input embedding layer.
#   - Highly efficient, production-grade design for financial time-series.

import os
import argparse
import random
import time
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam

from dotenv import load_dotenv
import torchsde  # for SDE integration

# Local imports
from influxdb_client import InfluxDBClient
from src.ml_models.module_5b import (
    prepare_market_data,
    chronological_split_raw,
    create_dataloaders,
)
from module_5b1 import InputEmbeddingNetwork  # The cyclical + feature embedding layer
from module_5a_sde import NeuralSDE, SDEFunction  # The Neural SDE architecture

###############################################################################
# 1) Set Seed for Reproducibility
###############################################################################
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


###############################################################################
# 2) Single Epoch SDE Training
###############################################################################
def train_sde_one_epoch(neural_sde, input_net, optimizer, train_loader, device):
    """
    Training routine that:
    1) Uses input_net to produce embeddings from raw numeric features.
    2) Defines an initial state from embeddings to feed into SDE.
    3) Integrates the SDE forward via torchsde.sdeint.
    4) Computes a loss (e.g., MSE) w.r.t. some target (e.g., final embedding or ground truth).
    5) Backpropagates the error to update neural_sde + input_net.
    """
    neural_sde.train()
    input_net.train()

    criterion = nn.MSELoss()
    total_loss = 0.0
    total_samples = 0

    for input_seq, labels in train_loader:
        # input_seq: [batch_size, seq_length-1, features]
        # labels: [batch_size, 1], e.g. some classification or target

        batch_size = input_seq.size(0)
        seq_len = input_seq.size(1)

        # Move to device
        input_seq = input_seq.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # For demonstration, timestamps can be zeros or real if dataset includes them.
        timestamps_dummy = torch.zeros(batch_size, seq_len, device=device, dtype=torch.long)

        # 1) Create embeddings from raw_features + cyclical time
        embeddings = input_net(raw_features=input_seq, timestamps=timestamps_dummy)
        # shape: [B, seq_len, latent_dim]

        # 2) Take the initial embedding as SDE initial state
        H0 = embeddings[:, 0, :]  # [B, latent_dim]

        # 3) We define a time grid for the SDE solver:
        sde_times = torch.linspace(0, 1, seq_len, device=device)

        # 4) Build the SDEFunction which references the NeuralSDE
        sde_func = SDEFunction(neural_sde=neural_sde, past_window=seq_len)

        # 5) Integrate
        # shape returned: [len(sde_times), B, latent_dim]
        H_traj = torchsde.sdeint(sde_func, H0, sde_times, method='euler')

        # 6) Final state from the SDE
        H_final_pred = H_traj[-1]  # [B, latent_dim]

        # For demonstration, let's define a toy loss = MSE(H_final_pred, embeddings[:, -1, :])
        H_final_true = embeddings[:, -1, :]  # last embedding
        loss = criterion(H_final_pred, H_final_true)
        loss.backward()
        nn.utils.clip_grad_norm_(list(input_net.parameters()) + list(neural_sde.parameters()), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


###############################################################################
# 3) Main Execution
###############################################################################
def main(args):
    # Logging setup
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # InfluxDB client
    client = InfluxDBClient(url=os.getenv("INFLUXDB_URL"),
                            token=os.getenv("INFLUXDB_TOKEN"),
                            org=os.getenv("INFLUXDB_ORG"))
    query_api = client.query_api()

    # 1) Read data from module_5b
    numeric_df, timestamps = prepare_market_data(
        query_api=query_api,
        stock_id=args.stock_id,
        start_date=args.start_date,
        end_date=args.end_date
    )
    logging.info(f"Data shape: {numeric_df.shape}, Timestamps: {len(timestamps)}")

    # 2) Split data
    train_df, train_times, val_df, val_times, test_df, test_times = chronological_split_raw(
        numeric_df, timestamps,
        train_ratio=args.train_ratio, val_ratio=args.val_ratio
    )
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, train_times,
        val_df, val_times,
        test_df, test_times,
        sequence_length=args.sequence_length,
        window_step=args.window_step,
        batch_size=args.batch_size
    )

    # 3) Build input embedding net
    input_feature_size = numeric_df.shape[1]
    embedding_net = InputEmbeddingNetwork(
        input_feature_size=input_feature_size,
        latent_dim=args.latent_dim,
        time_feature_count=2,
        time_encoding_dims=4,
        hidden_dims=[128, 128],
        dropout=0.1,
        event_vocab_size=0,       # If we don't have discrete events
        event_embedding_dim=16,
        use_layernorm=True
    ).to(device)

    # 4) Build Neural SDE
    h_dim = args.latent_dim
    param_dim_f = 32
    param_dim_g = 32

    neural_sde = NeuralSDE(
        h_dim=h_dim,
        param_dim_f=param_dim_f,
        param_dim_g=param_dim_g,
        event_vocab_size=0,
        latent_dim=h_dim
    ).to(device)

    # 5) Setup optimizer
    all_params = list(embedding_net.parameters()) + list(neural_sde.parameters())
    optimizer = Adam(all_params, lr=args.lr)

    best_val_loss = float('inf')
    no_improvement_count = 0

    # 6) Training loop
    for epoch in range(args.num_epochs):
        start_time = time.time()
        train_loss = train_sde_one_epoch(neural_sde, embedding_net, optimizer, train_loader, device)

        # Evaluate
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            neural_sde.eval()
            embedding_net.eval()
            for (input_seq, labels) in val_loader:
                input_seq = input_seq.to(device)
                labels = labels.to(device)
                B = input_seq.size(0)
                seq_len = input_seq.size(1)

                # Timestamps again can be real or dummy
                timestamps_dummy = torch.zeros(B, seq_len, device=device, dtype=torch.long)
                embeddings = embedding_net(raw_features=input_seq, timestamps=timestamps_dummy)
                H0 = embeddings[:, 0, :]
                sde_times = torch.linspace(0, 1, seq_len, device=device)

                sde_func = SDEFunction(neural_sde=neural_sde, past_window=seq_len)
                H_traj = torchsde.sdeint(sde_func, H0, sde_times, method='euler')

                H_final_pred = H_traj[-1]  # [B, latent_dim]
                H_final_true = embeddings[:, -1, :]  # ground truth as final embedding
                loss = nn.functional.mse_loss(H_final_pred, H_final_true)
                val_loss += loss.item() * B
                val_count += B

        val_loss = val_loss / (val_count if val_count > 0 else 1)
        duration = time.time() - start_time
        logging.info(f"Epoch {epoch}/{args.num_epochs} - Train Loss: {train_loss:.4f}, "
                     f"Val Loss: {val_loss:.4f}, Time: {duration:.2f}s")

        # Check improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_count = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict_embedding': embedding_net.state_dict(),
                'model_state_dict_sde': neural_sde.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }, args.model_path)
        else:
            no_improvement_count += 1
            if no_improvement_count >= args.patience:
                logging.info("Early stopping triggered.")
                break

    # Final test evaluation (skipped in detail)
    logging.info("Training completed. Evaluate on test_loader if needed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Production-grade training of a Neural SDE model.")
    parser.add_argument('--stock_id', type=str, default="ADBE", help="Stock ID for data fetch")
    parser.add_argument('--start_date', type=str, default="2020-01-01T00:00:00Z", help="Data start date")
    parser.add_argument('--end_date', type=str, default="2025-12-31T23:59:59Z", help="Data end date")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training")
    parser.add_argument('--sequence_length', type=int, default=30, help="Sequence length for MarketDataset")
    parser.add_argument('--window_step', type=int, default=15, help="Sliding window step")
    parser.add_argument('--latent_dim', type=int, default=64, help="Latent dimension for embeddings + SDE state")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--num_epochs', type=int, default=30, help="Max training epochs")
    parser.add_argument('--train_ratio', type=float, default=0.7, help="Train split ratio")
    parser.add_argument('--val_ratio', type=float, default=0.15, help="Validation split ratio")
    parser.add_argument('--patience', type=int, default=5, help="Early stopping patience")
    parser.add_argument('--model_path', type=str, default='neural_sde_checkpoint.pth', help="Checkpoint path")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    args = parser.parse_args()

    main(args)
