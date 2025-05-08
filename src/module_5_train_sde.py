"""
module_5_train_sde.py
-----------------------------------------
Trains a Neural SDE model on a custom PyTorch Dataset.
"""

import logging
import argparse
import torch
from torch.utils.data import DataLoader, Subset

# If your dataset is in module_4_sequence_dataset, adjust the import accordingly
from module_4_tokenization import MarketSDEDataset
from module_6_training import SDETrainer

def parse_arguments():
    """
    Parse command-line arguments for hyperparameters and environment configs.
    """
    parser = argparse.ArgumentParser(description="Train a Neural SDE model.")
    
    # Data parameters
    parser.add_argument("--tickers", nargs="+", default=["MSFT", "AMZN", "GOOGL"], help="List of ticker symbols to train on.")
    parser.add_argument("--start_date", type=str, default="2020-01-01", help="Data start date (YYYY-MM-DD).")
    parser.add_argument("--end_date", type=str, default="2024-12-31", help="Data end date (YYYY-MM-DD).")
    parser.add_argument("--data_dir", type=str, default="data/indicators", help="Directory where the *with_indicators.parquet files reside.")
    
    # Dataset parameters
    parser.add_argument("--sequence_length", type=int, default=30, help="Number of time steps in each sequence window.")
    parser.add_argument("--window_step", type=int, default=15, help="Step size for the sliding window (default=sequence_length // 2).")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Ratio of dataset for training split (chronological).")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Ratio of dataset for validation split (chronological).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoader.")
    
    # Model/trainer parameters
    parser.add_argument("--token_dim", type=int, default=8, help="Number of features in each token (depends on your pipeline).")
    parser.add_argument("--embed_dim", type=int, default=64, help="Embedding dimension for token embedding.")
    parser.add_argument("--num_stocks", type=int, default=100, help="Max number of stocks recognized by the identity embedding.")
    parser.add_argument("--num_sectors", type=int, default=10, help="Max number of sectors recognized by the identity embedding.")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads for the Transformer encoder.")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of Transformer encoder layers.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability in the Transformer layers.")
    parser.add_argument("--param_dim_f", type=int, default=32, help="Hidden dimension for the drift network in NeuralSDE.")
    parser.add_argument("--param_dim_g", type=int, default=32, help="Hidden dimension for the diffusion network in NeuralSDE.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer.")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs to train.")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience.")
    
    # Model Checkpointing
    parser.add_argument("--checkpoint_path", type=str, default="neural_sde_checkpoint.pth", help="File path to save the model checkpoint.")
    
    args = parser.parse_args()
    return args

def chronological_split(dataset_size, train_ratio, val_ratio):
    """
    Chronologically splits the dataset into train, val, test indices.
    Typically recommended for time series tasks.
    
    :param dataset_size: total number of sequences in the dataset
    :param train_ratio: fraction of data for training
    :param val_ratio: fraction of data for validation
    :return: (train_start, train_end, val_start, val_end, test_start, test_end) indices
    """
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    train_start, train_end = 0, train_size
    val_start, val_end = train_end, train_end + val_size
    test_start, test_end = val_end, val_end + test_size
    
    return (train_start, train_end, val_start, val_end, test_start, test_end)

def main():
    # 0) Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    logger = logging.getLogger(__name__)
    
    # 1) Parse arguments
    args = parse_arguments()
    logger.info("Starting training script with args: %s", args)
    
    # 2) Load dataset
    logger.info("Loading dataset from %s for tickers: %s", args.data_dir, args.tickers)
    dataset = MarketSDEDataset(
        tickers=args.tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        sequence_length=args.sequence_length,
        window_step=args.window_step,
        data_dir=args.data_dir
    )
    dataset_size = len(dataset)
    print(f"Dataset size: {dataset_size}")
    if dataset_size == 0:
        logger.error("Dataset is empty. Exiting.")
        return
    
    # 3) Chronological split
    train_start, train_end, val_start, val_end, test_start, test_end = chronological_split(
        dataset_size, args.train_ratio, args.val_ratio
    )
    indices = list(range(dataset_size))
    
    train_indices = indices[train_start:train_end]
    val_indices   = indices[val_start:val_end]
    test_indices  = indices[test_start:test_end]
    
    train_subset = Subset(dataset, train_indices)
    print(f"Train subset: {len(train_subset)}")
    print(f"train_subset[0]: {len(train_subset[0])}")
    print(f"train_subset[0][0] shape: {train_subset[0][0].shape}")
    val_subset   = Subset(dataset, val_indices)
    test_subset  = Subset(dataset, test_indices)  # if needed
    
    # 4) Initialize DataLoader
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=False)
    print(f"Train loader batch size: {args.batch_size}")
    print(f"Train loader: {len(train_loader)}")
    val_loader   = DataLoader(val_subset,   batch_size=args.batch_size, shuffle=False)
    # test_loader  = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)
    
    logger.info("Train samples: %d, Val samples: %d, Test samples: %d", len(train_subset), len(val_subset), len(test_subset))
    
    # 5) Initialize Trainer
    logger.info("Initializing SDETrainer...")
    trainer = SDETrainer(args)
    
    # 6) Train
    logger.info("Commencing training...")
    trainer.fit(train_loader, val_loader)
    logger.info("Training complete.")
    
    # Potentially evaluate test set:
    # test_loss = trainer.evaluate(test_loader)
    # logger.info(f"Final test loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()
