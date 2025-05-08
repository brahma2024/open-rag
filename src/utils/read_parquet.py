"""
read_parquet.py
-----------------------------------------
Read and print the first few rows of all Parquet files in a given directory. 
"""

import os
import pandas as pd

def read_parquet_files(directory: str) -> None:
    """
    Read and print the first few rows of all Parquet files in the given directory.
    """
    if not os.path.isdir(directory):
        print(f"Directory {directory} does not exist.")
        return

    parquet_files = [f for f in os.listdir(directory) if f.endswith('.parquet')]
    
    if not parquet_files:
        print("No Parquet files found.")
        return

    for file in parquet_files:
        file_path = os.path.join(directory, file)
        try:
            df = pd.read_parquet(file_path)
            print(f"\nFile: {file}")
            print(df.head())
            print(df.columns)
        except (pd.errors.EmptyDataError, IOError, pd.errors.ParserError) as e:
            print(f"Error reading {file_path}: {e}")

if __name__ == "__main__":
    # Adjust the RAW_DATA_DIR path if necessary.
    current_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    # Assume the parquet files are in data/raw relative to the project root.
    # data_dir = os.path.join(current_dir, "data", "raw") # read from data/raw
    # data_dir = os.path.join(current_dir, "data", "processed") # read from data/processed
    data_dir = os.path.join(current_dir, "data", "indicators") # read from data/processed
    read_parquet_files(data_dir)