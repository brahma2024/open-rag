import numpy as np
import pandas as pd

def tokenize_data(df: pd.DataFrame) -> np.ndarray:
    """
    Converts a processed DataFrame into tokens.
    Required columns:
      'log_return_prevclose_to_close', 'log_return_prevclose_to_open',
      'log_return_open_to_close', 'log_return_open_to_high', 'log_return_open_to_low',
      'S_t', 'L_t', 'IsOpen'
      
    Parameters:
      df (pd.DataFrame): Processed market data.
      
    Returns:
      np.ndarray: Tokens of shape [num_timesteps, token_dim]
    """
    required_cols = ['log_return_prevclose_to_close', 'log_return_prevclose_to_open',
                     'log_return_open_to_close', 'log_return_open_to_high', 'log_return_open_to_low',
                     'S_t', 'L_t', 'IsOpen']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    tokens = df[required_cols].values.astype(np.float32)
    return tokens
