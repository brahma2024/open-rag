"""
module_8_positional_encoding.py
-----------------------------------------
Function:
In this module, we will define the PositionalEncoding class which injects positional information into embeddings using sine/cosine functions.
"""


import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Injects positional information into embeddings using sine/cosine functions.
    
    Args:
      d_model (int): Embedding dimension.
      max_len (int): Maximum sequence length.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
          x (Tensor): [batch_size, seq_length, d_model]
        Returns:
          Tensor: Positional encoded embeddings.
        """
        return x + self.pe[:, :x.size(1)]
