"""
module_10_sequence_model.py
-----------------------------------------
Function:
In this module, we will define the SequenceModel class which is a Transformer encoder for sequence modeling.
"""

import torch
import torch.nn as nn

class SequenceModel(nn.Module):
    """
    Transformer encoder for sequence modeling.
    """
    def __init__(self, embed_dim: int, num_heads: int, num_layers: int, dropout: float = 0.1):
        super(SequenceModel, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [batch_size, seq_length, embed_dim]
        :return: [batch_size, seq_length, embed_dim]
        """
        # PyTorch Transformer expects [seq_len, batch_size, d_model]
        x = x.transpose(0, 1)  # [seq_length, batch_size, embed_dim]
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)  # [batch_size, seq_length, embed_dim]
        return x
