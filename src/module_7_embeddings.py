"""
module_7_embeddings.py
-----------------------------------------
Function:
In this module, we will define the TokenEmbedding class which embeds tokens into a higher-dimensional space.
"""

import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    """
    Embeds tokens into a higher-dimensional space.
    
    Args:
      input_dim (int): Dimension of input token.
      embed_dim (int): Dimension of the output embedding.
    """
    def __init__(self, input_dim: int, embed_dim: int):
        super(TokenEmbedding, self).__init__()
        self.fc = nn.Linear(input_dim, embed_dim)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        # x: [batch_size, seq_length, input_dim]
        x = self.fc(x)  # [batch_size, seq_length, embed_dim]
        x = self.activation(x)
        return x
