# module_12_output_head.py
# -----------------------------------------
# Function:
#   A small module that maps from embed_dim -> label_dim.
#   For multi-task or multi-dimensional output.
# """
# Why a small MLP?
# A single linear layer could suffice, but an extra hidden layer can learn non-linear transformations if your label tasks are more complex.
# Adjust hidden_dim or add dropout if needed.
#

import torch
import torch.nn as nn

class FinalHead(nn.Module):
    """
    A simple final head mapping from embed_dim to label_dim (e.g. 5).
    Optionally includes a hidden layer or non-linearities.
    """
    def __init__(self, embed_dim: int, label_dim: int):
        super(FinalHead, self).__init__()
        # Example: a small MLP with one hidden layer
        hidden_dim = embed_dim // 2  # or choose your own
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, label_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, embed_dim]
        return self.net(x)  # [B, label_dim]
