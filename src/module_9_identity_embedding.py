"""
module_9_identity_embedding.py
-----------------------------------------
Function:
In this module, we will define the IdentityEmbedding class which learns stock + sector embeddings, then merges them into a single vector.
"""

import torch
import torch.nn as nn

class IdentityEmbedding(nn.Module):
    """
    Learns stock + sector embeddings, then merges them into a single vector.
    """
    def __init__(self, num_stocks: int, num_sectors: int, embed_dim: int):
        super(IdentityEmbedding, self).__init__()
        self.stock_emb = nn.Embedding(num_stocks, embed_dim)
        self.sector_emb = nn.Embedding(num_sectors, embed_dim)
        self.comb_layer = nn.Linear(embed_dim * 2, embed_dim)
    
    def forward(self, stock_ids: torch.Tensor, sector_ids: torch.Tensor) -> torch.Tensor:
        """
        :param stock_ids: shape [batch_size]
        :param sector_ids: shape [batch_size]
        :return: shape [batch_size, embed_dim]
        """
        s_emb = self.stock_emb(stock_ids)   # [B, embed_dim]
        sec_emb = self.sector_emb(sector_ids)  # [B, embed_dim]
        combined = torch.cat([s_emb, sec_emb], dim=-1)  # [B, 2*embed_dim]
        out = self.comb_layer(combined)  # [B, embed_dim]
        return out
