# module_5a_sde.py
# version: 1.1
# purpose: Production-grade Neural SDE architecture for financial time series, featuring:
#   - LTC-based param adaptation
#   - Historical attention
#   - Drift/Diffusion sub-networks

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math
import torchsde

################################################################################
# LTC Layer
################################################################################
class LTCLayer(nn.Module):
    """
    Liquid Time-Constant (LTC) layer, producing dynamic parameters for drift/diff networks
    based on current state + context.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, h_t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h_t, context], dim=-1)  # [batch, h_dim + context_dim]
        return self.mlp(x)

################################################################################
# Multi-Head Attention Over Past States
################################################################################
class HistoricalAttention(nn.Module):
    """
    Scaled-down multi-head self-attention that uses current state as query
    and entire past state sequence as key/value to yield a context vector.
    """
    def __init__(self, latent_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=latent_dim, 
                                          num_heads=num_heads, 
                                          dropout=dropout, 
                                          batch_first=True)
        self.layernorm = nn.LayerNorm(latent_dim)

    def forward(self, current_h: torch.Tensor, past_states: torch.Tensor) -> torch.Tensor:
        """
        current_h: [batch, 1, latent_dim]
        past_states: [batch, seq_len, latent_dim]
        return: context -> [batch, 1, latent_dim]
        """
        attn_output, _ = self.attn(query=current_h, key=past_states, value=past_states)
        context = self.layernorm(current_h + attn_output)
        return context

################################################################################
# Drift & Diffusion Param Networks
################################################################################
class DriftDiffusionParamNet(nn.Module):
    """
    Produces drift or diffusion outputs from:
      - Current state h_t
      - LTC parameters (f_params, g_params)
    """
    def __init__(self, h_dim: int, param_dim: int, output_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(h_dim + param_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, h_t: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h_t, params], dim=-1)
        return self.mlp(x)

################################################################################
# Neural SDE
################################################################################
class NeuralSDE(nn.Module):
    """
    Composes:
      - Historical attention for context
      - LTC layers to produce dynamic parameters for drift/diffusion
      - f_net & g_net for final drift/diff calculations
    """
    def __init__(self,
                 h_dim: int,
                 param_dim_f: int,
                 param_dim_g: int,
                 event_vocab_size: int = 0,
                 latent_dim: int = 64):
        super().__init__()
        self.h_dim = h_dim
        self.latent_dim = latent_dim
        # LTC
        self.ltc_f = LTCLayer(input_dim=h_dim + latent_dim, output_dim=param_dim_f)
        self.ltc_g = LTCLayer(input_dim=h_dim + latent_dim, output_dim=param_dim_g)
        # Param Networks
        self.f_net = DriftDiffusionParamNet(h_dim=h_dim, param_dim=param_dim_f, output_dim=h_dim)
        self.g_net = DriftDiffusionParamNet(h_dim=h_dim, param_dim=param_dim_g, output_dim=h_dim)
        # Attention
        self.attention = HistoricalAttention(latent_dim=latent_dim)

    def forward(self, h_t: torch.Tensor, past_states: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        h_t: [batch, h_dim]  current state
        past_states: [batch, seq_len, latent_dim] for attention
        Returns: drift [batch, h_dim], diffusion [batch, h_dim]
        """
        # Expand current state for attention
        curr_h_expanded = h_t.unsqueeze(1)  # [batch, 1, h_dim]
        # If h_dim != latent_dim, add a linear projection
        context = self.attention(curr_h_expanded, past_states).squeeze(1)  # [batch, latent_dim]

        # LTC param productions
        f_params = self.ltc_f(h_t, context)  # [batch, param_dim_f]
        g_params = self.ltc_g(h_t, context)  # [batch, param_dim_g]

        # Drift & Diffusion
        drift = self.f_net(h_t, f_params)
        diffusion = self.g_net(h_t, g_params)
        return drift, diffusion

################################################################################
# SDEFunction for torchsde integration
################################################################################
class SDEFunction(torchsde.SDEIto):
    """
    The SDE function to pass to torchsde.sdeint.
    """
    def __init__(self, neural_sde: NeuralSDE, past_window: int = 30):
        super().__init__()
        self.neural_sde = neural_sde
        self.past_window = past_window
        self.cached_past_states = None  # or handle externally

    def f(self, t: float, y: torch.Tensor) -> torch.Tensor:
        """
        Drift: f(t, y)
        y: [batch, h_dim]
        """
        # Retrieve or build past_states
        past_states = self.get_past_states(t, y)
        drift, _ = self.neural_sde(h_t=y, past_states=past_states)
        return drift

    def g(self, t: float, y: torch.Tensor) -> torch.Tensor:
        """
        Diffusion: g(t, y)
        y: [batch, h_dim]
        Must return shape [batch, h_dim, Brownian_dim] or a diagonal matrix
        """
        past_states = self.get_past_states(t, y)
        _, diffusion = self.neural_sde(h_t=y, past_states=past_states)
        # Diagonal diffusion: shape => [batch, h_dim, h_dim]
        return torch.diag_embed(diffusion)

    def get_past_states(self, t: float, y: torch.Tensor) -> torch.Tensor:
        """
        This is domain-specific. You might store a rolling window of states
        or replicate them. For demonstration, we return a dummy zero
        or anything that matches [batch, seq_len, latent_dim].
        """
        # Example: zero placeholder
        batch_size, h_dim = y.shape
        # Suppose we want seq_len= self.past_window
        # If h_dim == self.neural_sde.latent_dim:
        return torch.zeros(batch_size, self.past_window, self.neural_sde.latent_dim,
                           device=y.device, dtype=y.dtype)
