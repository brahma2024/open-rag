"""
module_5b1_input_processing.py

Purpose:
Provides a high-quality, production-level "Input Processing Layer" for advanced financial time series modeling.
1) Encodes cyclical time features (hour_of_day, day_of_week) with sine/cos expansions.
2) Optionally integrates learned embeddings for discrete event types (e.g., earnings, macro announcements).
3) Concatenates raw features (OHLC-based log returns, advanced volatility/trend/volume/distribution factors) with time/event embeddings.
4) Projects the combined vector into a latent dimension via an MLP (multi-layer perceptron).
5) Outputs the embedding that becomes the input to a Neural SDE or other continuous-time network.

Real-world considerations:
- Flexible dimensioning for new indicators or additional cyclical features.
- Potential for batch normalization, large-batch training, or environment toggles.
- Maintains clarity and short-circuit checks for missing data or event IDs.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict

#######################################
# Time Positional Encoding
#######################################

class TimePositionalEncoding(nn.Module):
    """
    Encodes cyclical time features (in [0,1]) using sine/cosine transformations.
    Typically, these features might be hour_of_day_normalized (0..1),
    day_of_week_normalized (0..1), or others.

    Each cyclical feature f_i is expanded into:
        sin(2*pi * f_i * freq_j), cos(2*pi * f_i * freq_j)
    for j in some set of frequencies.

    A single time feature dimension can be encoded with multiple frequency pairs for multi-scale periodicity.
    """

    def __init__(self, encoding_dims: int = 4, time_feature_count: int = 2):
        """
        Args:
            encoding_dims: Number of sine/cos frequency pairs per time feature.
            time_feature_count: Number of cyclical features (e.g. hour_of_day, day_of_week).
        """
        super().__init__()
        self.encoding_dims = encoding_dims
        self.time_feature_count = time_feature_count

    def forward(self, time_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time_features: Tensor of shape [batch_size, seq_len, time_feature_count],
                           e.g. [B, S, 2] for hour_of_day and day_of_week in normalized form.

        Returns:
            out: [batch_size, seq_len, time_feature_count + 2*(encoding_dims*time_feature_count)]
                 Because for each cyclical time feature, we produce 2*encoding_dims expansions.
        """
        B, S, T = time_features.shape
        if T != self.time_feature_count:
            raise ValueError(f"Expected {self.time_feature_count} cyclical time features, got {T}.")

        # We create the frequencies for each cyclical feature
        # For instance, freq_j = [1, 2, 4, 8] or [2^0, 2^1, ...].
        # We'll do a universal freq set and apply to each time feature.
        # shape [1, 1, encoding_dims]
        freqs = torch.tensor([2**i for i in range(self.encoding_dims)],
                             device=time_features.device, dtype=time_features.dtype).view(1,1,self.encoding_dims)

        # We'll iterate across each cyclical dimension, do sin/cos expansions,
        # and then concat them all.
        expansions = []
        for dim_i in range(self.time_feature_count):
            # shape [B, S, 1]
            feature_i = time_features[..., dim_i:dim_i+1]

            # angle = 2 pi * feature_i * freq
            # => shape [B, S, encoding_dims]
            angle = 2 * math.pi * feature_i * freqs

            sin_enc = torch.sin(angle)  # [B, S, encoding_dims]
            cos_enc = torch.cos(angle)  # [B, S, encoding_dims]
            expansions.append(sin_enc)
            expansions.append(cos_enc)

        # expansions shape: a list of 2*(time_feature_count) items, each [B, S, encoding_dims]
        expansions_cat = torch.cat(expansions, dim=-1)
        # expansions_cat shape: [B, S, encoding_dims*(2*time_feature_count)]

        out = torch.cat([time_features, expansions_cat], dim=-1)
        return out


#######################################
# Input Embedding Network
#######################################

class InputEmbeddingNetwork(nn.Module):
    """
    A robust input processing layer for:
      1) Combining raw feature vectors (OHLC log returns, advanced volatility/trend, distribution features).
      2) Appending cyclical time encodings from TimePositionalEncoding.
      3) Optionally appending event embeddings if event_vocab_size > 0.
      4) Passing the combined vector through an MLP to project into a latent dimension for the Neural SDE.

    Real-world production aspects:
      - Flexible dimensioning
      - Potential dropout, layer norm, etc.
      - Event ID embedding for major events
      - Clear docstrings + typed arguments
    """

    def __init__(self,
                 input_feature_size: int,
                 latent_dim: int,
                 time_feature_count: int = 2,
                 time_encoding_dims: int = 4,
                 hidden_dims: Optional[List[int]] = None,
                 dropout: float = 0.1,
                 event_vocab_size: int = 0,
                 event_embedding_dim: int = 16,
                 use_layernorm: bool = True):
        """
        Args:
            input_feature_size: Number of raw features per time step (already includes advanced feats).
            latent_dim: Dimension of final output for each time step.
            time_feature_count: Number of cyclical time features (e.g., hour_of_day, day_of_week).
            time_encoding_dims: Number of freq pairs used by the TimePositionalEncoding for each cyclical feature.
            hidden_dims: List of MLP layer sizes. If None, defaults to [128, 128].
            dropout: Dropout rate for MLP.
            event_vocab_size: If > 0, enable event embedding. 0 means no event embeddings.
            event_embedding_dim: Dimension of the event embedding vector.
            use_layernorm: If True, apply LayerNorm in each hidden layer.
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128]

        self.time_encoder = TimePositionalEncoding(encoding_dims=time_encoding_dims,
                                                  time_feature_count=time_feature_count)

        self.event_vocab_size = event_vocab_size
        self.event_embedding_dim = event_embedding_dim
        if self.event_vocab_size > 0:
            self.event_embedding = nn.Embedding(event_vocab_size, event_embedding_dim)
        else:
            self.event_embedding = None

        # We'll figure out the final input dimension to the MLP
        # raw_features: input_feature_size
        # cyclical expansions: for each cyclical dimension: 2 + 2*(time_encoding_dims)
        #   (2 is the original cyclical feature dimension, but we'll add them as a separate dimension)
        # Actually we are appending them at the final step, so let's do it systematically:
        # time_encoded_dim = time_feature_count + 2*(time_feature_count * time_encoding_dims)
        # but note that the time_encoder returns shape: [B, S, time_feature_count + 2*(time_feature_count * encoding_dims)] 
        # We'll call that final_time_dim
        final_time_dim = time_feature_count + 2*(time_feature_count * time_encoding_dims)

        event_dim = event_embedding_dim if event_vocab_size > 0 else 0

        # MLP input dimension
        self.mlp_input_dim = input_feature_size + final_time_dim + event_dim

        # Build MLP
        layers = []
        in_dim = self.mlp_input_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(in_dim, hd))
            if use_layernorm:
                layers.append(nn.LayerNorm(hd))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hd
        layers.append(nn.Linear(in_dim, latent_dim))
        self.mlp = nn.Sequential(*layers)

    def _extract_time_features(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Convert timestamps [B, S] in ms or sec to cyclical features like hour_of_day_norm, day_of_week_norm.
        Returns [B, S, time_feature_count], e.g. 2 dims: [hour_of_day_norm, day_of_week_norm]
        """
        device = timestamps.device
        dtype = timestamps.dtype

        # For demonstration, we assume we want hour_of_day and day_of_week from millisecond timestamps
        # This is the same logic as before
        sec_per_day = 24 * 3600
        ms_per_day = sec_per_day * 1000

        hour_of_day = ((timestamps % ms_per_day) // (3600*1000)).float()
        hour_of_day_norm = hour_of_day / 24.0

        days_since_epoch = (timestamps // ms_per_day).long()
        day_of_week = (days_since_epoch % 7).float()
        day_of_week_norm = day_of_week / 7.0

        cyc_feats = torch.stack([hour_of_day_norm, day_of_week_norm], dim=-1)
        return cyc_feats.to(device=device, dtype=dtype)

    def forward(self,
                raw_features: torch.Tensor,
                timestamps: torch.Tensor,
                event_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            raw_features: [B, S, input_feature_size], e.g. advanced feats from modules 4a/4b (log_return, figarch_vol, trend, etc.)
            timestamps: [B, S], integer timestamps in ms from epoch (used for cyclical time features).
            event_ids: [B, S], optional integer-coded events.

        Returns:
            latent: [B, S, latent_dim]
        """
        B, S, F = raw_features.shape
        # 1) Extract cyclical time features
        cyc_feats = self._extract_time_features(timestamps)  # [B, S, 2]
        # 2) Sine/cos expansions
        cyc_encoded = self.time_encoder(cyc_feats)  # [B, S, 2 + expansions]

        # 3) Possibly get event embedding
        if self.event_embedding is not None and event_ids is not None:
            # event_ids: [B, S]
            event_emb = self.event_embedding(event_ids)  # [B, S, event_embedding_dim]
        else:
            event_emb = None

        # 4) Concatenate raw_features + cyc_encoded + event_emb
        if event_emb is not None:
            combined = torch.cat([raw_features, cyc_encoded, event_emb], dim=-1)  # [B, S, ???]
        else:
            combined = torch.cat([raw_features, cyc_encoded], dim=-1)

        # 5) MLP
        latent = self.mlp(combined)  # [B, S, latent_dim]
        return latent


# Example usage block for local testing
if __name__ == "__main__":
    # Suppose we have a batch of data with advanced features:
    # e.g. input_feature_size = 10 => [log_return, figarch_vol, trend_factor, distribution_kurt, ...]
    batch_size = 16
    seq_len = 30
    input_feature_size = 10
    latent_dim = 64
    event_vocab_size = 5
    event_embedding_dim = 16

    # Create random input
    raw_feats = torch.randn(batch_size, seq_len, input_feature_size)
    # Timestamps in ms
    timestamps = torch.randint(low=1670000000000, high=1679990400000, size=(batch_size, seq_len))
    # Random event IDs
    event_ids = torch.randint(low=0, high=event_vocab_size, size=(batch_size, seq_len))

    input_net = InputEmbeddingNetwork(input_feature_size=input_feature_size,
                                      latent_dim=latent_dim,
                                      time_feature_count=2,
                                      time_encoding_dims=4,
                                      hidden_dims=[128, 128],
                                      dropout=0.1,
                                      event_vocab_size=event_vocab_size,
                                      event_embedding_dim=event_embedding_dim,
                                      use_layernorm=True)

    output_latent = input_net(raw_feats, timestamps, event_ids)
    print(f"Output latent shape: {output_latent.shape}")
    # Expecting [16, 30, 64]
