"""PatchTST-inspired tire degradation model.

Architecture overview
---------------------
1. Left-pad ``x`` so that ``seq_len_padded`` is a multiple of ``patch_len``
2. Reshape: (batch, n_patches, patch_len * n_features)
3. Linear patch embedding → d_model per patch
4. Add a learnable positional embedding (one vector per patch position)
5. Standard ``nn.TransformerEncoder`` with ``batch_first=True``, ``norm_first=True``
6. Flatten: (batch, n_patches * d_model)
7. LayerNorm → Linear(n_patches*d_model, d_model) → GELU →
   Linear(d_model, 1) → scalar

Reference: Nie et al. "A Time Series is Worth 64 Words" (ICLR 2023).
We use patch tokenisation only — the channel-independence trick from the paper
is not applied here because features are semantically heterogeneous.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class PatchTSTTireModel(nn.Module):
    """PatchTST-inspired model for per-lap tire degradation prediction.

    Divides the input sequence into non-overlapping patches and processes
    them as tokens with a Transformer encoder.  Left-padding ensures that
    the most recent laps are always aligned to the right edge of the last
    patch, which is the most informative position for regression.

    Args:
        n_features: Number of input features per time step.
        seq_len: Input sequence length (number of laps in the window).
        patch_len: Number of time steps per patch token.
        d_model: Patch embedding dimension.
        nhead: Number of attention heads in the Transformer encoder.
        num_layers: Number of Transformer encoder layers.
        dropout: Dropout probability used in the Transformer and the head.
    """

    def __init__(
        self,
        n_features: int,
        seq_len: int = 10,
        patch_len: int = 5,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self._patch_len = patch_len
        self._n_features = n_features
        self._seq_len = seq_len

        # Compute n_patches once so the head size is fixed at init time.
        self._n_patches: int = math.ceil(seq_len / patch_len)

        self._patch_embed = nn.Linear(patch_len * n_features, d_model)
        # Positional embedding: one vector per patch position
        self._pos_embed = nn.Parameter(torch.zeros(1, self._n_patches, d_model))
        nn.init.trunc_normal_(self._pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self._transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        flat_dim = self._n_patches * d_model
        self._head = nn.Sequential(
            nn.LayerNorm(flat_dim),
            nn.Linear(flat_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Run the PatchTST forward pass.

        Args:
            x: Input tensor of shape ``(batch, seq_len, n_features)``.

        Returns:
            Predicted lap-time delta, shape ``(batch,)``.
        """
        batch = x.size(0)
        padded_len = self._n_patches * self._patch_len
        pad_size = padded_len - self._seq_len

        # Left-pad with zeros so recent laps align to the right of each patch.
        if pad_size > 0:
            padding = torch.zeros(batch, pad_size, self._n_features, device=x.device, dtype=x.dtype)
            x = torch.cat([padding, x], dim=1)

        # (batch, padded_len, n_features) → (batch, n_patches, patch_len*n_features)
        x = x.reshape(batch, self._n_patches, self._patch_len * self._n_features)

        # Patch embedding + positional embedding
        out: Tensor = self._patch_embed(x) + self._pos_embed

        # Transformer encoder: (batch, n_patches, d_model)
        out = self._transformer(out)

        # Flatten and project to scalar
        out = out.reshape(batch, -1)
        result: Tensor = self._head(out).squeeze(-1)
        return result
