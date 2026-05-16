"""TCN + GRU tire degradation model.

Architecture overview
---------------------
1. Linear input projection: n_features → d_model
2. Transpose to (batch, d_model, seq_len) for 1-D convolutions
3. N causal TCN blocks with exponentially growing dilations [1, 2, 4, 8, …]
4. Transpose back to (batch, seq_len, d_model)
5. GRU(d_model → gru_hidden, batch_first=True) — captures temporal order that
   dilated convolutions can miss at moderate seq_len values
6. Take the final hidden state (last time-step output)
7. Two-layer MLP head: gru_hidden → gru_hidden//2 → 1
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class _CausalConv1d(nn.Module):
    """Causal 1-D convolution with no look-ahead into the future.

    Standard ``nn.Conv1d`` with symmetric padding would leak future context.
    We instead add ``(kernel_size - 1) * dilation`` zeros to **both** sides
    and then trim the right tail so the output length matches the input.

    Args:
        in_channels: Number of input feature maps.
        out_channels: Number of output feature maps.
        kernel_size: Convolutional kernel size.
        dilation: Dilation factor.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self._pad = (kernel_size - 1) * dilation
        self._conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self._pad,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply causal convolution.

        Args:
            x: Input tensor of shape ``(batch, channels, seq_len)``.

        Returns:
            Output tensor of shape ``(batch, out_channels, seq_len)``.
        """
        out = self._conv(x)
        # Trim the right tail that was produced by right-side padding,
        # ensuring strict causality — position t sees only t-1, t-2, …
        return torch.narrow(out, 2, 0, x.size(2))


class _TCNBlock(nn.Module):
    """One residual TCN block: two causal convolutions with GELU and dropout.

    The residual connection uses a LayerNorm on the *sum* (post-LN residual)
    which keeps the gradient scale healthy across many stacked blocks.

    Args:
        channels: Number of feature map channels (d_model).
        kernel_size: Convolutional kernel size.
        dilation: Dilation factor for both convolutions.
        dropout: Dropout probability applied after each GELU.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self._conv1 = _CausalConv1d(channels, channels, kernel_size, dilation)
        self._conv2 = _CausalConv1d(channels, channels, kernel_size, dilation)
        self._act = nn.GELU()
        self._drop = nn.Dropout(dropout)
        self._norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        """Apply two causal convolutions then add the residual.

        Args:
            x: Input tensor of shape ``(batch, channels, seq_len)``.

        Returns:
            Output tensor of the same shape.
        """
        out: Tensor = self._drop(self._act(self._conv1(x)))
        out = self._drop(self._act(self._conv2(out)))
        # LayerNorm expects (batch, seq_len, channels) — transpose in/out
        residual: Tensor = (x + out).permute(0, 2, 1)
        residual = self._norm(residual).permute(0, 2, 1)
        return residual


class TCNGRUTireModel(nn.Module):
    """TCN + GRU sequence model for per-lap tire degradation prediction.

    The TCN front-end captures multi-scale local patterns across the stint
    history; the GRU layer then integrates long-range sequential context
    before a small MLP head regresses to a scalar lap-time delta.

    Args:
        n_features: Number of input features per time step.
        d_model: Hidden dimension after input projection and inside TCN blocks.
        n_tcn_layers: Number of TCN blocks; dilation doubles each block
            (1, 2, 4, 8, …).
        gru_hidden: Hidden state size of the GRU layer.
        kernel_size: Kernel size for all causal convolutions.
        dropout: Dropout probability applied inside each TCN block and before
            the MLP head.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        n_tcn_layers: int = 4,
        gru_hidden: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self._input_proj = nn.Linear(n_features, d_model)
        self._tcn_blocks = nn.ModuleList(
            [
                _TCNBlock(d_model, kernel_size, dilation=2**i, dropout=dropout)
                for i in range(n_tcn_layers)
            ]
        )
        self._gru = nn.GRU(d_model, gru_hidden, batch_first=True)
        self._drop = nn.Dropout(dropout)
        self._head = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden // 2),
            nn.GELU(),
            nn.Linear(gru_hidden // 2, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Run the TCN+GRU forward pass.

        Args:
            x: Input tensor of shape ``(batch, seq_len, n_features)``.

        Returns:
            Predicted lap-time delta, shape ``(batch,)``.
        """
        # (batch, seq_len, n_features) → (batch, seq_len, d_model)
        out: Tensor = self._input_proj(x)
        # TCN expects (batch, d_model, seq_len)
        out = out.permute(0, 2, 1)
        for block in self._tcn_blocks:
            out = block(out)
        # GRU expects (batch, seq_len, d_model)
        out = out.permute(0, 2, 1)
        _, hidden = self._gru(out)
        # hidden: (1, batch, gru_hidden) — take the single-layer state
        last: Tensor = self._drop(hidden.squeeze(0))
        result: Tensor = self._head(last).squeeze(-1)
        return result
