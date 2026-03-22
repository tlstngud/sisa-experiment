"""
Mamba-2 baseline wrapper.
Uses the official mamba-ssm package (mamba_ssm.modules.mamba2.Mamba2).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import RMSNorm


class Mamba2Block(nn.Module):
    """Single Mamba-2 block with pre-norm and residual."""

    def __init__(self, d_model: int, d_state: int = 64, expand: int = 2, d_conv: int = 4):
        super().__init__()
        self.norm = RMSNorm(d_model)

        from mamba_ssm.modules.mamba2 import Mamba2

        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mamba(self.norm(x)) + x


class Mamba2Model(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layer: int,
        d_state: int = 64,
        expand: int = 2,
        d_conv: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [Mamba2Block(d_model, d_state, expand, d_conv) for _ in range(n_layer)]
        )
        self.final_norm = RMSNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        std = 0.02
        nn.init.normal_(self.embed.weight, std=std)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)
        logits = F.linear(x, self.embed.weight)  # tied embedding
        return logits
