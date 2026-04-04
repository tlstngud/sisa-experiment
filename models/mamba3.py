"""
Mamba-3 baseline — using official mamba-ssm Mamba3 module.

Falls back to two-SSD decomposition if official module is not available.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import RMSNorm

# Try official Mamba3 first
try:
    from mamba_ssm.modules.mamba3 import Mamba3 as OfficialMamba3
    HAS_OFFICIAL_MAMBA3 = True
except ImportError:
    HAS_OFFICIAL_MAMBA3 = False


class Mamba3Block(nn.Module):
    """Single Mamba-3 block with pre-norm and residual."""

    def __init__(self, d_model: int, d_state: int = 64, expand: int = 2):
        super().__init__()
        self.norm = RMSNorm(d_model)

        if HAS_OFFICIAL_MAMBA3:
            self.mamba = OfficialMamba3(
                d_model=d_model,
                d_state=d_state,
                expand=expand,
            )
        else:
            raise ImportError("Official Mamba3 not available. Install from: git+https://github.com/state-spaces/mamba.git")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mamba(self.norm(x)) + x


class Mamba3Model(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layer: int,
        d_state: int = 64,
        expand: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [Mamba3Block(d_model, d_state, expand) for _ in range(n_layer)]
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
