"""
Hybrid architectures: Mamba front-end + Attention/SISA back-end.

Mamba+Transformer: Mamba layers compress first, Transformer layers retrieve/decide
Mamba+SISA: Mamba layers compress first, SISA layers retrieve with importance guidance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import RMSNorm, SwiGLU, precompute_rope_freqs
from .transformer import TransformerLayer
from .sisa import SISALayer
from .mamba2 import Mamba2Block


class HybridModel(nn.Module):
    """
    Hybrid model: Mamba front layers + Attention/SISA back layers.

    Args:
        backend: 'transformer' or 'sisa'
        n_mamba: number of Mamba layers (front)
        n_attn: number of Transformer/SISA layers (back)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_head: int,
        n_mamba: int,
        n_attn: int,
        d_ff: int,
        d_state_mamba: int = 64,
        expand_mamba: int = 2,
        d_conv_mamba: int = 4,
        d_state_sisa: int = 32,
        max_seq_len: int = 2048,
        backend: str = 'transformer',  # 'transformer' or 'sisa'
    ):
        super().__init__()
        self.d_model = d_model
        self.backend = backend

        self.embed = nn.Embedding(vocab_size, d_model)

        # Front: Mamba layers
        self.mamba_layers = nn.ModuleList([
            Mamba2Block(d_model, d_state_mamba, expand_mamba, d_conv_mamba)
            for _ in range(n_mamba)
        ])

        # Back: Transformer or SISA layers
        d_head = d_model // n_head
        if backend == 'transformer':
            self.attn_layers = nn.ModuleList([
                TransformerLayer(d_model, n_head, d_ff)
                for _ in range(n_attn)
            ])
        elif backend == 'sisa':
            self.attn_layers = nn.ModuleList([
                SISALayer(d_model, n_head, d_state_sisa, d_ff)
                for _ in range(n_attn)
            ])
        else:
            raise ValueError(f"Unknown backend: {backend}")

        self.final_norm = RMSNorm(d_model)

        # RoPE for attention layers
        cos, sin = precompute_rope_freqs(d_head, max_seq_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self._init_weights()

    def _init_weights(self):
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=std)

        # Re-apply SISA-specific inits
        if self.backend == 'sisa':
            for layer in self.attn_layers:
                nn.init.constant_(layer.alpha_proj.bias, -5.0)
                nn.init.normal_(layer.theta_proj.weight, std=0.01)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)

        # Front: Mamba
        for layer in self.mamba_layers:
            x = layer(x)

        # Back: Transformer or SISA
        for layer in self.attn_layers:
            if self.backend == 'transformer':
                x = layer(x, self.rope_cos, self.rope_sin)
            elif self.backend == 'sisa':
                x = layer(x, self.rope_cos, self.rope_sin)

        x = self.final_norm(x)
        logits = F.linear(x, self.embed.weight)  # tied embedding
        return logits
