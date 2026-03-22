import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .common import RMSNorm, SwiGLU, precompute_rope_freqs, apply_rotary_emb


class TransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_ff: int):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head

        # Attention
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        # Norms
        self.attn_norm = RMSNorm(d_model)
        self.ffn_norm = RMSNorm(d_model)

        # FFN
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        B, L, D = x.shape
        H, d_h = self.n_head, self.d_head

        # === Attention ===
        residual = x
        h = self.attn_norm(x)

        q = self.W_Q(h).view(B, L, H, d_h).transpose(1, 2)
        k = self.W_K(h).view(B, L, H, d_h).transpose(1, 2)
        v = self.W_V(h).view(B, L, H, d_h).transpose(1, 2)

        q, k = apply_rotary_emb(q, k, cos[:L], sin[:L])

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, L, D)
        x = self.W_O(y) + residual

        # === FFN ===
        residual = x
        x = self.ffn(self.ffn_norm(x)) + residual

        return x


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_head: int,
        n_layer: int,
        d_ff: int,
        max_seq_len: int = 2048,
        grad_checkpoint: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.grad_checkpoint = grad_checkpoint
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [TransformerLayer(d_model, n_head, d_ff) for _ in range(n_layer)]
        )
        self.final_norm = RMSNorm(d_model)

        # Precompute RoPE
        d_head = d_model // n_head
        cos, sin = precompute_rope_freqs(d_head, max_seq_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # Init
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

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (B, L) long
        Returns: logits (B, L, vocab_size)
        """
        x = self.embed(input_ids)

        for layer in self.layers:
            if self.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, self.rope_cos, self.rope_sin,
                    use_reentrant=False,
                )
            else:
                x = layer(x, self.rope_cos, self.rope_sin)

        x = self.final_norm(x)
        logits = F.linear(x, self.embed.weight)  # tied embedding
        return logits
