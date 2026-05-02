"""
SSM-Informed Softmax Attention (SISA)

Core equation:
    score_ij = q_i^T k_j / sqrt(d_h)  +  lambda * C_bar_i^T * B_bar_j

Implemented via augmented Q/K:
    Q_hat = [q,  s * C_bar],  K_hat = [k,  s * B_bar]
    s = d_h^{1/4} * sqrt(lambda)
    Y = SDPA(Q_hat, K_hat, V, scale=1/sqrt(d_h), is_causal=True)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .common import RMSNorm, SwiGLU, precompute_rope_freqs, apply_rotary_emb, apply_ssm_rope


class SISALayer(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_state: int, d_ff: int):
        super().__init__()
        assert d_model % n_head == 0
        assert d_state % 2 == 0  # RoPE needs even d_state

        self.n_head = n_head
        self.d_head = d_model // n_head
        self.d_state = d_state
        self.d_model = d_model

        # --- Attention projections ---
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        # --- SSM projections ---
        self.W_B = nn.Linear(d_model, n_head * d_state, bias=False)
        self.W_C = nn.Linear(d_model, n_head * d_state, bias=False)

        # --- SSM dynamics ---
        self.alpha_proj = nn.Linear(d_model, n_head, bias=True)
        nn.init.constant_(self.alpha_proj.bias, -5.0)  # half-life ~100 tokens

        self.theta_proj = nn.Linear(d_model, n_head * (d_state // 2), bias=False)
        nn.init.normal_(self.theta_proj.weight, std=0.01)

        # --- lambda: per-head, positive via softplus ---
        self.lambda_raw = nn.Parameter(torch.full((n_head,), -1.0))

        # --- Norms ---
        self.attn_norm = RMSNorm(d_model)
        self.ffn_norm = RMSNorm(d_model)

        # --- FFN ---
        self.ffn = SwiGLU(d_model, d_ff)

        # --- Monitoring buffers (populated during forward, read in get_monitor_metrics) ---
        self._last_g: torch.Tensor | None = None
        self._last_c: torch.Tensor | None = None
        self._last_log_alpha: torch.Tensor | None = None
        self._last_lambda: torch.Tensor | None = None

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """
        x: (B, L, D)
        cos, sin: positional RoPE tables (L, d_head//2)
        Returns: (B, L, D)
        """
        B, L, D = x.shape
        H = self.n_head
        d_h = self.d_head
        d_s = self.d_state

        # ═══════ Attention Sublayer ═══════
        residual = x
        x_norm = self.attn_norm(x)

        # --- Q, K, V ---
        q = self.W_Q(x_norm).view(B, L, H, d_h).transpose(1, 2)  # (B, H, L, d_h)
        k = self.W_K(x_norm).view(B, L, H, d_h).transpose(1, 2)
        v = self.W_V(x_norm).view(B, L, H, d_h).transpose(1, 2)

        # Positional RoPE — q, k channels ONLY
        q, k = apply_rotary_emb(q, k, cos[:L], sin[:L])

        # --- SSM projections ---
        B_ssm = self.W_B(x_norm).view(B, L, H, d_s).transpose(1, 2)  # (B, H, L, d_state)
        C_ssm = self.W_C(x_norm).view(B, L, H, d_s).transpose(1, 2)

        # --- Decay g (FP32, cumsum along L) ---
        log_alpha = -F.softplus(self.alpha_proj(x_norm))  # (B, L, H)
        log_alpha = log_alpha.transpose(1, 2).float()  # (B, H, L)
        g = torch.cumsum(log_alpha, dim=2)  # (B, H, L)

        # --- Phase Phi (FP32, cumsum along L) ---
        theta = self.theta_proj(x_norm)  # (B, L, H * d_state//2)
        theta = theta.view(B, L, H, d_s // 2).permute(0, 2, 1, 3).float()  # (B, H, L, d_s//2)
        Phi = torch.cumsum(theta, dim=2)  # (B, H, L, d_s//2)

        # --- Sequence-global c offset (minimax) ---
        c = (g.amax(dim=2, keepdim=True) + g.amin(dim=2, keepdim=True)) / 2  # (B, H, 1)

        # --- Store monitoring tensors (no .item() calls — compile-safe) ---
        self._last_g = g.detach()
        self._last_c = c.detach()
        self._last_log_alpha = log_alpha.detach()

        # --- Apply decay scaling ---
        g_minus_c = (g - c).unsqueeze(-1)  # (B, H, L, 1)
        # Clamp to prevent bf16 overflow: exp(11) ≈ 59874 < bf16 max (65504)
        g_minus_c = g_minus_c.clamp(-11.0, 11.0)
        C_bar = C_ssm.float() * torch.exp(g_minus_c)   # (B, H, L, d_state)
        B_bar = B_ssm.float() * torch.exp(-g_minus_c)

        # --- Data-dependent RoPE on SSM channels ---
        C_bar = apply_ssm_rope(C_bar, Phi).to(x.dtype)
        B_bar = apply_ssm_rope(B_bar, Phi).to(x.dtype)

        # ═══════ Augmented Q/K ═══════
        lam = F.softplus(self.lambda_raw)  # (H,)
        self._last_lambda = lam.detach()

        s = (d_h ** 0.25) * torch.sqrt(lam)  # (H,)
        s = s.view(1, H, 1, 1)  # broadcast

        Q_aug = torch.cat([q, s * C_bar], dim=-1)  # (B, H, L, d_h + d_state)
        K_aug = torch.cat([k, s * B_bar], dim=-1)  # (B, H, L, d_h + d_state)

        # ═══════ Single Attention ═══════
        # Use memory-efficient attention backend (handles Q/K dim != V dim)
        with torch.nn.attention.sdpa_kernel([
            torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
            torch.nn.attention.SDPBackend.MATH,
        ]):
            Y = F.scaled_dot_product_attention(
                Q_aug, K_aug, v,
                scale=1.0 / math.sqrt(d_h),  # MUST be d_h, not augmented_dim
                is_causal=True,
            )  # (B, H, L, d_h)

        Y = Y.transpose(1, 2).reshape(B, L, D)
        x = self.W_O(Y) + residual

        # ═══════ FFN Sublayer ═══════
        residual = x
        x = self.ffn(self.ffn_norm(x)) + residual

        return x


class SISAModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_head: int,
        n_layer: int,
        d_ff: int,
        d_state: int = 32,
        max_seq_len: int = 2048,
        grad_checkpoint: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.grad_checkpoint = grad_checkpoint
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [SISALayer(d_model, n_head, d_state, d_ff) for _ in range(n_layer)]
        )
        self.final_norm = RMSNorm(d_model)

        # Precompute positional RoPE for q, k
        d_head = d_model // n_head
        cos, sin = precompute_rope_freqs(d_head, max_seq_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self._init_weights()

    def _init_weights(self):
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Skip alpha_proj — its bias is set to -5.0 intentionally
                if hasattr(module, "_sisa_skip_init"):
                    continue
                nn.init.normal_(module.weight, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=std)

        # Re-apply alpha_proj specific init after generic init
        for layer in self.layers:
            nn.init.constant_(layer.alpha_proj.bias, -5.0)
            nn.init.normal_(layer.theta_proj.weight, std=0.01)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
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

    def get_monitor_metrics(self) -> dict:
        """Extract SISA-specific monitoring metrics from last forward pass (.item() calls here, not in forward)."""
        metrics = {}
        for i, layer in enumerate(self.layers):
            if layer._last_g is not None and layer._last_c is not None:
                metrics[f"monitor/max_g_c_L{i}"] = (layer._last_g - layer._last_c).abs().max().item()
            if layer._last_log_alpha is not None:
                mean_la = layer._last_log_alpha.mean().item()
                metrics[f"monitor/half_life_L{i}"] = math.log(0.5) / mean_la if mean_la < 0 else float("inf")
            if layer._last_lambda is not None:
                for h in range(layer.n_head):
                    metrics[f"lambda/L{i}_H{h}"] = layer._last_lambda[h].item()
        return metrics
