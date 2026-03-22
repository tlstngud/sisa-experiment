import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return ((x.float() * norm) * self.weight.float()).to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10000.0):
    """Precompute cos/sin tables for rotary embeddings (interleaved format)."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    angles = torch.outer(t, freqs)  # (max_seq_len, dim//2)
    cos = angles.cos()
    sin = angles.sin()
    return cos, sin  # each (max_seq_len, dim//2)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to q and k (interleaved format).

    q, k: (B, H, L, d_h)
    cos, sin: (L, d_h//2) — will be broadcast
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, L, d_h//2)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q1, q2 = q[..., 0::2], q[..., 1::2]
    k1, k2 = k[..., 0::2], k[..., 1::2]

    q_out = torch.stack([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    q_out = q_out.flatten(-2)

    k_out = torch.stack([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
    k_out = k_out.flatten(-2)

    return q_out, k_out


def apply_ssm_rope(z: torch.Tensor, Phi: torch.Tensor) -> torch.Tensor:
    """
    Data-dependent RoPE on SSM channels (interleaved format).

    z:   (B, H, L, d_state)
    Phi: (B, H, L, d_state//2)  — cumulative phase
    """
    z_even = z[..., 0::2]
    z_odd = z[..., 1::2]

    cos_phi = torch.cos(Phi)
    sin_phi = torch.sin(Phi)

    out_even = z_even * cos_phi - z_odd * sin_phi
    out_odd = z_even * sin_phi + z_odd * cos_phi

    out = torch.stack([out_even, out_odd], dim=-1)
    return out.flatten(-2)
