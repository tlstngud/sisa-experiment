"""
Mamba-3 baseline — using Mamba-2 SSD kernel for speed.

Key improvements over Mamba-2:
- Exponential-trapezoidal discretization (no short conv1d)
- Complex-valued state via data-dependent RoPE on B, C
- BCNorm (RMSNorm on B/C projections + learnable bias initialized to ones)
- No post-gate normalization (removed from Mamba-2)

Reference: "Mamba-3: Improved Sequence Modeling using State Space Principles"
           (arXiv: 2603.15569, ICLR 2026)

Recurrence:
    h_t = α_t h_{t-1} + β_t B_{t-1} x_{t-1} + γ_t B_t x_t
    y_t = C_t^T h_t
where α = exp(Δ·A), β = (1-λ)·Δ·exp(Δ·A), γ = λ·Δ

Two-SSD decomposition using mamba_ssm's optimized Triton kernel:
    y = SSD_kernel(λ·x, dt, A, B, C) + SSD_kernel((1-λ)·α·x_{t-1}, dt, A, B_{t-1}, C)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import RMSNorm

# Try to import the optimized SSD kernel
try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    HAS_SSD_KERNEL = True
except ImportError:
    HAS_SSD_KERNEL = False


def _apply_rope_pairs(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """
    Data-dependent RoPE on SSM state channels (interleaved pairs).
    x:      (..., d_state)       — even d_state required
    angles: (..., d_state // 2)  — cumulative phase
    """
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    out = torch.stack([x1 * cos_a - x2 * sin_a,
                       x1 * sin_a + x2 * cos_a], dim=-1)
    return out.flatten(-2)


class Mamba3Mixer(nn.Module):
    """
    Mamba-3 SSM mixer using Mamba-2's SSD kernel.

    Flow: in_proj → split → BCNorm → RoPE → two-SSD-kernel-calls → +D·x → ×SiLU(z) → out_proj
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        expand: int = 2,
        headdim: int = 64,
        chunk_size: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.headdim = headdim
        self.nheads = self.d_inner // headdim
        self.chunk_size = chunk_size
        assert d_state % 2 == 0, "d_state must be even for RoPE"

        # Input projection: z, x, B, C, dt, lambda, theta
        bc_dim = self.nheads * d_state
        self.d_in_proj = (
            2 * self.d_inner      # z + x
            + 2 * bc_dim          # B + C
            + 2 * self.nheads     # dt + lambda_trap
            + d_state // 2        # theta (shared across heads)
        )
        self.in_proj = nn.Linear(d_model, self.d_in_proj, bias=False)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # A_log: log of decay rate; A = -exp(A_log) ensures A < 0
        self.A_log = nn.Parameter(torch.empty(self.nheads))

        # dt bias (added before softplus)
        self.dt_bias = nn.Parameter(torch.empty(self.nheads))

        # Skip connection D (per-head scalar)
        self.D = nn.Parameter(torch.empty(self.nheads))

        # BCNorm: RMSNorm on B and C (paper: QK-Norm analog)
        self.B_norm = RMSNorm(d_state)
        self.C_norm = RMSNorm(d_state)

        # BC bias (initialized to ones, added after norm, before RoPE)
        self.B_bias = nn.Parameter(torch.ones(self.nheads, d_state))
        self.C_bias = nn.Parameter(torch.ones(self.nheads, d_state))

        self._init_params()

    def _init_params(self):
        nn.init.normal_(self.A_log, mean=math.log(0.05), std=0.1)
        nn.init.constant_(self.dt_bias, -5.0)
        nn.init.ones_(self.D)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """u: (B, L, d_model) -> (B, L, d_model)"""
        B_sz, L, _ = u.shape
        H = self.nheads
        P = self.headdim
        N = self.d_state

        # ═══════ Input projection & split ═══════
        proj = self.in_proj(u)
        idx = 0
        z = proj[..., idx:idx + self.d_inner];  idx += self.d_inner
        x = proj[..., idx:idx + self.d_inner];  idx += self.d_inner
        B_raw = proj[..., idx:idx + H * N];     idx += H * N
        C_raw = proj[..., idx:idx + H * N];     idx += H * N
        dt_raw = proj[..., idx:idx + H];        idx += H
        lam_raw = proj[..., idx:idx + H];       idx += H
        theta = proj[..., idx:idx + N // 2];    idx += N // 2

        # ═══════ dt, lambda, A ═══════
        dt = F.softplus(dt_raw + self.dt_bias)  # (B, L, H)
        lam = torch.sigmoid(lam_raw)            # (B, L, H)
        A = -torch.exp(self.A_log)              # (H,), always negative
        alpha = torch.exp(dt * A)               # (B, L, H), per-step decay

        # ═══════ BCNorm: RMSNorm → + bias ═══════
        B_proj = self.B_norm(B_raw.view(B_sz, L, H, N)) + self.B_bias  # (B,L,H,N)
        C_proj = self.C_norm(C_raw.view(B_sz, L, H, N)) + self.C_bias

        # ═══════ Data-dependent RoPE on B, C ═══════
        raw_angle = dt.unsqueeze(-1) * theta.unsqueeze(2)   # (B,L,H,N//2)
        cum_angle = -torch.cumsum(raw_angle, dim=1)         # negative per paper

        B_rot = _apply_rope_pairs(B_proj.float(), cum_angle.float()).to(u.dtype)  # (B,L,H,N)
        C_rot = _apply_rope_pairs(C_proj.float(), cum_angle.float()).to(u.dtype)

        # ═══════ Reshape x ═══════
        x_heads = x.view(B_sz, L, H, P)  # (B, L, H, P)

        if HAS_SSD_KERNEL:
            y = self._forward_ssd_kernel(x_heads, dt, A, B_rot, C_rot, lam, alpha)
        else:
            y = self._forward_pure_pytorch(x_heads, dt, A, B_rot, C_rot, lam, alpha)

        # ═══════ Skip connection ═══════
        y = y + self.D.view(1, 1, H, 1) * x_heads  # (B, L, H, P)

        # ═══════ Reshape → gate → project ═══════
        y = y.reshape(B_sz, L, self.d_inner)
        y = y * F.silu(z)
        return self.out_proj(y)

    def _forward_ssd_kernel(self, x_heads, dt, A, B_rot, C_rot, lam, alpha):
        """
        Two-SSD decomposition using Mamba-2's optimized Triton kernel.

        The kernel computes: SSD(x * dt, A * dt, B, C)
        So to get our desired effective_input, we pre-scale x.

        Gamma term: want effective_input = γ·x = λ·dt·x
            → pass x_gamma = λ·x, kernel multiplies by dt → λ·dt·x = γ·x  ✓

        Beta term: want effective_input = β·x_{t-1} = (1-λ)·dt·α·x_{t-1}
            → pass x_beta = (1-λ)·α·x_{t-1}, kernel multiplies by dt → (1-λ)·α·dt·x_{t-1} = β·x_{t-1}  ✓
        """
        B_sz, L, H, P = x_heads.shape
        N = self.d_state

        # Ensure all kernel inputs have the SAME dtype (avoids Triton dot mismatch in backward).
        # Use bf16 instead of fp32 for speed and memory.
        kernel_dtype = x_heads.dtype  # bf16 under autocast
        x_heads = x_heads.to(kernel_dtype)
        dt = dt.to(kernel_dtype)
        B_rot = B_rot.to(kernel_dtype)
        C_rot = C_rot.to(kernel_dtype)
        lam = lam.to(kernel_dtype)
        alpha = alpha.to(kernel_dtype)

        # --- Gamma SSD: current input ---
        x_gamma = lam.unsqueeze(-1) * x_heads  # (B, L, H, P)

        # --- Beta SSD: previous input (time-shifted) ---
        x_prev = F.pad(x_heads[:, :-1, :, :], (0, 0, 0, 0, 1, 0))  # shift right by 1
        B_prev = F.pad(B_rot[:, :-1, :, :], (0, 0, 0, 0, 1, 0))
        x_beta = (1 - lam).unsqueeze(-1) * alpha.unsqueeze(-1) * x_prev  # (B, L, H, P)

        # B, C must be (B, L, ngroups, N) — here ngroups = nheads
        B_gamma = B_rot   # (B, L, H, N)
        B_beta = B_prev   # (B, L, H, N)
        C_kernel = C_rot  # (B, L, H, N) — same C for both calls

        # Call SSD kernel twice with dt_softplus=False, dt_bias=None
        # (dt is already processed)
        y_gamma = mamba_chunk_scan_combined(
            x_gamma, dt, A, B_gamma, C_kernel,
            chunk_size=self.chunk_size,
            D=None, z=None,
            dt_bias=None, dt_softplus=False,
        )

        y_beta = mamba_chunk_scan_combined(
            x_beta, dt, A, B_beta, C_kernel,
            chunk_size=self.chunk_size,
            D=None, z=None,
            dt_bias=None, dt_softplus=False,
        )

        return y_gamma + y_beta  # (B, L, H, P)

    def _forward_pure_pytorch(self, x_heads, dt, A, B_rot, C_rot, lam, alpha):
        """Fallback: pure PyTorch chunked scan (slow but works without mamba-ssm)."""
        B_sz, L, H, P = x_heads.shape
        N = self.d_state
        CS = self.chunk_size

        # Transpose to (B, H, L, ...) for computation
        x_h = x_heads.permute(0, 2, 1, 3)       # (B, H, L, P)
        B_r = B_rot.permute(0, 2, 1, 3)          # (B, H, L, N)
        C_r = C_rot.permute(0, 2, 1, 3)

        dA = (dt * A).permute(0, 2, 1).float()   # (B, H, L)
        gamma_h = (lam * dt).permute(0, 2, 1).unsqueeze(-1)    # (B, H, L, 1)
        beta_h = ((1 - lam) * dt * alpha).permute(0, 2, 1).unsqueeze(-1)

        B_gamma = B_r * gamma_h
        x_gamma = x_h

        x_prev = F.pad(x_h[:, :, :-1, :], (0, 0, 1, 0))
        B_prev = F.pad(B_r[:, :, :-1, :], (0, 0, 1, 0))
        B_beta = B_prev * beta_h
        x_beta = x_prev

        # Pad to chunk multiple
        pad = (CS - L % CS) % CS
        if pad > 0:
            dA = F.pad(dA, (0, pad))
            B_gamma = F.pad(B_gamma, (0, 0, 0, pad))
            x_gamma = F.pad(x_gamma, (0, 0, 0, pad))
            B_beta = F.pad(B_beta, (0, 0, 0, pad))
            x_beta = F.pad(x_beta, (0, 0, 0, pad))
            C_r = F.pad(C_r, (0, 0, 0, pad))
        Lp = L + pad
        nc = Lp // CS

        dA = dA.view(B_sz, H, nc, CS)
        Bg = B_gamma.view(B_sz, H, nc, CS, N)
        xg = x_gamma.view(B_sz, H, nc, CS, P)
        Bb = B_beta.view(B_sz, H, nc, CS, N)
        xb = x_beta.view(B_sz, H, nc, CS, P)
        Cc = C_r.view(B_sz, H, nc, CS, N)

        cum_la = torch.cumsum(dA, dim=3)
        decay_diff = cum_la.unsqueeze(-1) - cum_la.unsqueeze(-2)
        causal = torch.tril(torch.ones(CS, CS, device=x_h.device))
        W = (torch.exp(decay_diff) * causal).to(x_h.dtype)

        score_g = torch.einsum('bhcin,bhcjn->bhcij', Cc, Bg)
        score_b = torch.einsum('bhcin,bhcjn->bhcij', Cc, Bb)

        y_intra = (torch.einsum('bhcij,bhcjp->bhcip', (W * score_g).to(xg.dtype), xg)
                 + torch.einsum('bhcij,bhcjp->bhcip', (W * score_b).to(xb.dtype), xb))

        decay_to_end = torch.exp(cum_la[:, :, :, -1:] - cum_la).to(x_h.dtype)
        h_end = (torch.einsum('bhcj,bhcjn,bhcjp->bhcnp', decay_to_end, Bg, xg)
               + torch.einsum('bhcj,bhcjn,bhcjp->bhcnp', decay_to_end, Bb, xb))
        chunk_decay = torch.exp(cum_la[:, :, :, -1])

        carry = torch.zeros(B_sz, H, N, P, device=x_h.device, dtype=x_h.dtype)
        y_carry_list = []
        for c in range(nc):
            d = torch.exp(cum_la[:, :, c, :]).to(x_h.dtype)
            Cc_carry = torch.einsum('bhtn,bhnp->bhtp', Cc[:, :, c], carry)
            y_carry_list.append(d.unsqueeze(-1) * Cc_carry)
            carry = chunk_decay[:, :, c].unsqueeze(-1).unsqueeze(-1) * carry + h_end[:, :, c]

        y_carry = torch.stack(y_carry_list, dim=2)
        y = (y_intra + y_carry).view(B_sz, H, Lp, P)
        if pad > 0:
            y = y[:, :, :L, :]

        return y.permute(0, 2, 1, 3)  # (B, L, H, P)


class Mamba3Block(nn.Module):
    """Single Mamba-3 block: pre-norm → mixer → + residual."""

    def __init__(self, d_model: int, d_state: int = 64, expand: int = 2):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mixer = Mamba3Mixer(d_model, d_state=d_state, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mixer(self.norm(x)) + x


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
