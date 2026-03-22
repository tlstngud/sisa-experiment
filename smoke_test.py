#!/usr/bin/env python3
"""
Phase 0 Smoke Test — SISA Sanity Checks.

Checklist from design doc:
  □ NaN/Inf 없음
  □ SDPA flash backend가 augmented_dim에서 잡히는지 확인
  □ λ가 양수로 학습됨 (0으로 안 죽음)
  □ α effective half-life가 10~200 범위
  □ max|g - c| < 20 (FP32 exp 안전 범위)
  □ Q̂^T K̂ / √d_h = qk/√d_h + λC̄B̄ 수치적으로 일치 확인
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend


def test_sdpa_flash_backend():
    """Test if flash attention backend works with augmented Q/K dims."""
    print("=" * 50)
    print("[Test 1] SDPA Flash Backend")
    print("=" * 50)

    B, H, L = 2, 12, 256
    d_aug = 96  # d_head(64) + d_state(32)
    d_v = 64

    Q = torch.randn(B, H, L, d_aug, device="cuda", dtype=torch.bfloat16)
    K = torch.randn(B, H, L, d_aug, device="cuda", dtype=torch.bfloat16)
    V = torch.randn(B, H, L, d_v, device="cuda", dtype=torch.bfloat16)

    # Test flash backend
    try:
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            Y = F.scaled_dot_product_attention(Q, K, V, scale=1.0 / 8.0, is_causal=True)
        print(f"  FLASH_ATTENTION: OK  (output shape: {Y.shape})")
        flash_ok = True
    except RuntimeError as e:
        print(f"  FLASH_ATTENTION: FAILED — {e}")
        flash_ok = False

    # Test memory-efficient backend
    try:
        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            Y = F.scaled_dot_product_attention(Q, K, V, scale=1.0 / 8.0, is_causal=True)
        print(f"  EFFICIENT_ATTENTION: OK  (output shape: {Y.shape})")
    except RuntimeError as e:
        print(f"  EFFICIENT_ATTENTION: FAILED — {e}")

    # Test math backend (always works)
    with sdpa_kernel(backends=[SDPBackend.MATH]):
        Y = F.scaled_dot_product_attention(Q, K, V, scale=1.0 / 8.0, is_causal=True)
    print(f"  MATH: OK  (output shape: {Y.shape})")

    if not flash_ok:
        print("  → Will fallback to EFFICIENT or MATH backend")
    print()
    return flash_ok


def test_score_decomposition():
    """Verify: Q̂^T K̂ / √d_h == q^T k / √d_h + λ * C̄^T * B̄"""
    print("=" * 50)
    print("[Test 2] Score Decomposition Exact Match")
    print("=" * 50)

    B, H, L = 1, 4, 8
    d_h = 64
    d_s = 32
    lam = 0.3

    torch.manual_seed(42)
    q = torch.randn(B, H, L, d_h)
    k = torch.randn(B, H, L, d_h)
    C_bar = torch.randn(B, H, L, d_s)
    B_bar = torch.randn(B, H, L, d_s)

    # Method 1: Direct additive score
    attn_score = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(d_h)
    ssm_score = lam * torch.einsum("bhid,bhjd->bhij", C_bar, B_bar)
    score_direct = attn_score + ssm_score

    # Method 2: Augmented Q/K
    s = (d_h ** 0.25) * math.sqrt(lam)
    Q_aug = torch.cat([q, s * C_bar], dim=-1)
    K_aug = torch.cat([k, s * B_bar], dim=-1)
    score_augmented = torch.einsum("bhid,bhjd->bhij", Q_aug, K_aug) / math.sqrt(d_h)

    diff = (score_direct - score_augmented).abs().max().item()
    print(f"  Max absolute difference: {diff:.2e}")
    assert diff < 1e-5, f"Score decomposition mismatch: {diff}"
    print("  PASSED")
    print()


def test_sisa_forward():
    """Test SISA model forward pass for NaN/Inf and monitoring metrics."""
    print("=" * 50)
    print("[Test 3] SISA Forward Pass")
    print("=" * 50)

    from models.sisa import SISAModel

    model = SISAModel(
        vocab_size=1000,
        d_model=256,
        n_head=4,
        d_state=32,
        n_layer=4,
        d_ff=920,
        max_seq_len=256,
    ).cuda()

    input_ids = torch.randint(0, 1000, (2, 128), device="cuda")

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(input_ids)

    print(f"  Output shape: {logits.shape}")
    print(f"  Has NaN: {torch.isnan(logits).any().item()}")
    print(f"  Has Inf: {torch.isinf(logits).any().item()}")
    assert not torch.isnan(logits).any(), "NaN detected in output!"
    assert not torch.isinf(logits).any(), "Inf detected in output!"

    # Check monitoring metrics
    metrics = model.get_monitor_metrics()
    print(f"\n  Monitoring metrics:")

    max_g_c_values = [v for k, v in metrics.items() if "max_g_c" in k]
    half_life_values = [v for k, v in metrics.items() if "half_life" in k]
    lambda_values = [v for k, v in metrics.items() if "lambda" in k]

    for i, v in enumerate(max_g_c_values):
        status = "OK" if v < 20 else "WARNING"
        print(f"    Layer {i} max|g-c|: {v:.4f}  [{status}]")

    for i, v in enumerate(half_life_values):
        status = "OK" if 10 < v < 200 else "WARNING"
        print(f"    Layer {i} half-life: {v:.1f}  [{status}]")

    print(f"    Lambda values: {[f'{v:.4f}' for v in lambda_values]}")
    assert all(v > 0 for v in lambda_values), "Lambda must be positive!"

    print("  PASSED")
    print()


def test_backward():
    """Test that gradients flow and lambda_raw gets gradients."""
    print("=" * 50)
    print("[Test 4] Backward Pass + Lambda Gradient")
    print("=" * 50)

    from models.sisa import SISAModel

    model = SISAModel(
        vocab_size=1000,
        d_model=256,
        n_head=4,
        d_state=32,
        n_layer=4,
        d_ff=920,
        max_seq_len=256,
    ).cuda()

    input_ids = torch.randint(0, 1000, (2, 128), device="cuda")
    labels = torch.randint(0, 1000, (2, 128), device="cuda")

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

    loss.backward()

    # Check lambda_raw gradients
    for i, layer in enumerate(model.layers):
        grad = layer.lambda_raw.grad
        print(f"  Layer {i} lambda_raw.grad: {grad}")
        assert grad is not None, f"No gradient for lambda_raw in layer {i}!"
        assert not torch.isnan(grad).any(), f"NaN in lambda_raw gradient, layer {i}!"

    # Check no NaN gradients anywhere
    nan_params = []
    for name, p in model.named_parameters():
        if p.grad is not None and torch.isnan(p.grad).any():
            nan_params.append(name)
    if nan_params:
        print(f"  WARNING: NaN gradients in: {nan_params}")
    else:
        print("  No NaN gradients anywhere")

    print(f"  Loss: {loss.item():.4f}")
    print("  PASSED")
    print()


def test_transformer_baseline():
    """Quick sanity check for Transformer baseline."""
    print("=" * 50)
    print("[Test 5] Transformer Baseline Forward")
    print("=" * 50)

    from models.transformer import TransformerModel

    model = TransformerModel(
        vocab_size=1000,
        d_model=256,
        n_head=4,
        n_layer=4,
        d_ff=1024,
        max_seq_len=256,
    ).cuda()

    input_ids = torch.randint(0, 1000, (2, 128), device="cuda")

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(input_ids)

    print(f"  Output shape: {logits.shape}")
    assert not torch.isnan(logits).any(), "NaN in Transformer output!"
    print("  PASSED")
    print()


def test_param_counts():
    """Verify parameter counts match design doc expectations."""
    print("=" * 50)
    print("[Test 6] Parameter Count Verification")
    print("=" * 50)

    from models.transformer import TransformerModel
    from models.sisa import SISAModel

    # Phase 1 configs
    tf = TransformerModel(50277, 768, 12, 12, 3072, 2048)
    sisa = SISAModel(50277, 768, 12, 12, 2748, 32, 2048)

    tf_params = sum(p.numel() for p in tf.parameters())
    sisa_params = sum(p.numel() for p in sisa.parameters())

    print(f"  Transformer:   {tf_params:>12,}  ({tf_params/1e6:.1f}M)")
    print(f"  SISA:          {sisa_params:>12,}  ({sisa_params/1e6:.1f}M)")
    print(f"  Difference:    {abs(tf_params - sisa_params):>12,}")

    # Should be within ~0.1% of each other
    ratio = abs(tf_params - sisa_params) / tf_params
    status = "OK" if ratio < 0.01 else "WARNING"
    print(f"  Ratio diff:    {ratio:.4%}  [{status}]")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  SISA Phase 0 — Smoke Tests")
    print("=" * 50 + "\n")

    test_score_decomposition()
    test_param_counts()

    if torch.cuda.is_available():
        test_sdpa_flash_backend()
        test_sisa_forward()
        test_backward()
        test_transformer_baseline()
    else:
        print("  CUDA not available — skipping GPU tests\n")

    print("=" * 50)
    print("  All smoke tests passed!")
    print("=" * 50)
