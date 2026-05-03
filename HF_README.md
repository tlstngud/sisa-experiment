---
license: apache-2.0
language:
- en
tags:
- sisa
- ssm
- attention
- mamba
- transformer
- hybrid
---

# SISA: SSM-Informed Softmax Attention

This repository contains model checkpoints for the SISA paper.
SISA augments standard softmax attention with SSM-derived importance bias
via score-level fusion: `Q_aug = [q, s·C̄]`, `K_aug = [k, s·B̄]`,
implemented as a single `scaled_dot_product_attention` call (FlashAttention
compatible).

## Repository Structure

```
369M/                        # 369M-param scale, 5B tokens, mb=4 grad_accum=64
  sisa_ds16/final.pt         # SISA, d_state=16  (step 3000 only — incomplete)
  sisa_ds32/final.pt         # SISA, d_state=32
  sisa_ds64/final.pt         # SISA, d_state=64
  sisa_ds128/final.pt        # SISA, d_state=128
  mamba3/final.pt            # Mamba-3 baseline (mb=4 retrain)

152M/                        # 152M-param scale, 5B tokens
  sisa_ds16/final.pt
  sisa_ds64/final.pt
  sisa_ds128/final.pt

50M/                         # 50M-param scale, 5B tokens
  sisa_ds16/final.pt
  sisa_ds64/final.pt
  sisa_ds128/final.pt

results/                     # Benchmark JSON outputs
  p1_benchmark_results.json  # 152M
  p2_benchmark_results.json  # 369M
  p3_benchmark_results.json  # 50M
  ...
```

## Key Findings

- **152M scale**: SISA `d_state=16` reaches LAMBADA 17.27%, beating
  Mamba-3 (15.51%) and Transformer (13.88%).
- **NIAH**: SISA matches Transformer at 100% across all scales;
  Mamba-2/3 underperform on small scale.
- **369M scale**: Mamba-3 wins LAMBADA (17.10%); SISA undertrained at
  Chinchilla 13.5×.
- Best `d_state` per scale: 50M=64, 152M=16, 369M=128 (U-shape on extremes).

## Loading

```python
import torch
from models.sisa import SISA  # see github.com/tlstngud/sisa-experiment

ckpt = torch.load("369M/sisa_ds128/final.pt", map_location="cpu")
model = SISA(d_model=1024, n_layers=24, n_heads=16, d_state=128, d_ff=1232)
model.load_state_dict(ckpt["model"])
```

## Code & Paper

- Code: https://github.com/tlstngud/sisa-experiment
- Paper: see `paper_final/sisa_neurips.tex` in the github repo

## Citation

```bibtex
@misc{sisa2026,
  title  = {SISA: SSM-Informed Softmax Attention},
  author = {tlstngud},
  year   = {2026}
}
```
