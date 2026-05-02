#!/bin/bash
export CUDA_VISIBLE_DEVICES="GPU-507ecf4f-e53a-fca2-38be-2255782b529d"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/oem/sisa-experiment

# d_state=64 is bigger (augmented_dim=128 vs 80 for d_state=16)
# Use smaller micro_batch to fit: 12 with grad_accum=21 gives ~516K effective (close to 524K)
# Actually use 8 with grad_accum=32 for exact 524K effective batch

echo "===== RUN 2: d_state=64, d_ff=2085 (micro_batch=8) ====="
python3 -u train.py \
  --phase 2 --model sisa \
  --d-state 64 --d-ff 2085 \
  --micro-batch 8 --grad-accum 32 \
  --max-tokens 1572864000 \
  --no-wandb --no-compile \
  2>&1
