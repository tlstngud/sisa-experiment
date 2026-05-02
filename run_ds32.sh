#!/bin/bash
export CUDA_VISIBLE_DEVICES="GPU-507ecf4f-e53a-fca2-38be-2255782b529d"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/oem/sisa-experiment

# Rename RUN 2 checkpoints before this overwrites them
mv checkpoints/phase2_sisa checkpoints/phase2_sisa_ds64 2>/dev/null

echo "===== RUN 3: d_state=32, d_ff=2512 (original config) ====="
python3 -u train.py \
  --phase 2 --model sisa \
  --d-state 32 --d-ff 2512 \
  --micro-batch 12 --grad-accum 22 \
  --max-tokens 1572864000 \
  --no-wandb --no-compile \
  2>&1
