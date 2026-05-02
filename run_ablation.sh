#!/bin/bash
export CUDA_VISIBLE_DEVICES="GPU-507ecf4f-e53a-fca2-38be-2255782b529d"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/oem/sisa-experiment

echo "===== RUN 1: d_state=16, d_ff=2725 (resume from step 2000) ====="
python3 -u train.py \
  --phase 2 --model sisa \
  --d-state 16 --d-ff 2725 \
  --micro-batch 16 --grad-accum 16 \
  --max-tokens 1572864000 \
  --resume checkpoints/phase2_sisa/step_002000.pt \
  --no-wandb --no-compile \
  2>&1

echo ""
echo "===== RUN 2: d_state=64, d_ff=2085 ====="
# rename d_state=16 checkpoints to preserve them before RUN 2 overwrites
mv checkpoints/phase2_sisa checkpoints/phase2_sisa_ds16 2>/dev/null

python3 -u train.py \
  --phase 2 --model sisa \
  --d-state 64 --d-ff 2085 \
  --micro-batch 16 --grad-accum 16 \
  --max-tokens 1572864000 \
  --no-wandb --no-compile \
  2>&1
