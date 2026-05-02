#!/bin/bash
export CUDA_VISIBLE_DEVICES="GPU-507ecf4f-e53a-fca2-38be-2255782b529d"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/oem/sisa-experiment

python3 -u train.py \
  --phase 2 --model sisa \
  --d-state 128 --d-ff 1232 \
  --micro-batch 4 --grad-accum 64 \
  --max-tokens 5000000000 \
  --output-dir ckpts_3way/ds128 \
  --resume ckpts_3way/ds128/phase2_sisa/step_001000.pt \
  --no-wandb --no-compile
