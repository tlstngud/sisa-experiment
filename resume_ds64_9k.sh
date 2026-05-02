#!/bin/bash
export CUDA_VISIBLE_DEVICES="GPU-507ecf4f-e53a-fca2-38be-2255782b529d"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/oem/sisa-experiment

python3 -u train.py \
  --phase 2 --model sisa \
  --d-state 64 --d-ff 2085 \
  --micro-batch 8 --grad-accum 32 \
  --max-tokens 5000000000 \
  --output-dir ckpts_3way/ds64 \
  --resume ckpts_3way/ds64/phase2_sisa/final.pt \
  --no-wandb --no-compile
