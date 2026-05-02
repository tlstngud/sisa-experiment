#!/bin/bash
export CUDA_VISIBLE_DEVICES="GPU-507ecf4f-e53a-fca2-38be-2255782b529d"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/oem/sisa-experiment

# Resume Mamba-3 369M from step 3000 to step 9536 (5B tokens)
python3 -u train.py \
  --phase 2 --model mamba3 \
  --micro-batch 4 --grad-accum 64 \
  --max-tokens 5000000000 \
  --output-dir ckpts_3way/mamba3_v2 \
  --resume ckpts_3way/mamba3_v2/phase2_mamba3/final.pt \
  --no-wandb --no-compile
