#!/bin/bash
export CUDA_VISIBLE_DEVICES="GPU-3c0a7072-5e78-6467-72e5-de92eb812e6b"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/oem/sisa-experiment

# 369M d_state=128, d_ff=1232, full training to step 9536
python3 -u train.py \
  --phase 2 --model sisa \
  --d-state 128 --d-ff 1232 \
  --micro-batch 4 --grad-accum 64 \
  --max-tokens 5000000000 \
  --output-dir ckpts_3way/ds128 \
  --no-wandb --no-compile
