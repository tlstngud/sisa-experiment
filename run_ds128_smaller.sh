#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/oem/sisa-experiment

# 50M ds=128 on GPU 0
CUDA_VISIBLE_DEVICES="GPU-3c0a7072-5e78-6467-72e5-de92eb812e6b" \
python3 -u train.py \
  --phase 3 --model sisa \
  --d-state 128 --d-ff 1192 \
  --micro-batch 8 --grad-accum 32 \
  --max-tokens 5000000000 \
  --output-dir ckpts_ablation/p3_ds128 \
  --no-wandb --no-compile \
  > /home/oem/p3_ds128.log 2>&1 &
echo "P3 ds128 PID: $!"

# 152M ds=128 on GPU 0 (parallel)
CUDA_VISIBLE_DEVICES="GPU-3c0a7072-5e78-6467-72e5-de92eb812e6b" \
python3 -u train.py \
  --phase 1 --model sisa \
  --d-state 128 --d-ff 1788 \
  --micro-batch 8 --grad-accum 32 \
  --max-tokens 5000000000 \
  --output-dir ckpts_ablation/p1_ds128 \
  --no-wandb --no-compile \
  > /home/oem/p1_ds128.log 2>&1 &
echo "P1 ds128 PID: $!"

wait
echo "Both done"
