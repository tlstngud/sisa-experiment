#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/oem/sisa-experiment

# All runs: 1.57B tokens (step 3000 at effective batch 524K)
# 4 independent MIG slices → no SM contention

# Phase 1 (152M), d_state=16 — bigger FFN
CUDA_VISIBLE_DEVICES="MIG-31646ad1-b13a-5a51-8802-c2c248db11bd" \
python3 -u train.py \
  --phase 1 --model sisa \
  --d-state 16 --d-ff 2908 \
  --micro-batch 2 --grad-accum 128 \
  --max-tokens 5000000000 \
  --output-dir ckpts_ablation/p1_ds16 \
  --no-wandb --no-compile \
  > /home/oem/p1_ds16.log 2>&1 &
echo "P1 ds16 PID: $!"

# Phase 1 (152M), d_state=64 — bigger SSM
CUDA_VISIBLE_DEVICES="MIG-bbcf01e8-c8c5-5324-8b98-a8fdccbdb583" \
python3 -u train.py \
  --phase 1 --model sisa \
  --d-state 64 --d-ff 2428 \
  --micro-batch 2 --grad-accum 128 \
  --max-tokens 5000000000 \
  --output-dir ckpts_ablation/p1_ds64 \
  --no-wandb --no-compile \
  > /home/oem/p1_ds64.log 2>&1 &
echo "P1 ds64 PID: $!"

# Phase 3 (50M), d_state=16 — bigger FFN
CUDA_VISIBLE_DEVICES="MIG-c3f0b735-aedd-5327-87b0-27f5bffd3b4e" \
python3 -u train.py \
  --phase 3 --model sisa \
  --d-state 16 --d-ff 1939 \
  --micro-batch 4 --grad-accum 64 \
  --max-tokens 5000000000 \
  --output-dir ckpts_ablation/p3_ds16 \
  --no-wandb --no-compile \
  > /home/oem/p3_ds16.log 2>&1 &
echo "P3 ds16 PID: $!"

# Phase 3 (50M), d_state=64 — bigger SSM
CUDA_VISIBLE_DEVICES="MIG-843018f0-bcc1-58b8-aeba-c2300642c1e9" \
python3 -u train.py \
  --phase 3 --model sisa \
  --d-state 64 --d-ff 1619 \
  --micro-batch 4 --grad-accum 64 \
  --max-tokens 5000000000 \
  --output-dir ckpts_ablation/p3_ds64 \
  --no-wandb --no-compile \
  > /home/oem/p3_ds64.log 2>&1 &
echo "P3 ds64 PID: $!"

wait
echo "All 4 completed"
