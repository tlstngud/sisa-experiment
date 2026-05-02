#!/bin/bash
export CUDA_VISIBLE_DEVICES="GPU-507ecf4f-e53a-fca2-38be-2255782b529d"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/oem/sisa-experiment

# Exact paper config: micro_batch=2, grad_accum=128, effective 524,288
# 3 processes × ~10GB = ~30GB (fits 80GB)
# max-tokens 1.57B = step 3000

# d_state=16
python3 -u train.py \
  --phase 2 --model sisa \
  --d-state 16 --d-ff 2725 \
  --micro-batch 4 --grad-accum 64 \
  --max-tokens 1572864000 \
  --output-dir ckpts_3way/ds16 \
  --no-wandb --no-compile \
  > /home/oem/p2_ds16_v2.log 2>&1 &
PID1=$!
echo "d_state=16 PID: $PID1"

# d_state=32 (paper config)
python3 -u train.py \
  --phase 2 --model sisa \
  --d-state 32 --d-ff 2512 \
  --micro-batch 4 --grad-accum 64 \
  --max-tokens 1572864000 \
  --output-dir ckpts_3way/ds32 \
  --no-wandb --no-compile \
  > /home/oem/p2_ds32_v2.log 2>&1 &
PID2=$!
echo "d_state=32 PID: $PID2"

# d_state=64
python3 -u train.py \
  --phase 2 --model sisa \
  --d-state 64 --d-ff 2085 \
  --micro-batch 4 --grad-accum 64 \
  --max-tokens 1572864000 \
  --output-dir ckpts_3way/ds64 \
  --no-wandb --no-compile \
  > /home/oem/p2_ds64_v2.log 2>&1 &
PID3=$!
echo "d_state=64 PID: $PID3"

wait $PID1 $PID2 $PID3
echo "All 3 completed"
