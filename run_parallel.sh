#!/bin/bash
export CUDA_VISIBLE_DEVICES="GPU-507ecf4f-e53a-fca2-38be-2255782b529d"
cd /home/oem/sisa-experiment

# Run both d_state variants concurrently on same GPU
# micro_batch=4, grad_accum=64 → effective batch = 524,288 (same as original)
# ~20GB per process, 40GB total (fits in 80GB with headroom)

python3 -u train.py \
  --phase 2 --model sisa \
  --d-state 16 --d-ff 2725 \
  --micro-batch 4 --grad-accum 64 \
  --max-tokens 1572864000 \
  --no-wandb --no-compile \
  > /home/oem/sisa_ds16.log 2>&1 &
PID1=$!
echo "RUN 1 (d_state=16) PID: $PID1"

python3 -u train.py \
  --phase 2 --model sisa \
  --d-state 64 --d-ff 2085 \
  --micro-batch 4 --grad-accum 64 \
  --max-tokens 1572864000 \
  --no-wandb --no-compile \
  > /home/oem/sisa_ds64.log 2>&1 &
PID2=$!
echo "RUN 2 (d_state=64) PID: $PID2"

wait $PID1 $PID2
echo "Both runs completed"
