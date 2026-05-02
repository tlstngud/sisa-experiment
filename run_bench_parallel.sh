#!/bin/bash
cd /home/oem/sisa-experiment

# d_state=16 on MIG 0
CUDA_VISIBLE_DEVICES="MIG-31646ad1-b13a-5a51-8802-c2c248db11bd" \
python3 -u run_bench_original.py \
  --model sisa --phase 2 \
  --ckpt checkpoints/phase2_sisa_ds16/final.pt \
  --d-state 16 --d-ff 2725 \
  --tag "d_state=16" \
  > /home/oem/bench_ds16.log 2>&1 &
PID1=$!
echo "d_state=16 PID: $PID1 on MIG-0"

# d_state=64 on MIG 1
CUDA_VISIBLE_DEVICES="MIG-bbcf01e8-c8c5-5324-8b98-a8fdccbdb583" \
python3 -u run_bench_original.py \
  --model sisa --phase 2 \
  --ckpt checkpoints/phase2_sisa_ds64/final.pt \
  --d-state 64 --d-ff 2085 \
  --tag "d_state=64" \
  > /home/oem/bench_ds64.log 2>&1 &
PID2=$!
echo "d_state=64 PID: $PID2 on MIG-1"

wait $PID1 $PID2
echo "Done"
