#!/bin/bash
export CUDA_VISIBLE_DEVICES="GPU-507ecf4f-e53a-fca2-38be-2255782b529d"
cd /home/oem/sisa-experiment

echo "===== NIAH: d_state=16 ====="
python3 -u run_niah.py \
  --model sisa --phase 2 \
  --ckpt checkpoints/phase2_sisa_ds16/step_003000.pt \
  --d-state 16 --d-ff 2725 \
  --tag "d_state=16" \
  --lengths 512 1024 2048 \
  --n-samples 50 \
  2>&1

echo ""
echo "===== NIAH: d_state=64 ====="
python3 -u run_niah.py \
  --model sisa --phase 2 \
  --ckpt checkpoints/phase2_sisa/final.pt \
  --d-state 64 --d-ff 2085 \
  --tag "d_state=64" \
  --lengths 512 1024 2048 \
  --n-samples 50 \
  2>&1
