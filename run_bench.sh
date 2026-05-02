#!/bin/bash
export CUDA_VISIBLE_DEVICES="GPU-507ecf4f-e53a-fca2-38be-2255782b529d"
cd /home/oem/sisa-experiment

mkdir -p results

echo "===== Benchmark: d_state=16 ====="
python3 -u run_eval.py \
  --model sisa \
  --phase 2 \
  --ckpt checkpoints/phase2_sisa_ds16/step_003000.pt \
  --d-state 16 --d-ff 2725 \
  --tag "ds16_step3000" \
  --tasks lambada_openai hellaswag arc_easy winogrande \
  2>&1

echo ""
echo "===== Benchmark: d_state=64 ====="
python3 -u run_eval.py \
  --model sisa \
  --phase 2 \
  --ckpt checkpoints/phase2_sisa/final.pt \
  --d-state 64 --d-ff 2085 \
  --tag "ds64_step3000" \
  --tasks lambada_openai hellaswag arc_easy winogrande \
  2>&1
