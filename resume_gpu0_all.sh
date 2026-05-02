#!/bin/bash
export CUDA_VISIBLE_DEVICES="GPU-3c0a7072-5e78-6467-72e5-de92eb812e6b"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/oem/sisa-experiment

# 1. P3 ds64 finish (~15 min)
echo "===== P3 ds64 resume from step 9000 ====="
python3 -u train.py \
  --phase 3 --model sisa \
  --d-state 64 --d-ff 1619 \
  --micro-batch 16 --grad-accum 16 \
  --max-tokens 5000000000 \
  --output-dir ckpts_ablation/p3_ds64 \
  --resume ckpts_ablation/p3_ds64/phase3_sisa/step_009000.pt \
  --no-wandb --no-compile

# 2. P1 ds16 resume from step 4000
echo "===== P1 ds16 resume from step 4000 ====="
python3 -u train.py \
  --phase 1 --model sisa \
  --d-state 16 --d-ff 2908 \
  --micro-batch 16 --grad-accum 16 \
  --max-tokens 5000000000 \
  --output-dir ckpts_ablation/p1_ds16 \
  --resume ckpts_ablation/p1_ds16/phase1_sisa/step_004000.pt \
  --no-wandb --no-compile

# 3. P1 ds64 resume from step 4000
echo "===== P1 ds64 resume from step 4000 ====="
python3 -u train.py \
  --phase 1 --model sisa \
  --d-state 64 --d-ff 2428 \
  --micro-batch 16 --grad-accum 16 \
  --max-tokens 5000000000 \
  --output-dir ckpts_ablation/p1_ds64 \
  --resume ckpts_ablation/p1_ds64/phase1_sisa/step_004000.pt \
  --no-wandb --no-compile

echo "All done"
