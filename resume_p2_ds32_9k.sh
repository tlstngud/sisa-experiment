#!/bin/bash
export CUDA_VISIBLE_DEVICES="GPU-3c0a7072-5e78-6467-72e5-de92eb812e6b"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/oem/sisa-experiment

# Resume 369M ds=32 from step 3000 → step 9000
# d_ff=2512, max-tokens 5B, mb=16 (aug_dim=96 fits easily)
python3 -u train.py \
  --phase 2 --model sisa \
  --d-state 32 --d-ff 2512 \
  --micro-batch 4 --grad-accum 64 \
  --max-tokens 5000000000 \
  --output-dir ckpts_3way/ds32 \
  --resume ckpts_3way/ds32/phase2_sisa/final.pt \
  --no-wandb --no-compile
