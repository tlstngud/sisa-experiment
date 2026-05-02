#!/bin/bash
export CUDA_VISIBLE_DEVICES="GPU-507ecf4f-e53a-fca2-38be-2255782b529d"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/oem/sisa-experiment

# Wait for GPU 1 to free up (369M ds=128 finishing)
echo "Waiting for GPU 1 to free up..."
while true; do
  used=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits)
  if [ "$used" -lt 1000 ]; then
    echo "GPU 1 free, starting Mamba-3"
    break
  fi
  sleep 60
done

# Mamba-3 369M from scratch, mb=4 (matching our SISA retrains)
# max-tokens 1.57B = step 3000
python3 -u train.py \
  --phase 2 --model mamba3 \
  --micro-batch 4 --grad-accum 64 \
  --max-tokens 1572864000 \
  --output-dir ckpts_3way/mamba3_v2 \
  --no-wandb --no-compile
