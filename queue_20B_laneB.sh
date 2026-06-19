#!/bin/bash
# 20B 베이스라인 레인 B: mamba2 → mamba3 (순차). GPU0, SISA2 v11 + 레인A와 공존.
cd /home/oem/sisa-experiment
run () {
  local m=$1
  echo "[laneB $(date +%m-%d_%H:%M)] START $m 20B"
  CUDA_VISIBLE_DEVICES=0 python -u train.py --phase 3 --model $m \
    --max-tokens 20000000000 --data-dir /data/sisa_tokens \
    --output-dir /home/oem/sisa-experiment/ckpts_20B --run-name p3_${m}_20B \
    --no-wandb --no-compile > /home/oem/sisa-experiment/logs/p3_${m}_20B.log 2>&1
  echo "[laneB $(date +%m-%d_%H:%M)] DONE $m 20B"
}
run mamba2
run mamba3
echo "[laneB] ALL DONE"
