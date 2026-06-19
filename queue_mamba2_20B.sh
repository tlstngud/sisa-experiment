#!/bin/bash
# mamba2 20B: transformer(20B) 종료로 GPU0 자리 나면 mamba20b venv로 실행.
cd /home/oem/sisa-experiment
TPID=${1:?transformer PID}
VENV=/home/oem/anaconda3/envs/mamba20b/bin/python
echo "[mamba2-q] transformer(PID $TPID) 종료 대기..."
while kill -0 $TPID 2>/dev/null; do sleep 120; done
echo "[mamba2-q $(date +%m-%d_%H:%M)] transformer 종료. mamba2 20B START (venv)"
CUDA_VISIBLE_DEVICES=0 $VENV -u train.py --phase 3 --model mamba2 \
  --max-tokens 20000000000 --data-dir /data/sisa_tokens \
  --output-dir /home/oem/sisa-experiment/ckpts_20B --run-name p3_mamba2_20B \
  --no-wandb --no-compile > /home/oem/sisa-experiment/logs/p3_mamba2_20B.log 2>&1
echo "[mamba2-q $(date +%m-%d_%H:%M)] mamba2 20B DONE"
