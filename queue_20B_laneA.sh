#!/bin/bash
# 20B 베이스라인 레인 A: (실행중 transformer 종료 대기) → sisa(d_state=32) 순차.
cd /home/oem/sisa-experiment
TPID=${1:?transformer PID}
echo "[laneA] transformer(PID $TPID) 종료 대기..."
while kill -0 $TPID 2>/dev/null; do sleep 120; done
echo "[laneA $(date +%m-%d_%H:%M)] transformer 종료. START sisa 20B"
CUDA_VISIBLE_DEVICES=0 python -u train.py --phase 3 --model sisa \
  --max-tokens 20000000000 --data-dir /data/sisa_tokens \
  --output-dir /home/oem/sisa-experiment/ckpts_20B --run-name p3_sisa_20B \
  --no-wandb --no-compile > /home/oem/sisa-experiment/logs/p3_sisa_20B.log 2>&1
echo "[laneA $(date +%m-%d_%H:%M)] DONE sisa 20B"
