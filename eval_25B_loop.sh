#!/bin/bash
# SISA1 25B continue-training 곡선: 등간격 7개 ckpt 벤치 (run_eval.py, 단일 ckpt 단위)
# d_state=64 d_ff=1619 phase=3 (25B 학습 설정과 동일). piqa는 깨져서 제외.
cd /home/oem/sisa-experiment
CKDIR=/home/oem/sisa-experiment/ckpts_25B/phase3_sisa
OUT=/home/oem/sisa-experiment/results/sisa1_25B_curve.log
echo "[sisa1-25B-eval] BEGIN $(date)" > $OUT
for step in 010000 014000 018000 022000 026000 030000 031000; do
  ck=$CKDIR/step_$step.pt
  [ -f "$ck" ] || { echo "[skip] $ck 없음" | tee -a $OUT; continue; }
  echo "" | tee -a $OUT
  echo "===== SISA1 25B step_$step =====" | tee -a $OUT
  CUDA_VISIBLE_DEVICES=0 python -u run_eval.py --model sisa --phase 3 \
    --ckpt "$ck" --d-state 64 --d-ff 1619 --tag "25B_step$step" \
    --tasks lambada_openai hellaswag arc_easy winogrande 2>&1 | tee -a $OUT
done
echo "[sisa1-25B-eval] DONE $(date)" | tee -a $OUT
