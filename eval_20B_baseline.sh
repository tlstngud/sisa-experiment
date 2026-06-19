#!/bin/bash
# 20B 베이스라인 spaced 벤치 (run_eval, 단일 ckpt). $1=model $2=gpu $3=steps...
cd /home/oem/sisa-experiment
MODEL=$1; GPU=$2; shift 2
OUT=/home/oem/sisa-experiment/results/${MODEL}_20B_curve.log
echo "[$MODEL-20B-eval] BEGIN $(date)" > $OUT
for step in "$@"; do
  ck=ckpts_20B/phase3_$MODEL/step_$(printf %06d $step).pt
  [ -f "$ck" ] || { echo "[skip] step$step 없음" | tee -a $OUT; continue; }
  echo "" | tee -a $OUT; echo "===== $MODEL 20B step_$step =====" | tee -a $OUT
  CUDA_VISIBLE_DEVICES=$GPU python -u run_eval.py --model $MODEL --phase 3 \
    --ckpt "$ck" --tag "20B_step$step" \
    --tasks lambada_openai hellaswag arc_easy winogrande 2>&1 | tee -a $OUT
done
echo "[$MODEL-20B-eval] DONE $(date)" | tee -a $OUT
