#!/bin/bash
cd /home/oem/sisa-experiment
mkdir -p /home/oem/bench_mig

run_bench() {
    local gpu=$1 phase=$2 ds=$3 dff=$4 ckpt=$5 tag=$6
    if grep -q "winogrande:" /home/oem/bench_mig/${tag}.log 2>/dev/null; then
        echo "[skip] $tag"
        return
    fi
    echo "[run] $tag"
    CUDA_VISIBLE_DEVICES="$gpu" python3 -u run_bench_original.py \
      --model sisa --phase $phase \
      --ckpt "$ckpt" --d-state $ds --d-ff $dff --tag "$tag" \
      > /home/oem/bench_mig/${tag}.log 2>&1
}

# Stream A on MIG-4: P3 ds16 step 6-9k (4개)
(
  for step in 006000 007000 008000 009000; do
    run_bench "MIG-5b5e111b-963b-5d5d-9e89-c3ac727750c1" 3 16 1939 \
      /home/oem/sisa-experiment/ckpts_ablation/p3_ds16/phase3_sisa/step_${step}.pt \
      "p3_ds16_step${step}"
  done
) &
PIDA=$!

# Stream B on MIG-5: P3 ds64 step 6-8k + P1 ds16 3k (4개)
sleep 5
(
  for step in 006000 007000 008000; do
    run_bench "MIG-53e9bd39-8943-517a-892f-518273dc66b4" 3 64 1619 \
      /home/oem/sisa-experiment/ckpts_ablation/p3_ds64/phase3_sisa/step_${step}.pt \
      "p3_ds64_step${step}"
  done
  run_bench "MIG-53e9bd39-8943-517a-892f-518273dc66b4" 1 16 2908 \
    /home/oem/sisa-experiment/ckpts_ablation/p1_ds16/phase1_sisa/step_003000.pt \
    "p1_ds16_step003000"
) &
PIDB=$!

# Stream C on MIG-6: P1 ds16 4k + P1 ds64 3k + 369M ds64 4-6k (5개)
sleep 10
(
  run_bench "MIG-11f0291b-e4a0-5502-9209-889f6c2fc04f" 1 16 2908 \
    /home/oem/sisa-experiment/ckpts_ablation/p1_ds16/phase1_sisa/step_004000.pt \
    "p1_ds16_step004000"
  run_bench "MIG-11f0291b-e4a0-5502-9209-889f6c2fc04f" 1 64 2428 \
    /home/oem/sisa-experiment/ckpts_ablation/p1_ds64/phase1_sisa/step_003000.pt \
    "p1_ds64_step003000"
  for step in 004000 005000 006000; do
    run_bench "MIG-11f0291b-e4a0-5502-9209-889f6c2fc04f" 2 64 2085 \
      /home/oem/sisa-experiment/ckpts_3way/ds64/phase2_sisa/step_${step}.pt \
      "p2_ds64_resume_step${step}"
  done
) &
PIDC=$!

wait $PIDA $PIDB $PIDC
echo "All done"
