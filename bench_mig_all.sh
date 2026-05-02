#!/bin/bash
cd /home/oem/sisa-experiment

mkdir -p /home/oem/bench_mig

run_bench() {
    local gpu=$1
    local phase=$2
    local ds=$3
    local dff=$4
    local ckpt=$5
    local tag=$6
    CUDA_VISIBLE_DEVICES="$gpu" python3 -u run_bench_original.py \
      --model sisa --phase $phase \
      --ckpt "$ckpt" --d-state $ds --d-ff $dff \
      --tag "$tag" \
      > /home/oem/bench_mig/${tag}.log 2>&1
}

# GPU 1 (full 80GB): P1 ds16 (2 ckpts sequential)
(
  for step in 001000 002000; do
    run_bench "GPU-507ecf4f-e53a-fca2-38be-2255782b529d" 1 16 2908 \
      /home/oem/sisa-experiment/ckpts_ablation/p1_ds16/phase1_sisa/step_${step}.pt \
      "p1_ds16_step${step}"
  done
) &
PID_GPU1=$!

# MIG-4: P1 ds64 (2 ckpts sequential)
(
  for step in 001000 002000; do
    run_bench "MIG-5b5e111b-963b-5d5d-9e89-c3ac727750c1" 1 64 2428 \
      /home/oem/sisa-experiment/ckpts_ablation/p1_ds64/phase1_sisa/step_${step}.pt \
      "p1_ds64_step${step}"
  done
) &
PID_MIG4=$!

# MIG-5: P3 ds16 (5 ckpts sequential)
(
  for step in 001000 002000 003000 004000 005000; do
    run_bench "MIG-53e9bd39-8943-517a-892f-518273dc66b4" 3 16 1939 \
      /home/oem/sisa-experiment/ckpts_ablation/p3_ds16/phase3_sisa/step_${step}.pt \
      "p3_ds16_step${step}"
  done
) &
PID_MIG5=$!

# MIG-6: P3 ds64 (5 ckpts sequential)
(
  for step in 001000 002000 003000 004000 005000; do
    run_bench "MIG-11f0291b-e4a0-5502-9209-889f6c2fc04f" 3 64 1619 \
      /home/oem/sisa-experiment/ckpts_ablation/p3_ds64/phase3_sisa/step_${step}.pt \
      "p3_ds64_step${step}"
  done
) &
PID_MIG6=$!

wait
echo "All done"
