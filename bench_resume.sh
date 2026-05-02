#!/bin/bash
cd /home/oem/sisa-experiment

run_bench() {
    local gpu=$1 phase=$2 ds=$3 dff=$4 ckpt=$5 tag=$6
    if grep -q "winogrande:" /home/oem/bench_mig/${tag}.log 2>/dev/null; then
        echo "[skip] $tag"
        return
    fi
    echo "[run] $tag"
    rm -f /home/oem/bench_mig/${tag}.log
    CUDA_VISIBLE_DEVICES="$gpu" python3 -u run_bench_original.py \
      --model sisa --phase $phase --ckpt "$ckpt" \
      --d-state $ds --d-ff $dff --tag "$tag" \
      > /home/oem/bench_mig/${tag}.log 2>&1
}

GPU0="GPU-3c0a7072-5e78-6467-72e5-de92eb812e6b"
GPU1="GPU-507ecf4f-e53a-fca2-38be-2255782b529d"

# Stream A on GPU 1: 369M ds=128 step 6000
( run_bench "$GPU1" 2 128 1232 ckpts_3way/ds128/phase2_sisa/step_006000.pt "p2_ds128_step006000" ) &
PIDA=$!

# Stream B on GPU 0: 50M ds=128 step 6-9k + 152M step 2,3k
(
  for step in 006000 007000 008000 009000; do
    run_bench "$GPU0" 3 128 1192 /home/oem/sisa-experiment/ckpts_ablation/p3_ds128/phase3_sisa/step_${step}.pt "p3_ds128_step${step}"
  done
  for step in 002000 003000; do
    run_bench "$GPU0" 1 128 1788 /home/oem/sisa-experiment/ckpts_ablation/p1_ds128/phase1_sisa/step_${step}.pt "p1_ds128_step${step}"
  done
) &
PIDB=$!

wait $PIDA $PIDB
echo "All done"
