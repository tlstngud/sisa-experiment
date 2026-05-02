#!/bin/bash
cd /home/oem/sisa-experiment

GPU="GPU-507ecf4f-e53a-fca2-38be-2255782b529d"

run_bench() {
    local phase=$1 ds=$2 dff=$3 ckpt=$4 tag=$5
    if grep -q "winogrande:" /home/oem/bench_mig/${tag}.log 2>/dev/null; then
        echo "[skip] $tag"
        return
    fi
    echo "[run] $tag"
    CUDA_VISIBLE_DEVICES="$GPU" python3 -u run_bench_original.py \
      --model sisa --phase $phase --ckpt "$ckpt" \
      --d-state $ds --d-ff $dff --tag "$tag" \
      > /home/oem/bench_mig/${tag}.log 2>&1
}

# 369M ds=128 step 4000
run_bench 2 128 1232 /home/oem/sisa-experiment/ckpts_3way/ds128/phase2_sisa/step_004000.pt "p2_ds128_step004000"

# 152M ds=128 step 1000
run_bench 1 128 1788 /home/oem/sisa-experiment/ckpts_ablation/p1_ds128/phase1_sisa/step_001000.pt "p1_ds128_step001000"

# 50M ds=128 step 1k, 2k, 3k
for step in 001000 002000 003000; do
  run_bench 3 128 1192 /home/oem/sisa-experiment/ckpts_ablation/p3_ds128/phase3_sisa/step_${step}.pt "p3_ds128_step${step}"
done

echo "All done"
