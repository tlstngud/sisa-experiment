#!/bin/bash
cd /home/oem/sisa-experiment
GPU="GPU-3c0a7072-5e78-6467-72e5-de92eb812e6b"

run_bench() {
    local model=$1 phase=$2 ckpt=$3 tag=$4 ds_args=$5
    if grep -q "winogrande:" /home/oem/bench_mig/${tag}.log 2>/dev/null; then
        echo "[skip] $tag"
        return
    fi
    echo "[run] $tag"
    CUDA_VISIBLE_DEVICES="$GPU" python3 -u run_bench_original.py \
      --model $model --phase $phase --ckpt "$ckpt" --tag "$tag" $ds_args \
      > /home/oem/bench_mig/${tag}.log 2>&1
}

# d_s=32 step 6000~9000
for step in 006000 007000 008000 009000; do
  run_bench sisa 2 ckpts_3way/ds32/phase2_sisa/step_${step}.pt "p2_ds32_v2_step${step}" "--d-state 32 --d-ff 2512"
done

# Mamba-3 step 3000~5000
for step in 003000 004000 005000; do
  run_bench mamba3 2 ckpts_3way/mamba3_v2/phase2_mamba3/step_${step}.pt "p2_mamba3_v2_step${step}" ""
done

echo "All done"
