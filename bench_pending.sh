#!/bin/bash
cd /home/oem/sisa-experiment
mkdir -p /home/oem/bench_mig

GPU="GPU-3c0a7072-5e78-6467-72e5-de92eb812e6b"

run_bench() {
    local phase=$1 ds=$2 dff=$3 ckpt=$4 tag=$5
    if grep -q "winogrande:" /home/oem/bench_mig/${tag}.log 2>/dev/null; then
        echo "[skip] $tag"
        return
    fi
    if [ ! -f "$ckpt" ]; then
        echo "[missing] $ckpt — skip"
        return
    fi
    echo "[run] $tag"
    CUDA_VISIBLE_DEVICES="$GPU" python3 -u run_bench_original.py \
      --model sisa --phase $phase --ckpt "$ckpt" \
      --d-state $ds --d-ff $dff --tag "$tag" \
      > /home/oem/bench_mig/${tag}.log 2>&1
}

# P1 ds16 step 5k-9k + final
for step in 005000 006000 007000 008000 009000; do
  run_bench 1 16 2908 ckpts_ablation/p1_ds16/phase1_sisa/step_${step}.pt "p1_ds16_step${step}"
done
run_bench 1 16 2908 ckpts_ablation/p1_ds16/phase1_sisa/final.pt "p1_ds16_final"

# P1 ds64 step 4k-9k
for step in 004000 005000 006000 007000 008000 009000; do
  run_bench 1 64 2428 ckpts_ablation/p1_ds64/phase1_sisa/step_${step}.pt "p1_ds64_step${step}"
done
run_bench 1 64 2428 ckpts_ablation/p1_ds64/phase1_sisa/final.pt "p1_ds64_final"

# P3 ds64 step 9k + final
run_bench 3 64 1619 ckpts_ablation/p3_ds64/phase3_sisa/step_009000.pt "p3_ds64_step009000"
run_bench 3 64 1619 ckpts_ablation/p3_ds64/phase3_sisa/final.pt "p3_ds64_final"

# 369M ds64 resume step 7k-9k + final
for step in 007000 008000 009000; do
  run_bench 2 64 2085 ckpts_3way/ds64/phase2_sisa/step_${step}.pt "p2_ds64_resume_step${step}"
done
run_bench 2 64 2085 ckpts_3way/ds64/phase2_sisa/final.pt "p2_ds64_resume_final"

# 369M ds128 step 1k, 2k
for step in 001000 002000; do
  run_bench 2 128 1232 ckpts_3way/ds128/phase2_sisa/step_${step}.pt "p2_ds128_step${step}"
done

echo "All done"
