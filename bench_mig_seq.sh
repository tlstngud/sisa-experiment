#!/bin/bash
cd /home/oem/sisa-experiment
mkdir -p /home/oem/bench_mig

# Save test result we already have
cat > /home/oem/bench_mig/p3_ds16_step001000.log << 'EOF'
===== p3_ds16_step001000 =====
  lambada: 0.0631
  niah: 1.0000
  hellaswag: 0.2454
  arc_easy: 0.2863
  winogrande: 0.5028
EOF

run_bench() {
    local phase=$1 ds=$2 dff=$3 ckpt=$4 tag=$5
    if grep -q "winogrande:" /home/oem/bench_mig/${tag}.log 2>/dev/null; then
        echo "[skip] $tag already done"
        return
    fi
    echo "[run] $tag"
    CUDA_VISIBLE_DEVICES="MIG-53e9bd39-8943-517a-892f-518273dc66b4" python3 -u run_bench_original.py \
      --model sisa --phase $phase \
      --ckpt "$ckpt" --d-state $ds --d-ff $dff --tag "$tag" \
      > /home/oem/bench_mig/${tag}.log 2>&1
}

# P3 (50M) - 10 runs
for step in 001000 002000 003000 004000 005000; do
  run_bench 3 16 1939 /home/oem/sisa-experiment/ckpts_ablation/p3_ds16/phase3_sisa/step_${step}.pt "p3_ds16_step${step}"
done
for step in 001000 002000 003000 004000 005000; do
  run_bench 3 64 1619 /home/oem/sisa-experiment/ckpts_ablation/p3_ds64/phase3_sisa/step_${step}.pt "p3_ds64_step${step}"
done

# P1 (152M) - 4 runs
for step in 001000 002000; do
  run_bench 1 16 2908 /home/oem/sisa-experiment/ckpts_ablation/p1_ds16/phase1_sisa/step_${step}.pt "p1_ds16_step${step}"
done
for step in 001000 002000; do
  run_bench 1 64 2428 /home/oem/sisa-experiment/ckpts_ablation/p1_ds64/phase1_sisa/step_${step}.pt "p1_ds64_step${step}"
done

echo "All done"
