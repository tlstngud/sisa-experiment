#!/bin/bash
# Upload all final.pt checkpoints to HF in organized layout
set -e
REPO=koreashin/sisa-experiment

upload() {
    local local_path=$1 remote_path=$2
    echo "[uploading] $local_path -> $remote_path"
    hf upload "$REPO" "$local_path" "$remote_path" --commit-message "Add $remote_path" 2>&1 | tail -2
}

# 369M (ckpts_3way)
upload ckpts_3way/ds16/phase2_sisa/final.pt    369M/sisa_ds16/final.pt
upload ckpts_3way/ds32/phase2_sisa/final.pt    369M/sisa_ds32/final.pt
upload ckpts_3way/ds64/phase2_sisa/final.pt    369M/sisa_ds64/final.pt
upload ckpts_3way/ds128/phase2_sisa/final.pt   369M/sisa_ds128/final.pt
upload ckpts_3way/mamba3_v2/phase2_mamba3/final.pt 369M/mamba3/final.pt

# 152M (p1)
upload ckpts_ablation/p1_ds16/phase1_sisa/final.pt   152M/sisa_ds16/final.pt
upload ckpts_ablation/p1_ds64/phase1_sisa/final.pt   152M/sisa_ds64/final.pt
upload ckpts_ablation/p1_ds128/phase1_sisa/final.pt  152M/sisa_ds128/final.pt

# 50M (p3)
upload ckpts_ablation/p3_ds16/phase3_sisa/final.pt   50M/sisa_ds16/final.pt
upload ckpts_ablation/p3_ds64/phase3_sisa/final.pt   50M/sisa_ds64/final.pt
upload ckpts_ablation/p3_ds128/phase3_sisa/final.pt  50M/sisa_ds128/final.pt

# results
upload results results

echo "All uploads done"
