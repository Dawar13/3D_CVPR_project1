#!/usr/bin/env bash
# E2-B audit rerun: dev clips first (plumbing check on the real model), then the 10 eval clips.
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mapanything
cd ~/3D-CVPR_Exp/MapAnything_Exp1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "GPU free MiB: $free"; [ "$free" -ge 20480 ] || { echo "less than 20 GB free, not starting"; exit 1; }
for split in dev eval; do
  python -m exp1.run_exp1 --config configs/e2b_audit.json --clips "data/clips/${split}__*.npz" \
    --out results/e2b_audit_$split --save_extras
done
echo E2B_DONE
