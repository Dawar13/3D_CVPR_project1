#!/usr/bin/env bash
# E2-C5 dev run: one bad camera (view 3, 15 deg) on the 4- and 20-view dev clips.
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mapanything
cd ~/3D-CVPR_Exp/MapAnything_Exp1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge 20480 ]; do
  echo "$(date) waiting: less than 20 GB free"; sleep 60; done
for n in 4 20; do
  python -m exp1.run_exp1 --config configs/e2c_views.json --clips "data/clips_n${n}/dev__*.npz" \
    --out results/e2c_n${n}_dev --save_extras
done
echo E2C_VIEWS_DEV_DONE
