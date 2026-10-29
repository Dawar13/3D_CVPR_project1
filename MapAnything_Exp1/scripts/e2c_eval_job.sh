#!/usr/bin/env bash
# E2-C eval run, ONCE, from tag e2-prereg-C (code on the VM verified against the tag by sha256).
# 24 ten-view clips x 132 + 10 four-view + 6 twenty-view clips x 4 = 3,232 inferences.
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mapanything
cd ~/3D-CVPR_Exp/MapAnything_Exp1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "frozen commit: $(cat E2C_FROZEN_COMMIT)"
wait_gpu() { until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge 20480 ]; do
  echo "$(date) waiting: less than 20 GB free"; sleep 60; done; }
wait_gpu
python -m exp1.run_exp1 --config configs/e2c_full.json --clips 'data/clips_e2/eval__*.npz' \
  --out results/e2c_eval --save_extras
for n in 4 20; do
  wait_gpu
  python -m exp1.run_exp1 --config configs/e2c_views.json --clips "data/clips_n${n}/eval__*.npz" \
    --out results/e2c_n${n}_eval --save_extras
done
echo E2C_EVAL_DONE
