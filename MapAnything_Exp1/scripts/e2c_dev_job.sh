#!/usr/bin/env bash
# E2-C dev (debug) run: the full E2-C grid on the 3 dev c0 clips. Waits for the E2-B audit to end.
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mapanything
cd ~/3D-CVPR_Exp/MapAnything_Exp1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while tmux has-session -t e2b 2>/dev/null; do sleep 30; done
python -m pytest tests/test_e2_tools.py -q 2>&1 | tail -2   # includes the MapAnything median-swap test
until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge 20480 ]; do
  echo "$(date) waiting: less than 20 GB free"; sleep 60; done
python -m exp1.run_exp1 --config configs/e2c_full.json --clips 'data/clips/dev__*__c0.npz' \
  --out results/e2c_dev --save_extras
echo E2C_DEV_DONE
