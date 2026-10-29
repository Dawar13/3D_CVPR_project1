#!/usr/bin/env bash
# E2-D step 1, queued: starts only after the E2-C eval run ends (one GPU job at a time, E2.md 0.3).
# Runs from ~/e2d (a separate copy), so the frozen E2-C code in ~/3D-CVPR_Exp is never touched.
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh && conda activate omnivggt
cd ~/e2d/MapAnything_Exp1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while tmux has-session -t e2ceval 2>/dev/null; do sleep 60; done
until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge 20480 ]; do
  echo "$(date) waiting: less than 20 GB free"; sleep 60; done
python -c "import torch; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m pytest tests/test_e2_tools.py -q -k omnivggt 2>&1 | tail -2
python -m exp1.run_exp1 --backend omnivggt --config configs/e2d_checks.json --clips 'data/clips/dev__*.npz' \
  --out results/e2d_checks --save_extras
python scripts/e2d_checks.py --run results/e2d_checks
echo E2D_CHECKS_DONE
