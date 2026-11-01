#!/usr/bin/env bash
# E2-C eval, once, from the frozen e2-prereg-C tree, after the session queue ends (Amendment 14: parity gate
# judged by the 99th percentile and the mean shift; user-approved 2026-11-01 before any E2-C eval result).
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mapanything
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
F=~/3D-CVPR_Exp/MapAnything_Exp1
log() { echo "$(date -u '+%F %T') $*" | tee -a ~/logs/session.txt; }
while tmux has-session -t =gpu 2>/dev/null; do sleep 20; done
cd ~/e2d/MapAnything_Exp1
log "2 E2-C eval (amended parity gate, Amendment 14) from $(cat "$F/E2C_FROZEN_COMMIT")"
python -m exp1.shard run --n 4 --workdir "$F" -- --config configs/e2c_full.json --clips 'data/clips_e2/eval__*.npz' --out results/e2c_eval --save_extras
for n in 4 20; do
  python -m exp1.shard run --n 4 --workdir "$F" -- --config configs/e2c_views.json --clips "data/clips_n${n}/eval__*.npz" --out "results/e2c_n${n}_eval" --save_extras
done
log "E2C_EVAL_DONE"
log "SESSION_ALL_DONE"
