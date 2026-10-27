#!/usr/bin/env bash
# E0 Phase C: E0 Steps 4.3-8 in one go (GPU busy for only part of it).
# Run from MapAnything_Exp1/ with `conda activate mapanything` and CUDA_VISIBLE_DEVICES set.
# Every printout is saved under reports/e0_logs/; results go to results/.
# Stops before any model run on clips if the GT self-consistency check fails.
#
#   ETH3D_ROOT=/data/eth3d bash scripts/e0_phase_c.sh
# SCENES="delivery_area facade terrace" limits data prep to a subset (e.g. the dev scenes while
# the eval scenes still download); the model only ever runs on dev clips anyway.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT=${ETH3D_ROOT:-/data/eth3d}
SC=${SCENES:+--scenes $SCENES}
LOG=reports/e0_logs
mkdir -p "$LOG"
step() { echo; echo "===== $1"; }

step "4.3 layout"
python scripts/e0_check_layout.py --eth3d_root "$ROOT" $SC | tee "$LOG/4_layout.txt" || { echo "layout BAD: fix before continuing"; exit 1; }

step "4.4 clips (depth undistorted)"
# No A/B with --no_depth_undistort: on real ETH3D the undistorted images (~6200x4130) and the
# depth maps (6048x4032, distorted grid) differ in size, so the old direct lookup cannot run.
python -m exp1.datasets --eth3d_root "$ROOT" --out data/clips --n_views 10 --size 518 336 $SC \
  | tee "$LOG/4_clips.txt" || exit 1
ls -la data/clips | tee -a "$LOG/4_clips.txt"

step "5 GT self-consistency (GT only, no model)"
python scripts/e0_check_clips.py --clips 'data/clips/*.npz' --out reports/e0_clip_check \
  | tee "$LOG/5_clip_check.txt"
CLIP_OK=$?

step "6 official README example (GPU)"
python scripts/e0_official_example.py --eth3d_root "$ROOT" | tee reports/e0_official_example.txt

if [ "$CLIP_OK" -ne 0 ]; then
  echo "GT self-consistency NOT PASS: stopping before the smoke runs (E0 Step 5 rule)"
  exit 1
fi

CLIP=$(ls data/clips/dev__*.npz | head -1)
echo "smoke clip: $CLIP" | tee "$LOG/7_smoke_clip.txt"

step "7 smoke runs (GPU)"
python -m exp1.run_exp1 --config configs/pilot.json --clips "$CLIP" --out results/smoke \
  --only rgb_only rgb_K clean_pose clean_pose_noK zero_noise 2>&1 | tee "$LOG/7_smoke.txt"
python -m exp1.run_exp1 --config configs/pilot.json --clips "$CLIP" --out results/smoke_ctrl \
  --only rgb_only clean_pose "outlier_rot_rot_deg=15" 2>&1 | tee "$LOG/7_smoke_ctrl.txt"
python scripts/e0_check_smoke.py --runs results/smoke results/smoke_ctrl | tee "$LOG/7_smoke_check.txt"

step "7b visual check: demo reconstructions of the smoke clip (GPU)"
python scripts/e0_demo_recon.py --clip "$CLIP" --out results/e0_demo | tee "$LOG/7b_demo_recon.txt"

step "8 determinism + timing (GPU, two processes)"
python scripts/e0_determinism.py --clip "$CLIP" --out results/det_a | tee "$LOG/8_det_a.txt"
python scripts/e0_determinism.py --clip "$CLIP" --out results/det_b --compare results/det_a \
  | tee "$LOG/8_det_b.txt"

step "provenance"
{ git -C third_party/map-anything rev-parse HEAD
  ls "${HF_HOME:-$HOME/.cache/huggingface}/hub/models--facebook--map-anything/snapshots/" 2>/dev/null \
    | sed 's/^/HF snapshot: /'
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
} | tee "$LOG/provenance.txt"
echo; echo "Phase C done. Hand reports/ and results/ back for Phase D."
