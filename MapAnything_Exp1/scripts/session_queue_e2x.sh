#!/usr/bin/env bash
# GPU queue of VM session A (experiments/E2X.md section 6). Started by vm_setup_e2x.sh when MapAnything and the clips
# are ready; resumable (every job resumes from its shard folders). Ordered by priority, so an early stop loses only
# the tail. A failed dev gate skips only the eval run it guards. mono_depth rewrites the SceneFun3D clips in place,
# so no job touches them before ~/logs/ready_mono.
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
T=~/x/MapAnything_Exp1
log() { echo "$(date -u '+%F %T') $*" | tee -a ~/logs/session.txt; }
if [ -z "${SHARDS:-}" ]; then  # one process per 16 GB free and per 4 CPUs, at most 4
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n 1p)
  SHARDS=$(( free / 16384 )); c=$(( $(nproc) / 4 ))
  [ "$c" -lt "$SHARDS" ] && SHARDS=$c
  [ "$SHARDS" -gt 4 ] && SHARDS=4
  [ "$SHARDS" -lt 1 ] && SHARDS=1
fi
shard() { python -m exp1.shard run --n "$SHARDS" --workdir "$T" -- "$@" --save_extras; }
gate() { python scripts/dev_gates.py "$@" 2>&1 | tee -a ~/logs/session.txt; return "${PIPESTATUS[0]}"; }
wait_for() {  # wait_for ready_x FAILED_x what
  until [ -f ~/logs/$1 ] || [ -f ~/logs/$2 ]; do sleep 10; done
  [ -f ~/logs/$1 ] || { log "$3 setup FAILED: its jobs are skipped"; return 1; }
}
ROOM_DEV='data/clips_e2x_room/dev__*.npz'; ROOM_EVAL='data/clips_e2x_room/eval__*.npz'
CAL='data/clips_e2x_*/cal__*.npz'
cd "$T"
log "start e2x session A, $SHARDS processes per job, tree $(cat E2X_FROZEN_COMMIT 2>/dev/null || echo '?')"

# ------------------------------------------------------------------ MapAnything
conda activate mapanything
log "MA 1 E3-cal synthetic on ETH3D dev"
shard --config configs/e2x_e3cal_synth_mapanything.json --clips 'data/clips_e2/dev__*__c0.npz' --out results/e2x_e3cal_synth_ma_eth3d
wait_for ready_mono FAILED_mono "AI depth" || true
log "MA 2 E3-cal on cal clips (synthetic, real) and SceneFun3D dev clips (real)"
shard --config configs/e2x_e3cal_synth_mapanything.json --clips "$CAL" --out results/e2x_e3cal_synth_ma_cal
shard --config configs/e2x_e3cal_real_mapanything.json --clips "$CAL" --out results/e2x_e3cal_real_ma_cal
shard --config configs/e2x_e3cal_real_mapanything.json --clips 'data/clips_e2x_room/dev__*.npz' --out results/e2x_e3cal_real_ma_roomdev
log "MA 3 room clips: dev, gate, eval"
shard --config configs/e2x_real_mapanything.json --clips "$ROOM_DEV" --out results/e2x_room_ma_dev
if gate e2e --run results/e2x_room_ma_dev --config configs/e2x_real_mapanything.json; then
  shard --config configs/e2x_real_mapanything.json --clips "$ROOM_EVAL" --out results/e2x_room_ma_eval
  log "ROOM_MA_EVAL_DONE"
fi
conda deactivate

# ------------------------------------------------------------------ Pi3X
if wait_for ready_pi3x FAILED_pi3x Pi3X; then
  conda activate pi3x
  log "PX 1 checks, ETH3D dev, gate, eval"
  python -m exp1.run_exp1 --backend pi3x --config configs/e2x_pi3x_checks.json --clips 'data/clips/dev__*.npz' \
    --out results/e2x_pi3x_checks --save_extras
  shard --backend pi3x --config configs/e2x_pi3x_eth3d.json --clips 'data/clips_e2/dev__*__c0.npz' --out results/e2x_pi3x_eth3d_dev
  if gate ov_e2d --run results/e2x_pi3x_eth3d_dev --config configs/e2x_pi3x_eth3d.json --checks results/e2x_pi3x_checks; then
    shard --backend pi3x --config configs/e2x_pi3x_eth3d.json --clips 'data/clips_e2/eval__*.npz' --out results/e2x_pi3x_eth3d_eval
    log "PX_ETH3D_EVAL_DONE"
  fi
  log "PX 2 real hints: E2-E clips and room clips"
  shard --backend pi3x --config configs/e2x_real_pi3x.json --clips 'data/clips_e2e/dev__*.npz' --out results/e2x_e2e_px_dev
  if gate e2e --run results/e2x_e2e_px_dev --config configs/e2x_real_pi3x.json; then
    shard --backend pi3x --config configs/e2x_real_pi3x.json --clips 'data/clips_e2e/eval__*.npz' --out results/e2x_e2e_px_eval
    log "PX_E2E_EVAL_DONE"
  fi
  shard --backend pi3x --config configs/e2x_real_pi3x.json --clips "$ROOM_DEV" --out results/e2x_room_px_dev
  if gate e2e --run results/e2x_room_px_dev --config configs/e2x_real_pi3x.json; then
    shard --backend pi3x --config configs/e2x_real_pi3x.json --clips "$ROOM_EVAL" --out results/e2x_room_px_eval
    log "ROOM_PX_EVAL_DONE"
  fi
  log "PX 3 E3-cal"
  shard --backend pi3x --config configs/e2x_e3cal_synth_pi3x.json --clips 'data/clips_e2/dev__*__c0.npz' --out results/e2x_e3cal_synth_px_eth3d
  shard --backend pi3x --config configs/e2x_e3cal_synth_pi3x.json --clips "$CAL" --out results/e2x_e3cal_synth_px_cal
  shard --backend pi3x --config configs/e2x_e3cal_real_pi3x.json --clips "$CAL" --out results/e2x_e3cal_real_px_cal
  conda deactivate
fi

# ------------------------------------------------------------------ OmniVGGT (lowest priority)
if wait_for ready_omnivggt FAILED_omnivggt OmniVGGT; then
  conda activate omnivggt
  log "OV 1 room clips: dev, gate, eval"
  shard --backend omnivggt --config configs/e2x_real_omnivggt.json --clips "$ROOM_DEV" --out results/e2x_room_ov_dev
  if gate e2e --run results/e2x_room_ov_dev --config configs/e2x_real_omnivggt.json; then
    shard --backend omnivggt --config configs/e2x_real_omnivggt.json --clips "$ROOM_EVAL" --out results/e2x_room_ov_eval
    log "ROOM_OV_EVAL_DONE"
  fi
  log "OV 2 E3-cal"
  shard --backend omnivggt --config configs/e2x_e3cal_synth_omnivggt.json --clips 'data/clips_e2/dev__*__c0.npz' --out results/e2x_e3cal_synth_ov_eth3d
  shard --backend omnivggt --config configs/e2x_e3cal_synth_omnivggt.json --clips "$CAL" --out results/e2x_e3cal_synth_ov_cal
  shard --backend omnivggt --config configs/e2x_e3cal_real_omnivggt.json --clips "$CAL" --out results/e2x_e3cal_real_ov_cal
fi
log "SESSION_GPU_DONE"
