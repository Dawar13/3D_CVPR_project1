#!/usr/bin/env bash
# Gate track, DEV clips only (experiments/G.md). Run on the GPU machine from anywhere:
#   bash MapAnything_Exp1/scripts/g_run_dev.sh smoke    AI depth, unit tests and a one-clip run on both models
#   bash MapAnything_Exp1/scripts/g_run_dev.sh full     every dev run and the summaries (resumes; repeats nothing done)
#   bash MapAnything_Exp1/scripts/g_run_dev.sh pack     results/g_dev_results.tgz, to send back
# Env names can be overridden: MA_ENV, OV_ENV, AI_ENV. GPU: CUDA_VISIBLE_DEVICES as usual.
set -euo pipefail
cd "$(dirname "$0")/.."
MA_ENV=${MA_ENV:-mapanything}
OV_ENV=${OV_ENV:-omnivggt}
AI_ENV=${AI_ENV:-aidepth}
ROOMS='data/clips_e2e/dev__*.npz'
ETH='data/clips_e2/dev__*.npz'
ROOM_ONE=$(ls data/clips_e2e/dev__*.npz | head -1)
ETH_ONE=$(ls data/clips_e2/dev__*.npz | head -1)
mkdir -p results/g_logs
source "$(conda info --base)/etc/profile.d/conda.sh"

log() { echo "[$(date -u '+%F %T')] $*" | tee -a results/g_logs/progress.txt; }

mono() {
  conda activate "$AI_ENV"
  log "AI depth on the dev rooms only"
  mkdir -p results/g0_dev
  python -m exp1.mono_depth --clips "$ROOMS" --record results/g0_dev/mono_depth_dev.csv 2>&1 | tee results/g_logs/mono.txt
  conda deactivate
  conda activate "$MA_ENV"
  python -m exp1.g_analysis verify_mono --record results/g0_dev/mono_depth_dev.csv \
    --reference results/e2e_mono/mono_depth.csv 2>&1 | tee -a results/g_logs/mono.txt
  python -m exp1.g_analysis decompose --clips "$ROOMS" --out results/g0_dev/depth_decompose.csv \
    2>&1 | tee results/g_logs/g0.txt
  conda deactivate
}

smoke() {
  mono
  for pair in "$MA_ENV mapanything ma" "$OV_ENV omnivggt ov"; do
    read -r env backend tag <<<"$pair"
    conda activate "$env"
    log "$backend: unit tests"
    python -m pytest tests/test_gate.py -q -rs 2>&1 | tee "results/g_logs/pytest_$tag.txt" | tail -15 || true
    grep -Eq '[0-9]+ passed' "results/g_logs/pytest_$tag.txt" && ! grep -Eq '[0-9]+ (failed|error)' "results/g_logs/pytest_$tag.txt" \
      || { log "$backend: UNIT TESTS FAILED, stop and report results/g_logs/pytest_$tag.txt"; exit 1; }
    log "$backend: one room clip and one ETH3D clip, with the gate checks"
    python -m exp1.run_gate --config configs/g1_oracle.json --clips "$ROOM_ONE" --out "results/g_smoke_${tag}_rooms" --backend "$backend" \
      2>&1 | tee "results/g_logs/smoke_${tag}_rooms.txt" | tail -8
    python -m exp1.run_gate --config configs/g1_oracle.json --clips "$ETH_ONE" --out "results/g_smoke_${tag}_eth" --backend "$backend" \
      2>&1 | tee "results/g_logs/smoke_${tag}_eth.txt" | tail -8
    python -m exp1.run_gate --config configs/g2_signal.json --clips "$ROOM_ONE" --out "results/g_smoke_${tag}_tokens" --backend "$backend" \
      --no_checks 2>&1 | tee "results/g_logs/smoke_${tag}_tokens.txt" | tail -4
    for d in "results/g_smoke_${tag}_rooms" "results/g_smoke_${tag}_eth" "results/g_smoke_${tag}_tokens"; do
      [ ! -f "$d/failures.jsonl" ] || { log "$backend: FAILED RUNS, stop and report $d/failures.jsonl"; exit 1; }
    done
    conda deactivate
  done
  log "SMOKE PASSED. Read the [check] lines in results/g_logs/smoke_*_rooms.txt, then run: full"
  grep -h '\[check\]' results/g_logs/smoke_*_rooms.txt results/g_logs/smoke_*_eth.txt || true
}

full() {
  [ -f results/g0_dev/mono_depth_dev.csv ] || mono
  for pair in "$MA_ENV mapanything ma" "$OV_ENV omnivggt ov"; do
    read -r env backend tag <<<"$pair"
    conda activate "$env"
    for job in "g1_oracle g1" "g2_signal g2"; do
      read -r cfg name <<<"$job"
      log "$backend $name rooms"
      python -m exp1.run_gate --config "configs/$cfg.json" --clips "$ROOMS" --out "results/${name}_${tag}_rooms_dev" --backend "$backend" \
        2>&1 | tee -a "results/g_logs/${name}_${tag}_rooms.txt"
      log "$backend $name eth"
      python -m exp1.run_gate --config "configs/$cfg.json" --clips "$ETH" --out "results/${name}_${tag}_eth_dev" --backend "$backend" \
        2>&1 | tee -a "results/g_logs/${name}_${tag}_eth.txt"
    done
    conda deactivate
  done
  conda activate "$MA_ENV"
  python -m exp1.g_analysis oracle --runs 'results/g1_*_dev' --out results/g1_summary 2>&1 | tee results/g_logs/g1_summary.txt
  python -m exp1.g_analysis signal --runs 'results/g2_*_dev' --out results/g2_summary 2>&1 | tee results/g_logs/g2_summary.txt
  conda deactivate
  log "FULL DONE. Run: pack"
}

pack() {
  tar czf results/g_dev_results.tgz -C results --exclude='g_dev_results.tgz' \
    $(cd results && ls -d g0_dev g1_* g2_* g_logs g_smoke_* 2>/dev/null)
  ls -lh results/g_dev_results.tgz
}

case "${1:-}" in
  smoke) smoke ;;
  full) full ;;
  pack) pack ;;
  *) echo "usage: $0 smoke|full|pack"; exit 2 ;;
esac
