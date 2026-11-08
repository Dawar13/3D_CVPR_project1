#!/usr/bin/env bash
# Session A2 queue (experiments/E2X.md X-M, frozen at tag e2x-a2-prereg). Runs from its own tree
# ~/e2x_a2/MapAnything_Exp1 (code from the tag; data/ and ../third_party linked to the session-A tree), beside
# session A on the same GPU. One lane per model:
#   bash scripts/session_queue_a2.sh ma|px
# Lane: X-M dev (3 clips) -> gate -> X-M eval (24 ETH3D eval clips, NPROC processes over alternate clips, merged);
#       M8 dev -> gate -> M8 eval (exp1.shard). A failed gate skips only the eval it guards. Resumable.
set -euo pipefail
LANE=${1:?usage: session_queue_a2.sh ma|px}
C=${E2X_CONDA:-~/e2x_conda}
L=~/e2x_logs
T=~/e2x_a2/MapAnything_Exp1
NPROC=${NPROC:-2}
FRAC=${FRAC:-0.13}
source "$C/etc/profile.d/conda.sh"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date -u '+%F %T') [a2 $LANE] $*" | tee -a "$L/session_a2.txt"; }
case "$LANE" in
  ma) ENV=mapanything; MODEL=mapanything; BK="" ;;
  px) ENV=pi3x; MODEL=pi3x; BK="--backend pi3x" ;;
  *) echo "lane must be ma or px" >&2; exit 2 ;;
esac
conda activate "$ENV"
cd "$T"
log "start, tree $(cat E2X_A2_COMMIT 2>/dev/null || echo '?')"
DEV='data/clips_e2/dev__*__c0.npz'; EVAL='data/clips_e2/eval__*.npz'

xm_eval() {  # NPROC processes over alternate clips, then one merged folder
  local out=results/xm_${LANE}_eval k pids=()
  mkdir -p "$out/_parts"
  python - "$out" "$EVAL" "$NPROC" <<'PY'
import glob, os, sys
out, pat, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
for k, c in enumerate(sorted(glob.glob(pat))):
    d = os.path.join(out, "_parts", str(k % n), "clips")
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, os.path.basename(c))
    if not os.path.lexists(dst):
        os.symlink(os.path.abspath(c), dst)
PY
  for k in $(seq 0 $((NPROC - 1))); do
    python -m exp1.run_xm --config "configs/e2x_xm_$MODEL.json" --clips "$out/_parts/$k/clips/*.npz" \
      --out "$out/_parts/$k/run" --allow_eval --mem_fraction "$FRAC" >> "$out/_parts/$k/log.txt" 2>&1 &
    pids+=($!)
    sleep 20
  done
  local ok=1
  for p in "${pids[@]}"; do wait "$p" || ok=0; done
  python - "$out" <<'PY'
import glob, os, sys
import pandas as pd
out = sys.argv[1]
for name in ("per_run.csv", "state_stats.csv"):
    parts = [pd.read_csv(p) for p in sorted(glob.glob(os.path.join(out, "_parts", "*", "run", name)))]
    if parts:
        pd.concat(parts, ignore_index=True).sort_values("run_key").to_csv(os.path.join(out, name), index=False)
fl = sorted(glob.glob(os.path.join(out, "_parts", "*", "run", "failures.jsonl")))
if fl:
    open(os.path.join(out, "failures.jsonl"), "w").write("".join(open(f).read() for f in fl))
info = sorted(glob.glob(os.path.join(out, "_parts", "*", "run", "run_info.json")))
if info:
    open(os.path.join(out, "run_info.json"), "w").write(open(info[0]).read())
PY
  [ "$ok" = 1 ]
}

log "X-M dev"
python -m exp1.run_xm --config "configs/e2x_xm_$MODEL.json" --clips "$DEV" --out "results/xm_${LANE}_dev" \
  --mem_fraction "$FRAC" >> "$L/a2_${LANE}.txt" 2>&1 || log "X-M dev runner exited with an error"
if python scripts/e2x/a2_gate.py xm --run "results/xm_${LANE}_dev" --backend "$MODEL" 2>&1 | tee -a "$L/session_a2.txt" | grep -q PASS; then
  log "X-M eval"
  if xm_eval; then log "A2_XM_${LANE^^}_EVAL_DONE"; else log "X-M eval: a process exited with an error (see _parts/*/log.txt)"; fi
fi

log "M8 dev"
python -m exp1.shard run --n 1 --workdir . -- $BK --config "configs/e2x_m8_$MODEL.json" --clips "$DEV" \
  --out "results/m8_${LANE}_dev" --save_extras --mem_fraction "$FRAC" >> "$L/a2_${LANE}.txt" 2>&1 || log "M8 dev error"
if python scripts/e2x/a2_gate.py m8 --run "results/m8_${LANE}_dev" 2>&1 | tee -a "$L/session_a2.txt" | grep -q PASS; then
  log "M8 eval"
  python -m exp1.shard run --n "$NPROC" --workdir . -- $BK --config "configs/e2x_m8_$MODEL.json" --clips "$EVAL" \
    --out "results/m8_${LANE}_eval" --save_extras --mem_fraction "$FRAC" >> "$L/a2_${LANE}.txt" 2>&1 &&
    log "A2_M8_${LANE^^}_EVAL_DONE" || log "M8 eval error"
fi
log "A2_${LANE^^}_DONE"
