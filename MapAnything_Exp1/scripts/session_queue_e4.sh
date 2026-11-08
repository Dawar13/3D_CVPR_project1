#!/usr/bin/env bash
# E4-A eval queue (experiments/E4.md, frozen at tag e4-prereg). One lane per model:
#   bash scripts/session_queue_e4.sh ma     (MapAnything, conda env mapanything)
#   bash scripts/session_queue_e4.sh px     (Pi3X, conda env pi3x)
# Runs in the E4 tree ~/e4/MapAnything_Exp1 (the code at the tag; data/ links to the session-A clips, read only).
# Same orchestration as E3 (scripts/session_queue_e3.sh, E3 amendment 1): offline model loading; a job is done only
# if every process exits 0, no run failed and every clip has rows; out-of-memory failures are reset and rerun
# (scripts/e2x/retry_oom.py), at most 3 passes; NPROC processes per job (exp1.shard). Dev gates first (0 failures and
# vtt rows present), then each eval set once.
set -uo pipefail
M=${1:?model: ma or px}
L=~/e2x_logs
NPROC=${NPROC:-3}
case $M in
  ma) ENV=mapanything; BACK=""; CS=configs/e4_eval_synth_mapanything.json; CR=configs/e4_eval_real_mapanything.json ;;
  px) ENV=pi3x; BACK="--backend pi3x"; CS=configs/e4_eval_synth_pi3x.json; CR=configs/e4_eval_real_pi3x.json ;;
  *) echo "model must be ma or px"; exit 2 ;;
esac
source ~/e2x_conda/etc/profile.d/conda.sh
conda activate $ENV
cd ~/e4/MapAnything_Exp1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export https_proxy=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2   # the VM's CPUs are shared; VtT is CPU work
log() { echo "$(date -u '+%F %T') [e4 $M] $*" | tee -a "$L/session_e4.txt"; }
waitmem() { until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge "$1" ]; do sleep 60; done; }
missing() {  # missing <out> <clips glob>: number of clips of the glob without any row in <out>/per_run.csv
  python - "$1" "$2" <<'PY'
import glob, os, sys
import pandas as pd
out, pat = sys.argv[1], sys.argv[2]
want = {os.path.splitext(os.path.basename(c))[0] for c in glob.glob(pat)}
p = os.path.join(out, "per_run.csv")
have = set(pd.read_csv(p).clip_id) if os.path.exists(p) else set()
print(len(want - have))
PY
}
vtt_rows() {  # vtt_rows <out>: number of vtt rows (the dev gate needs them)
  python -c "import pandas as pd, sys; R = pd.read_csv(sys.argv[1] + '/per_run.csv'); print(int((R.control == 'vtt').sum()))" "$1" 2>/dev/null || echo 0
}
job() {  # job <name> <config> <clips glob> <out> <nproc>
  local name=$1 cfg=$2 clips=$3 out=$4 n=$5 pass k ok miss
  for pass in 1 2 3; do
    waitmem $(( n * 10500 ))
    ok=1
    python -m exp1.shard run --n "$n" --workdir . -- $BACK --config "$cfg" --clips "$clips" --out "$out" --save_extras \
      --mem_fraction 0.13 >> "$L/e4_${M}_jobs.txt" 2>&1 || ok=0
    k=$(python scripts/e2x/retry_oom.py reset "$out" 2>/dev/null || echo 0)
    python scripts/e2x/retry_oom.py dedupe "$out" > /dev/null 2>&1
    python -m exp1.shard merge --out "$out" --clips "$clips" --workdir . > /dev/null 2>&1
    miss=$(missing "$out" "$clips")
    if [ "$ok" = 1 ] && [ "$k" = 0 ] && [ "$miss" = 0 ]; then break; fi
    log "$name: pass $pass incomplete (all processes ok: $ok, out-of-memory resets: $k, clips without rows: $miss)"
    sleep 120
  done
  if [ "$ok" != 1 ] || [ "$miss" != 0 ] || [ -f "$out/failures.jsonl" ]; then
    log "$name: FAILED after $pass passes (processes ok: $ok, clips without rows: $miss, failures: $(cat "$out/failures.jsonl" 2>/dev/null | wc -l))"
    return 1
  fi
  log "$name: done, all clips complete, 0 failures, vtt rows: $(vtt_rows "$out")"
}
log "start (code $(git -C .. describe --tags --always 2>/dev/null || cat ../E4_CODE_TAG 2>/dev/null); VtT constants $(python -c "import json; c = json.load(open('$CS')); print({k: round(v, 4) for k, v in c.items() if k.startswith('vtt_')})"))"
if job "dev synth" $CS 'data/clips_e2/dev__*__c0.npz' results/e4_eval_synth_${M}_dev 3 &&
   job "dev real" $CR 'data/clips_e2x_room/dev__*.npz' results/e4_eval_real_${M}_roomdev 3 &&
   [ "$(vtt_rows results/e4_eval_synth_${M}_dev)" -gt 0 ] && [ "$(vtt_rows results/e4_eval_real_${M}_roomdev)" -gt 0 ]; then
  log "GATE e4 dev: PASS"
  job "eval synth (ETH3D)" $CS 'data/clips_e2/eval__*.npz' results/e4_eval_synth_${M} "$NPROC" && log "E4_${M}_SYNTH_EVAL_DONE"
  job "eval real (rooms)" $CR 'data/clips_e2x_room/eval__*.npz' results/e4_eval_real_${M}_room "$NPROC" && log "E4_${M}_ROOM_EVAL_DONE"
  job "eval real (E2-E)" $CR 'data/clips_e2e/eval__*.npz' results/e4_eval_real_${M}_e2e "$NPROC" && log "E4_${M}_E2E_EVAL_DONE"
else
  log "GATE e4 dev: FAIL (eval not run)"
fi
log "E4_${M}_QUEUE_DONE"
