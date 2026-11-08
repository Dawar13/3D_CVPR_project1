#!/usr/bin/env bash
# E3 eval, E2-E job only, run in parallel with the main E3 queue (generated from session_queue_e3.sh; same job code)
# E3 eval queue (experiments/E3.md, frozen at tag e3-prereg). Runs in the session-A tree ~/e2x/MapAnything_Exp1 (the
# frozen experiment code of e2x-prereg, the same code the calibration ran with) with the e3_eval configs copied in.
# Dev gates (0 failures) on dev clips, then each eval set once. Out-of-memory failures are reset and rerun
# (scripts/e2x/retry_oom.py), at most 3 passes per job. NPROC processes per job (exp1.shard).
set -uo pipefail
L=~/e2x_logs
NPROC=${NPROC:-3}
source ~/e2x_conda/etc/profile.d/conda.sh
conda activate mapanything
cd ~/e2x/MapAnything_Exp1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# E3 amendment 1 (orchestration): every asset is cached; offline mode, so model loading never depends on GitHub or
# the Hugging Face Hub (a GitHub 504 crashed every model load at 09:00 UTC). torch.hub then falls back to its cache.
export https_proxy=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 HF_HUB_OFFLINE=1
log() { echo "$(date -u '+%F %T') [e3 e2e-lane] $*" | tee -a "$L/session_e3.txt"; }
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
job() {  # job <name> <config> <clips glob> <out> <nproc>: done only if every process exits 0, no run failed, no clip missing
  local name=$1 cfg=$2 clips=$3 out=$4 n=$5 pass k ok miss
  for pass in 1 2 3; do
    waitmem $(( n * 10500 ))
    ok=1
    python -m exp1.shard run --n "$n" --workdir . -- --config "$cfg" --clips "$clips" --out "$out" --save_extras       --mem_fraction 0.13 >> "$L/e3_jobs.txt" 2>&1 || ok=0
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
  log "$name: done, all clips complete, 0 failures"
}
R=configs/e3_eval_real_mapanything.json
job "eval real (E2-E)" $R 'data/clips_e2e/eval__*.npz' results/e3_eval_real_ma_e2e "$NPROC" && log "E3_E2E_EVAL_DONE (parallel lane)"
