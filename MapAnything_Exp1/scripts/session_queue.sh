#!/usr/bin/env bash
# The ONE GPU session for the rest of E2 (experiments/E2.md Amendments 10-12). Started by vm_setup_e2.sh as soon
# as MapAnything is installed; resumable (every job resumes from its shard folders). Each job is split over
# $SHARDS processes on the one GPU (exp1/shard.py; outputs identical to one process, tests/test_shard.py).
# MapAnything jobs first, OmniVGGT jobs last (its install finishes meanwhile). A failed gate skips only the
# eval run it guards.
#   1  parity gate vs E1, plus parallel-vs-single identity on one clip      (frozen e2-prereg-C tree)
#   2  E2-C eval, once                                                      (frozen tree; skipped if 1 fails)
#   3  MapAnything dev runs: e2df (E2-D companion + E2-F), e2e (real hints) (e2-prereg-DEF tree)
#   4  MapAnything eval runs whose dev gate passed
#   5  OmniVGGT: checks, dev runs e2d and e2e, then the eval runs whose dev gate passed
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
F=~/3D-CVPR_Exp/MapAnything_Exp1   # frozen tag e2-prereg-C
H=~/e2d/MapAnything_Exp1           # tag e2-prereg-DEF
log() { echo "$(date -u '+%F %T') $*" | tee -a ~/logs/session.txt; }
if [ -z "${SHARDS:-}" ]; then  # one process per 16 GB free and per 4 CPUs, at most 4
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n 1p)
  SHARDS=$(( free / 16384 )); c=$(( $(nproc) / 4 ))
  [ "$c" -lt "$SHARDS" ] && SHARDS=$c
  [ "$SHARDS" -gt 4 ] && SHARDS=4
  [ "$SHARDS" -lt 1 ] && SHARDS=1
fi
shard() { local tree=$1; shift; python -m exp1.shard run --n "$SHARDS" --workdir "$tree" -- "$@"; }
gate() { python scripts/dev_gates.py "$@" 2>&1 | tee -a ~/logs/session.txt; return "${PIPESTATUS[0]}"; }
cd "$H"
conda activate mapanything
log "start, $SHARDS processes per job, DEF tree $(cat "$H/E2DEF_FROZEN_COMMIT" 2>/dev/null || echo '?')"

# ---------------------------------------------------------------- 1-2  parity gate, E2-C eval
log "1 parity gate"
shard "$F" --config configs/e2b_audit.json --clips 'data/clips/eval__*.npz' --out results/parity_rtx
one=$(cd "$F" && ls data/clips/eval__*.npz | sed -n 1p)
(cd "$F" && python -m exp1.run_exp1 --config configs/e2b_audit.json --clips "$one" --out results/parity_single)
set +e
python - <<'PY' 2>&1 | tee -a ~/logs/session.txt
import csv, os, sys
import pandas as pd
from exp1.e2_audit import parity
F = os.path.expanduser("~/3D-CVPR_Exp/MapAnything_Exp1")
pa = parity(pd.read_csv(os.path.expanduser("~/e1_eval_per_run.csv")), pd.read_csv(f"{F}/results/parity_rtx/per_run.csv"))
pa.to_csv(f"{F}/results/parity_rtx/parity.csv", index=False)
print(pa.to_string(index=False))
lim = {"pts_relerr_scale": 0.001, "rre_deg": 0.025, "pts_relerr_metric": 0.005}  # 1/10 of the practical thresholds
m = pa.set_index("metric")
bad = [k for k, v in lim.items() if not m.loc[k, "max_abs_diff"] <= v] + ([] if (m.n_runs == 420).all() else ["n_runs"])
rows = lambda p: {r["run_key"]: {k: v for k, v in r.items() if k not in ("runtime_s", "peak_mem_gb")}
                  for r in csv.DictReader(open(p, newline=""))}
single, par = rows(f"{F}/results/parity_single/per_run.csv"), rows(f"{F}/results/parity_rtx/per_run.csv")
same = len(single) > 0 and all(par.get(k) == v for k, v in single.items())
print(f"PARALLEL_IDENTICAL {same} ({len(single)} rows compared)")
print("PARITY", ("FAIL " + ",".join(bad)) if bad else "PASS")
sys.exit((0 if not bad else 1) + (0 if same else 2))
PY
g=${PIPESTATUS[0]}
set -e
if [ $(( g & 2 )) -ne 0 ]; then SHARDS=1; log "parallel outputs differ from one process: 1 process from now on"; fi
if [ $(( g & 1 )) -ne 0 ]; then
  log "PARITY FAIL: E2-C eval NOT started (needs a decision)"
else
  log "2 E2-C eval from $(cat "$F/E2C_FROZEN_COMMIT")"
  shard "$F" --config configs/e2c_full.json --clips 'data/clips_e2/eval__*.npz' --out results/e2c_eval --save_extras
  for n in 4 20; do
    shard "$F" --config configs/e2c_views.json --clips "data/clips_n${n}/eval__*.npz" --out "results/e2c_n${n}_eval" --save_extras
  done
  log "E2C_EVAL_DONE"
fi

# ---------------------------------------------------------------- 3-4  MapAnything dev, gates, eval
log "3 MapAnything dev runs"
shard "$H" --config configs/e2df_mapanything.json --clips 'data/clips_e2/dev__*__c0.npz' --out results/e2df_dev --save_extras
until [ -f ~/logs/ready_mono ] || [ -f ~/logs/FAILED_mono ]; do sleep 10; done   # the AI depth hint (setup_ai)
[ -f ~/logs/FAILED_mono ] && log "AI depth FAILED (see ~/logs/setup_ai.txt): its E2-E conditions fail, so the E2-E gates skip the eval runs"
shard "$H" --config configs/e2e_mapanything.json --clips 'data/clips_e2e/dev__*.npz' --out results/e2e_ma_dev --save_extras
log "4 MapAnything eval runs"
if gate ma_e2df --run results/e2df_dev --config configs/e2df_mapanything.json; then
  shard "$H" --config configs/e2df_mapanything.json --clips 'data/clips_e2/eval__*.npz' --out results/e2df_eval --save_extras
  log "E2DF_MA_EVAL_DONE"
fi
if gate e2e --run results/e2e_ma_dev --config configs/e2e_mapanything.json; then
  shard "$H" --config configs/e2e_mapanything.json --clips 'data/clips_e2e/eval__*.npz' --out results/e2e_ma_eval --save_extras
  log "E2E_MA_EVAL_DONE"
fi

# ---------------------------------------------------------------- 5  OmniVGGT
conda deactivate
until [ -f ~/logs/ready_omnivggt ]; do
  [ -f ~/logs/FAILED_omnivggt ] && { log "OmniVGGT setup failed, see ~/logs/setup_ov.txt"; exit 1; }
  sleep 10
done
conda activate omnivggt
log "5 OmniVGGT checks and dev runs"
python -m exp1.run_exp1 --backend omnivggt --config configs/e2d_checks.json --clips 'data/clips/dev__*.npz' \
  --out results/e2d_checks --save_extras
python scripts/e2d_checks.py --run results/e2d_checks 2>&1 | tee ~/logs/e2d_checks.txt || true
shard "$H" --backend omnivggt --config configs/e2d_full.json --clips 'data/clips_e2/dev__*__c0.npz' --out results/e2d_dev --save_extras
shard "$H" --backend omnivggt --config configs/e2e_omnivggt.json --clips 'data/clips_e2e/dev__*.npz' --out results/e2e_ov_dev --save_extras
if gate ov_e2d --run results/e2d_dev --config configs/e2d_full.json --checks results/e2d_checks; then
  shard "$H" --backend omnivggt --config configs/e2d_full.json --clips 'data/clips_e2/eval__*.npz' --out results/e2d_eval --save_extras
  log "E2D_OV_EVAL_DONE"
fi
if gate e2e --run results/e2e_ov_dev --config configs/e2e_omnivggt.json; then
  shard "$H" --backend omnivggt --config configs/e2e_omnivggt.json --clips 'data/clips_e2e/eval__*.npz' --out results/e2e_ov_eval --save_extras
  log "E2E_OV_EVAL_DONE"
fi
log "SESSION_GPU_DONE"
