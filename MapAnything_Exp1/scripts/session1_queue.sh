#!/usr/bin/env bash
# Session 1 GPU queue (experiments/E2.md Amendments 10-11). Started by vm_setup_e2.sh once MapAnything is
# installed; resumable (every job resumes from its shard folders). Each job is split over $SHARDS processes
# on the one GPU (exp1/shard.py; outputs identical to one process, tests/test_shard.py).
#   1  parity gate: E1 conditions on the E1 eval clips vs E1, plus parallel-vs-single identity on one clip
#   2  E2-C eval, once, from the frozen e2-prereg-C tree            (skipped if the parity gate fails)
#   3  E2-D MapAnything companion, dev clips
#   4  E2-D OmniVGGT checks, dev clips                              (waits for the OmniVGGT install)
#   5  E2-D OmniVGGT grid, dev clips
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
F=~/3D-CVPR_Exp/MapAnything_Exp1   # frozen tag e2-prereg-C
H=~/e2d/MapAnything_Exp1           # HEAD
log() { echo "$(date -u '+%F %T') $*" | tee -a ~/logs/session1.txt; }
if [ -z "${SHARDS:-}" ]; then  # one process per 16 GB free and per 4 CPUs, at most 4
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  SHARDS=$(( free / 16384 )); c=$(( $(nproc) / 4 ))
  [ "$c" -lt "$SHARDS" ] && SHARDS=$c
  [ "$SHARDS" -gt 4 ] && SHARDS=4
  [ "$SHARDS" -lt 1 ] && SHARDS=1
fi
shard() { local tree=$1; shift; python -m exp1.shard run --n "$SHARDS" --workdir "$tree" -- "$@"; }
cd "$H"
conda activate mapanything
log "start, $SHARDS processes per job"

log "1 parity gate"
shard "$F" --config configs/e2b_audit.json --clips 'data/clips/eval__*.npz' --out results/parity_rtx
one=$(cd "$F" && ls data/clips/eval__*.npz | head -1)
(cd "$F" && python -m exp1.run_exp1 --config configs/e2b_audit.json --clips "$one" --out results/parity_single)
set +e
python - <<'PY' 2>&1 | tee -a ~/logs/session1.txt
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
gate=${PIPESTATUS[0]}
set -e
if [ $(( gate & 2 )) -ne 0 ]; then SHARDS=1; log "parallel outputs differ from one process: 1 process from now on"; fi
if [ $(( gate & 1 )) -ne 0 ]; then
  log "PARITY FAIL: E2-C eval NOT started (needs a decision); continuing with the E2-D dev jobs"
else
  log "2 E2-C eval from $(cat "$F/E2C_FROZEN_COMMIT")"
  shard "$F" --config configs/e2c_full.json --clips 'data/clips_e2/eval__*.npz' --out results/e2c_eval --save_extras
  for n in 4 20; do
    shard "$F" --config configs/e2c_views.json --clips "data/clips_n${n}/eval__*.npz" --out "results/e2c_n${n}_eval" --save_extras
  done
  log "E2C_EVAL_DONE"
fi

log "3 E2-D MapAnything companion, dev"
shard "$H" --config configs/e2d_mapanything.json --clips 'data/clips_e2/dev__*__c0.npz' --out results/e2d_ma_dev --save_extras
conda deactivate
until [ -f ~/logs/ready_omnivggt ]; do
  [ -f ~/logs/FAILED_omnivggt ] && { log "OmniVGGT setup failed, see ~/logs/setup_ov.txt"; exit 1; }
  sleep 10
done
conda activate omnivggt
log "4 E2-D checks"
python -m exp1.run_exp1 --backend omnivggt --config configs/e2d_checks.json --clips 'data/clips/dev__*.npz' \
  --out results/e2d_checks --save_extras
python scripts/e2d_checks.py --run results/e2d_checks 2>&1 | tee ~/logs/e2d_checks.txt
log "5 E2-D OmniVGGT grid, dev"
shard "$H" --backend omnivggt --config configs/e2d_full.json --clips 'data/clips_e2/dev__*__c0.npz' --out results/e2d_dev --save_extras
log "SESSION1_GPU_DONE"
