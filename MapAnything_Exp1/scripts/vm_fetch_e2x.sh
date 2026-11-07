#!/usr/bin/env bash
# LAPTOP side, from the repo root: copy every result folder of VM session A (experiments/E2X.md) and the VM logs into
# MapAnything_Exp1/results/ (tracked in git). Shard folders are skipped: merge() already moved their contents.
#   bash MapAnything_Exp1/scripts/vm_fetch_e2x.sh user@host [port]
set -euo pipefail
T=$1; P=${2:-22}
SSH="ssh ${SSH_KEY:+-i $SSH_KEY} -p $P -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30"   # SSH_KEY=/path/key.pem if needed
$SSH "$T" 'cd ~/e2x/MapAnything_Exp1 && tar czf - --exclude=_shards $(ls -d results/e2x_* results/g0_dev results/g1_* results/g2_* 2>/dev/null)' \
  | tar xzf - -C MapAnything_Exp1
mkdir -p MapAnything_Exp1/results/vm_logs_e2x
$SSH "$T" "cd ~/e2x_logs && tar czf - ." | tar xzf - -C MapAnything_Exp1/results/vm_logs_e2x
ls -d MapAnything_Exp1/results/e2x_* MapAnything_Exp1/results/g* 2>/dev/null
echo "fetched. Check SESSION_GPU_DONE in results/vm_logs_e2x/session.txt before switching the VM off."
