#!/usr/bin/env bash
# LAPTOP side, run from the repo root: copy every E2 result folder and the VM logs back into
# MapAnything_Exp1/results/ (tracked in git). Shard folders are skipped: merge() already moved their
# contents into the merged run folders.
#   bash MapAnything_Exp1/scripts/vm_fetch_e2.sh user@host [port]
set -euo pipefail
T=$1; P=${2:-22}
SSH="ssh -p $P -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30"
F_RUNS="parity_rtx parity_single e2c_eval e2c_n4_eval e2c_n20_eval"
H_RUNS="e2df_dev e2df_eval e2e_ma_dev e2e_ma_eval e2d_checks e2d_dev e2d_eval e2e_ov_dev e2e_ov_eval e2e_mono"
$SSH "$T" "cd ~/3D-CVPR_Exp/MapAnything_Exp1 && tar czf - --exclude=_shards \$(cd results && for r in $F_RUNS; do [ -d \$r ] && echo results/\$r; done)" \
  | tar xzf - -C MapAnything_Exp1
$SSH "$T" "cd ~/e2d/MapAnything_Exp1 && tar czf - --exclude=_shards \$(cd results && for r in $H_RUNS; do [ -d \$r ] && echo results/\$r; done)" \
  | tar xzf - -C MapAnything_Exp1
mkdir -p MapAnything_Exp1/results/vm_logs_e2
$SSH "$T" "cd ~/logs && tar czf - ." | tar xzf - -C MapAnything_Exp1/results/vm_logs_e2
ls MapAnything_Exp1/results
echo "fetched. Check, then tell the user the VM can be switched off."
