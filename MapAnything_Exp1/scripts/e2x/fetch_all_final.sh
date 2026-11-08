#!/usr/bin/env bash
# LAPTOP side, from the repo root, at the end of the 2026-11-07/08 A100 session: every result into
# MapAnything_Exp1/results/ (tracked in git), shard / part folders and clip links left on the VM.
#   SSH_KEY=key bash MapAnything_Exp1/scripts/e2x/fetch_all_final.sh user@host [port]
#   1. session A (~/e2x tree: e2x_*, g*), via vm_fetch_e2x.sh, plus the E3 eval runs (e3_eval_*) of the same tree;
#   2. session A2 (~/e2x_a2 tree: xm_*, m8_*), into results/ under the same names;
#   3. the TUM clip exclusions (data/clips_tum/excluded.csv) into results/e2x_tum_clip_check/;
#   4. every VM log (~/e2x_logs) into results/vm_logs_e2x/.
set -euo pipefail
T=$1; P=${2:-22}
SSH="ssh ${SSH_KEY:+-i $SSH_KEY} -p $P -o BatchMode=yes -o ConnectTimeout=25 -o ServerAliveInterval=30"
R=MapAnything_Exp1/results
SSH_KEY=${SSH_KEY:-} bash MapAnything_Exp1/scripts/vm_fetch_e2x.sh "$T" "$P"
$SSH "$T" 'cd ~/e2x/MapAnything_Exp1 && tar czf - --exclude=_shards $(ls -d results/e3_eval_* 2>/dev/null)' | tar xzf - -C MapAnything_Exp1
$SSH "$T" 'cd ~/e2x_a2/MapAnything_Exp1 && tar czf - --exclude=_parts --exclude=_shards --exclude=_discarded $(ls -d results/xm_* results/m8_* 2>/dev/null) E2X_A2_COMMIT' |
  tar xzf - -C MapAnything_Exp1 --transform 's#^E2X_A2_COMMIT#results/xm_E2X_A2_COMMIT.txt#'
$SSH "$T" 'cat ~/e2x/MapAnything_Exp1/data/clips_tum/excluded.csv' > "$R/e2x_tum_clip_check/excluded.csv"
$SSH "$T" 'cd ~/e2x_logs && tar czf - .' | tar xzf - -C "$R/vm_logs_e2x"
ls -d $R/e2x_* $R/e3_eval_* $R/xm_* $R/m8_* 2>/dev/null | tr '\n' ' '
echo
echo "fetched; check failures.jsonl files and the session logs before analysing."
