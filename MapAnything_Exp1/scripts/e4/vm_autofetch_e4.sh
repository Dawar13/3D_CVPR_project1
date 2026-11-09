#!/usr/bin/env bash
# LAPTOP side, from the repo root, while the E4 queues run (experiments/E4.md §7):
#   SSH_KEY=key [EVERY_MIN=20] bash MapAnything_Exp1/scripts/e4/vm_autofetch_e4.sh user@host [port]
# Every EVERY_MIN minutes, copies every E4 result file that is new or changed since the last successful copy (shard
# folders included) plus the E4 logs, into ../3D-CVPR_Exp_vm_backup_e4 (outside the repo: a backup, not the results).
# Same design as scripts/vm_autofetch_e2x.sh: the VM-side marker moves only after a copy has arrived and unpacked.
# Stops when both lanes have logged E4_<model>_QUEUE_DONE.
set -uo pipefail
T=$1; P=${2:-22}
K=${SSH_KEY:+-i $SSH_KEY}
SSH="ssh $K -p $P -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30"
D=${BACKUP:-$(cd .. && pwd)/3D-CVPR_Exp_vm_backup_e4}
EVERY=${EVERY_MIN:-20}
mkdir -p "$D"
log() { echo "$(date -u '+%F %T') $*" | tee -a "$D/autofetch.txt"; }
round() {
  $SSH "$T" 'cd ~/e4/MapAnything_Exp1 || exit 2
    touch ~/e2x_logs/.fetch_e4_now
    new=""; [ -f ~/e2x_logs/.fetch_e4_prev ] && new="-newer $HOME/e2x_logs/.fetch_e4_prev"
    dirs=$(ls -d results/e4_* 2>/dev/null)
    { [ -n "$dirs" ] && find $dirs -type f $new
      ls ~/e2x_logs/session_e4.txt ~/e2x_logs/e4_*_jobs.txt 2>/dev/null | sed "s|^$HOME/e2x_logs/|../../e2x_logs/|"; } |
      tar czf - -T - --ignore-failed-read --warning=no-file-changed 2>/dev/null
    [ $? -le 1 ]' > "$D/.part.tgz" || return 1
  tar xzf "$D/.part.tgz" -C "$D" 2>/dev/null || return 1
  local n; n=$(tar tzf "$D/.part.tgz" | wc -l)
  rm -f "$D/.part.tgz"
  $SSH "$T" 'mv ~/e2x_logs/.fetch_e4_now ~/e2x_logs/.fetch_e4_prev' || return 1
  log "copied $n new or changed files; backup now $(du -sm "$D" | cut -f1) MB"
}
log "E4 autofetch from $T every $EVERY min into $D"
while true; do
  round || log "copy failed (VM unreachable?); trying again in $EVERY min"
  if [ "$(grep -c 'QUEUE_DONE' "$D/e2x_logs/session_e4.txt" 2>/dev/null || echo 0)" -ge 2 ]; then
    log "both E4 lanes done; last copy taken"
    exit 0
  fi
  sleep $(( EVERY * 60 ))
done
