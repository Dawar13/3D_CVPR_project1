#!/usr/bin/env bash
# LAPTOP side, from the repo root, while VM session A runs (experiments/E2X_instructions.md):
#   [SSH_KEY=key] [EVERY_MIN=20] bash MapAnything_Exp1/scripts/vm_autofetch_e2x.sh user@host [port]
# Every EVERY_MIN minutes, copies to the laptop every result file that is new or changed since the last successful
# copy, shard folders included (they hold the work of a job that is still running), plus ~/e2x_logs.
# Destination: ../3D-CVPR_Exp_vm_backup_e2x/{results,e2x_logs} (outside the repo: a backup, not the results).
# A lost VM then costs at most one interval of GPU work, and the backup's shard folders can be resumed on a new VM.
# When the queue logs SESSION_GPU_DONE it runs scripts/vm_fetch_e2x.sh (the real fetch into
# MapAnything_Exp1/results/) and stops. A failed copy is repeated next time: the VM-side marker moves only after a
# copy has arrived and unpacked; a file still being written is copied again next time.
set -uo pipefail
T=$1; P=${2:-22}
K=${SSH_KEY:+-i $SSH_KEY}
SSH="ssh $K -p $P -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30"
D=${BACKUP:-$(cd .. && pwd)/3D-CVPR_Exp_vm_backup_e2x}
EVERY=${EVERY_MIN:-20}
mkdir -p "$D"
log() { echo "$(date -u '+%F %T') $*" | tee -a "$D/autofetch.txt"; }
round() {
  $SSH "$T" 'cd ~/e2x/MapAnything_Exp1 || exit 2
    touch ~/e2x_logs/.fetch_now
    new=""; [ -f ~/e2x_logs/.fetch_prev ] && new="-newer $HOME/e2x_logs/.fetch_prev"
    dirs=$(ls -d results/e2x_* results/g0_dev results/g1_* results/g2_* 2>/dev/null)
    { [ -n "$dirs" ] && find $dirs -type f $new
      find ~/e2x_logs -maxdepth 1 -type f -printf "../../e2x_logs/%f\n"; } |
      tar czf - -T - --ignore-failed-read --warning=no-file-changed 2>/dev/null
    [ $? -le 1 ]' > "$D/.part.tgz" || return 1
  tar xzf "$D/.part.tgz" -C "$D" 2>/dev/null || return 1   # tar drops the leading ../../ of the log files
  local n; n=$(tar tzf "$D/.part.tgz" | wc -l)
  rm -f "$D/.part.tgz"
  $SSH "$T" 'mv ~/e2x_logs/.fetch_now ~/e2x_logs/.fetch_prev' || return 1
  log "copied $n new or changed files; backup now $(du -sm "$D" | cut -f1) MB"
}
log "autofetch from $T every $EVERY min into $D"
while true; do
  round || log "copy failed (VM unreachable?); trying again in $EVERY min"
  if grep -q SESSION_GPU_DONE "$D/e2x_logs/session.txt" 2>/dev/null; then
    log "SESSION_GPU_DONE seen: final fetch into MapAnything_Exp1/results/"
    if SSH_KEY=${SSH_KEY:-} bash MapAnything_Exp1/scripts/vm_fetch_e2x.sh "$T" "$P"; then log "final fetch done"
    else log "final fetch FAILED: run vm_fetch_e2x.sh by hand"; fi
    exit 0
  fi
  sleep $(( EVERY * 60 ))
done
