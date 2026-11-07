#!/usr/bin/env bash
# LAPTOP side, from the repo root (experiments/E2X.md section 6).
#   bash MapAnything_Exp1/scripts/vm_launch_e2x.sh build             pack the tree: experiment code from the tag
#                                                                     e2x-prereg, job scripts from HEAD; clip checksums
#   [SSH_KEY=key] [GPU_BUDGET_GB=32] bash MapAnything_Exp1/scripts/vm_launch_e2x.sh send user@host [port]
# `send` uploads the code first and starts scripts/vm_setup_e2x.sh, so the installs begin at once. The clips are then
# uploaded while the envs install, RESUMABLY: the VM checks every file's sha256 and only missing or broken files are
# sent, in small batches, until all verify (a dropped connection loses nothing; 2026-11-07 a single stream was cut).
# The GPU queue waits for them. On the VM everything lives in ~/e2x, ~/e2x_conda and ~/e2x_logs (nothing else is
# touched). Start scripts/vm_autofetch_e2x.sh afterwards: it copies the results to the laptop every 20 minutes.
set -euo pipefail
B=${BUNDLE:-$(cd .. && pwd)/3D-CVPR_Exp_bundle_e2x}
FROZEN="MapAnything_Exp1/exp1 MapAnything_Exp1/configs MapAnything_Exp1/tests MapAnything_Exp1/results/e2e_mono/mono_depth.csv"

clip_files() {  # what the VM needs (run inside MapAnything_Exp1): excluded clips stay home
  { find data/clips data/clips_e2 data/clips_e2x_room data/clips_e2x_cal -type f -not -path '*/excluded/*'
    ls data/clips_e2e/*.npz data/clips_e2e/*.csv; } | LC_ALL=C sort
}

build() {
  local tag head
  tag=$(git rev-parse -q --verify "e2x-prereg^{commit}") || { echo "tag e2x-prereg missing: freeze E2X first" >&2; exit 1; }
  head=$(git rev-parse HEAD)
  # the experiment (code, configs, tests) must be exactly the frozen one; only job scripts may be newer
  git diff --quiet "$tag" HEAD -- MapAnything_Exp1/exp1 MapAnything_Exp1/configs MapAnything_Exp1/tests ||
    { echo "exp1/, configs/ or tests/ changed since e2x-prereg: not allowed for session A" >&2; exit 1; }
  git diff --quiet HEAD -- MapAnything_Exp1/scripts || { echo "uncommitted changes in scripts/: commit them first" >&2; exit 1; }
  # no clip may be added after the freeze; excluded clips live in an excluded/ subfolder, outside every list
  (cd MapAnything_Exp1 && for d in data/clips_e2x_room data/clips_e2x_cal; do
     [ -f "$d/excluded.csv" ] || { echo "$d/excluded.csv missing: run the clip checks first" >&2; exit 1; }; done)
  rm -rf "$B" && mkdir -p "$B/t"
  git archive "$tag" $FROZEN | tar x -C "$B/t"
  git archive HEAD MapAnything_Exp1/scripts | tar x -C "$B/t"
  echo "$tag" > "$B/t/MapAnything_Exp1/E2X_FROZEN_COMMIT"
  echo "$head" > "$B/t/MapAnything_Exp1/E2X_SCRIPTS_COMMIT"
  tar czf "$B/e2x_tree.tgz" -C "$B/t" MapAnything_Exp1
  rm -rf "$B/t"
  git show "HEAD:MapAnything_Exp1/scripts/vm_setup_e2x.sh" > "$B/vm_setup_e2x.sh"
  (cd MapAnything_Exp1 && clip_files | xargs -d '\n' sha256sum) > "$B/clips.sha"
  ls -la "$B"
  echo "$(wc -l < "$B/clips.sha") clip files hashed"
  echo "bundle ready in $B (experiment e2x-prereg ${tag:0:7}, scripts ${head:0:7})"
}

upload_clips() {  # upload_clips: resumable, verified by sha256 on the VM
  local round n f s size batch
  for round in $(seq 1 40); do
    if ! $SCP "$B/clips.sha" "$T:~/e2x_logs/clips.sha"; then sleep 15; continue; fi
    if ! n=$($SSH "$T" 'cd ~/e2x/MapAnything_Exp1 && sha256sum -c ~/e2x_logs/clips.sha 2>/dev/null | grep -v ": OK$" |
          sed "s/: FAILED.*//" > ~/e2x_logs/clips_missing.txt; wc -l < ~/e2x_logs/clips_missing.txt'); then
      sleep 15; continue
    fi
    n=$(echo "$n" | tr -d '\r ')
    echo "clips, round $round: $n of $(wc -l < "$B/clips.sha") files missing or broken"
    [ "$n" = 0 ] && return 0
    if ! $SCP "$T:~/e2x_logs/clips_missing.txt" "$B/clips_missing.txt"; then sleep 15; continue; fi
    batch=(); size=0
    while IFS= read -r f; do
      f=${f%$'\r'}; [ -n "$f" ] || continue
      s=$(stat -c %s "MapAnything_Exp1/$f")
      if [ ${#batch[@]} -gt 0 ] && [ $((size + s)) -gt 41943040 ]; then   # batches of at most 40 MB
        tar cf - -C MapAnything_Exp1 "${batch[@]}" | $SSH "$T" 'cd ~/e2x/MapAnything_Exp1 && tar xf -' ||
          echo "  batch interrupted (checked again next round)"
        batch=(); size=0
      fi
      batch+=("$f"); size=$((size + s))
    done < "$B/clips_missing.txt"
    if [ ${#batch[@]} -gt 0 ]; then
      tar cf - -C MapAnything_Exp1 "${batch[@]}" | $SSH "$T" 'cd ~/e2x/MapAnything_Exp1 && tar xf -' ||
        echo "  batch interrupted (checked again next round)"
    fi
  done
  echo "clips did not verify after 40 rounds" >&2
  return 1
}

send() {
  T=$1; local P=${2:-22}
  local K=${SSH_KEY:+-i $SSH_KEY}   # SSH_KEY=/path/to/key when the VM needs its own key
  SSH="ssh $K -p $P -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30"
  SCP="scp -q $K -P $P -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new"
  [ -f "$B/e2x_tree.tgz" ] && [ -f "$B/clips.sha" ] || { echo "run '$0 build' first" >&2; exit 1; }
  $SSH "$T" 'command -v tmux >/dev/null || (sudo apt-get -qq update && sudo apt-get -qq install -y tmux)'
  $SCP "$B/e2x_tree.tgz" "$B/vm_setup_e2x.sh" "$T:~/"
  $SSH "$T" "mkdir -p ~/e2x_logs && tmux has-session -t =e2x_main 2>/dev/null || tmux new-session -d -s e2x_main \"GPU_BUDGET_GB=${GPU_BUDGET_GB:-32} bash ~/vm_setup_e2x.sh 2>&1 | tee -a ~/e2x_logs/e2x_main.txt\""
  $SSH "$T" 'until [ -d ~/e2x/MapAnything_Exp1 ]; do sleep 3; done'
  upload_clips
  $SSH "$T" 'bash ~/vm_setup_e2x.sh clips && tmux ls'
  echo "started. progress: $SSH $T 'cd ~/e2x/MapAnything_Exp1 && ~/e2x_conda/envs/mapanything/bin/python scripts/session_progress_e2x.py; tail -5 ~/e2x_logs/session.txt'"
  echo "now start the backups: SSH_KEY=${SSH_KEY:-} bash MapAnything_Exp1/scripts/vm_autofetch_e2x.sh $T $P"
}

case "${1:-}" in
  build) build ;;
  send) shift; send "$@" ;;
  *) echo "usage: $0 build | send user@host [port]" >&2; exit 2 ;;
esac
