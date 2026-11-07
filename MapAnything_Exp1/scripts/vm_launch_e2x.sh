#!/usr/bin/env bash
# LAPTOP side, from the repo root (experiments/E2X.md section 6).
#   bash MapAnything_Exp1/scripts/vm_launch_e2x.sh build             pack the tree: experiment code from the tag
#                                                                     e2x-prereg, job scripts from HEAD
#   [SSH_KEY=key.pem] [GPU_BUDGET_GB=32] bash MapAnything_Exp1/scripts/vm_launch_e2x.sh send user@host [port]
# `send` uploads the code first and starts scripts/vm_setup_e2x.sh, so the installs begin at once; the clips are
# streamed (no local archive) while the envs install, and the GPU queue waits for them. On the VM everything lives in
# ~/e2x, ~/e2x_conda and ~/e2x_logs (shared machine: nothing else is touched).
set -euo pipefail
B=${BUNDLE:-$(cd .. && pwd)/3D-CVPR_Exp_bundle_e2x}
FROZEN="MapAnything_Exp1/exp1 MapAnything_Exp1/configs MapAnything_Exp1/tests MapAnything_Exp1/results/e2e_mono/mono_depth.csv"
CLIPS="data/clips data/clips_e2 data/clips_e2e/*.npz data/clips_e2e/*.csv data/clips_e2x_room data/clips_e2x_cal"

build() {
  local tag head
  tag=$(git rev-parse -q --verify "e2x-prereg^{commit}") || { echo "tag e2x-prereg missing: freeze E2X first" >&2; exit 1; }
  head=$(git rev-parse HEAD)
  # the experiment (code, configs, tests) must be exactly the frozen one; only job scripts may be newer
  git diff --quiet "$tag" HEAD -- MapAnything_Exp1/exp1 MapAnything_Exp1/configs MapAnything_Exp1/tests ||
    { echo "exp1/, configs/ or tests/ changed since e2x-prereg: not allowed for session A" >&2; exit 1; }
  git diff --quiet HEAD -- MapAnything_Exp1/scripts || { echo "uncommitted changes in scripts/: commit them first" >&2; exit 1; }
  rm -rf "$B" && mkdir -p "$B/t"
  git archive "$tag" $FROZEN | tar x -C "$B/t"
  git archive HEAD MapAnything_Exp1/scripts | tar x -C "$B/t"
  echo "$tag" > "$B/t/MapAnything_Exp1/E2X_FROZEN_COMMIT"
  echo "$head" > "$B/t/MapAnything_Exp1/E2X_SCRIPTS_COMMIT"
  tar czf "$B/e2x_tree.tgz" -C "$B/t" MapAnything_Exp1
  rm -rf "$B/t"
  git show "HEAD:MapAnything_Exp1/scripts/vm_setup_e2x.sh" > "$B/vm_setup_e2x.sh"
  # no clip may be added after the freeze; excluded clips live in an excluded/ subfolder, outside every glob
  (cd MapAnything_Exp1 && for d in data/clips_e2x_room data/clips_e2x_cal; do
     [ -f "$d/excluded.csv" ] || { echo "$d/excluded.csv missing: run the clip checks first" >&2; exit 1; }; done)
  ls -la "$B"
  echo "bundle ready in $B (experiment e2x-prereg ${tag:0:7}, scripts ${head:0:7})"
}

send() {
  local T=$1 P=${2:-22}
  local K=${SSH_KEY:+-i $SSH_KEY}   # SSH_KEY=/path/to/key.pem when the VM needs its own key
  local SSH="ssh $K -p $P -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30"
  local SCP="scp $K -P $P -o StrictHostKeyChecking=accept-new"
  [ -f "$B/e2x_tree.tgz" ] || { echo "run '$0 build' first" >&2; exit 1; }
  $SSH "$T" 'command -v tmux >/dev/null || (sudo apt-get -qq update && sudo apt-get -qq install -y tmux)'
  $SCP "$B/e2x_tree.tgz" "$B/vm_setup_e2x.sh" "$T:~/"
  $SSH "$T" "mkdir -p ~/e2x_logs && tmux new-session -d -s e2x_main \"GPU_BUDGET_GB=${GPU_BUDGET_GB:-32} bash ~/vm_setup_e2x.sh 2>&1 | tee -a ~/e2x_logs/e2x_main.txt\""
  (cd MapAnything_Exp1 && tar czf - --exclude='*/excluded/*' $CLIPS) |
    $SSH "$T" 'cat > ~/clips_e2x.tgz.part && mv ~/clips_e2x.tgz.part ~/clips_e2x.tgz'
  $SSH "$T" 'until [ -d ~/e2x/MapAnything_Exp1 ]; do sleep 5; done; bash ~/vm_setup_e2x.sh clips && tmux ls'
  echo "started. progress: $SSH $T 'cd ~/e2x/MapAnything_Exp1 && ~/e2x_conda/envs/mapanything/bin/python scripts/session_progress_e2x.py; tail -5 ~/e2x_logs/session.txt'"
}

case "${1:-}" in
  build) build ;;
  send) shift; send "$@" ;;
  *) echo "usage: $0 build | send user@host [port]" >&2; exit 2 ;;
esac
