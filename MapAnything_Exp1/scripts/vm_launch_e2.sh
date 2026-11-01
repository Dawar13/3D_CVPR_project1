#!/usr/bin/env bash
# LAPTOP side, run from the repo root (experiments/E2.md Amendments 10-12).
#   bash MapAnything_Exp1/scripts/vm_launch_e2.sh build            BEFORE switching the VM on: pack the bundle
#   bash MapAnything_Exp1/scripts/vm_launch_e2.sh send user@host [port]
# `send` uploads the code (~0.5 MB) first and starts scripts/vm_setup_e2.sh, so installing begins at once; the
# clips (~0.4 GB, streamed, no local copy) upload while the envs install, and the GPU queue waits for them.
set -euo pipefail
B=${BUNDLE:-$(cd .. && pwd)/3D-CVPR_Exp_bundle}   # absolute, outside the repo
CODE="MapAnything_Exp1/exp1 MapAnything_Exp1/configs MapAnything_Exp1/scripts MapAnything_Exp1/tests"
CLIPS="data/clips data/clips_e2 data/clips_n4 data/clips_n20 data/clips_e2e/*.npz data/clips_e2e/*.csv"

build() {
  rm -rf "$B" && mkdir -p "$B/frozen" "$B/def"
  # frozen E2-C tree: the tag's code + the two job scripts added after it + the commit id
  git archive e2-prereg-C $CODE | tar x -C "$B/frozen"
  for s in e2c_eval_job.sh e2_progress.sh; do git show "HEAD:MapAnything_Exp1/scripts/$s" > "$B/frozen/MapAnything_Exp1/scripts/$s"; done
  git rev-parse 'e2-prereg-C^{commit}' > "$B/frozen/MapAnything_Exp1/E2C_FROZEN_COMMIT"
  tar czf "$B/e2c_frozen.tgz" -C "$B/frozen" MapAnything_Exp1
  # D/E/F tree: the tag e2-prereg-DEF
  local def
  def=$(git rev-parse -q --verify "e2-prereg-DEF^{commit}") || { echo "tag e2-prereg-DEF missing: freeze D/E/F first" >&2; exit 1; }
  git archive "$def" $CODE | tar x -C "$B/def"
  echo "$def" > "$B/def/MapAnything_Exp1/E2DEF_FROZEN_COMMIT"
  tar czf "$B/e2_head.tgz" -C "$B/def" MapAnything_Exp1
  rm -rf "$B/frozen" "$B/def"
  cp MapAnything_Exp1/results/e1_eval/per_run.csv "$B/e1_eval_per_run.csv"
  git show "$def:MapAnything_Exp1/scripts/vm_setup_e2.sh" > "$B/vm_setup_e2.sh"
  local n
  local x
  n=$(ls MapAnything_Exp1/data/clips_e2e/*__c*.npz | wc -l)
  x=$(tail -n +2 MapAnything_Exp1/data/clips_e2e/excluded.csv | wc -l)   # clips that failed the GT check (E2.md)
  [ "$n" -eq $(( 54 - x )) ] || { echo "data/clips_e2e has $n clips, expected 54 - $x excluded" >&2; exit 1; }
  ls -la "$B"
  echo "bundle ready in $B  (e2-prereg-C $(git rev-parse --short e2-prereg-C), e2-prereg-DEF ${def:0:7})"
}

send() {
  local T=$1 P=${2:-22}
  local SSH="ssh -p $P -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30"
  local SCP="scp -P $P -o StrictHostKeyChecking=accept-new"
  [ -f "$B/e2_head.tgz" ] || { echo "run '$0 build' first" >&2; exit 1; }
  $SSH "$T" 'command -v tmux >/dev/null || (sudo apt-get -qq update && sudo apt-get -qq install -y tmux)'
  $SCP "$B/e2c_frozen.tgz" "$B/e2_head.tgz" "$B/e1_eval_per_run.csv" "$B/vm_setup_e2.sh" "$T:~/"
  $SSH "$T" 'bash ~/vm_setup_e2.sh'
  # clips streamed straight to the VM (no local archive: the laptop is short of disk); globs expand in MapAnything_Exp1
  (cd MapAnything_Exp1 && tar czf - $CLIPS) | $SSH "$T" 'cat > ~/clips_all.tgz'
  $SSH "$T" 'bash ~/vm_setup_e2.sh clips && tmux ls'
  echo "started. progress: $SSH $T 'cd ~/e2d/MapAnything_Exp1 && ~/miniconda3/envs/mapanything/bin/python scripts/session_progress.py; tail -5 ~/logs/session.txt'"
}

case "${1:-}" in
  build) build ;;
  send) shift; send "$@" ;;
  *) echo "usage: $0 build | send user@host [port]" >&2; exit 2 ;;
esac
