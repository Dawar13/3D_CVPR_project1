#!/usr/bin/env bash
# LAPTOP side, run from the repo root: build the upload bundle, send it to a fresh GPU VM and start
# scripts/vm_setup_e2.sh there. The code (~0.3 MB) goes first so installing starts at once; the clips
# (~320 MB) upload while the envs install, and the GPU queue waits for them.
#   bash MapAnything_Exp1/scripts/vm_launch_e2.sh user@host [port]
set -euo pipefail
T=$1; P=${2:-22}
B=${BUNDLE:-$(mktemp -d)}
SSH="ssh -p $P -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30"
SCP="scp -P $P -o StrictHostKeyChecking=accept-new"
CODE="MapAnything_Exp1/exp1 MapAnything_Exp1/configs MapAnything_Exp1/scripts MapAnything_Exp1/tests"
# frozen tree: the tag's code + the two job scripts added after it + the commit id
mkdir -p "$B/frozen"
git archive e2-prereg-C $CODE | tar x -C "$B/frozen"
for s in e2c_eval_job.sh e2_progress.sh; do git show "HEAD:MapAnything_Exp1/scripts/$s" > "$B/frozen/MapAnything_Exp1/scripts/$s"; done
git rev-parse 'e2-prereg-C^{commit}' > "$B/frozen/MapAnything_Exp1/E2C_FROZEN_COMMIT"
tar czf "$B/e2c_frozen.tgz" -C "$B/frozen" MapAnything_Exp1
git archive --format=tar.gz HEAD $CODE > "$B/e2_head.tgz"
cp MapAnything_Exp1/results/e1_eval/per_run.csv "$B/e1_eval_per_run.csv"
cp MapAnything_Exp1/scripts/vm_setup_e2.sh "$B/"
[ -f "$B/clips_all.tgz" ] || tar czf "$B/clips_all.tgz" -C MapAnything_Exp1 data/clips data/clips_e2 data/clips_n4 data/clips_n20
echo "bundle: $B  (HEAD $(git rev-parse --short HEAD))"
$SSH "$T" 'command -v tmux >/dev/null || (sudo apt-get -qq update && sudo apt-get -qq install -y tmux)'
$SCP "$B/e2c_frozen.tgz" "$B/e2_head.tgz" "$B/e1_eval_per_run.csv" "$B/vm_setup_e2.sh" "$T:~/"
$SSH "$T" 'bash ~/vm_setup_e2.sh'
$SCP "$B/clips_all.tgz" "$T:~/"
$SSH "$T" 'bash ~/vm_setup_e2.sh clips && tmux ls'
echo "started. progress: $SSH $T 'cd ~/e2d/MapAnything_Exp1 && ~/miniconda3/envs/mapanything/bin/python scripts/session_progress.py; tail -5 ~/logs/session1.txt'"
