#!/usr/bin/env bash
# E4-A: put the code of tag e4-prereg on the VM as ~/e4 (results/ excluded; data and third_party linked read-only to
# the session-A tree ~/e2x, which stays untouched), then check that the copy is the tag.
#   SSH_KEY=~/.ssh/a100_dawar_ed25519 bash MapAnything_Exp1/scripts/e4/deploy_vm.sh dawar@35.193.8.245 [tag]
set -euo pipefail
HOST=${1:?user@host}
TAG=${2:-e4-prereg}
KEY=${SSH_KEY:-~/.ssh/id_ed25519}
cd "$(git rev-parse --show-toplevel)"
COMMIT=$(git rev-list -n 1 "$TAG")
ssh -i "$KEY" "$HOST" "test ! -e ~/e4 || { echo '~/e4 exists; refusing to overwrite'; exit 1; }"
git archive "$TAG" -- MapAnything_Exp1 ':(exclude)MapAnything_Exp1/results' | ssh -i "$KEY" "$HOST" 'mkdir -p ~/e4 && tar -x -C ~/e4'
ssh -i "$KEY" "$HOST" "set -e
  ln -s ~/e2x/MapAnything_Exp1/data ~/e4/MapAnything_Exp1/data
  ln -s ~/e2x/MapAnything_Exp1/third_party ~/e4/MapAnything_Exp1/third_party
  ln -s ~/e2x/third_party ~/e4/third_party
  mkdir -p ~/e4/MapAnything_Exp1/results
  echo '$TAG $COMMIT' > ~/e4/E4_CODE_TAG
  cd ~/e4/MapAnything_Exp1 && find exp1 scripts configs tests -type f | sort | xargs sha256sum > ~/e4/code.sha256"
T=$(mktemp -d)
git archive "$TAG" -- MapAnything_Exp1 ':(exclude)MapAnything_Exp1/results' | tar -x -C "$T"
(cd "$T/MapAnything_Exp1" && find exp1 scripts configs tests -type f | sort | xargs sha256sum) | tr -d '*' | tr -s ' ' > "$T/local.sha256"
ssh -i "$KEY" "$HOST" "cat ~/e4/code.sha256" | tr -s ' ' > "$T/vm.sha256"   # Git Bash prints 'hash *file', Linux 'hash  file'
if diff -q "$T/local.sha256" "$T/vm.sha256" > /dev/null; then
  echo "deployed $TAG ($COMMIT) to $HOST:~/e4: $(wc -l < "$T/vm.sha256") files, checksums identical to the tag"
else
  echo "CHECKSUM MISMATCH between the tag and ~/e4"; diff "$T/local.sha256" "$T/vm.sha256" | head; exit 1
fi
rm -r "$T"
