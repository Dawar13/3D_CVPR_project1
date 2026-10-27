#!/usr/bin/env bash
# Run ON THE GPU SERVER (not the Mac). Clones only if absent; never touches an existing checkout.
# Creates a dedicated conda env so no shared environment is altered.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party
if [ ! -d third_party/map-anything ]; then
  git clone https://github.com/facebookresearch/map-anything.git third_party/map-anything
fi
git -C third_party/map-anything rev-parse HEAD | tee reports/mapanything_commit.txt
if ! conda env list | grep -q '^mapanything '; then
  conda create -y -n mapanything python=3.12
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mapanything
pip install -e third_party/map-anything            # follow the repo README if it says otherwise
pip install pandas matplotlib pytest
python -c "import mapanything, torch; print('mapanything ok, torch', torch.__version__, torch.cuda.is_available())"
