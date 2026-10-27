#!/usr/bin/env bash
# Run ON THE GPU SERVER (not the Mac). Clones only if absent; never touches an existing checkout.
# Creates a dedicated conda env so no shared environment is altered.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party
# The interface was verified at this commit (reports/e0_interface.md); override only deliberately.
MAPANYTHING_COMMIT=${MAPANYTHING_COMMIT:-3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9}
if [ ! -d third_party/map-anything ]; then
  git clone https://github.com/facebookresearch/map-anything.git third_party/map-anything
  git -C third_party/map-anything checkout -q "$MAPANYTHING_COMMIT"
fi
HEAD_NOW=$(git -C third_party/map-anything rev-parse HEAD)
echo "$HEAD_NOW" | tee reports/mapanything_commit.txt
if [ "$HEAD_NOW" != "$MAPANYTHING_COMMIT" ]; then
  echo "WARNING: map-anything is at $HEAD_NOW, not the verified $MAPANYTHING_COMMIT" >&2
fi
if ! conda env list | grep -q '^mapanything '; then
  conda create -y -n mapanything python=3.12
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mapanything
# Optional CUDA-matched torch first (MapAnything does not pin torch), e.g.
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 bash scripts/setup_server.sh
if [ -n "${TORCH_INDEX_URL:-}" ]; then
  pip install torch torchvision --index-url "$TORCH_INDEX_URL"
fi
pip install -e third_party/map-anything            # follow the repo README if it says otherwise
pip install pandas matplotlib pytest
python -c "import mapanything, torch; print('mapanything ok, torch', torch.__version__, torch.cuda.is_available())"
