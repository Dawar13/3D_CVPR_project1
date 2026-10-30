#!/usr/bin/env bash
# Run ON THE GPU SERVER, from the copy used for E2-D. OmniVGGT gets its own conda env; the mapanything
# env is not touched. Clones only if absent.
set -euo pipefail
cd "$(dirname "$0")/.."
# model_io_omnivggt.py was read at this commit; override only deliberately.
OMNIVGGT_COMMIT=${OMNIVGGT_COMMIT:-c8cbfc6bcd57a07bf08a1f97804da9827847af8c}
# The README pins torch 2.7.0 / torchvision 0.22.0 with CUDA 12.8 wheels (Blackwell GPUs need >= 12.8).
# A driver older than 570 needs TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 instead.
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}
mkdir -p third_party
if [ ! -d third_party/OmniVGGT-official ]; then
  git clone -q https://github.com/Livioni/OmniVGGT-official.git third_party/OmniVGGT-official
  git -C third_party/OmniVGGT-official checkout -q "$OMNIVGGT_COMMIT"
fi
HEAD_NOW=$(git -C third_party/OmniVGGT-official rev-parse HEAD)
echo "OmniVGGT-official at $HEAD_NOW"
if [ "$HEAD_NOW" != "$OMNIVGGT_COMMIT" ]; then
  echo "WARNING: OmniVGGT-official is at $HEAD_NOW, not the verified $OMNIVGGT_COMMIT" >&2
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list | grep -q '^omnivggt ' || conda create -y -q -n omnivggt python=3.10
conda activate omnivggt
pip install -q torch==2.7.0 torchvision==0.22.0 --index-url "$TORCH_INDEX_URL"
# inference dependencies only (first block of the repo's requirements.txt); training and demo extras skipped
pip install -q numpy Pillow opencv-python-headless scipy einops safetensors huggingface_hub trimesh imageio \
  tqdm requests evo pandas matplotlib pytest
python -c "import torch; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "from huggingface_hub import snapshot_download; print('weights at', snapshot_download('Livioni/OmniVGGT'))"
