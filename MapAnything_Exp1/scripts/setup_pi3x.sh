#!/usr/bin/env bash
# Run ON THE GPU SERVER from MapAnything_Exp1/. Pi3X gets its own conda env `pi3x`; no other env is touched.
# Clones only if absent. exp1/model_io_pi3x.py was read at PI3_COMMIT; override only deliberately.
set -euo pipefail
cd "$(dirname "$0")/.."
PI3_COMMIT=${PI3_COMMIT:-9fa3ddb}
# The repo pins torch 2.5.1 (requirements.txt), which has no Blackwell (sm_120) kernels; 2.7.0 with CUDA 12.8 wheels
# is the first that does (as for OmniVGGT). A driver older than 570 needs TORCH_INDEX_URL=.../whl/cu126 instead.
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}
mkdir -p ../third_party
if [ ! -d ../third_party/Pi3 ]; then
  git clone -q https://github.com/yyfz/Pi3.git ../third_party/Pi3
  git -C ../third_party/Pi3 checkout -q "$PI3_COMMIT"
fi
HEAD_NOW=$(git -C ../third_party/Pi3 rev-parse --short=7 HEAD)
echo "Pi3 at $HEAD_NOW"
[ "$HEAD_NOW" = "${PI3_COMMIT:0:7}" ] || echo "WARNING: Pi3 is at $HEAD_NOW, not the verified $PI3_COMMIT" >&2
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list | grep -q '^pi3x ' || conda create -y -q -n pi3x python=3.10
conda activate pi3x
pip install -q torch==2.7.0 torchvision==0.22.0 --index-url "$TORCH_INDEX_URL"
# requirements.txt minus torch/torchvision; plus what our runner and tests import
pip install -q numpy==1.26.4 pillow opencv-python-headless plyfile huggingface_hub safetensors \
  scipy pandas pytest
python -c "import torch; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "from huggingface_hub import snapshot_download; print('weights at', snapshot_download('yyfz233/Pi3X'))"
