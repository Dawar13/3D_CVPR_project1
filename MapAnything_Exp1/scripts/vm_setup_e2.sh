#!/usr/bin/env bash
# Run ON A FRESH GPU VM from ~, after uploading e2c_frozen.tgz, e2_head.tgz, clips_all.tgz,
# e1_eval_per_run.csv and this script (experiments/E2.md Amendments 10-11):
#   bash ~/vm_setup_e2.sh        unpack, install miniconda, create both envs, then start three tmux sessions:
#     setup_ma  MapAnything env for the frozen e2-prereg-C tree   -> ~/logs/ready_mapanything
#     setup_ov  OmniVGGT env for the HEAD tree (E2-D)             -> ~/logs/ready_omnivggt
#     gpu       waits for ready_mapanything, then scripts/session1_queue.sh (the GPU never waits for OmniVGGT)
# Trees: ~/3D-CVPR_Exp = tag e2-prereg-C (+ E2C_FROZEN_COMMIT); ~/e2d = HEAD. Both share ~/3D-CVPR_Exp/.../data.
set -euo pipefail
F=~/3D-CVPR_Exp/MapAnything_Exp1
H=~/e2d/MapAnything_Exp1
log() { echo "$(date -u '+%F %T') $*" | tee -a ~/logs/setup.txt; }

cu_build() {  # Blackwell (sm_120) needs CUDA >= 12.8 wheels: cu130 needs driver >= 580 (as in E1), cu128 >= 570
  local drv
  drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed -n 1p | cut -d. -f1)
  if [ "$drv" -ge 580 ]; then echo cu130; elif [ "$drv" -ge 570 ]; then echo cu128; else
    echo "driver $drv is too old for this GPU" >&2; exit 1; fi
}

setup_ma() {
  trap 'touch ~/logs/FAILED_mapanything' ERR
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate mapanything
  python -c "import torch" 2>/dev/null ||
    pip install -q torch==2.14.1 torchvision --index-url "https://download.pytorch.org/whl/$(cu_build)"
  cd "$F" && bash scripts/setup_server.sh
  python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
assert torch.isfinite(x @ x).all(), "bf16 matmul not finite"
PY
  python -c "from exp1 import model_io; model_io.load_model('facebook/map-anything', 'cpu'); print('checkpoint cached')"
  local snap
  snap=$(ls ~/.cache/huggingface/hub/models--facebook--map-anything/snapshots/)
  echo "MapAnything snapshot: $snap"
  echo "$snap" | grep -q '^a1d87e90' || { echo "weights are not snapshot a1d87e90 (E0-E2)"; false; }
  python -m pytest tests/test_e2_tools.py -q -rs -k "median or swap" 2>&1 | tail -3   # swap runs only here
  touch ~/logs/ready_mapanything
  python -m pytest tests -q > ~/logs/pytest_frozen.txt 2>&1 || true   # CPU only, while the GPU queue runs
  tail -2 ~/logs/pytest_frozen.txt
}

setup_ov() {
  trap 'touch ~/logs/FAILED_omnivggt' ERR
  source ~/miniconda3/etc/profile.d/conda.sh
  cd "$H" && PIP_CACHE_DIR=~/.cache/pip_ov bash scripts/setup_omnivggt.sh
  local snap
  snap=$(ls ~/.cache/huggingface/hub/models--Livioni--OmniVGGT/snapshots/)
  echo "OmniVGGT snapshot: $snap"
  echo "$snap" | grep -q '^29c67625' || { echo "weights are not snapshot 29c67625"; false; }
  conda activate omnivggt
  python -m pytest tests/test_e2_tools.py tests/test_e2d.py tests/test_shard.py -q 2>&1 | tail -2 || true  # passed on the laptop
  touch ~/logs/ready_omnivggt
}

gpu() {
  until [ -f ~/logs/ready_mapanything ] && [ -f ~/logs/ready_clips ]; do
    [ -f ~/logs/FAILED_mapanything ] && { echo "MapAnything setup failed, see ~/logs/setup_ma.txt"; exit 1; }
    sleep 10
  done
  bash "$H/scripts/session1_queue.sh"
}

case "${1:-main}" in
  ma) setup_ma ;;
  ov) setup_ov ;;
  gpu) gpu ;;
  clips)  # after clips_all.tgz has arrived (uploaded while the envs install)
    [ -d "$F/data/clips_e2" ] || tar xzf ~/clips_all.tgz -C "$F"
    ln -sfn "$F/data" "$H/data"
    touch ~/logs/ready_clips
    ;;
  main)
    mkdir -p ~/logs ~/3D-CVPR_Exp ~/e2d
    [ -d "$F" ] || tar xzf ~/e2c_frozen.tgz -C ~/3D-CVPR_Exp
    [ -d "$H" ] || tar xzf ~/e2_head.tgz -C ~/e2d
    mkdir -p "$F/reports" "$H/reports"
    { nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; echo "cpus $(nproc)";
      free -g | sed -n 2p; df -h ~ | tail -1; echo "torch wheels $(cu_build)"; } | tee ~/logs/machine.txt
    if [ ! -d ~/miniconda3 ]; then
      wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/mc.sh
      bash ~/mc.sh -b -p ~/miniconda3 && rm ~/mc.sh
    fi
    source ~/miniconda3/etc/profile.d/conda.sh
    conda config --set auto_activate_base false || true
    for ch in main r; do
      conda tos accept --override-channels --channel "https://repo.anaconda.com/pkgs/$ch" >/dev/null 2>&1 || true
    done
    grep -q miniconda3/etc/profile.d/conda.sh ~/.bashrc || echo 'source ~/miniconda3/etc/profile.d/conda.sh' >> ~/.bashrc
    # envs are created one after the other (shared package cache), then filled in parallel
    conda env list | grep -q '^mapanything ' || conda create -y -q -n mapanything python=3.12
    conda env list | grep -q '^omnivggt ' || conda create -y -q -n omnivggt python=3.10
    for s in setup_ma setup_ov gpu; do
      tmux has-session -t "$s" 2>/dev/null ||
        tmux new-session -d -s "$s" "bash ~/vm_setup_e2.sh ${s#setup_} 2>&1 | tee -a ~/logs/$s.txt"
    done
    log "started: setup_ma, setup_ov, gpu"
    tmux ls
    ;;
esac
