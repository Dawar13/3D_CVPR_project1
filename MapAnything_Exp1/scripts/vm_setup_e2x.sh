#!/usr/bin/env bash
# Run ON A FRESH GPU VM from ~ (experiments/E2X.md section 6), after scripts/vm_launch_e2x.sh has uploaded
# e2x_tree.tgz (the e2x-prereg tag) and this script:
#   bash ~/vm_setup_e2x.sh       unpack, install miniconda, create the envs, then start tmux sessions:
#     setup_ma  MapAnything env                      -> ~/logs/ready_mapanything
#     setup_px  Pi3X env + weights                   -> ~/logs/ready_pi3x
#     setup_ov  OmniVGGT env + weights               -> ~/logs/ready_omnivggt
#     setup_ai  AI depth on the E2-E, room and cal clips -> ~/logs/ready_mono
#     gpu       waits for MapAnything and the clips, then scripts/session_queue_e2x.sh
#   bash ~/vm_setup_e2x.sh clips  after clips_e2x.tgz has arrived (vm_launch_e2x.sh does this)
set -euo pipefail
T=~/x/MapAnything_Exp1
log() { echo "$(date -u '+%F %T') $*" | tee -a ~/logs/setup.txt; }

cu_build() {  # Blackwell (sm_120) needs CUDA >= 12.8 wheels: cu130 needs driver >= 580, cu128 >= 570
  local drv
  drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed -n 1p | cut -d. -f1)
  if [ "$drv" -ge 580 ]; then echo cu130; elif [ "$drv" -ge 570 ]; then echo cu128; else echo cu126; fi
}

setup_ma() {
  trap 'touch ~/logs/FAILED_mapanything' ERR
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate mapanything
  python -c "import torch" 2>/dev/null ||
    pip install -q torch==2.14.1 torchvision --index-url "https://download.pytorch.org/whl/$(cu_build)"
  cd "$T" && bash scripts/setup_server.sh
  python -c "from exp1 import model_io; model_io.load_model('facebook/map-anything', 'cpu'); print('checkpoint cached')"
  local snap
  snap=$(ls ~/.cache/huggingface/hub/models--facebook--map-anything/snapshots/)
  echo "MapAnything snapshot: $snap"
  echo "$snap" | grep -q '^a1d87e90' || { echo "weights are not snapshot a1d87e90 (E0-E2)"; false; }
  touch ~/logs/ready_mapanything
  python -m pytest tests -q -x --ignore=tests/test_gate.py > ~/logs/pytest_ma.txt 2>&1 || true   # CPU, while the GPU runs
  tail -2 ~/logs/pytest_ma.txt
}

setup_px() {
  trap 'touch ~/logs/FAILED_pi3x' ERR
  source ~/miniconda3/etc/profile.d/conda.sh
  local idx="https://download.pytorch.org/whl/$(cu_build | sed 's/cu130/cu128/')"   # torch 2.7.0 has cu128 at most
  cd "$T" && PIP_CACHE_DIR=~/.cache/pip_px TORCH_INDEX_URL="$idx" bash scripts/setup_pi3x.sh
  conda activate pi3x
  python -m pytest tests/test_pi3x.py tests/test_e3.py -q 2>&1 | tail -2 || true
  touch ~/logs/ready_pi3x
}

setup_ov() {
  trap 'touch ~/logs/FAILED_omnivggt' ERR
  source ~/miniconda3/etc/profile.d/conda.sh
  local idx="https://download.pytorch.org/whl/$(cu_build | sed 's/cu130/cu128/')"
  cd "$T" && PIP_CACHE_DIR=~/.cache/pip_ov TORCH_INDEX_URL="$idx" bash scripts/setup_omnivggt.sh
  local snap
  snap=$(ls ~/.cache/huggingface/hub/models--Livioni--OmniVGGT/snapshots/)
  echo "OmniVGGT snapshot: $snap"
  echo "$snap" | grep -q '^29c67625' || { echo "weights are not snapshot 29c67625"; false; }
  touch ~/logs/ready_omnivggt
}

setup_ai() {  # Depth Anything V2 (E2-E's AI depth hint) on the room and cal clips, before any run on them
  trap 'touch ~/logs/FAILED_mono' ERR
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate aidepth
  PIP_CACHE_DIR=~/.cache/pip_ai pip install -q torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
  PIP_CACHE_DIR=~/.cache/pip_ai pip install -q "transformers>=4.45,<5" pillow numpy
  until [ -f ~/logs/ready_clips ]; do sleep 10; done
  cd "$T"
  # E2-E clips first: the laptop copies have no AI depth (it was made on the E2 VM). It must match E2-E's record.
  python -m exp1.mono_depth --clips 'data/clips_e2e/*.npz' --record results/e2x_mono/mono_depth_e2e.csv
  python - <<'PY' 2>&1 | tee ~/logs/mono_verify.txt
import csv
new = {r["clip"]: r for r in csv.DictReader(open("results/e2x_mono/mono_depth_e2e.csv"))}
ref = {r["clip"]: r for r in csv.DictReader(open("results/e2e_mono/mono_depth.csv"))}
same = sum(new[c]["mono_depth_sha256"] == ref[c]["mono_depth_sha256"] for c in new if c in ref)
rev = all(new[c]["model"] == ref[c]["model"] for c in new if c in ref)
print(f"AI depth on the E2-E clips: {same}/{len(new)} bit-identical to E2-E; same model revision: {rev}")
if not rev:
    raise SystemExit("different Depth Anything revision from E2-E: stop and report")
PY
  python -m exp1.mono_depth --clips 'data/clips_e2x_room/*.npz' --record results/e2x_mono/mono_depth_room.csv
  python -m exp1.mono_depth --clips 'data/clips_e2x_cal/*.npz' --record results/e2x_mono/mono_depth_cal.csv
  touch ~/logs/ready_mono
}

gpu() {
  until [ -f ~/logs/ready_mapanything ] && [ -f ~/logs/ready_clips ]; do
    [ -f ~/logs/FAILED_mapanything ] && { echo "MapAnything setup failed, see ~/logs/setup_ma.txt"; exit 1; }
    sleep 10
  done
  bash "$T/scripts/session_queue_e2x.sh"
}

case "${1:-main}" in
  ma) setup_ma ;;
  px) setup_px ;;
  ov) setup_ov ;;
  ai) setup_ai ;;
  gpu) gpu ;;
  clips)
    [ -d "$T/data/clips_e2x_room" ] || tar xzf ~/clips_e2x.tgz -C "$T"
    touch ~/logs/ready_clips
    ;;
  main)
    mkdir -p ~/logs ~/x
    [ -d "$T" ] || tar xzf ~/e2x_tree.tgz -C ~/x
    mkdir -p "$T/reports" "$T/results/e2x_mono"
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
    conda env list | grep -q '^pi3x ' || conda create -y -q -n pi3x python=3.10
    conda env list | grep -q '^omnivggt ' || conda create -y -q -n omnivggt python=3.10
    conda env list | grep -q '^aidepth ' || conda create -y -q -n aidepth python=3.11
    for s in setup_ma setup_px setup_ov setup_ai gpu; do
      tmux has-session -t "=$s" 2>/dev/null ||   # "=": exact name
        tmux new-session -d -s "$s" "bash ~/vm_setup_e2x.sh ${s#setup_} 2>&1 | tee -a ~/logs/$s.txt"
    done
    log "started: setup_ma, setup_px, setup_ov, setup_ai, gpu"
    tmux ls
    ;;
esac
