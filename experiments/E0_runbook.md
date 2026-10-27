# E0 runbook: what to run on the VM (Phases B and C)

Phase A (laptop, no GPU) is done; see `MapAnything_Exp1/reports/e0_interface.md`.
This file is the exact command sequence for the VM. Paths assume the repo at `~/3D-CVPR_Exp`
and data at `/data`. Change them if your VM differs.

## 0. Copy the repo to the VM (laptop, Git Bash)

```bash
cd ~/Downloads
tar --exclude=third_party --exclude=data --exclude=results --exclude=__pycache__ \
    --exclude='*.pdf' -czf e0_repo.tgz 3D-CVPR_Exp
scp e0_repo.tgz USER@VM:~/
# on the VM
tar xzf e0_repo.tgz && cd ~/3D-CVPR_Exp/MapAnything_Exp1
```

## Phase B: GPU idle. Run B1 and B2 in parallel (two tmux panes)

Prerequisites (E0 Step 1):
```bash
nvidia-smi                                   # note a free GPU with >= 24 GB
df -h ~ /data                                # ~50 GB free where data/results live
which conda git 7z wget || echo "missing: install p7zip-full / wget"
export HF_HOME=/data/hf_cache                # only if the home disk is small (also in ~/.bashrc)
export CUDA_VISIBLE_DEVICES=0                # the GPU you picked
```

**B1. ETH3D.** This is the longest step. It takes 13 scenes × 3 archives, dev scenes first.
```bash
bash scripts/e0_download_eth3d.sh /data/eth3d
# only if wget fails on the TLS certificate:
# bash scripts/e0_download_eth3d.sh /data/eth3d --no-check-certificate
```
The third archive per scene (`dslr_jpg`) is new compared with the E0 plan. Its calibration
is needed to undistort the GT depth (`reports/e0_interface.md`, "Other findings" 1).

**B2. Environment and checkpoint.**
```bash
# pick the cuXXX wheel that matches the driver in nvidia-smi; omit TORCH_INDEX_URL to let pip choose
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 bash scripts/setup_server.sh
conda activate mapanything
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
bash scripts/env_report.sh > reports/env_dump.txt
python -m pytest tests -q                    # expect: 30 passed
# download + cache the checkpoint without touching the GPU (needs several GB of RAM)
python -c "from exp1 import model_io; model_io.load_model('facebook/map-anything', 'cpu'); print('checkpoint cached')"
```
`setup_server.sh` prints a WARNING if MapAnything is not at the verified commit `3d10cf7a`.

**Phase B is done when:**
- `cuda.is_available()` is True;
- pytest reports 30 passed;
- "checkpoint cached" is printed;
- B1 printed `[ok]` for all 13 scenes.

## Phase C: GPU busy for part of it

```bash
conda activate mapanything
export CUDA_VISIBLE_DEVICES=0
ETH3D_ROOT=/data/eth3d bash scripts/e0_phase_c.sh 2>&1 | tee reports/e0_phase_c_console.txt
```
It runs E0 Steps 4.3, 4.4, 5, 6, 7, the visual check 7b and 8 in order, and saves every printout in
`reports/e0_logs/`. Step 7b writes demo reconstructions of the smoke clip to `results/e0_demo/`:
- `.ply` point clouds for GT, images-only, clean hints and the 15° bad-pose case, which open in MeshLab or CloudCompare;
- one `__overview.png` with bird's-eye and side views and the metrics.

It stops by itself in two cases:
- the ETH3D layout is wrong;
- the GT self-consistency check does not pass. The official example still runs, but the smoke runs are skipped, as Step 5 requires.

It never runs the model on `eval__` clips. Step 5 checks GT geometry on all clips, but no model is involved there.

## Hand back for Phase D

```bash
# on the VM, from ~/3D-CVPR_Exp
tar --exclude='pred.npz' -czf e0_out.tgz MapAnything_Exp1/reports MapAnything_Exp1/results
# on the laptop
scp USER@VM:~/3D-CVPR_Exp/e0_out.tgz ~/Downloads/ && tar xzf ~/Downloads/e0_out.tgz -C ~/Downloads/3D-CVPR_Exp
```
Then tell Claude Code "Phase C done". If the script stopped early, send `reports/e0_phase_c_console.txt` instead.
