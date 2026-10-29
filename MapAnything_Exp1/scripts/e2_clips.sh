#!/usr/bin/env bash
# E2-C clips (CPU only): 3 non-overlapping 10-view clips per scene, plus 4- and 20-view clips from image 0.
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mapanything
cd ~/3D-CVPR_Exp/MapAnything_Exp1
python -m exp1.datasets --eth3d_root ~/data/eth3d --out data/clips_e2 --n_views 10 --clips_per_scene 3 --non_overlapping --size 518 336
python -m exp1.datasets --eth3d_root ~/data/eth3d --out data/clips_n4 --n_views 4 --clips_per_scene 1 --non_overlapping --size 518 336
python -m exp1.datasets --eth3d_root ~/data/eth3d --out data/clips_n20 --n_views 20 --clips_per_scene 1 --non_overlapping --size 518 336
# GT self-consistency check on every new clip
python scripts/e0_check_clips.py --clips "data/clips_e2/*.npz" --out reports/e2_clip_check 2>&1 | tail -40
python scripts/e0_check_clips.py --clips "data/clips_n20/*.npz" --out reports/e2_clip_check_n20 2>&1 | tail -20
echo CLIPS_DONE
