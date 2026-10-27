#!/usr/bin/env bash
# E0 Phase B: download + unpack ETH3D high-res multi-view TRAINING data (13 scenes).
# Licence: CC BY-NC-SA 4.0. Per-scene archives, URL pattern as in map-anything
# data_processing/wai_processing/download_scripts/download_eth3d.py:52-76.
#   dslr_undistorted  undistorted images + PINHOLE calibration
#   dslr_depth        GT depth (on the DISTORTED image grid)
#   dslr_jpg          distorted calibration (THIN_PRISM_FISHEYE) needed to undistort the depth;
#                     its distorted JPGs are not used and are deleted after unpacking.
# Dev scenes come first, so E0 can start before the eval scenes finish.
# Resumable (wget -c). Extra args go to wget, e.g. --no-check-certificate if TLS fails
# (map-anything's own downloader disables certificate checks, download_eth3d.py:18-24).
#
#   bash scripts/e0_download_eth3d.sh /data/eth3d [wget args]
#   SCENES="relief terrains" bash scripts/e0_download_eth3d.sh /data/eth3d   # resume a subset
set -euo pipefail
ROOT=${1:?usage: e0_download_eth3d.sh <eth3d_root> [wget args]}
shift
mkdir -p "$ROOT/_archives"
SCENES=${SCENES:-"delivery_area facade terrace courtyard electro kicker meadow office pipes playground relief relief_2 terrains"}
for scene in $SCENES; do
  for kind in dslr_undistorted dslr_depth dslr_jpg; do
    f="${scene}_${kind}.7z"
    wget -c -q --show-progress "$@" -O "$ROOT/_archives/$f" "https://www.eth3d.net/data/$f"
    7z x -y -o"$ROOT" "$ROOT/_archives/$f" > /dev/null
  done
  rm -rf "${ROOT:?}/$scene/images/dslr_images"
  printf '\n[ok] %s  %s\n' "$scene" "$(du -sh "$ROOT/$scene" | cut -f1)"
done
echo "done; archives kept in $ROOT/_archives (delete to free space once E0 passes)"
