"""E0 Step 4.3: check the unpacked ETH3D layout against what exp1/datasets.py reads.

  python scripts/e0_check_layout.py --eth3d_root /data/eth3d [--n_views 10]

Per scene: calibration files and camera models, image and depth files for every
registered image, depth file sizes (raw float32 = W*H*4 bytes), agreement of the two
calibrations, and which images the first clip will use (flagging ETH3D images stored
rotated by 90 deg). Exit code 1 if anything datasets.py needs is missing.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from exp1.datasets import (ETH3D_ROTATED, ETH3D_TRAIN_SCENES,  # noqa: E402
                           read_colmap_cameras, read_colmap_distorted_cameras,
                           read_colmap_images, split_of)


def check_scene(root, scene, n_views):
    problems, notes = [], []
    s = os.path.join(root, scene)
    if not os.path.isdir(s):
        return [f"scene folder missing: {s}"], notes
    try:
        cams = read_colmap_cameras(os.path.join(s, "dslr_calibration_undistorted", "cameras.txt"))
        imgs = read_colmap_images(os.path.join(s, "dslr_calibration_undistorted", "images.txt"))
    except (OSError, ValueError) as e:
        return [f"undistorted calibration: {e!r}"], notes
    try:
        dcams = read_colmap_distorted_cameras(os.path.join(s, "dslr_calibration_jpg", "cameras.txt"))
        dimgs = {os.path.basename(k): v for k, v in
                 read_colmap_images(os.path.join(s, "dslr_calibration_jpg", "images.txt")).items()}
    except (OSError, ValueError) as e:
        problems.append(f"distorted calibration (needed for depth undistortion): {e!r}")
        dcams, dimgs = None, None
    sizes = {c: (W, H) for c, (W, H, _) in cams.items()}
    notes.append(f"{len(imgs)} registered images; undistorted cameras {sizes}")
    if dcams:
        notes.append(f"distorted cameras {{{', '.join(f'{c}: {W}x{H}' for c, (W, H, _) in dcams.items())}}}")
    n_img = n_dep = 0
    for name, (cam_id, c2w) in imgs.items():
        base = os.path.basename(name)
        if os.path.exists(os.path.join(s, "images", name)):
            n_img += 1
        else:
            problems.append(f"missing image images/{name}")
        dpath = os.path.join(s, "ground_truth_depth", "dslr_images", base)
        if not os.path.exists(dpath):
            problems.append(f"missing depth ground_truth_depth/dslr_images/{base}")
            continue
        n_dep += 1
        if dcams is not None:
            if base not in dimgs:
                problems.append(f"{base} not in dslr_calibration_jpg/images.txt")
                continue
            d_cam, d_c2w = dimgs[base]
            if d_cam != cam_id or not np.allclose(d_c2w, c2w, atol=1e-5):
                problems.append(f"{base}: distorted/undistorted calibrations disagree")
            W, H, _ = dcams[d_cam]
        else:
            W, H, _ = cams[cam_id]
        if os.path.getsize(dpath) != W * H * 4:
            problems.append(f"{base}: depth file {os.path.getsize(dpath)} bytes != {W}x{H}x4")
    notes.append(f"{n_img} images and {n_dep} depth files found")
    first = [os.path.basename(n) for n in sorted(imgs)[:n_views]]
    notes.append(f"first clip: {first[0]} .. {first[-1]} ({len(first)} views)")
    rot = sorted(set(first) & set(ETH3D_ROTATED.get(scene, [])))
    if rot:
        notes.append(f"NOTE first clip contains images stored rotated by 90 deg: {rot}")
    if len(first) < n_views:
        problems.append(f"only {len(first)} images < n_views={n_views}")
    return problems, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eth3d_root", required=True)
    ap.add_argument("--n_views", type=int, default=10)
    ap.add_argument("--scenes", nargs="*", default=ETH3D_TRAIN_SCENES)
    a = ap.parse_args()
    bad = 0
    for scene in a.scenes:
        problems, notes = check_scene(a.eth3d_root, scene, a.n_views)
        tag = "OK " if not problems else "BAD"
        print(f"[{tag}] {scene} ({split_of(scene, a.scenes)})")
        for n in notes:
            print(f"      {n}")
        for p in problems[:10]:
            print(f"   !! {p}")
        if len(problems) > 10:
            print(f"   !! ... {len(problems) - 10} more")
        bad += bool(problems)
    print("OVERALL:", "OK" if not bad else f"{bad} scene(s) with problems")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
