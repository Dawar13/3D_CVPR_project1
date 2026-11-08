"""TUM RGB-D freiburg3 clips for the E3 SECONDARY calibration (E2X, E3-cal amendment 1): clips whose ground-truth
poses come from motion capture, i.e. independent of the photos and accurate, unlike SceneFun3D (E2X.md §2.2).

  python -m exp1.tum_clips --raw ~/e2x_tum/raw --out data/clips_tum

Facts used (https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats, read 2026-11-08):
  - groundtruth.txt lines "timestamp tx ty tz qx qy qz qw": position and orientation (unit quaternion) of the optical
    centre of the colour camera with respect to the world origin (motion capture, 100 Hz);
  - depth PNGs scaled by 5000 (5000 = 1 m), 0 = no data;
  - freiburg3 colour camera fx 535.4, fy 539.2, cx 320.1, cy 247.6; images already undistorted (distortion all 0);
    depth pre-registered to the colour image (OpenNI).
The camera-axis convention (x right, y down, z forward) is checked by the E0 ground-truth check on every clip
(depth of one view reprojected into the others with the GT poses), as for every other clip set.

Frames: every colour frame with a depth frame within DEPTH_DT and ground truth on both sides within GT_DT; the pose at
the colour timestamp is interpolated (translation linear, rotation slerp). Keyframes: a new frame when the camera
moved >= KF_TRANS m or turned >= KF_ROT deg since the last keyframe (as exp1/arkit_clips.keyframes, smaller steps:
TUM sequences are short). Clips: up to CLIPS_PER_SEQ non-overlapping runs of N_VIEWS consecutive keyframes, evenly
spaced (datasets.clip_starts). Size 518 x 392 (as the SceneFun3D clips), bicubic resize + centre crop, K adjusted,
depth nearest-sampled. Clip ids: cal__tum_<sequence>__c<k> (calibration only, never eval).
"""

import argparse
import glob
import os

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

from .arkit_clips import keyframes
from .datasets import adjust_K, clip_starts, resize_crop_params, resize_depth_nearest
from .geometry import relative_to_first, validate_pose

K_FR3 = np.array([[535.4, 0.0, 320.1], [0.0, 539.2, 247.6], [0.0, 0.0, 1.0]])
W, H = 640, 480
TW, TH = 518, 392
DEPTH_SCALE = 5000.0
N_VIEWS, CLIPS_PER_SEQ = 10, 4
KF_TRANS, KF_ROT = 0.20, 10.0
DEPTH_DT, GT_DT = 0.02, 0.02


def read_list(path):
    """[(timestamp, rest...)] from a TUM list file ('#' lines skipped)."""
    out = []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        out.append((float(p[0]), p[1:]))
    return out


def gt_interpolator(gt):
    ts = np.array([t for t, _ in gt])
    vals = np.array([[float(x) for x in v] for _, v in gt])
    rot = Rot.from_quat(vals[:, 3:7])            # qx qy qz qw (scipy order)
    slerp = Slerp(ts, rot)

    def at(t):
        i = np.searchsorted(ts, t)
        if i == 0 or i >= len(ts) or t - ts[i - 1] > GT_DT or ts[i] - t > GT_DT:
            return None
        a = (t - ts[i - 1]) / (ts[i] - ts[i - 1])
        P = np.eye(4)
        P[:3, :3] = slerp([t]).as_matrix()[0]
        P[:3, 3] = (1 - a) * vals[i - 1, :3] + a * vals[i, :3]
        return P
    return at


def frames(seq_dir):
    """[(rgb_path, depth_path, cam2world)] for usable colour frames, in time order."""
    rgb = read_list(os.path.join(seq_dir, "rgb.txt"))
    dep = read_list(os.path.join(seq_dir, "depth.txt"))
    dts = np.array([t for t, _ in dep])
    at = gt_interpolator(read_list(os.path.join(seq_dir, "groundtruth.txt")))
    out = []
    for t, (name,) in rgb:
        j = int(np.argmin(np.abs(dts - t)))
        if abs(dts[j] - t) > DEPTH_DT:
            continue
        P = at(t)
        if P is None:
            continue
        out.append((os.path.join(seq_dir, name), os.path.join(seq_dir, dep[j][1][0]), P))
    return out


def build_sequence(seq_dir, out_dir, log=print):
    seq = os.path.basename(seq_dir.rstrip("/")).replace("rgbd_dataset_freiburg3_", "fr3_")
    fr = frames(seq_dir)
    if len(fr) < N_VIEWS:
        raise ValueError(f"{seq}: {len(fr)} usable frames")
    kf = keyframes(np.stack([P for _, _, P in fr]), KF_TRANS, KF_ROT)
    n = min(CLIPS_PER_SEQ, len(kf) // N_VIEWS)
    if n < 1:
        raise ValueError(f"{seq}: {len(kf)} keyframes")
    nw, nh, left, top = resize_crop_params(W, H, TW, TH)
    K = adjust_K(K_FR3, W, H, nw, nh, left, top)
    paths = []
    for c, s in enumerate(clip_starts(len(kf), N_VIEWS, n, non_overlapping=True)):
        sel = [fr[kf[k]] for k in range(s, s + N_VIEWS)]
        imgs, deps, Ps = [], [], []
        for rgb_path, dep_path, P in sel:
            im = Image.open(rgb_path).convert("RGB")
            if im.size != (W, H):
                raise ValueError(f"{rgb_path}: size {im.size}")
            imgs.append(np.asarray(im.resize((nw, nh), Image.BICUBIC).crop((left, top, left + TW, top + TH)), np.uint8))
            d = np.asarray(Image.open(dep_path), np.float32) / DEPTH_SCALE
            deps.append(resize_depth_nearest(d, W, H, nw, nh, left, top, TW, TH))
            Ps.append(P)
        poses = relative_to_first(np.stack(Ps))
        for i, P in enumerate(poses):
            validate_pose(P, f"{seq} view {i}")
        clip = dict(images=np.stack(imgs), K=np.stack([K] * N_VIEWS), poses=poses,
                    depth=np.stack(deps).astype(np.float32), names=np.array([os.path.basename(x[0]) for x in sel]),
                    scene=np.array(f"tum_{seq}"))
        path = os.path.join(out_dir, f"cal__tum_{seq}__c{c}.npz")
        np.savez_compressed(path, **clip)
        paths.append(path)
        log(f"[ok] {path}  keyframes {s}..{s + N_VIEWS - 1} of {len(kf)}  depth valid {(clip['depth'] > 0).mean():.0%}")
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="folder with rgbd_dataset_freiburg3_* sequence folders")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for d in sorted(glob.glob(os.path.join(a.raw, "rgbd_dataset_freiburg3_*"))):
        if os.path.isdir(d):
            try:
                build_sequence(d, a.out)
            except ValueError as e:
                print(f"[skip] {e}")


if __name__ == "__main__":
    main()
