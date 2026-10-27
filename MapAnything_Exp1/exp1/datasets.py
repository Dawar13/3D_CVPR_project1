"""Dataset preparation: ETH3D high-res multi-view (training split) -> clip .npz files.

Each clip file holds exactly what every experimental condition sees, so the
images, their order, resolution and preprocessing are fixed across conditions:

  images  uint8   (N, H, W, 3)
  K       float64 (N, 3, 3)   pinhole intrinsics at (H, W)
  poses   float64 (N, 4, 4)   OpenCV cam2world, re-expressed so view 0 = I (metres)
  depth   float32 (N, H, W)   GT z-depth in metres, 0 = invalid
  names   str     (N,)
  scene   str

Expected raw layout (the two official ETH3D archives unpacked into one root):
  <root>/<scene>/images/dslr_images_undistorted/DSC_*.JPG
  <root>/<scene>/dslr_calibration_undistorted/{cameras.txt,images.txt}
  <root>/<scene>/ground_truth_depth/dslr_images/DSC_*.JPG   (raw float32, not JPEG)

Usage:
  python -m exp1.datasets --eth3d_root /data/eth3d --out data/clips \
      --n_views 8 --clips_per_scene 1 --size 518 336
"""

import argparse
import hashlib
import os

import numpy as np
from PIL import Image

from .geometry import invert_pose, quat_wxyz_to_R, relative_to_first, validate_pose

ETH3D_TRAIN_SCENES = [
    "courtyard", "delivery_area", "electro", "facade", "kicker", "meadow",
    "office", "pipes", "playground", "relief", "relief_2", "terrace", "terrains",
]
N_DEV_SCENES = 3


def dev_eval_split(scenes, n_dev=N_DEV_SCENES):
    """Fixed split by hash of the scene name: not chosen by looking at results."""
    order = sorted(scenes, key=lambda s: hashlib.sha1(s.encode()).hexdigest())
    return sorted(order[:n_dev]), sorted(order[n_dev:])


def read_colmap_cameras(path):
    cams = {}
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        tok = line.split()
        cam_id, model, W, H = int(tok[0]), tok[1], int(tok[2]), int(tok[3])
        if model != "PINHOLE":
            raise ValueError(f"expected PINHOLE (undistorted) camera, got {model}")
        fx, fy, cx, cy = map(float, tok[4:8])
        cams[cam_id] = (W, H, np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]))
    return cams


def read_colmap_images(path):
    """Returns {name: (cam_id, cam2world 4x4)}. COLMAP stores world2cam."""
    out = {}
    lines = [l for l in open(path) if not l.startswith("#")]
    for header in lines[0::2]:
        tok = header.split()
        if len(tok) < 10:
            continue
        q = np.array(list(map(float, tok[1:5])))
        t = np.array(list(map(float, tok[5:8])))
        w2c = np.eye(4)
        w2c[:3, :3] = quat_wxyz_to_R(q)
        w2c[:3, 3] = t
        out[tok[9]] = (int(tok[8]), invert_pose(w2c))
    return out


def resize_crop_params(W, H, tw, th):
    s = max(tw / W, th / H)
    nw, nh = round(W * s), round(H * s)
    return nw, nh, (nw - tw) // 2, (nh - th) // 2


def adjust_K(K, W, H, nw, nh, left, top):
    sx, sy = nw / W, nh / H
    K2 = K.copy()
    K2[0, 0] *= sx
    K2[1, 1] *= sy
    K2[0, 2] = (K[0, 2] + 0.5) * sx - 0.5 - left
    K2[1, 2] = (K[1, 2] + 0.5) * sy - 0.5 - top
    return K2


def resize_depth_nearest(depth, nw, nh, left, top, tw, th):
    H, W = depth.shape
    u = np.arange(left, left + tw)
    v = np.arange(top, top + th)
    src_u = np.clip(np.round((u + 0.5) * W / nw - 0.5).astype(int), 0, W - 1)
    src_v = np.clip(np.round((v + 0.5) * H / nh - 0.5).astype(int), 0, H - 1)
    return depth[np.ix_(src_v, src_u)]


def load_eth3d_depth(path, W, H):
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size != W * H:
        raise ValueError(f"{path}: {raw.size} floats, expected {W}x{H}")
    d = raw.reshape(H, W)
    d[~np.isfinite(d) | (d <= 0)] = 0.0
    return d


def build_clip(root, scene, names, cams, imgs, tw, th):
    images, Ks, poses, depths = [], [], [], []
    for name in names:
        cam_id, c2w = imgs[name]
        W, H, K = cams[cam_id]
        nw, nh, left, top = resize_crop_params(W, H, tw, th)
        im = Image.open(os.path.join(root, scene, "images", name)).convert("RGB")
        if im.size != (W, H):
            raise ValueError(f"{name}: image {im.size} != calibration {(W, H)}")
        im = im.resize((nw, nh), Image.BICUBIC).crop((left, top, left + tw, top + th))
        dpath = os.path.join(root, scene, "ground_truth_depth", "dslr_images",
                             os.path.basename(name))
        d = load_eth3d_depth(dpath, W, H)
        images.append(np.asarray(im, dtype=np.uint8))
        Ks.append(adjust_K(K, W, H, nw, nh, left, top))
        poses.append(c2w)
        depths.append(resize_depth_nearest(d, nw, nh, left, top, tw, th))
    poses = relative_to_first(np.stack(poses))
    for i, T in enumerate(poses):
        validate_pose(T, f"{scene}/{names[i]}")
    return dict(images=np.stack(images), K=np.stack(Ks), poses=poses,
                depth=np.stack(depths).astype(np.float32),
                names=np.array(names), scene=np.array(scene))


def prepare_eth3d(root, out_dir, n_views, clips_per_scene, stride, tw, th, scenes):
    os.makedirs(out_dir, exist_ok=True)
    dev, ev = dev_eval_split(scenes)
    written = []
    for scene in scenes:
        calib = os.path.join(root, scene, "dslr_calibration_undistorted")
        cams = read_colmap_cameras(os.path.join(calib, "cameras.txt"))
        imgs = read_colmap_images(os.path.join(calib, "images.txt"))
        names = sorted(imgs)  # ETH3D DSLR names follow capture order
        span = (n_views - 1) * stride + 1
        if len(names) < span:
            print(f"[skip] {scene}: {len(names)} images < {span} needed")
            continue
        # clip starts evenly spaced over the sequence; fixed before any run
        starts = np.linspace(0, len(names) - span, clips_per_scene).round().astype(int)
        for c, s in enumerate(starts):
            sel = names[s:s + span:stride]
            clip = build_clip(root, scene, sel, cams, imgs, tw, th)
            split = "dev" if scene in dev else "eval"
            path = os.path.join(out_dir, f"{split}__{scene}__c{c}.npz")
            np.savez_compressed(path, **clip)
            valid = (clip["depth"] > 0).mean()
            print(f"[ok] {path}  views={len(sel)}  valid_depth={valid:.1%}")
            written.append(path)
    return written


def load_clip(path):
    z = np.load(path, allow_pickle=False)
    clip = {k: z[k] for k in z.files}
    clip["scene"] = str(clip["scene"])
    clip["clip_id"] = os.path.splitext(os.path.basename(path))[0]
    return clip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eth3d_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_views", type=int, default=8)
    ap.add_argument("--clips_per_scene", type=int, default=1)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--size", type=int, nargs=2, default=[518, 336], metavar=("W", "H"))
    ap.add_argument("--scenes", nargs="*", default=ETH3D_TRAIN_SCENES)
    a = ap.parse_args()
    prepare_eth3d(a.eth3d_root, a.out, a.n_views, a.clips_per_scene, a.stride,
                  a.size[0], a.size[1], a.scenes)


if __name__ == "__main__":
    main()
