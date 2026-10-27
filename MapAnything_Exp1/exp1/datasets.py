"""Dataset preparation: ETH3D high-res multi-view (training split) -> clip .npz files.

Each clip file holds exactly what every experimental condition sees, so the
images, their order, resolution and preprocessing are fixed across conditions:

  images  uint8   (N, H, W, 3)
  K       float64 (N, 3, 3)   pinhole intrinsics at (H, W)
  poses   float64 (N, 4, 4)   OpenCV cam2world, re-expressed so view 0 = I (metres)
  depth   float32 (N, H, W)   GT z-depth in metres, 0 = invalid
  names   str     (N,)
  scene   str
  depth_undistorted bool      True if GT depth was re-sampled onto the undistorted grid

Expected raw layout (official ETH3D archives unpacked into one root):
  <root>/<scene>/images/dslr_images_undistorted/DSC_*.JPG
  <root>/<scene>/dslr_calibration_undistorted/{cameras.txt,images.txt}
  <root>/<scene>/ground_truth_depth/dslr_images/DSC_*.JPG   (raw float32, not JPEG)
  <root>/<scene>/dslr_calibration_jpg/{cameras.txt,images.txt}   (from the dslr_jpg archive)

The GT depth maps belong to the DISTORTED images (dslr_images, THIN_PRISM_FISHEYE),
not the undistorted ones. Each undistorted target pixel is therefore re-projected
into the distorted depth map, as MapAnything's own ETH3D converter does
(map-anything data_processing/wai_processing/scripts/conversion/eth3d.py:253-414).
--no_depth_undistort restores the old direct lookup (diagnostics only).

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
# Images stored rotated by 90 deg (originally portrait), per map-anything
# data_processing/wai_processing/scripts/conversion/eth3d.py:210-249. Geometry is
# still consistent; only "up" differs. Reported by the E0 checks, not altered.
ETH3D_ROTATED = {
    "delivery_area": [f"DSC_{i:04d}.JPG" for i in range(711, 715)],
    "playground": [f"DSC_{i:04d}.JPG" for i in range(587, 593)],
    "relief": [f"DSC_{i:04d}.JPG" for i in range(427, 440)],
    "relief_2": [f"DSC_{i:04d}.JPG" for i in range(458, 469)],
}


def dev_eval_split(scenes, n_dev=N_DEV_SCENES):
    """Fixed split by hash of the scene name: not chosen by looking at results."""
    order = sorted(scenes, key=lambda s: hashlib.sha1(s.encode()).hexdigest())
    return sorted(order[:n_dev]), sorted(order[n_dev:])


def split_of(scene, scenes=()):
    """'dev' or 'eval'. ETH3D scenes are split over the FULL fixed list, so preparing a
    subset can never relabel an eval scene as dev; other scenes (tests) over `scenes`."""
    pool = ETH3D_TRAIN_SCENES if scene in ETH3D_TRAIN_SCENES else list(scenes)
    return "dev" if scene in dev_eval_split(pool)[0] else "eval"


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


def read_colmap_distorted_cameras(path):
    """dslr_calibration_jpg/cameras.txt -> {cam_id: (W, H, 12 THIN_PRISM_FISHEYE params)}."""
    cams = {}
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        tok = line.split()
        if tok[1] != "THIN_PRISM_FISHEYE" or len(tok) != 16:
            raise ValueError(f"expected THIN_PRISM_FISHEYE with 12 params, got {tok[1:]}")
        cams[int(tok[0])] = (int(tok[2]), int(tok[3]), np.array(list(map(float, tok[4:]))))
    return cams


def thin_prism_fisheye_project(x, y, params):
    """COLMAP THIN_PRISM_FISHEYE: normalised coords (X/Z, Y/Z) -> distorted pixel coords.

    params = fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1 (COLMAP pixel convention).
    """
    fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1 = params
    r = np.hypot(x, y)
    eps = np.finfo(np.float64).eps
    s = np.where(r > eps, np.arctan(r) / np.maximum(r, eps), 1.0)
    u, v = x * s, y * s
    r2 = u * u + v * v
    radial = k1 * r2 + k2 * r2**2 + k3 * r2**3 + k4 * r2**4
    du = u * radial + 2 * p1 * u * v + p2 * (r2 + 2 * u * u) + sx1 * r2
    dv = v * radial + 2 * p2 * u * v + p1 * (r2 + 2 * v * v) + sy1 * r2
    return fx * (u + du) + cx, fy * (v + dv) + cy


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


def resize_depth_nearest(depth, W, H, nw, nh, left, top, tw, th, distortion=None):
    """Nearest-sample GT depth at every target pixel of the resized+cropped (W, H) image.

    distortion: None if `depth` is on the undistorted (W, H) grid; otherwise
    (K_undistorted, thin-prism params) and `depth` is on the distorted grid.
    """
    u = np.arange(left, left + tw)
    v = np.arange(top, top + th)
    src_u = np.clip(np.round((u + 0.5) * W / nw - 0.5).astype(int), 0, W - 1)
    src_v = np.clip(np.round((v + 0.5) * H / nh - 0.5).astype(int), 0, H - 1)
    if distortion is None:
        return depth[np.ix_(src_v, src_u)]
    K, params = distortion
    su, sv = np.meshgrid(src_u, src_v)
    # COLMAP pixel convention: pixel i covers [i, i+1), centre at i + 0.5
    xd, yd = thin_prism_fisheye_project((su + 0.5 - K[0, 2]) / K[0, 0],
                                        (sv + 0.5 - K[1, 2]) / K[1, 1], params)
    iu, iv = np.floor(xd).astype(int), np.floor(yd).astype(int)
    inside = (iu >= 0) & (iu < depth.shape[1]) & (iv >= 0) & (iv < depth.shape[0])
    out = np.zeros((th, tw), depth.dtype)
    out[inside] = depth[iv[inside], iu[inside]]
    return out


def load_eth3d_depth(path, W, H):
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size != W * H:
        raise ValueError(f"{path}: {raw.size} floats, expected {W}x{H}")
    d = raw.reshape(H, W)
    d[~np.isfinite(d) | (d <= 0)] = 0.0
    return d


def build_clip(root, scene, names, cams, imgs, tw, th, dist=None):
    """dist: (distorted cams, distorted images by basename) from dslr_calibration_jpg,
    or None to read the GT depth as if it were on the undistorted grid."""
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
        if dist is None:
            d, distortion = load_eth3d_depth(dpath, W, H), None
        else:
            d_cams, d_imgs = dist
            d_cam_id, d_c2w = d_imgs[os.path.basename(name)]
            if d_cam_id != cam_id or not np.allclose(d_c2w, c2w, atol=1e-5):
                raise ValueError(f"{name}: distorted and undistorted calibrations disagree")
            Wd, Hd, params = d_cams[d_cam_id]
            d, distortion = load_eth3d_depth(dpath, Wd, Hd), (K, params)
        images.append(np.asarray(im, dtype=np.uint8))
        Ks.append(adjust_K(K, W, H, nw, nh, left, top))
        poses.append(c2w)
        depths.append(resize_depth_nearest(d, W, H, nw, nh, left, top, tw, th, distortion))
    poses = relative_to_first(np.stack(poses))
    for i, T in enumerate(poses):
        validate_pose(T, f"{scene}/{names[i]}")
    return dict(images=np.stack(images), K=np.stack(Ks), poses=poses,
                depth=np.stack(depths).astype(np.float32),
                names=np.array(names), scene=np.array(scene),
                depth_undistorted=np.array(dist is not None))


def load_distortion(root, scene):
    calib = os.path.join(root, scene, "dslr_calibration_jpg")
    if not os.path.exists(os.path.join(calib, "cameras.txt")):
        raise SystemExit(
            f"{calib}/cameras.txt missing. ETH3D GT depth is on the DISTORTED image grid; "
            "unpack dslr_calibration_jpg from multi_view_training_dslr_jpg.7z, "
            "or pass --no_depth_undistort (diagnostics only).")
    imgs = read_colmap_images(os.path.join(calib, "images.txt"))
    return (read_colmap_distorted_cameras(os.path.join(calib, "cameras.txt")),
            {os.path.basename(k): v for k, v in imgs.items()})


def prepare_eth3d(root, out_dir, n_views, clips_per_scene, stride, tw, th, scenes,
                  undistort_depth=True):
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for scene in scenes:
        calib = os.path.join(root, scene, "dslr_calibration_undistorted")
        cams = read_colmap_cameras(os.path.join(calib, "cameras.txt"))
        imgs = read_colmap_images(os.path.join(calib, "images.txt"))
        dist = load_distortion(root, scene) if undistort_depth else None
        names = sorted(imgs)  # ETH3D DSLR names follow capture order
        span = (n_views - 1) * stride + 1
        if len(names) < span:
            print(f"[skip] {scene}: {len(names)} images < {span} needed")
            continue
        # clip starts evenly spaced over the sequence; fixed before any run
        starts = np.linspace(0, len(names) - span, clips_per_scene).round().astype(int)
        for c, s in enumerate(starts):
            sel = names[s:s + span:stride]
            clip = build_clip(root, scene, sel, cams, imgs, tw, th, dist)
            split = split_of(scene, scenes)
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
    ap.add_argument("--no_depth_undistort", action="store_true",
                    help="read GT depth as if on the undistorted grid (diagnostics only)")
    a = ap.parse_args()
    prepare_eth3d(a.eth3d_root, a.out, a.n_views, a.clips_per_scene, a.stride,
                  a.size[0], a.size[1], a.scenes, undistort_depth=not a.no_depth_undistort)


if __name__ == "__main__":
    main()
