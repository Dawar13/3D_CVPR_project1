"""exp1/tum_clips.py on a synthetic sequence in the TUM file format: frame association, pose interpolation at the
colour timestamps, keyframes, intrinsics after resize/crop, depth scale."""

import os

import numpy as np
import pytest
from PIL import Image

from exp1 import tum_clips as T
from exp1.datasets import load_clip


def write_seq(root, n=45, step=0.05):
    d = os.path.join(root, "rgbd_dataset_freiburg3_synthetic")
    os.makedirs(os.path.join(d, "rgb"))
    os.makedirs(os.path.join(d, "depth"))
    t0 = 1305031100.0
    with open(os.path.join(d, "rgb.txt"), "w") as fr, open(os.path.join(d, "depth.txt"), "w") as fd:
        fr.write("# color images\n")
        for i in range(n):
            t = t0 + i / 30.0
            Image.fromarray(np.full((480, 640, 3), (i * 5) % 255, np.uint8)).save(os.path.join(d, "rgb", f"{t:.6f}.png"))
            Image.fromarray(np.full((480, 640), 10000, np.uint16)).save(os.path.join(d, "depth", f"{t + 0.004:.6f}.png"))
            fr.write(f"{t:.6f} rgb/{t:.6f}.png\n")
            fd.write(f"{t + 0.004:.6f} depth/{t + 0.004:.6f}.png\n")
    with open(os.path.join(d, "groundtruth.txt"), "w") as f:   # 100 Hz, straight line along x, 1.5 m/s
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for k in range(int(n / 30.0 * 100) + 5):
            t = t0 - 0.02 + k / 100.0
            f.write(f"{t:.6f} {1.0 + step * 30 * (t - t0):.6f} 2.0 3.0 0 0 0 1\n")
    return d


def test_build_synthetic_sequence(tmp_path):
    seq = write_seq(str(tmp_path))
    out = str(tmp_path / "clips")
    os.makedirs(out)
    paths = T.build_sequence(seq, out, log=lambda *a: None)
    assert len(paths) == 1 and os.path.basename(paths[0]) == "cal__tum_fr3_synthetic__c0.npz"
    c = load_clip(paths[0])
    assert c["images"].shape == (10, 392, 518, 3) and c["depth"].shape == (10, 392, 518)
    assert np.allclose(c["depth"], 2.0)                        # 10000 / 5000
    assert np.allclose(c["poses"][0], np.eye(4))
    steps = np.diff(c["poses"][:, 0, 3])                      # keyframes every 0.20 m along x
    assert np.all(steps >= T.KF_TRANS - 1e-6) and np.all(steps < T.KF_TRANS + 0.06)
    assert np.allclose(c["poses"][:, 1:3, 3], 0, atol=1e-9)
    nw, nh, left, top = T.resize_crop_params(640, 480, 518, 392)
    assert (nw, nh) == (523, 392)
    K = c["K"][0]
    assert np.isclose(K[0, 0], 535.4 * nw / 640) and np.isclose(K[1, 1], 539.2 * nh / 480)
    assert np.isclose(K[0, 2], (320.1 + 0.5) * nw / 640 - 0.5 - left)


def test_interpolation_and_gaps():
    gt = [(0.0, ["0", "0", "0", "0", "0", "0", "1"]), (0.01, ["1", "0", "0", "0", "0", "0", "1"])]
    at = T.gt_interpolator(gt)
    assert np.isclose(at(0.005)[0, 3], 0.5)
    assert at(0.02) is None and at(-0.001) is None
