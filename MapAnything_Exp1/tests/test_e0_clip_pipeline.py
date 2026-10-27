"""End-to-end CPU test of clip preparation + the E0 Step-5 GT self-consistency check.

A synthetic ETH3D-style scene (slanted plane, 4 cameras) is written in the raw ETH3D
layout, with GT depth rendered on the DISTORTED (THIN_PRISM_FISHEYE) grid as in the
real dataset. Then exp1.datasets builds a clip and scripts/e0_check_clips.py scores it.

  python -m pytest tests -q
"""

import importlib.util
import os

import numpy as np
import pytest
from PIL import Image

from exp1.datasets import prepare_eth3d, load_clip, thin_prism_fisheye_project
from exp1.geometry import invert_pose, so3_exp

W, H = 360, 240
PINHOLE = (300.0, 300.0, 180.0, 120.0)
DIST = np.array([300.0, 300.0, 180.0, 120.0, 0.05, -0.01, 1e-3, -1e-3, 0.0, 0.0, 1e-3, -1e-3])
PLANE_N, PLANE_D = np.array([0.6, 0.3, 1.0]), 6.0  # world plane n . X = d

_spec = importlib.util.spec_from_file_location(
    "e0_check_clips", os.path.join(os.path.dirname(__file__), "..", "scripts", "e0_check_clips.py"))
e0_check_clips = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e0_check_clips)


def R_to_quat_wxyz(R):
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return np.array([w, x, y, z])


def plane_z(c2w, x, y, n=PLANE_N):
    """z-depth of the plane n . X = PLANE_D along camera rays with normalised coords (x, y)."""
    d = np.stack([x, y, np.ones_like(x)], -1) @ c2w[:3, :3].T
    s = (PLANE_D - n @ c2w[:3, 3]) / (d @ n)
    return np.where(s > 0, s, 0.0)


def undistort_normalised(xd, yd, p, iters=100):
    """Invert the thin-prism fisheye model: distorted normalised coords -> pinhole x, y."""
    k1, k2, p1, p2, k3, k4, sx1, sy1 = p[4:]
    u, v = xd.copy(), yd.copy()
    for _ in range(iters):
        r2 = u * u + v * v
        rad = k1 * r2 + k2 * r2**2 + k3 * r2**3 + k4 * r2**4
        du = u * rad + 2 * p1 * u * v + p2 * (r2 + 2 * u * u) + sx1 * r2
        dv = v * rad + 2 * p2 * u * v + p1 * (r2 + 2 * v * v) + sy1 * r2
        u, v = xd - du, yd - dv
    th = np.hypot(u, v)
    s = np.where(th > 1e-12, np.tan(th) / np.maximum(th, 1e-12), 1.0)
    return u * s, v * s


def write_scene(root, n=4, seed=0):
    rng = np.random.default_rng(seed)
    scene = os.path.join(root, "synth")
    for sub in ("images/dslr_images_undistorted", "dslr_calibration_undistorted",
                "dslr_calibration_jpg", "ground_truth_depth/dslr_images"):
        os.makedirs(os.path.join(scene, sub))
    with open(os.path.join(scene, "dslr_calibration_undistorted", "cameras.txt"), "w") as f:
        f.write("# Camera list\n0 PINHOLE %d %d %s\n" % (W, H, " ".join(map(str, PINHOLE))))
    with open(os.path.join(scene, "dslr_calibration_jpg", "cameras.txt"), "w") as f:
        f.write("# Camera list\n0 THIN_PRISM_FISHEYE %d %d %s\n" % (W, H, " ".join(map(str, DIST))))
    lines_u, lines_d = ["# header\n"], ["# header\n"]
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)  # COLMAP pixel centres
    xd, yd = undistort_normalised((u - DIST[2]) / DIST[0], (v - DIST[3]) / DIST[1], DIST)
    for k in range(n):
        c2w = np.eye(4)  # camera 0 = world, so clip poses (relative to view 0) = world poses
        if k > 0:
            c2w[:3, :3] = so3_exp(rng.normal(0, 0.05, 3))
            c2w[:3, 3] = rng.normal(0, 0.3, 3)
        w2c = invert_pose(c2w)
        q, t = R_to_quat_wxyz(w2c[:3, :3]), w2c[:3, 3]
        name = f"DSC_{k:04d}.JPG"
        rec = f"{k + 1} {' '.join(map(str, q))} {' '.join(map(str, t))} 0"
        lines_u.append(f"{rec} dslr_images_undistorted/{name}\n\n")
        lines_d.append(f"{rec} dslr_images/{name}\n\n")
        img = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
        Image.fromarray(img).save(os.path.join(scene, "images", "dslr_images_undistorted", name))
        plane_z(c2w, xd, yd).astype(np.float32).tofile(
            os.path.join(scene, "ground_truth_depth", "dslr_images", name))
    for sub, lines in (("dslr_calibration_undistorted", lines_u), ("dslr_calibration_jpg", lines_d)):
        with open(os.path.join(scene, sub, "images.txt"), "w") as f:
            f.writelines(lines)


def analytic_depth(clip, n=PLANE_N):
    """Exact undistorted z-depth at every clip pixel (OpenCV pixel convention)."""
    N, h, w = clip["depth"].shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    out = []
    for K, T in zip(clip["K"], clip["poses"]):
        out.append(plane_z(T, (u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], n))
    return np.stack(out)


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("eth3d"))
    write_scene(root)
    out = {}
    for und in (True, False):
        d = os.path.join(root, f"clips_{und}")
        prepare_eth3d(root, d, n_views=4, clips_per_scene=1, stride=1, tw=170, th=100,
                      scenes=["synth"], undistort_depth=und)
        out[und] = load_clip(os.path.join(d, "dev__synth__c0.npz"))
    return out


def _median_rel(clip, n=PLANE_N):
    gt = analytic_depth(clip, n)
    m = (clip["depth"] > 0) & (gt > 0)
    return float(np.median(np.abs(clip["depth"][m] - gt[m]) / gt[m]))


def test_flags_and_reference_frame(clips):
    assert np.allclose(clips[True]["poses"][0], np.eye(4), atol=1e-9)
    assert bool(clips[True]["depth_undistorted"]) and not bool(clips[False]["depth_undistorted"])


def test_undistorted_depth_matches_analytic(clips):
    err_und, err_raw = _median_rel(clips[True]), _median_rel(clips[False])
    assert err_und < 0.005, err_und
    assert err_raw > 3 * err_und, (err_raw, err_und)


def test_clip_check_passes_on_correct_clip(clips):
    rows, verdict = e0_check_clips.check_clip(clips[True])
    assert verdict == "PASS", rows
    assert all(r["overlap"] > 0.5 for r in rows)


def test_clip_check_fails_on_world2cam_poses(clips):
    bad = dict(clips[True])
    bad["poses"] = np.stack([invert_pose(T) for T in bad["poses"]])
    rows, verdict = e0_check_clips.check_clip(bad)
    assert verdict != "PASS", rows


def test_clip_check_sees_distortion_mismatch(clips):
    good, _ = e0_check_clips.check_clip(clips[True])
    raw, _ = e0_check_clips.check_clip(clips[False])
    assert np.median([r["median_rel"] for r in raw]) > 2 * np.median([r["median_rel"] for r in good])


def test_split_is_fixed_over_full_eth3d_list():
    from exp1.datasets import ETH3D_TRAIN_SCENES, dev_eval_split, split_of
    dev, ev = dev_eval_split(ETH3D_TRAIN_SCENES)
    assert dev == ["delivery_area", "facade", "terrace"]
    # preparing any subset must never relabel an eval scene as dev (2026-10-27 bug)
    subset = ["courtyard", "electro", "kicker", "meadow", "office", "pipes", "playground"]
    assert all(split_of(s, subset) == "eval" for s in subset)
    assert all(split_of(s, ["facade"]) == "dev" for s in dev)
    assert split_of("synth", ["synth"]) == "dev"  # non-ETH3D scenes: split over themselves


def test_thin_prism_identity_at_centre():
    u, v = thin_prism_fisheye_project(np.zeros(1), np.zeros(1), DIST)
    assert np.allclose([u[0], v[0]], DIST[2:4])


def test_upright_rotation_keeps_geometry(tmp_path, monkeypatch):
    """Views listed as stored-sideways are turned 90 deg clockwise: image, depth, K and pose
    must stay consistent (GT check passes, depth matches the analytic plane)."""
    import exp1.datasets as ds
    root = str(tmp_path)
    write_scene(root, n=4)
    monkeypatch.setitem(ds.ETH3D_ROTATED, "synth", [f"DSC_{k:04d}.JPG" for k in range(4)])
    side = os.path.join(root, "side")
    up = os.path.join(root, "up")
    prepare_eth3d(root, side, 4, 1, 1, 170, 100, ["synth"], upright=False)
    prepare_eth3d(root, up, 4, 1, 1, 170, 100, ["synth"], upright=True)
    a, b = load_clip(os.path.join(side, "dev__synth__c0.npz")), load_clip(os.path.join(up, "dev__synth__c0.npz"))
    assert a["images"].shape == (4, 100, 170, 3) and b["images"].shape == (4, 170, 100, 3)
    assert b["rotated_upright"].all() and not a["rotated_upright"].any()
    assert np.array_equal(b["images"][0], np.rot90(a["images"][0], k=-1))
    assert np.array_equal(b["depth"], np.rot90(a["depth"], k=-1, axes=(1, 2)))
    assert np.allclose(b["poses"][0], np.eye(4), atol=1e-9)
    assert e0_check_clips.check_clip(b)[1] == "PASS"
    # same camera centres; depth along each pixel ray matches the rotated camera model
    # new world frame = the turned camera 0: centres and the plane rotate by R_CW
    assert np.allclose(a["poses"][:, :3, 3] @ ds.R_CW.T, b["poses"][:, :3, 3], atol=1e-9)
    assert _median_rel(b, ds.R_CW @ PLANE_N) < 0.005


def test_mixed_sideways_clip_is_refused(tmp_path, monkeypatch):
    import exp1.datasets as ds
    root = str(tmp_path)
    write_scene(root, n=4)
    monkeypatch.setitem(ds.ETH3D_ROTATED, "synth", ["DSC_0001.JPG"])
    with pytest.raises(ValueError, match="mixes sideways"):
        prepare_eth3d(root, os.path.join(root, "o"), 4, 1, 1, 170, 100, ["synth"])
