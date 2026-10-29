"""E2 additions: one-channel pose corruptions, partial poses, per-view shape / pixel-size readouts,
the mean -> median pose-scale probe. Pure numpy except the probe (torch; the swap itself needs
MapAnything and is skipped on machines without it)."""

import numpy as np
import pytest

from exp1.corrupt_geometry import corrupt_poses
from exp1.geometry import (invert_pose, rotation_angle_deg, rotation_error_deg, rotation_error_deg_precise,
                           so3_exp, trajectory_scale)
from exp1.metrics import reproj_shift_px, sim3_shape_err, umeyama
from exp1.run_exp1 import posed_mask
from test_corrupt_geometry import random_poses


def _centres(P):
    return P[:, :3, 3] - P[0, :3, 3]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_radial_changes_distance_only(seed):
    clean = random_poses(10, seed)
    L = trajectory_scale(clean)
    out, e = corrupt_poses(clean, {"type": "radial", "mode": "subset", "n": 1, "trans_frac": 0.5}, seed, "c", "r")
    (i,) = e["affected_views"]
    a, b = _centres(clean)[i], _centres(out)[i]
    assert np.linalg.norm(b) - np.linalg.norm(a) == pytest.approx(0.5 * L)
    assert np.allclose(a / np.linalg.norm(a), b / np.linalg.norm(b))
    assert np.allclose(out[i][:3, :3], clean[i][:3, :3])
    assert np.allclose(np.delete(out, i, 0), np.delete(clean, i, 0))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_orbit_changes_direction_only(seed):
    clean = random_poses(10, seed)
    L = trajectory_scale(clean)
    d = 0.25 * L
    out, e = corrupt_poses(clean, {"type": "orbit", "mode": "subset", "n": 1, "trans_frac": 0.25}, seed, "c", "o")
    (i,) = e["affected_views"]
    a, b = _centres(clean)[i], _centres(out)[i]
    assert np.linalg.norm(b) == pytest.approx(np.linalg.norm(a))
    assert np.linalg.norm(b - a) == pytest.approx(d)
    assert np.allclose(out[i][:3, :3], clean[i][:3, :3])


def test_orbit_never_picks_a_view_too_close_to_view_0():
    clean = random_poses(10, 3)
    dist = np.linalg.norm(_centres(clean), axis=1)
    frac = 1.9 * np.median(dist[1:]) / trajectory_scale(clean)  # d / 2 = 0.95 x median distance
    for s in range(20):
        _, e = corrupt_poses(clean, {"type": "orbit", "mode": "subset", "n": 1, "trans_frac": frac}, s, "c", "o")
        assert dist[e["affected_views"][0]] >= frac * trajectory_scale(clean) / 2


def test_drift_jitter_matches_drift_sizes_with_independent_directions():
    clean = random_poses(10, 4)
    lv = {"rot_deg": 1.0, "scale_growth": 0.02}
    drift, _ = corrupt_poses(clean, {"type": "drift", **lv}, 1, "c", "c6_drift_level=2")
    jit, e = corrupt_poses(clean, {"type": "drift_jitter", **lv, "match": "c6_drift_level=2"}, 1, "c",
                           "c6_drift_jitter_level=2")
    for i in range(1, 10):
        assert rotation_angle_deg(clean[i][:3, :3].T @ jit[i][:3, :3]) == pytest.approx(
            rotation_angle_deg(clean[i][:3, :3].T @ drift[i][:3, :3]), abs=1e-6)
        assert np.linalg.norm(jit[i][:3, 3] - clean[i][:3, 3]) == pytest.approx(
            np.linalg.norm(drift[i][:3, 3] - clean[i][:3, 3]), abs=1e-9)
    axes = np.array([e["per_view"][i]["rot_axis_cam"] for i in range(1, 10)])
    assert np.abs(axes @ axes.T - np.eye(9)).max() < 0.999  # not one shared axis


def test_eligible_restricts_the_corrupted_views():
    clean = random_poses(10, 5)
    for s in range(10):
        _, e = corrupt_poses(clean, {"mode": "subset", "n": 1, "rot_deg": 15, "eligible": [2, 7]}, s, "c", "x")
        assert e["affected_views"][0] in (2, 7)


def test_posed_mask_is_shared_by_conditions_and_includes_view_0():
    for frac, k in ((0.3, 3), (0.6, 6), (1.0, 10)):
        m = posed_mask(10, frac, 1, "eval__x__c0")
        assert m[0] and m.sum() == k
        assert np.array_equal(m, posed_mask(10, frac, 1, "eval__x__c0"))
    masks = {tuple(posed_mask(10, 0.3, s, "c")) for s in range(5)}
    assert len(masks) > 1  # the posed set varies with the seed


def test_umeyama_and_view_shape_error():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3)) + [0, 0, 5]
    R = so3_exp([0.3, -0.2, 0.5])
    Y = 2.5 * X @ R.T + [1, 2, 3]
    s, R2, t = umeyama(X, Y)
    assert s == pytest.approx(2.5) and np.allclose(R2, R) and np.allclose(t, [1, 2, 3])
    assert sim3_shape_err(X, Y, np.zeros(3)) < 1e-9
    assert sim3_shape_err(X + rng.normal(scale=0.1, size=X.shape), X, np.zeros(3)) > 0.01


def _pinhole_clip(n=3, H=40, W=60):
    K = np.array([[50.0, 0, 30], [0, 50.0, 20], [0, 0, 1]])
    poses = np.stack([np.eye(4)] * n)
    for i in range(n):
        poses[i][:3, 3] = [0.2 * i, 0, 0]
    return np.full((n, H, W), 4.0), np.stack([K] * n), poses


def test_reproj_shift_px_zero_for_true_hints_and_exact_for_focal():
    depth, K, poses = _pinhole_clip()
    assert reproj_shift_px(1, depth, K, poses, K.copy(), poses.copy()) == pytest.approx(0, abs=1e-9)
    assert reproj_shift_px(1, depth, K, poses, hint_depth=depth.copy()) == pytest.approx(0, abs=1e-9)
    Kh = K.copy()
    Kh[1, 0, 0] *= 1.1
    Kh[1, 1, 1] *= 1.1
    v, u = np.nonzero(depth[1] > 0)
    expect = 0.1 * np.linalg.norm(np.stack([u - 30.0, v - 20.0], -1), axis=1).mean()
    assert reproj_shift_px(1, depth, K, poses, Kh, max_px=10 ** 6) == pytest.approx(expect, rel=1e-6)
    assert reproj_shift_px(1, depth, K, poses, hint_depth=1.5 * depth) > 0


def test_median_pose_scale_ignores_zero_translations():
    torch = pytest.importorskip("torch")
    from exp1.model_io import median_pose_translations
    t = torch.tensor([[[0.0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 10], [0, 0, 0]]])
    out, f = median_pose_translations(t, return_norm_factor=True)
    assert float(f[0]) == pytest.approx(2.0)  # median of {1, 2, 10}; zeros excluded
    assert torch.allclose(out, t / 2.0)


def test_pose_scale_statistic_swap():
    pytest.importorskip("mapanything.models.mapanything.model")
    import mapanything.models.mapanything.model as mm
    from mapanything.utils import geometry
    from exp1 import model_io
    model_io.set_pose_scale_statistic("median")
    assert mm.normalize_pose_translations is model_io.median_pose_translations
    model_io.set_pose_scale_statistic("mean")
    assert mm.normalize_pose_translations is geometry.normalize_pose_translations
    t = __import__("torch").rand(2, 5, 3) + 0.1
    a = geometry.normalize_pose_translations(t, return_norm_factor=True)[1]
    assert __import__("torch").allclose(a, t.norm(dim=-1).mean(1))


def test_seed_as_reproduces_another_conditions_corruption():
    clean = random_poses(10, 6)
    spec = {"mode": "subset", "n": 1, "trans_frac": 1.0}
    a, ea = corrupt_poses(clean, spec, 2, "c", "c1_trans_trans_frac=1")
    b, eb = corrupt_poses(clean, dict(spec, seed_as="c1_trans_trans_frac=1"), 2, "c", "c4m_trans_trans_frac=1")
    c, _ = corrupt_poses(clean, spec, 2, "c", "c4m_trans_trans_frac=1")
    assert np.array_equal(a, b) and ea["affected_views"] == eb["affected_views"]
    assert not np.array_equal(a, c)


def test_precise_rotation_error_agrees_and_resolves_tiny_angles():
    rng = np.random.default_rng(7)
    for deg in (0.001, 0.01, 0.5, 15, 90, 179.9):
        u = rng.normal(size=3); u /= np.linalg.norm(u)
        A = so3_exp(rng.normal(size=3))
        B = A @ so3_exp(np.radians(deg) * u)
        assert rotation_error_deg_precise(A, B) == pytest.approx(deg, rel=1e-6, abs=1e-9)
        if deg >= 0.5:
            assert rotation_error_deg_precise(A, B) == pytest.approx(rotation_error_deg(A, B), abs=1e-6)
    A32 = so3_exp([0.3, 0.2, 0.1]).astype(np.float32).astype(np.float64)  # float32-rounded identity pair
    assert rotation_error_deg_precise(A32, A32 @ np.eye(3)) < 1e-4 < 0.01
