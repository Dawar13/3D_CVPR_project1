"""Unit tests for the E1 hint-damage types (pose, lens, depth) and the E1 grid. Pure numpy.

  python -m pytest tests -q
"""

import json
import os

import numpy as np
import pytest

from exp1.corrupt_geometry import corrupt_poses
from exp1.corrupt_inputs import corrupt_depth, corrupt_intrinsics
from exp1.geometry import (invert_pose, relative_to_first, rotation_angle_deg, rotation_error_deg,
                           so3_exp, trajectory_scale)
from exp1.run_exp1 import expand_conditions, seeds_for

CFG = os.path.join(os.path.dirname(__file__), "..", "configs", "e1_full.json")


def random_poses(n=10, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        T = np.eye(4)
        T[:3, :3] = so3_exp(rng.normal(size=3))
        T[:3, 3] = rng.normal(size=3) * 3
        out.append(T)
    return relative_to_first(np.stack(out))


def K_stack(n=10):
    K = np.array([[300.0, 0, 259], [0, 301.0, 168], [0, 0, 1]])
    return np.stack([K.copy() for _ in range(n)])


def depth_stack(n=10, seed=0):
    rng = np.random.default_rng(seed)
    d = rng.uniform(2, 9, (n, 12, 16)).astype(np.float32)
    d[rng.random(d.shape) < 0.7] = 0  # sparse, like ETH3D
    return d


# ---------- pose types: size 0 / factor 1 is exactly clean

@pytest.mark.parametrize("spec", [
    {"type": "gauge", "rot_deg": 0.0, "trans_frac": 0.0},
    {"type": "scale", "factor": 1.0},
    {"type": "drift", "rot_deg": 0.0, "scale_growth": 0.0},
    {"type": "shared_rot", "rot_deg": 0.0},
    {"views": [0], "rot_deg": 0.0, "trans_frac": 0.0},
])
def test_pose_types_identity_at_zero(spec):
    P = random_poses()
    out, e = corrupt_poses(P, spec, 0, "c", "x")
    assert np.array_equal(out, P)


def test_gauge_keeps_relative_poses_and_moves_view0():
    P = random_poses()
    out, e = corrupt_poses(P, {"type": "gauge", "rot_deg": 30.0, "trans_frac": 1.0}, 1, "c", "g")
    assert rotation_angle_deg(np.array(e["params"]["G"])[:3, :3]) == pytest.approx(30.0, abs=1e-6)
    assert not np.allclose(out[0], np.eye(4))
    assert np.allclose(relative_to_first(out), P, atol=1e-9)


def test_scale_multiplies_every_translation_only():
    P = random_poses()
    out, e = corrupt_poses(P, {"type": "scale", "factor": 1.5}, 0, "c", "s")
    assert np.allclose(out[:, :3, 3], 1.5 * P[:, :3, 3])
    assert np.array_equal(out[:, :3, :3], P[:, :3, :3])
    assert np.allclose(out[0], P[0], atol=1e-15) and e["affected_views"] == list(range(1, 10))


def test_drift_steps_follow_the_model_exactly():
    P = random_poses()
    b, g = 2.0, 0.04
    out, e = corrupt_poses(P, {"type": "drift", "rot_deg": b, "scale_growth": g}, 0, "c", "d")
    assert np.array_equal(out[0], P[0])
    axis = np.array(e["params"]["axis_cam"])
    dR = so3_exp(e["params"]["sign"] * np.radians(b) * axis)
    for i in range(1, len(P)):
        D = invert_pose(P[i - 1]) @ P[i]
        D2 = invert_pose(out[i - 1]) @ out[i]
        assert np.allclose(D2[:3, :3], dR @ D[:3, :3], atol=1e-9)
        assert np.allclose(D2[:3, 3], (1 + g) ** i * D[:3, 3], atol=1e-9)
    # first step: exactly the per-step bias; the bias axis is within ~17 deg of the camera y axis
    assert rotation_error_deg(out[1][:3, :3], P[1][:3, :3]) == pytest.approx(b, abs=1e-6)
    assert np.degrees(np.arccos(abs(axis[1]))) < 45


def test_shared_rot_is_the_same_turn_on_every_non_reference_view():
    P = random_poses()
    out, e = corrupt_poses(P, {"type": "shared_rot", "rot_deg": 5.0}, 2, "c", "h")
    turns = [P[i][:3, :3].T @ out[i][:3, :3] for i in range(1, 10)]
    for t in turns:
        assert np.allclose(t, turns[0], atol=1e-12)
        assert rotation_angle_deg(t) == pytest.approx(5.0, abs=1e-6)
    assert np.array_equal(out[0], P[0]) and np.allclose(out[:, :3, 3], P[:, :3, 3])


@pytest.mark.parametrize("rot,frac", [(15.0, 0.0), (0.0, 0.25)])
def test_anchor_error_changes_only_view0_by_exact_amount(rot, frac):
    P = random_poses()
    out, e = corrupt_poses(P, {"views": [0], "rot_deg": rot, "trans_frac": frac}, 0, "c", "a")
    assert e["affected_views"] == [0]
    assert np.array_equal(out[1:], P[1:])
    assert rotation_error_deg(P[0][:3, :3], out[0][:3, :3]) == pytest.approx(rot, abs=1e-4)  # arccos near 0
    assert np.linalg.norm(out[0][:3, 3] - P[0][:3, 3]) == pytest.approx(frac * trajectory_scale(P), rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("spec", [
    {"type": "gauge", "rot_deg": 30.0, "trans_frac": 1.0},
    {"type": "drift", "rot_deg": 1.0, "scale_growth": 0.02},
    {"type": "shared_rot", "rot_deg": 5.0},
    {"views": [0], "rot_deg": 15.0},
])
def test_pose_types_deterministic_by_seed(spec):
    P = random_poses()
    a, _ = corrupt_poses(P, spec, 0, "clipA", "c")
    b, _ = corrupt_poses(P, spec, 0, "clipA", "c")
    c, _ = corrupt_poses(P, spec, 1, "clipA", "c")
    assert np.array_equal(a, b) and not np.array_equal(a, c)


# ---------- intrinsics

def test_focal_all_exact_and_identity():
    K = K_stack()
    out, e = corrupt_intrinsics(K, {"type": "focal", "factor": 1.2, "views": "all"}, 0, "c", "f", 518)
    assert np.allclose(out[:, 0, 0], 1.2 * K[:, 0, 0]) and np.allclose(out[:, 1, 1], 1.2 * K[:, 1, 1])
    assert np.array_equal(out[:, :2, 2], K[:, :2, 2]) and e["affected_views"] == list(range(10))
    same, _ = corrupt_intrinsics(K, {"type": "focal", "factor": 1.0, "views": "all"}, 0, "c", "f", 518)
    assert np.array_equal(same, K) and np.array_equal(K, K_stack())


def test_focal_one_picks_one_non_reference_view_deterministically():
    K = K_stack()
    picks = set()
    for seed in range(20):
        out, e = corrupt_intrinsics(K, {"type": "focal", "factor": 1.3, "views": "one"}, seed, "c", "f1", 518)
        (v,) = e["affected_views"]
        assert v != 0 and np.allclose(out[v, 0, 0], 1.3 * K[v, 0, 0])
        assert np.array_equal(np.delete(out, v, 0), np.delete(K, v, 0))
        picks.add(v)
    again, e2 = corrupt_intrinsics(K, {"type": "focal", "factor": 1.3, "views": "one"}, 0, "c", "f1", 518)
    assert e2["affected_views"] == corrupt_intrinsics(K, {"type": "focal", "factor": 1.3, "views": "one"}, 0, "c", "f1", 518)[1]["affected_views"]
    assert len(picks) > 1


def test_principal_shift_exact_and_identity():
    K = K_stack()
    out, _ = corrupt_intrinsics(K, {"type": "principal", "shift": 0.1, "views": "all"}, 0, "c", "p", 518)
    assert np.allclose(out[:, 0, 2], K[:, 0, 2] + 51.8) and np.allclose(out[:, 1, 2], K[:, 1, 2] + 51.8)
    assert np.array_equal(out[:, 0, 0], K[:, 0, 0])
    same, _ = corrupt_intrinsics(K, {"type": "principal", "shift": 0.0, "views": "all"}, 0, "c", "p", 518)
    assert np.array_equal(same, K)


# ---------- depth

def test_depth_scale_only_on_valid_pixels_of_hint_views():
    d = depth_stack()
    out, e = corrupt_depth(d, (3, 7), {"type": "scale", "factor": 1.5}, 0, "c", "ds")
    for i in range(10):
        if i in (3, 7):
            v = d[i] > 0
            assert np.allclose(out[i][v], 1.5 * d[i][v]) and np.all(out[i][~v] == 0)
        else:
            assert np.array_equal(out[i], d[i])
    one, e1 = corrupt_depth(d, (3, 7), {"type": "scale", "factor": 1.5, "views": [3]}, 0, "c", "d1")
    assert e1["affected_views"] == [3] and np.array_equal(one[7], d[7])
    same, _ = corrupt_depth(d, (3, 7), {"type": "scale", "factor": 1.0}, 0, "c", "ds")
    assert np.array_equal(same, d) and np.array_equal(d, depth_stack())


def test_depth_noise_size_determinism_and_identity():
    d = depth_stack(n=10)
    d[3] = np.where(d[3] > 0, 5.0, 0).astype(np.float32)
    d[3][:, :] = 5.0  # dense view for statistics
    out, _ = corrupt_depth(d, (3, 7), {"type": "noise", "sigma": 0.1}, 0, "c", "dn")
    r = out[3] / d[3]
    assert abs(r.std() - 0.1) < 0.04 and abs(r.mean() - 1) < 0.04 and r.min() >= 0.05 - 1e-6
    again, _ = corrupt_depth(d, (3, 7), {"type": "noise", "sigma": 0.1}, 0, "c", "dn")
    other, _ = corrupt_depth(d, (3, 7), {"type": "noise", "sigma": 0.1}, 1, "c", "dn")
    assert np.array_equal(out, again) and not np.array_equal(out, other)
    zero, _ = corrupt_depth(d, (3, 7), {"type": "noise", "sigma": 0.0}, 0, "c", "dn")
    assert np.array_equal(zero, d)


def test_depth_corruption_outside_hint_views_is_refused():
    with pytest.raises(ValueError):
        corrupt_depth(depth_stack(), (3, 7), {"type": "scale", "factor": 1.5, "views": [2]}, 0, "c", "x")


# ---------- the E1 grid

def test_e1_grid_expands_and_is_complete():
    cfg = json.load(open(CFG))
    conds = expand_conditions(cfg)
    names = {c["name"] for c in conds}
    for ref in ("rgb_only", "rgb_K", "clean_pose", "clean_pose_noK", "rgb_K_depth2", "clean_pose_depth2"):
        assert ref in names
    for c in conds:
        if c["family"] not in ("reference", "sanity"):
            assert c["floor"] in names and c["ceiling"] in names, c["name"]
            assert isinstance(c["realistic"], bool), c["name"]
    for fam in cfg["conditions"]:
        if "values" in fam and "realistic" in fam:
            assert set(fam["realistic"]) <= set(fam["values"]), fam["family"]
    runs = sum(len(seeds_for(c, cfg)) * (1 + len(c.get("controls", []))) for c in conds)
    assert 200 <= runs <= 260, runs  # E1 plan: about 230 inferences per clip
