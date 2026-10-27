"""Unit tests for the pose utilities. Pure numpy; no model or GPU needed.

  python -m pytest tests -q
"""

import numpy as np
import pytest

from exp1.corrupt_geometry import corrupt_poses, select_views
from exp1.geometry import (invert_pose, quat_wxyz_to_R, relative_to_first,
                           rotation_angle_deg, rotation_error_deg, so3_exp,
                           trajectory_scale, validate_pose)
from exp1.controls import rgbcheck_mask


def random_poses(n=10, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        T = np.eye(4)
        T[:3, :3] = so3_exp(rng.normal(size=3))
        T[:3, 3] = rng.normal(size=3) * 3
        out.append(T)
    return relative_to_first(np.stack(out))


def test_so3_exp_angle_exact():
    rng = np.random.default_rng(1)
    for deg in [0.5, 1, 5, 30, 179]:
        u = rng.normal(size=3); u /= np.linalg.norm(u)
        R = so3_exp(np.radians(deg) * u)
        validate_pose(np.block([[R, np.zeros((3, 1))], [np.zeros((1, 3)), 1]]))
        assert rotation_angle_deg(R) == pytest.approx(deg, abs=1e-6)


def test_quat_identity_and_inverse():
    assert np.allclose(quat_wxyz_to_R(np.array([1., 0, 0, 0])), np.eye(3))
    T = random_poses(2)[1]
    assert np.allclose(invert_pose(T) @ T, np.eye(4), atol=1e-10)


def test_relative_to_first_makes_ref_identity():
    P = random_poses()
    assert np.allclose(P[0], np.eye(4), atol=1e-12)


def test_validate_rejects_bad():
    T = np.eye(4); T[0, 0] = 2
    with pytest.raises(ValueError):
        validate_pose(T)
    T = np.eye(4); T[0, 0] = -1  # reflection
    with pytest.raises(ValueError):
        validate_pose(T)
    T = np.eye(4); T[1, 3] = np.nan
    with pytest.raises(ValueError):
        validate_pose(T)


def test_zero_noise_is_bitwise_clean():
    P = random_poses()
    out, e = corrupt_poses(P, {"mode": "all", "rot_deg": 0.0, "trans_frac": 0.0}, 0, "c", "z")
    assert np.array_equal(out, P)


def test_clean_input_not_modified():
    P = random_poses(); P0 = P.copy()
    corrupt_poses(P, {"mode": "all", "rot_deg": 5, "trans_frac": 0.05}, 0, "c", "x")
    assert np.array_equal(P, P0)


@pytest.mark.parametrize("deg", [0.5, 2, 15, 30])
def test_rotation_magnitude_and_centre_fixed(deg):
    P = random_poses()
    out, e = corrupt_poses(P, {"mode": "all", "rot_deg": deg, "trans_frac": 0.0}, 3, "c", "r")
    for i in e["affected_views"]:
        assert rotation_error_deg(P[i][:3, :3], out[i][:3, :3]) == pytest.approx(deg, abs=1e-6)
        assert np.allclose(P[i][:3, 3], out[i][:3, 3])


@pytest.mark.parametrize("frac", [0.005, 0.05, 0.5])
def test_translation_magnitude_and_rotation_fixed(frac):
    P = random_poses()
    L = trajectory_scale(P)
    out, e = corrupt_poses(P, {"mode": "all", "rot_deg": 0.0, "trans_frac": frac}, 3, "c", "t")
    for i in e["affected_views"]:
        assert np.linalg.norm(out[i][:3, 3] - P[i][:3, 3]) == pytest.approx(frac * L, rel=1e-9)
        assert np.allclose(P[i][:3, :3], out[i][:3, :3])
    assert e["trans_m"] == pytest.approx(frac * L)


def test_reference_view_excluded_and_others_untouched():
    P = random_poses()
    out, e = corrupt_poses(P, {"mode": "subset", "n": 1, "rot_deg": 30}, 0, "c", "o")
    assert e["affected_views"] != [0] and 0 not in e["affected_views"]
    for i in range(len(P)):
        if i not in e["affected_views"]:
            assert np.array_equal(out[i], P[i])


def test_subset_counts_and_fraction():
    rng = np.random.default_rng(0)
    assert len(select_views(10, {"mode": "subset", "frac": 0.1}, rng)) == 1
    assert len(select_views(10, {"mode": "subset", "frac": 0.2}, rng)) == 2
    assert len(select_views(10, {"mode": "subset", "frac": 0.4}, rng)) == 4
    assert len(select_views(10, {"mode": "all"}, rng)) == 9
    assert 0 in select_views(10, {"mode": "all", "exclude_ref": False}, rng)


def test_determinism_and_seed_dependence():
    P = random_poses()
    spec = {"mode": "subset", "n": 2, "rot_deg": 15}
    a, ea = corrupt_poses(P, spec, 0, "clipA", "c")
    b, eb = corrupt_poses(P, spec, 0, "clipA", "c")
    c, ec = corrupt_poses(P, spec, 1, "clipA", "c")
    assert np.array_equal(a, b) and ea["affected_views"] == eb["affected_views"]
    assert not np.array_equal(a, c)


def test_rgbcheck_flags_rotation_outlier_only():
    P = random_poses()
    out, e = corrupt_poses(P, {"mode": "subset", "n": 1, "rot_deg": 30}, 0, "c", "o")
    keep = rgbcheck_mask(out, P, rot_tau=10, dir_tau=15)  # "RGB-only" = perfect here
    assert list(np.where(~keep)[0]) == e["affected_views"]
    assert rgbcheck_mask(P, P, 10, 15).all()
