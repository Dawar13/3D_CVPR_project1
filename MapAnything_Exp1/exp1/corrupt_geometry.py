"""Deterministic, logged corruption of camera poses.

Perturbation model (applied to cam2world T = [R | c]):
  rotation:    R' = R @ Exp(theta * u),  u ~ uniform on S^2, theta FIXED (deg)
               -> the camera turns about its own optical centre by exactly theta
                  degrees, axis expressed in the camera frame. c is unchanged.
  translation: c' = c + (frac * L) * v,  v ~ uniform on S^2, magnitude FIXED
               L = trajectory_scale(clean poses of the clip) (metres)

Using a fixed magnitude with a random direction (instead of Gaussian noise)
makes every severity level an exact, preregistered error size.

Modes:
  "all"    -> every eligible view is perturbed (dense noise)
  "subset" -> exactly n views (or round(frac * N_eligible), min 1) are perturbed
              (isolated outliers / outlier fraction)

The reference view (index 0) is excluded by default, because poses are fed
relative to view 0 and the model may normalise by the first view; corrupting
it would corrupt every relative pose at once. Set exclude_ref=False to test
that case explicitly.
"""

import hashlib
import json

import numpy as np

from .geometry import random_unit_vector, so3_exp, trajectory_scale, validate_pose


def _seed_for(base_seed, clip_id, cond_name):
    """Stable per-(seed, clip, condition) RNG seed, independent of run order."""
    h = hashlib.sha256(f"{base_seed}|{clip_id}|{cond_name}".encode()).hexdigest()
    return int(h[:8], 16)


def perturb_pose(T, rot_deg, trans_mag, rng):
    T = np.asarray(T, dtype=np.float64)
    u = random_unit_vector(rng)
    v = random_unit_vector(rng)
    dR = so3_exp(np.radians(rot_deg) * u)
    out = T.copy()
    out[:3, :3] = T[:3, :3] @ dR
    out[:3, 3] = T[:3, 3] + trans_mag * v
    return out, {"rot_axis_cam": u.tolist(), "trans_dir_world": v.tolist()}


def select_views(n_views, spec, rng):
    eligible = list(range(1 if spec.get("exclude_ref", True) else 0, n_views))
    mode = spec["mode"]
    if mode == "all":
        return eligible
    if mode == "subset":
        if "n" in spec:
            k = int(spec["n"])
        else:
            k = max(1, int(round(spec["frac"] * len(eligible))))
        k = min(k, len(eligible))
        return sorted(rng.choice(eligible, size=k, replace=False).tolist())
    raise ValueError(f"unknown corruption mode {mode!r}")


def corrupt_poses(clean_poses, spec, base_seed, clip_id, cond_name):
    """Return (corrupted_poses, manifest_entry). clean_poses is never modified."""
    clean = np.asarray(clean_poses, dtype=np.float64)
    for i, T in enumerate(clean):
        validate_pose(T, f"{clip_id} clean view {i}")
    seed = _seed_for(base_seed, clip_id, cond_name)
    rng = np.random.default_rng(seed)
    L = trajectory_scale(clean)
    rot_deg = float(spec.get("rot_deg", 0.0))
    trans_frac = float(spec.get("trans_frac", 0.0))
    views = select_views(len(clean), spec, rng)

    out = clean.copy()
    per_view = {}
    for i in views:
        out[i], info = perturb_pose(clean[i], rot_deg, trans_frac * L, rng)
        validate_pose(out[i], f"{clip_id} corrupted view {i}")
        per_view[int(i)] = info

    entry = {
        "clip_id": clip_id,
        "condition": cond_name,
        "base_seed": base_seed,
        "rng_seed": seed,
        "spec": spec,
        "scene_length_L_m": L,
        "rot_deg": rot_deg,
        "trans_m": trans_frac * L,
        "affected_views": [int(i) for i in views],
        "actual_fraction": len(views) / len(clean),
        "per_view": per_view,
    }
    return out, entry


def write_manifest(entries, path):
    with open(path, "w") as f:
        json.dump(entries, f, indent=1)
