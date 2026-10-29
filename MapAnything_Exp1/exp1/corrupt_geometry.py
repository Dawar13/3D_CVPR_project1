"""Deterministic, logged corruption of camera poses.

Perturbation model (applied to cam2world T = [R | c]):
  rotation:    R' = R @ Exp(theta * u),  u ~ uniform on S^2, theta FIXED (deg)
               -> the camera turns about its own optical centre by exactly theta
                  degrees, axis expressed in the camera frame. c is unchanged.
  translation: c' = c + (frac * L) * v,  v ~ uniform on S^2, magnitude FIXED
               L = trajectory_scale(clean poses of the clip) (metres)

Using a fixed magnitude with a random direction (instead of Gaussian noise)
makes every severity level an exact, preregistered error size.

Types (spec["type"], default "perturb"):
  "perturb" -> the model above, on the views chosen by `mode`:
               "all"    every eligible view (dense noise)
               "subset" exactly n views (or round(frac * N_eligible), min 1)
               spec["views"] = [..] fixes the views instead (e.g. [0] for an anchor error)
  "gauge"   -> one random rigid transform G applied to ALL poses, view 0 included:
               T' = G @ T. Relative poses are unchanged, so this should be harmless.
  "scale"   -> every translation times `factor` (all agree with each other).
  "drift"   -> re-integrate the relative motions D_i = T_{i-1}^-1 T_i with a per-step
               heading bias (rot_deg about the camera's own y axis, tilted at random by
               up to ~17 deg, random sign) and a compounding translation scale
               (1 + scale_growth)^i on step i. Odometry-style drift.
  "shared_rot" -> the same R' = R @ Exp(theta * u) (same u) on every non-reference view.
  E2 (one channel at a time; rotations unchanged unless stated):
  "radial"  -> distance only: c' = c + (frac * L) * c/|c| on the chosen views (direction from
               view 0 kept, distance from view 0 grows by frac * L).
  "orbit"   -> direction only: c turned about view 0 so that |c'| = |c| and |c' - c| = frac * L;
               only views with |c| >= frac * L / 2 are eligible.
  "drift_jitter" -> the per-view rotation and translation error SIZES of the "drift" run with the
               same rot_deg / scale_growth whose condition name is spec["match"] (same seed, so the
               same drift), applied with independent random axes / directions.
  spec["eligible"] = [..] restricts which views "subset"/"all" may pick (partial poses, E2-C7).

The reference view (index 0) is excluded by default, because poses are fed
relative to view 0 and the model may normalise by the first view; corrupting
it would corrupt every relative pose at once. Set exclude_ref=False or
views=[0] to test that case explicitly.
"""

import hashlib
import json

import numpy as np

from .geometry import (invert_pose, random_unit_vector, so3_exp, trajectory_scale,
                       validate_pose)


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
    if "views" in spec:
        return sorted(int(i) for i in spec["views"])
    eligible = list(range(1 if spec.get("exclude_ref", True) else 0, n_views))
    if "eligible" in spec:
        eligible = [i for i in eligible if i in set(int(j) for j in spec["eligible"])]
    if not eligible:
        raise ValueError("no eligible view for this corruption")
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


def drift_poses(clean, rot_deg, scale_growth, rng):
    axis = np.array([0.0, 1.0, 0.0]) + 0.3 * rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    sign = float(rng.choice([-1.0, 1.0]))
    out = clean.copy()
    if rot_deg == 0 and scale_growth == 0:
        return out, {"axis_cam": axis.tolist(), "sign": sign}
    dR = so3_exp(sign * np.radians(rot_deg) * axis)
    for i in range(1, len(clean)):
        D = invert_pose(clean[i - 1]) @ clean[i]
        D[:3, :3] = dR @ D[:3, :3]
        D[:3, 3] *= (1.0 + scale_growth) ** i
        out[i] = out[i - 1] @ D
    return out, {"axis_cam": axis.tolist(), "sign": sign}


def corrupt_poses(clean_poses, spec, base_seed, clip_id, cond_name):
    """Return (corrupted_poses, manifest_entry). clean_poses is never modified."""
    clean = np.asarray(clean_poses, dtype=np.float64)
    for i, T in enumerate(clean):
        validate_pose(T, f"{clip_id} clean view {i}")
    # seed_as: draw the same random corruption as another condition (paired variants, e.g. E2-C4 median)
    seed = _seed_for(base_seed, clip_id, spec.get("seed_as", cond_name))
    rng = np.random.default_rng(seed)
    L = trajectory_scale(clean)
    kind = spec.get("type", "perturb")
    rot_deg = float(spec.get("rot_deg", 0.0))
    trans_frac = float(spec.get("trans_frac", 0.0))
    out = clean.copy()
    per_view, params = {}, {}

    if kind == "perturb":
        views = select_views(len(clean), spec, rng)
        for i in views:
            out[i], info = perturb_pose(clean[i], rot_deg, trans_frac * L, rng)
            per_view[int(i)] = info
    elif kind == "gauge":
        views = list(range(len(clean)))
        G, info = perturb_pose(np.eye(4), rot_deg, trans_frac * L, rng)
        out = np.stack([G @ T for T in clean])
        params = {"G": G.tolist(), **info}
    elif kind == "scale":
        views = list(range(1, len(clean)))
        out[:, :3, 3] = clean[:, :3, 3] * float(spec["factor"])
        params = {"factor": float(spec["factor"])}
    elif kind == "drift":
        views = list(range(1, len(clean)))
        g = float(spec.get("scale_growth", 0.0))
        out, info = drift_poses(clean, rot_deg, g, rng)
        params = {"scale_growth": g, **info}
    elif kind == "shared_rot":
        views = list(range(1, len(clean)))
        u = random_unit_vector(rng)
        dR = so3_exp(np.radians(rot_deg) * u)
        for i in views:
            out[i][:3, :3] = clean[i][:3, :3] @ dR
        params = {"rot_axis_cam": u.tolist()}
    elif kind in ("radial", "orbit"):
        c0 = clean[0][:3, 3]
        dist = np.linalg.norm(clean[:, :3, 3] - c0, axis=1)
        d = trans_frac * L
        ok = [i for i in range(1, len(clean)) if dist[i] > 1e-9 and (kind == "radial" or dist[i] >= d / 2)]
        views = select_views(len(clean), dict(spec, eligible=[i for i in ok if i in spec.get("eligible", ok)]), rng)
        for i in views:
            r = clean[i][:3, 3] - c0
            if kind == "radial":
                new = r * (1 + d / dist[i])
            else:  # turn r about a random axis perpendicular to it by phi, chord = d
                a = random_unit_vector(rng)
                a = a - (a @ r) / (r @ r) * r
                a /= np.linalg.norm(a)
                phi = 2 * np.arcsin(min(1.0, d / (2 * dist[i])))
                new = so3_exp(phi * a) @ r
                per_view[int(i)] = {"axis_world": a.tolist(), "phi_deg": float(np.degrees(phi))}
            out[i][:3, 3] = c0 + new
    elif kind == "drift_jitter":
        drift_spec = {"type": "drift", "rot_deg": rot_deg, "scale_growth": float(spec.get("scale_growth", 0.0))}
        ref, _ = corrupt_poses(clean, drift_spec, base_seed, clip_id, spec["match"])
        views = list(range(1, len(clean)))
        for i in views:
            th = np.degrees(np.arccos(np.clip((np.trace(clean[i][:3, :3].T @ ref[i][:3, :3]) - 1) / 2, -1, 1)))
            t = float(np.linalg.norm(ref[i][:3, 3] - clean[i][:3, 3]))
            out[i], info = perturb_pose(clean[i], th, t, rng)
            per_view[int(i)] = {**info, "rot_deg": float(th), "trans_m": t}
        params = {"match": spec["match"]}
    else:
        raise ValueError(f"unknown pose corruption type {kind!r}")

    for i, T in enumerate(out):
        validate_pose(T, f"{clip_id} corrupted view {i}")
    entry = {
        "clip_id": clip_id,
        "condition": cond_name,
        "target": "pose",
        "type": kind,
        "base_seed": base_seed,
        "rng_seed": seed,
        "spec": spec,
        "scene_length_L_m": L,
        "rot_deg": rot_deg,
        "trans_m": trans_frac * L,
        "affected_views": [int(i) for i in views],
        "actual_fraction": len(views) / len(clean),
        "params": params,
        "per_view": per_view,
    }
    return out, entry


def write_manifest(entries, path):
    with open(path, "w") as f:
        json.dump(entries, f, indent=1)
