"""E3 method: check hints against the photos, repair what an explicit error model explains, drop the rest.
Training-free and ground-truth-free: it only calls the model (through the backend's model_io) and looks at hints.

Steps, for one clip and one hint set (poses H, intrinsics K, depth D):
  1. Witness: images-only passes with the views cyclically re-ordered (n_orders, a different reference view each),
     mapped back to view 0. Their mean is the photos' answer, their spread per view (and of the shared part) is how
     much the photos' answer itself moves: the witness uncertainty sigma. Computed once per clip (hint-free).
  2. Pose check (exp1/witness.fit) against the witness mean: a shared rotation L, a mount rotation S, a drift w and
     per-view residuals E_i.
     A part is acted on only if it exceeds z * sqrt(sigma^2 + floor^2) (z, floors from the config, calibrated on the
     'cal' split, never on eval):
       - shared (frame of view 0) / mount (camera frame) / drift: REPAIR, i.e. the fitted part is undone
         (exp1/witness.py for the error model);
       - a single view: DROP its pose hint.
     Pose scale is not checked: the photos' metric scale is too uncertain (E2-C8: 13% typical, 52% worst 10%).
  3. Depth check (only with depth hints): one pass with the repaired pose hints and K but no depth; its depth is
     put in the pose hints' units (pose_units: Pi3X and OmniVGGT normalise hint scale away). Per view
     r_i = median(D_hint / D_pass) over valid pixels; k = median_i r_i.
       - |log k| > tau_depth_shared: REPAIR, every depth hint is divided by k;
       - |log(r_i / k)| > tau_depth_view: DROP that view's depth hint.
     Without pose hints the images-only witness gives the scale (much less certain, so it rarely acts); a model
     without metric output (OmniVGGT) then skips the depth check.
  4. Final pass with the repaired hints. Decisions are returned for logging.

The 'drop' variant (control witness_drop) drops every pose hint when a group test (shared / mount / drift) fires and
every depth hint when the depth-scale test fires, instead of repairing: an ablation of the repair step.
"""

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from . import witness as W

DEFAULTS = dict(e3_n_orders=3, e3_z=3.0, e3_floor_shared_deg=0.5, e3_floor_view_deg=1.0, e3_floor_drift_deg=0.5,
                e3_tau_depth_shared=0.10, e3_tau_depth_view=0.15, e3_variant="repair")


def cfg_get(cfg, k):
    return cfg.get(k, DEFAULTS[k])


def _rel(P):
    T0 = np.linalg.inv(P[0])
    return np.stack([T0 @ T for T in P])


def witness_ensemble(run_images_only, N, n_orders):
    """run_images_only(order) -> pred dict for images in that order. Returns poses (n_orders, N, 4, 4) relative to
    view 0 in the ORIGINAL order, conf (n_orders, N) per-view mean confidence, and the first pass (order 0)."""
    poses, confs, first = [], [], None
    for k in range(n_orders):
        shift = (k * N) // n_orders
        order = np.roll(np.arange(N), -shift)
        p = run_images_only(order)
        if first is None:
            first = p
        inv = np.argsort(order)                       # position of original view i in `order`
        P = p["pose"][inv]
        poses.append(_rel(P))
        confs.append(p["conf"].reshape(N, -1).mean(1)[inv])
    return np.stack(poses), np.stack(confs), first


def witness_mean(poses):
    """Per-view chordal mean over the ensemble (rotations) and mean position; view 0 stays identity."""
    n_o, N = poses.shape[:2]
    out = np.tile(np.eye(4), (N, 1, 1))
    for i in range(1, N):
        out[i, :3, :3] = Rot.from_matrix(poses[:, i, :3, :3]).mean().as_matrix()
        out[i, :3, 3] = poses[:, i, :3, 3].mean(0)
    return out


def _spread_deg(R):
    return float(np.sqrt(np.mean(np.degrees((R.mean().inv() * R).magnitude()) ** 2)))


def witness_sigma(poses, mean):
    """Spread of the ensemble, each member fitted against the mean: per-view rotation spread (deg) after removing the
    member's shared parts, and the spread of the shared part L, the mount part S and the drift (total deg).
    With one member everything is 0 (the floors then rule)."""
    n_o, N = poses.shape[:2]
    if n_o < 2:
        return np.zeros(N), {"shared": 0.0, "mount": 0.0, "drift": 0.0}
    fits = [W.fit(p, mean) for p in poses]
    dr = np.array([f["w"] for f in fits]) * (N - 1)
    sig = {"shared": _spread_deg(Rot.concatenate([f["L"] for f in fits])),
           "mount": _spread_deg(Rot.concatenate([f["S"] for f in fits])),
           "drift": float(np.degrees(np.sqrt(np.mean(np.sum((dr - dr.mean(0)) ** 2, 1)))))}
    res = np.array([f["rot_res_deg"] for f in fits])
    return np.sqrt(np.mean(res ** 2, 0)), sig


def pose_decision(hint_rel, wit_mean, sig_view, sig, cfg):
    """Fit the error model and test each part against z * sqrt(sigma^2 + floor^2)."""
    z = cfg_get(cfg, "e3_z")
    f = W.fit(hint_rel, wit_mean)
    N = len(hint_rel)
    th = {"shared": z * np.hypot(sig["shared"], cfg_get(cfg, "e3_floor_shared_deg")),
          "mount": z * np.hypot(sig["mount"], cfg_get(cfg, "e3_floor_shared_deg")),
          "drift": z * np.hypot(sig["drift"], cfg_get(cfg, "e3_floor_drift_deg"))}
    th_v = z * np.hypot(sig_view, cfg_get(cfg, "e3_floor_view_deg"))
    drop = np.zeros(N, bool)
    drop[1:] = f["rot_res_deg"][1:] > th_v[1:]
    return dict(fit=f, fire_shared=bool(f["shared_deg"] > th["shared"]), fire_mount=bool(f["mount_deg"] > th["mount"]),
                fire_drift=bool(f["drift_total_deg"] > th["drift"]), drop_views=drop, th=th, th_view=th_v)


def group_fired(dec):
    return dec["fire_shared"] or dec["fire_mount"] or dec["fire_drift"]


def repair_poses(hint_rel, dec):
    """Undo the fitted parts that fired. Left (frame of view 0), with L_i = M exp(a + i w) the full left error of view
    i and L = M exp(a) its shared part: shared and drift -> L_i^-1; shared only -> L^-1; drift only -> L L_i^-1.
    Positions turn with the left correction. Mount: rotations times S^-1 on the right (positions unchanged)."""
    f = dec["fit"]
    out = hint_rel.copy()
    for k in range(1, len(hint_rel)):
        Li = f["Li"][k - 1]
        if dec["fire_shared"] and dec["fire_drift"]:
            corr = Li.inv()
        elif dec["fire_shared"]:
            corr = f["L"].inv()
        elif dec["fire_drift"]:
            corr = f["L"] * Li.inv()
        else:
            corr = Rot.identity()
        R = corr.as_matrix() @ hint_rel[k, :3, :3]
        if dec["fire_mount"]:
            R = R @ f["S"].inv().as_matrix()
        out[k, :3, :3] = R
        out[k, :3, 3] = corr.apply(hint_rel[k, :3, 3])
    return out


def pose_units(hint_poses, pred_poses, mask):
    """Factor from the model's output units to the pose hints' units: median over hinted views i > 0 of
    |t_hint,i| / |t_pred,i| (both relative to view 0). MapAnything with metric pose hints gives ~1; Pi3X and
    OmniVGGT normalise hint scale away, so their depth must be converted before it is compared with depth hints."""
    th, tp = _rel(hint_poses)[1:, :3, 3], _rel(pred_poses)[1:, :3, 3]
    nh, n_p = np.linalg.norm(th, axis=1), np.linalg.norm(tp, axis=1)
    ok = np.asarray(mask)[1:] & (n_p > 1e-6) & (nh > 1e-6)
    return float(np.median(nh[ok] / n_p[ok])) if ok.any() else float("nan")


def no_depth_decision(N):
    """No scale to compare with (no pose hints and a model without metric output): the depth check does nothing."""
    return dict(k=1.0, ratios=np.full(N, np.nan), fire_scale=False, drop_views=np.zeros(N, bool))


def depth_decision(depth_hint, depth_pass, views, cfg):
    """views: indices with a depth hint. Returns k, per-view ratios, drop mask (over all N) and whether k fired."""
    N = len(depth_hint)
    r = np.full(N, np.nan)
    for i in views:
        m = (depth_hint[i] > 0) & (depth_pass[i] > 1e-6) & np.isfinite(depth_pass[i])
        if m.sum() > 100:
            r[i] = float(np.median(depth_hint[i][m] / depth_pass[i][m]))
    ok = np.isfinite(r)
    k = float(np.median(r[ok])) if ok.any() else 1.0
    fire = bool(abs(np.log(k)) > cfg_get(cfg, "e3_tau_depth_shared"))
    drop = np.zeros(N, bool)
    drop[ok] = np.abs(np.log(r[ok] / k)) > cfg_get(cfg, "e3_tau_depth_view")
    return dict(k=k, ratios=r, fire_scale=fire, drop_views=drop)
