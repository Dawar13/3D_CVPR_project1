"""E3: check pose hints against the model's own photos-only prediction (the "witness"). NO ground truth.

Hints and the photos-only prediction are both camera-to-world poses relative to view 0 (OpenCV axes). Error model
for the hints H_i given the photos-only poses P_i (i = 1..N-1):

    R(H_i) = L_i R(P_i) S E_i,   L_i = M exp(a + i w)   (L = M exp(a): one rotation shared by all views in the
                                                         frame of view 0; w: drift per view; S: one rotation shared
                                                         by all views in their own camera frame)
    t(H_i) = s L_i t(P_i) + e_i                          (s: one scale; e_i: per-view position error)

L is what a wrong first camera or a world-frame heading bias looks like after making poses relative to view 0, w is
accumulated drift, S is a camera-mount (camera vs IMU) rotation error on the non-reference views (E1's shared_rot),
E_i and e_i are single bad cameras. L and S are fitted as two competing explanations ('world' with L, 'mount' with
S; both with drift) and the one that better explains rotations AND movement directions is kept (see fit). The fits
are robust (Huber weights), so a few bad cameras do not move the shared parts.

Outputs per clip: |L| (shared_deg), |S| (mount_deg), drift (deg per view and total), scale s, per-view residual
rotation |E_i| (deg) and position error |e_i| relative to the clip size. Decisions are in exp1/e3.py.
"""

import numpy as np
from scipy.spatial.transform import Rotation as Rot


def _rot(P):
    return Rot.from_matrix(np.asarray(P)[..., :3, :3])


def robust_mean_rotation(R, iters=10, c=2.0):
    """Weighted chordal mean with Huber-like weights in degrees (c = transition, deg)."""
    w = np.ones(len(R))
    m = R.mean(weights=w)
    for _ in range(iters):
        r = np.degrees((m.inv() * R).magnitude())
        w = np.where(r <= c, 1.0, c / np.maximum(r, 1e-9))
        m = R.mean(weights=w)
    return m


def _left_fit(left, drift, iters, c, intercept=True):
    """Robust line fit log(M^-1 left_i) ~ a + i w in the tangent space at the robust mean M (intercept=False: M = I,
    a = 0, drift only). Returns L = M exp(a), w, L_i = M exp(a + i w)."""
    n = len(left)
    idx = np.arange(1, n + 1, dtype=float)
    M = robust_mean_rotation(left, c=c) if intercept else Rot.identity()
    x = (M.inv() * left).as_rotvec()
    cols = ([np.ones(n)] if intercept else []) + ([idx] if drift and n >= 3 else [])
    if not cols:
        return M, np.zeros(3), Rot.concatenate([M] * n)
    A = np.stack(cols, 1)
    wt = np.ones(n)
    for _ in range(iters):
        sw = np.sqrt(wt)[:, None]
        coef = np.linalg.lstsq(A * sw, x * sw, rcond=None)[0]
        res = np.linalg.norm(x - A @ coef, axis=1)
        wt = np.where(np.degrees(res) <= c, 1.0, np.radians(c) / np.maximum(res, 1e-12))
    a = coef[0] if intercept else np.zeros(3)
    w = coef[-1] if drift and n >= 3 else np.zeros(3)
    return M * Rot.from_rotvec(a), w, M * Rot.from_rotvec(a + idx[:, None] * w)


def _huber(r, c):
    r = np.abs(r)
    return float(np.sum(np.where(r <= c, 0.5 * r ** 2, c * (r - 0.5 * c))))


def _assemble(H, P, tH, tP, L, w, Li, S, c):
    E = S.inv() * P.inv() * Li.inv() * H                 # per-view residual in the camera frame
    rot_res = np.degrees(E.magnitude())
    tPL = Li.apply(tP)
    nP, nH = np.linalg.norm(tPL, axis=1), np.linalg.norm(tH, axis=1)
    ok = (nP > 1e-6) & (nH > 1e-6)
    cos = np.sum(tH[ok] * tPL[ok], 1) / (nH[ok] * nP[ok])
    dir_res = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    cost = _huber(rot_res, c) + _huber(dir_res, c)
    s = float(np.median(nH[ok] / nP[ok])) if ok.any() else np.nan
    extent = max(nH.max(), 1e-9)
    pos_res = np.linalg.norm(tH - s * tPL, axis=1) / extent
    n = len(H)
    return {"shared_deg": float(np.degrees(L.magnitude())), "mount_deg": float(np.degrees(S.magnitude())),
            "drift_deg_per_view": float(np.degrees(np.linalg.norm(w))),
            "drift_total_deg": float(np.degrees(np.linalg.norm(w)) * n), "scale": s,
            "rot_res_deg": np.r_[0.0, rot_res], "pos_res_rel": np.r_[0.0, pos_res], "cost": cost,
            "L": L, "w": w, "Li": Li, "S": S}


def fit(hint_rel, pred_rel, drift=True, mount=True, iters=10, rounds=3, c=2.0):
    """hint_rel, pred_rel: (N, 4, 4) cam2world relative to view 0. Returns a dict (see module docstring).
    Two explanations of a shared rotation are fitted and the one with the lower robust cost on rotation residuals
    AND movement directions is kept: 'world' (L in the frame of view 0, positions turn with it, plus drift) or
    'mount' (S in the camera frame, positions do not turn, plus drift). Rotations alone cannot tell them apart when
    the views barely turn relative to each other; the camera positions can. drift=False fixes w = 0; mount=False
    keeps only the 'world' explanation."""
    H, P = _rot(hint_rel[1:]), _rot(pred_rel[1:])
    tH, tP = np.asarray(hint_rel)[1:, :3, 3], np.asarray(pred_rel)[1:, :3, 3]
    I = Rot.identity()
    L, w, Li = _left_fit(H * P.inv(), drift, iters, c)
    world = _assemble(H, P, tH, tP, L, w, Li, I, c)
    world["explanation"] = "world"
    if not mount:
        return world
    S = robust_mean_rotation(P.inv() * H, c=c)
    for _ in range(rounds):                              # mount S, then drift on what is left, alternately
        L0, w, Li = _left_fit((H * S.inv()) * P.inv(), drift, iters, c, intercept=False)
        S = robust_mean_rotation(P.inv() * Li.inv() * H, c=c)
    mnt = _assemble(H, P, tH, tP, I, w, Li, S, c)
    mnt["explanation"] = "mount"
    return mnt if mnt["cost"] < world["cost"] else world
