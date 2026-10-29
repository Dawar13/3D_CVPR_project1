"""Evaluation metrics. All are computed identically for every condition.

Frames: predictions and GT are both re-expressed in the camera frame of view 0
(prediction: via the predicted pose of view 0; GT: poses are already relative
to view 0). No rotation/translation alignment is ever applied, so pose errors
remain visible in the point-map metrics.

Two scale protocols are reported for each geometric metric:
  *_metric : raw prediction, metres (MapAnything predicts metric scale)
  *_scale  : ONE scale factor per clip (median GT/pred depth ratio over all
             valid pixels of all views), so a global scale error is removed
             but per-view inconsistencies are not.

Direction: every metric is lower-is-better except delta1 and RRA/RTA (higher).
"""

import numpy as np

from .geometry import invert_pose, rotation_error_deg

LOWER_IS_BETTER = {
    "depth_absrel_metric": True, "depth_absrel_scale": True,
    "depth_delta1_metric": False, "depth_delta1_scale": False,
    "pts_relerr_metric": True, "pts_relerr_scale": True,
    "rre_deg": True, "rte_deg": True, "rra5": False, "rta5": False,
    "centre_err_rel": True,
}


def gt_points_ref0(depth, K, poses):
    """Back-project GT depth into the view-0 frame. Returns (N,H,W,3), valid (N,H,W)."""
    N, H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    pts = np.zeros((N, H, W, 3))
    for i in range(N):
        z = depth[i]
        x = (u - K[i, 0, 2]) / K[i, 0, 0] * z
        y = (v - K[i, 1, 2]) / K[i, 1, 1] * z
        pc = np.stack([x, y, z], -1)
        pts[i] = pc @ poses[i][:3, :3].T + poses[i][:3, 3]
    return pts, depth > 0


def clip_gt_points(clip):
    """gt_points_ref0 of a clip, computed once and kept in the clip dict (same numbers, less CPU)."""
    if "_gt_pts" not in clip:
        clip["_gt_pts"] = gt_points_ref0(clip["depth"], clip["K"], clip["poses"])
    return clip["_gt_pts"]


def umeyama(src, dst):
    """Least-squares similarity (s, R, t) with dst ~ s R src + t (Umeyama 1991). src, dst (M,3)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / (a ** 2).sum(1).mean())
    return s, R, mu_d - s * R @ mu_s


def sim3_shape_err(pred_pts, gt_pts, gt_centre):
    """A view's own shape: mean point error after fitting that view alone to the truth with one
    rotation, shift and scale, relative to the mean true distance from its camera."""
    s, R, t = umeyama(pred_pts, gt_pts)
    res = np.linalg.norm(s * pred_pts @ R.T + t - gt_pts, axis=-1).mean()
    return float(res / np.linalg.norm(gt_pts - gt_centre, axis=-1).mean())


def _project(pts_world, T_c2w, K):
    T = invert_pose(T_c2w)
    pc = pts_world @ T[:3, :3].T + T[:3, 3]
    z = pc[:, 2]
    uv = pc[:, :2] / np.where(np.abs(z) > 1e-9, z, 1e-9)[:, None] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    return uv, z


def reproj_shift_px(i, depth, K, poses, hint_K=None, hint_poses=None, hint_depth=None, max_px=5000):
    """Size of view i's hint error in pixels: mean image shift of its true 3D points when the hint
    replaces the truth. Pose / intrinsics hints: re-projected into view i itself. Depth hint:
    the hinted depth back-projected (true pose and K) and projected into every other view."""
    H, W = depth.shape[1:]
    v, u = np.nonzero(depth[i] > 0)
    if len(u) == 0:
        return np.nan
    sel = np.linspace(0, len(u) - 1, min(max_px, len(u))).astype(int)
    u, v = u[sel], v[sel]

    def backproject(z, Ki, Ti):
        pc = np.stack([(u - Ki[0, 2]) / Ki[0, 0] * z, (v - Ki[1, 2]) / Ki[1, 1] * z, z], -1)
        return pc @ Ti[:3, :3].T + Ti[:3, 3]

    X = backproject(depth[i][v, u], K[i], poses[i])
    if hint_depth is not None:
        Xh = backproject(hint_depth[i][v, u], K[i], poses[i])
        shifts = []
        for j in range(len(depth)):
            if j == i:
                continue
            a, za = _project(X, poses[j], K[j])
            b, zb = _project(Xh, poses[j], K[j])
            ok = (za > 1e-6) & (zb > 1e-6) & (a[:, 0] >= 0) & (a[:, 0] < W) & (a[:, 1] >= 0) & (a[:, 1] < H)
            if ok.any():
                shifts.append(np.linalg.norm(a[ok] - b[ok], axis=-1).mean())
        return float(np.mean(shifts)) if shifts else np.nan
    Th = poses[i] if hint_poses is None else hint_poses[i]
    Kh = K[i] if hint_K is None else hint_K[i]
    uv, z = _project(X, Th, Kh)
    ok = z > 1e-6
    if not ok.any():
        return np.nan
    return float(np.linalg.norm(uv[ok] - np.stack([u, v], -1)[ok], axis=-1).mean())


def _absrel_delta(pred, gt):
    r = np.abs(pred - gt) / gt
    ratio = np.maximum(pred / gt, gt / np.maximum(pred, 1e-8))
    return float(r.mean()), float((ratio < 1.25).mean())


def pose_metrics(pred_poses, gt_poses, views=None):
    """Pairwise relative-pose errors over all pairs (i<j) touching `views`."""
    N = len(gt_poses)
    rre, rte = [], []
    for i in range(N):
        for j in range(i + 1, N):
            if views is not None and i not in views and j not in views:
                continue
            Rp = invert_pose(pred_poses[i]) @ pred_poses[j]
            Rg = invert_pose(gt_poses[i]) @ gt_poses[j]
            rre.append(rotation_error_deg(Rp[:3, :3], Rg[:3, :3]))
            tp, tg = Rp[:3, 3], Rg[:3, 3]
            c = tp @ tg / (np.linalg.norm(tp) * np.linalg.norm(tg) + 1e-12)
            rte.append(float(np.degrees(np.arccos(np.clip(c, -1, 1)))))
    rre, rte = np.array(rre), np.array(rte)
    return {"rre_deg": float(rre.mean()), "rte_deg": float(rte.mean()),
            "rra5": float((rre < 5).mean()), "rta5": float((rte < 5).mean())}


def evaluate(pred, clip, views=None):
    """pred: dict from model_io.run. views: optional list restricting pixel metrics."""
    depth, K, gt_poses = clip["depth"], clip["K"], clip["poses"]
    N = len(depth)
    if pred["depth_z"].shape != depth.shape:
        raise ValueError(f"prediction {pred['depth_z'].shape} vs GT {depth.shape}: "
                         "model resized inputs; fix the clip resolution")
    views = list(range(N)) if views is None else list(views)

    # predicted points in predicted view-0 frame
    T0_inv = invert_pose(pred["pose"][0])
    pts_p = pred["pts3d"] @ T0_inv[:3, :3].T + T0_inv[:3, 3]
    pred_poses = np.stack([T0_inv @ T for T in pred["pose"]])
    pts_g, valid = clip_gt_points(clip)

    # one global scale per clip, computed on ALL views so it does not change with `views`
    vall = valid & (pred["depth_z"] > 1e-6)
    s = float(np.median(depth[vall] / pred["depth_z"][vall]))

    m = np.zeros_like(vall)
    m[views] = vall[views]
    dp, dg = pred["depth_z"][m], depth[m]
    out = {"n_valid_px": int(m.sum()), "scale_factor": s}
    out["depth_absrel_metric"], out["depth_delta1_metric"] = _absrel_delta(dp, dg)
    out["depth_absrel_scale"], out["depth_delta1_scale"] = _absrel_delta(s * dp, dg)

    pp, pg = pts_p[m], pts_g[m]
    gnorm = np.linalg.norm(pg, axis=-1).mean()
    out["pts_relerr_metric"] = float(np.linalg.norm(pp - pg, axis=-1).mean() / gnorm)
    out["pts_relerr_scale"] = float(np.linalg.norm(s * pp - pg, axis=-1).mean() / gnorm)

    out.update(pose_metrics(pred_poses, gt_poses, None if len(views) == N else views))
    cg, cp = gt_poses[:, :3, 3], s * pred_poses[:, :3, 3]
    L = np.median(np.linalg.norm(cg - cg.mean(0), axis=1)) + 1e-12
    out["centre_err_rel"] = float(np.linalg.norm(cp[views] - cg[views], axis=1).mean() / L)

    out["mean_conf"] = float(pred["conf"][views].mean())
    # predicted / true size of the scene (1 = metric scale right); all views, like s
    out["metric_scale_ratio"] = float(np.median(pred["depth_z"][vall] / depth[vall]))
    return out


def per_view(pred, clip, s):
    """Per-view quantities for per_view.csv. s: the clip's scale factor from evaluate()."""
    depth, K = clip["depth"], clip["K"]
    N = len(depth)
    T0_inv = invert_pose(pred["pose"][0])
    pts_p = pred["pts3d"] @ T0_inv[:3, :3].T + T0_inv[:3, 3]
    pts_g, valid = clip_gt_points(clip)
    out = {k: np.full(N, np.nan) for k in ("view_pts_err_scale", "view_pts_err_metric",
                                           "view_depth_absrel_scale", "view_depth_absrel_metric",
                                           "pred_depth_scale", "conf_mean", "conf_median",
                                           "view_shape_err_sim3")}
    for i in range(N):
        m = valid[i] & (pred["depth_z"][i] > 1e-6)
        out["conf_mean"][i] = float(pred["conf"][i].mean())
        out["conf_median"][i] = float(np.median(pred["conf"][i]))
        if not m.any():
            continue
        gnorm = np.linalg.norm(pts_g[i][m], axis=-1).mean()
        out["view_pts_err_metric"][i] = np.linalg.norm(pts_p[i][m] - pts_g[i][m], axis=-1).mean() / gnorm
        out["view_pts_err_scale"][i] = np.linalg.norm(s * pts_p[i][m] - pts_g[i][m], axis=-1).mean() / gnorm
        dp, dg = pred["depth_z"][i][m], depth[i][m]
        out["view_depth_absrel_metric"][i] = float(np.mean(np.abs(dp - dg) / dg))
        out["view_depth_absrel_scale"][i] = float(np.mean(np.abs(s * dp - dg) / dg))
        out["pred_depth_scale"][i] = float(np.median(dp / dg))
        if m.sum() >= 10:
            out["view_shape_err_sim3"][i] = sim3_shape_err(pts_p[i][m], pts_g[i][m], clip["poses"][i][:3, 3])
    out["pred_poses_rel"] = np.stack([T0_inv @ T for T in pred["pose"]])
    out["pred_focal"] = 0.5 * (pred["intrinsics"][:, 0, 0] + pred["intrinsics"][:, 1, 1])
    return out
