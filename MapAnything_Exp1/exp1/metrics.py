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
    pts_g, valid = gt_points_ref0(depth, K, gt_poses)

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
    pts_g, valid = gt_points_ref0(depth, K, clip["poses"])
    out = {k: np.full(N, np.nan) for k in ("view_pts_err_scale", "view_pts_err_metric",
                                           "view_depth_absrel_scale", "view_depth_absrel_metric",
                                           "pred_depth_scale", "conf_mean", "conf_median")}
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
    out["pred_poses_rel"] = np.stack([T0_inv @ T for T in pred["pose"]])
    out["pred_focal"] = 0.5 * (pred["intrinsics"][:, 0, 0] + pred["intrinsics"][:, 1, 1])
    return out
