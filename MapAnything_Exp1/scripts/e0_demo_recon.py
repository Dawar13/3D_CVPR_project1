"""E0 visual check: demo reconstructions of one dev clip, shown next to their numbers.

  python scripts/e0_demo_recon.py --clip data/clips/dev__delivery_area__c0.npz --out results/e0_demo

Runs rgb_only, clean_pose and outlier_rot_rot_deg=15 (seed 0, the same corruption the
runner uses) and writes, in the GT view-0 camera frame at raw metric scale (no
alignment, the same frame as exp1/metrics.py):
  <clip>__gt.ply, <clip>__<condition>.ply   GT-valid pixels only, every --stride-th pixel,
                                            image colours; open in MeshLab / CloudCompare
  <clip>__overview.png   bird's-eye and side views: GT (grey) vs prediction (image colours),
                         GT cameras (black) vs predicted (red), corrupted hint (orange)
  <clip>__metrics.json   exp1.metrics.evaluate for each condition
A sanity check for E0, not a result. Dev clips only.
"""

import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from exp1 import model_io  # noqa: E402
from exp1.datasets import load_clip  # noqa: E402
from exp1.geometry import invert_pose  # noqa: E402
from exp1.metrics import evaluate, gt_points_ref0  # noqa: E402
from exp1.run_exp1 import apply_corruption, expand_conditions  # noqa: E402

CONDITIONS = ["rgb_only", "clean_pose", "outlier_rot_rot_deg=15"]


def write_ply(path, xyz, rgb):
    v = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                  ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    v["x"], v["y"], v["z"] = xyz.T
    v["red"], v["green"], v["blue"] = rgb.T
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(v)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                 "end_header\n").encode())
        f.write(v.tobytes())


def pred_in_ref0(pred):
    """Predicted points and poses re-expressed in the predicted view-0 camera frame."""
    T0_inv = invert_pose(pred["pose"][0])
    pts = pred["pts3d"] @ T0_inv[:3, :3].T + T0_inv[:3, 3]
    return pts, np.stack([T0_inv @ T for T in pred["pose"]])


def draw(ax, gt_pts, pts, rgb, gt_poses, poses, hint_poses, affected, plane, title, lim):
    i, j, sj = plane  # horizontal axis, vertical axis, sign of the vertical axis
    ax.scatter(gt_pts[:, i], sj * gt_pts[:, j], s=0.3, c="#c8c8c8", linewidths=0, rasterized=True)
    if pts is not None:
        ax.scatter(pts[:, i], sj * pts[:, j], s=0.3, c=rgb / 255.0, linewidths=0, rasterized=True)
    L = 0.06 * max(lim[1] - lim[0], lim[3] - lim[2])
    sets = [(gt_poses, "k", None)]
    if poses is not None:
        sets.append((poses, "#d62728", None))
    if hint_poses is not None:
        sets.append((hint_poses, "#ff9f1c", affected))
    for P, col, only in sets:
        for k, T in enumerate(P):
            if only is not None and k not in only:
                continue
            c, f = T[:3, 3], T[:3, 2]
            ax.plot([c[i], c[i] + L * f[i]], [sj * c[j], sj * (c[j] + L * f[j])], color=col, lw=1.2)
            ax.plot(c[i], sj * c[j], "o", color=col, ms=2.5)
    ax.set_xlim(lim[0], lim[1])
    ax.set_ylim(lim[2], lim[3])
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=7)
    ax.tick_params(labelsize=6)


def overview(path, clip, gt_pts, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    planes = [((0, 2, 1), "bird's-eye: x right, z forward (m)"),
              ((2, 1, -1), "side: z forward, up = -y (m)")]
    rng = np.random.default_rng(0)
    sub = rng.permutation(len(gt_pts))[:30000]
    fig, axes = plt.subplots(2, 1 + len(results), figsize=(4.2 * (1 + len(results)), 8.4))
    for row, (plane, label) in enumerate(planes):
        i, j, sj = plane
        g = gt_pts[sub][:, [i, j]] * [1, sj]
        C = np.concatenate([r["poses"][:, :3, 3] for r in results.values()]
                           + [clip["poses"][:, :3, 3]])[:, [i, j]] * [1, sj]
        lo = np.minimum(np.percentile(g, 1, axis=0), C.min(0))  # always show every camera
        hi = np.maximum(np.percentile(g, 99, axis=0), C.max(0))
        pad = 0.1 * (hi - lo).max()
        lim = (lo[0] - pad, hi[0] + pad, lo[1] - pad, hi[1] + pad)
        draw(axes[row, 0], gt_pts[sub], None, None, clip["poses"], None, None, None, plane,
             f"GT ({label})", lim)
        for col, (name, r) in enumerate(results.items(), start=1):
            s = rng.permutation(len(r["pts"]))[:30000]
            m = r["metrics"]
            title = (f"{name}\npts_relerr metric {m['pts_relerr_metric']:.3f} / scale "
                     f"{m['pts_relerr_scale']:.3f} | RRE {m['rre_deg']:.2f} deg | s={m['scale_factor']:.3f}")
            draw(axes[row, col], gt_pts[sub], r["pts"][s], r["rgb"][s], clip["poses"], r["poses"],
                 r["hint_poses"], r["affected"], plane, title, lim)
    fig.suptitle(f"{clip['clip_id']}: grey = GT points, colour = prediction; black = GT cameras, "
                 "red = predicted, orange = corrupted hint. Axes fit to GT. E0 sanity check, "
                 "not a result.", fontsize=8)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/pilot.json")
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stride", type=int, default=2, help="keep every n-th pixel in the PLYs")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if os.path.basename(a.clip).startswith("eval__"):
        raise SystemExit("E0 visual checks use dev clips only")

    cfg = json.load(open(a.config))
    conds = {c["name"]: c for c in expand_conditions(cfg)}
    clip = load_clip(a.clip)
    cid = clip["clip_id"]
    os.makedirs(a.out, exist_ok=True)
    gt, valid = gt_points_ref0(clip["depth"], clip["K"], clip["poses"])
    keep = np.zeros_like(valid)
    keep[:, ::a.stride, ::a.stride] = True
    keep &= valid
    rgb = clip["images"][keep]
    write_ply(os.path.join(a.out, f"{cid}__gt.ply"), gt[keep], rgb)

    model = model_io.load_model(cfg["checkpoint"], a.device)
    depth_views = tuple(cfg.get("depth_views", ()))
    results = {}
    for name in a.conditions:
        cond = conds[name]
        # same damage as the runner (pose, lens or depth), same seed
        poses, K, depth, entry = apply_corruption(clip, cond, a.seed, depth_views)
        affected = entry["affected_views"] if entry and entry["target"] == "pose" else None
        pred = model_io.run(model, model_io.build_views(clip["images"], K, poses, cond["inputs"], None,
                                                        bool(cond.get("metric", True)), depth, depth_views),
                            a.device)
        pts, pposes = pred_in_ref0(pred)
        results[name] = {"pts": pts[keep], "rgb": rgb, "poses": pposes,
                         "hint_poses": poses if affected else None, "affected": affected,
                         "metrics": evaluate(pred, clip)}
        write_ply(os.path.join(a.out, f"{cid}__{re.sub(r'[^\w.-]', '_', name)}.ply"),
                  pts[keep], rgb)
        print(f"{name:28s} " + "  ".join(f"{k}={results[name]['metrics'][k]:.4f}" for k in
                                        ("pts_relerr_metric", "pts_relerr_scale", "rre_deg")))
    json.dump({n: r["metrics"] for n, r in results.items()},
              open(os.path.join(a.out, f"{cid}__metrics.json"), "w"), indent=1)
    overview(os.path.join(a.out, f"{cid}__overview.png"), clip, gt[keep], results)
    print(f"written to {a.out}")


if __name__ == "__main__":
    main()
