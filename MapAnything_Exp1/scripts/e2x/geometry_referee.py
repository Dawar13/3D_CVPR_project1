"""Geometry referee (planned baseline R5, experiments/E2X.md section 6b, L2): can plain multi-view geometry, with no 3D
model, tell which hints are wrong? DEV and CAL clips only (the script refuses eval clips). CPU only.

For every clip, RANSAC-cleaned SIFT matches between its own images are computed once (exp1/photo_check.match_pairs;
no pose is used). Then, for every hint set of a config (the same corruptions the runner applies,
run_exp1.apply_corruption):
  - pose check: per view, the median Sampson distance (px) of its matches under the HINTED poses and intrinsics
    (exp1/photo_check.score). A wrong camera hint shows as a large score.
  - depth check (hint sets with depth): matches triangulated with the hinted poses give a depth per match in view i;
    r_i = median(depth hint / triangulated depth) at the match pixels; k = median_i r_i. With metric pose hints, k is
    the depth hint's scale relative to the poses (1 = consistent).
Outputs <out>/views.csv (per view) and <out>/runs.csv (per hint set), then summary.csv: detection AUROCs
(corrupted vs good views within a run; and per hint set vs the clean hint sets of the same clips).

  python scripts/e2x/geometry_referee.py --config configs/e2x_e3cal_synth_mapanything.json \
      --clips 'data/clips_e2/dev__*__c0.npz' 'data/clips_e2x_cal/cal__*.npz' 'data/clips_e2x_room/cal__*.npz' \
      --out results/e2x_geometry_referee/synth
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from exp1 import photo_check  # noqa: E402
from exp1.datasets import load_clip  # noqa: E402
from exp1.e1_analysis import auroc  # noqa: E402
from exp1.run_exp1 import apply_corruption, expand_conditions, seeds_for  # noqa: E402

MIN_PARALLAX_DEG = 1.0


def depth_ratios(pairs, K, poses, depth, views):
    """Per view i in `views`: median(depth hint / triangulated depth) over its matches (NaN if too few)."""
    N = len(poses)
    acc = {i: [] for i in range(N)}
    W2C = [np.linalg.inv(P) for P in poses]
    for (i, j), (xi, xj) in pairs.items():
        Pi = K[i] @ W2C[i][:3]
        Pj = K[j] @ W2C[j][:3]
        Xh = cv2.triangulatePoints(Pi, Pj, xi.T.astype(np.float64), xj.T.astype(np.float64))
        X = (Xh[:3] / Xh[3:]).T
        ci, cj = poses[i][:3, 3], poses[j][:3, 3]
        a, b = X - ci, X - cj
        cos = np.sum(a * b, 1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)
        good = np.degrees(np.arccos(np.clip(cos, -1, 1))) > MIN_PARALLAX_DEG
        for v, x in ((i, xi), (j, xj)):
            if v not in views:
                continue
            z = (W2C[v][:3, :3] @ X.T + W2C[v][:3, 3:4])[2]
            u = np.clip(np.round(x[:, 0]).astype(int), 0, depth.shape[2] - 1)
            w = np.clip(np.round(x[:, 1]).astype(int), 0, depth.shape[1] - 1)
            dh = depth[v][w, u]
            ok = good & (z > 1e-6) & (dh > 0)
            acc[v].extend((dh[ok] / z[ok]).tolist())
    return np.array([float(np.median(acc[i])) if len(acc[i]) >= 20 else np.nan for i in range(N)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    paths = sorted({p for g in a.clips for p in glob.glob(g)})
    bad = [p for p in paths if os.path.basename(p).startswith("eval__")]
    if bad:
        raise SystemExit(f"eval clips are not used here: {bad[:3]}")
    cfg = json.load(open(a.config))
    conds = [c for c in expand_conditions(cfg) if c["inputs"]]
    depth_views = tuple(cfg.get("depth_views", ()))
    os.makedirs(a.out, exist_ok=True)
    vrows, rrows = [], []
    for p in paths:
        clip = load_clip(p)
        pairs = photo_check.match_pairs(clip["images"])
        N = len(clip["images"])
        for cond in conds:
            needs = cond.get("corruption", {}).get("use", [])
            if "mono_depth" in needs and "mono_depth" not in clip:
                continue
            if cond.get("corruption", {}).get("target") == "device" and "dev_poses" not in clip:
                continue
            for seed in seeds_for(cond, cfg):
                poses, K, depth, entry = apply_corruption(clip, cond, seed, depth_views)
                affected = set(entry["affected_views"]) if entry else set()
                key = {"clip_id": clip["clip_id"], "condition": cond["name"], "family": cond["family"], "seed": seed}
                if "pose" in cond["inputs"]:
                    _, view = photo_check.score(pairs, K, poses, N)
                    for v in range(N):   # view 0 included: a wrong first camera (anchor_error) shows there
                        vrows.append({**key, "view": v, "sampson_px": view[v], "was_corrupted": int(v in affected)})
                row = {**key, "n_pairs": len(pairs)}
                if "pose" in cond["inputs"]:
                    row["max_view_px"] = float(np.nanmax(view)) if np.isfinite(view).any() else np.nan
                    row["median_view_px"] = float(np.nanmedian(view)) if np.isfinite(view).any() else np.nan
                if "depth" in cond["inputs"] and "pose" in cond["inputs"]:
                    dv = list(depth_views) if not entry or entry.get("type") != "device" else list(range(N))
                    r = depth_ratios(pairs, K, poses, depth, set(dv))
                    row["depth_k"] = float(np.nanmedian(r)) if np.isfinite(r).any() else np.nan
                rrows.append(row)
        print(f"[ok] {clip['clip_id']}: {len(pairs)} matched pairs", flush=True)
    V, R = pd.DataFrame(vrows), pd.DataFrame(rrows)
    V.to_csv(os.path.join(a.out, "views.csv"), index=False)
    R.to_csv(os.path.join(a.out, "runs.csv"), index=False)
    summary(V, R).to_csv(os.path.join(a.out, "summary.csv"), index=False)
    print(summary(V, R).to_string(index=False))


def summary(V, R):
    rows = []
    clean = {"clean_pose", "clean_pose_depth2", "gt_pose", "gt_all"}
    for (fam, cond), g in V.groupby(["family", "condition"]):
        if cond in clean or g.was_corrupted.sum() == 0 or (g.was_corrupted == 0).sum() == 0:
            continue
        vals = []
        for _, run in g.groupby(["clip_id", "seed"]):
            x = auroc(run[run.was_corrupted == 1].sampson_px, run[run.was_corrupted == 0].sampson_px)
            if np.isfinite(x):
                vals.append(x)
        rows.append({"check": "pose: bad view vs good views (within run)", "condition": cond,
                     "auroc": float(np.mean(vals)) if vals else np.nan, "n_runs": len(vals),
                     "scored_bad_views": float(g[g.was_corrupted == 1].sampson_px.notna().mean())})
    ref = R[R.condition.isin(clean)]
    for cond, g in R[~R.condition.isin(clean)].groupby("condition"):
        if "median_view_px" in g and g.median_view_px.notna().any():
            for stat in ("median_view_px", "max_view_px"):
                rows.append({"check": f"pose: hint set vs clean hint sets ({stat})", "condition": cond,
                             "auroc": auroc(g[stat], ref[stat]), "n_runs": len(g)})
        if "depth_k" in g and g.depth_k.notna().any():
            base = ref.depth_k.dropna()
            rows.append({"check": "depth: |log k| vs clean hint sets", "condition": cond,
                         "auroc": auroc(np.abs(np.log(g.depth_k.dropna())), np.abs(np.log(base))) if len(base) else np.nan,
                         "median_k": float(g.depth_k.median()), "clean_median_k": float(base.median()) if len(base) else np.nan,
                         "n_runs": len(g)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
