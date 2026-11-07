"""Geometry referee in degrees (E2X L2, planned baseline R5; experiments/E2X.md section 6b). DEV and CAL clips only
(the script refuses eval clips). CPU only, no 3D model.

geometry_referee.py scores hints in pixels (Sampson distance). Pixels are not comparable across scenes and do not say
how wrong a camera is. Here the photos give their own relative pose per pair of views, and the hints are compared
with it in degrees:
  - per pair (i, j): RANSAC-clean SIFT matches (exp1/photo_check.match_pairs, no pose used), normalised with the
    HINTED intrinsics; essential matrix + cheirality (cv2.findEssentialMat / recoverPose) -> photo relative rotation
    R_ij and translation direction t_ij. Pairs with < MIN_INLIERS inliers are not used.
  - for a hint set: rot_deg(i, j) = angle(R_ij_photo^T R_ij_hint); dir_deg(i, j) = angle between the photo and the
    hinted translation directions (pairs whose hinted baseline is < 1e-6 are skipped).
  - per view v: the median over its pairs (rot_view_deg, dir_view_deg); per hint set: the median over pairs
    (rot_set_deg, dir_set_deg).
  A single wrong camera shows in its own pairs only; an error shared by all cameras (one world rotation) leaves the
  relative rotations unchanged and shows in the translation directions; a camera-mount error shows in both.

Decision rule (Neyman-Pearson, the same false-action budget as E3, fixed here before looking at detections):
  - view drop: rot_view_deg > tau_view, tau_view = the smallest value of VIEW_GRID that drops <= 2% of views (i > 0)
    on ground-truth hints of the cal + dev clips;
  - group action (drop all pose hints): max(rot_set_deg, dir_set_deg) > tau_set, tau_set = the smallest value of
    SET_GRID that acts on <= 5% of clips with ground-truth hints;
  - if no grid value meets the budget, that part is NOT CALIBRATED (reported, never loosened).
  Revision history (cal/dev data only, 2026-11-07):
  - rev 1 (results/e2x_geometry_referee_deg_rev1/): every view / hint set with >= 1 posed pair scored. NOT
    CALIBRATED: on ground-truth hints of the SceneFun3D clips the median clip has 1 posed pair and the scores are
    wide (95th percentile 21.6 deg rotation, 48.4 deg direction), against 1.1 / 4.0 deg on ETH3D dev.
  - rev 2 (this default): abstain where the photos give too little evidence: a view needs >= 3 posed pairs
    (--min_view_pairs), a hint set >= 10 (--min_set_pairs). Abstaining is never an action. Coverage is reported.
  Reported, not used to choose: detection AUROCs and action rates per condition at the chosen cut-offs, and the
  action rate on the real iPad (ARKit) poses, which help the models (E2-E): an action there is a practical false
  alarm. Whether acting improves the 3D output needs model runs (session A2, control R5).

  python scripts/e2x/geometry_referee_deg.py --config configs/e2x_e3cal_synth_mapanything.json \
      --clips 'data/clips_e2/dev__*__c0.npz' 'data/clips_e2x_cal/cal__*.npz' 'data/clips_e2x_room/cal__*.npz' \
      --out results/e2x_geometry_referee_deg/synth
  (and --config configs/e2x_e3cal_real_mapanything.json with the SceneFun3D dev + cal clips, --out .../real)
  python scripts/e2x/geometry_referee_deg.py --calibrate results/e2x_geometry_referee_deg
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

MIN_INLIERS = 30
VIEW_GRID = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0]
SET_GRID = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0]
GT_CONDITIONS = {"clean_pose", "clean_pose_depth2", "gt_pose", "gt_all"}


def _angle(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def _dir_angle(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(a @ b / (na * nb), -1, 1))))


def photo_relposes(pairs, K):
    """{(i, j): (R, t)}: camera i -> camera j (x_j = R x_i + t, t unit) from the matches and the given intrinsics."""
    out = {}
    for (i, j), (xi, xj) in pairs.items():
        ni = cv2.undistortPoints(xi.reshape(-1, 1, 2).astype(np.float64), K[i], None).reshape(-1, 2)
        nj = cv2.undistortPoints(xj.reshape(-1, 1, 2).astype(np.float64), K[j], None).reshape(-1, 2)
        f = 0.5 * (K[i][0, 0] + K[j][0, 0])
        E, inl = cv2.findEssentialMat(ni, nj, np.eye(3), cv2.RANSAC, 0.999, photo_check.RANSAC_PX / f)
        if E is None or E.shape != (3, 3) or inl is None or int(inl.sum()) < MIN_INLIERS:
            continue
        n, R, t, _ = cv2.recoverPose(E, ni, nj, np.eye(3), mask=inl.copy())
        if n < MIN_INLIERS:
            continue
        out[(i, j)] = (R, t.ravel() / max(np.linalg.norm(t), 1e-12))
    return out


def score(rel, poses, n):
    rot, dirs = {}, {}
    for (i, j), (R, t) in rel.items():
        T = np.linalg.inv(poses[j]) @ poses[i]        # hinted camera i -> camera j
        rot[(i, j)] = _angle(R, T[:3, :3])
        if np.linalg.norm(poses[i][:3, 3] - poses[j][:3, 3]) > 1e-6:
            dirs[(i, j)] = _dir_angle(t, T[:3, 3])
    view_rot, view_dir, view_pairs = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, int)
    for v in range(n):
        r = [x for p, x in rot.items() if v in p]
        d = [x for p, x in dirs.items() if v in p and np.isfinite(x)]
        view_pairs[v] = len(r)
        if r:
            view_rot[v] = float(np.median(r))
        if d:
            view_dir[v] = float(np.median(d))
    set_rot = float(np.median(list(rot.values()))) if rot else np.nan
    dv = [x for x in dirs.values() if np.isfinite(x)]
    set_dir = float(np.median(dv)) if dv else np.nan
    return view_rot, view_dir, set_rot, set_dir, view_pairs


def run(a):
    paths = sorted({p for g in a.clips for p in glob.glob(g)})
    bad = [p for p in paths if os.path.basename(p).startswith("eval__")]
    if bad:
        raise SystemExit(f"eval clips are not used here: {bad[:3]}")
    cfg = json.load(open(a.config))
    conds = [c for c in expand_conditions(cfg) if "pose" in c["inputs"]]
    depth_views = tuple(cfg.get("depth_views", ()))
    os.makedirs(a.out, exist_ok=True)
    vrows, rrows = [], []
    for p in paths:
        clip = load_clip(p)
        pairs = photo_check.match_pairs(clip["images"])
        N = len(clip["images"])
        cache = {}
        for cond in conds:
            needs = cond.get("corruption", {}).get("use", [])
            if "mono_depth" in needs and "mono_depth" not in clip:
                continue
            if cond.get("corruption", {}).get("target") == "device" and "dev_poses" not in clip:
                continue
            for seed in seeds_for(cond, cfg):
                poses, K, depth, entry = apply_corruption(clip, cond, seed, depth_views)
                kk = K.tobytes()
                if kk not in cache:   # photo relative poses depend only on the matches and the hinted intrinsics
                    cache[kk] = photo_relposes(pairs, K)
                rel = cache[kk]
                vr, vd, sr, sd, vp = score(rel, poses, N)
                affected = set(entry["affected_views"]) if entry and "affected_views" in entry else set()
                key = {"clip_id": clip["clip_id"], "condition": cond["name"], "family": cond["family"], "seed": seed,
                       "gt_hint": int(cond["name"] in GT_CONDITIONS)}
                for v in range(N):
                    vrows.append({**key, "view": v, "rot_view_deg": vr[v], "dir_view_deg": vd[v],
                                  "n_view_pairs": int(vp[v]), "was_corrupted": int(v in affected)})
                rrows.append({**key, "n_pairs_matched": len(pairs), "n_pairs_posed": len(rel),
                              "rot_set_deg": sr, "dir_set_deg": sd})
        print(f"[ok] {clip['clip_id']}: {len(pairs)} matched pairs", flush=True)
    pd.DataFrame(vrows).to_csv(os.path.join(a.out, "views.csv"), index=False)
    pd.DataFrame(rrows).to_csv(os.path.join(a.out, "runs.csv"), index=False)


def _smallest(grid, rate, budget):
    for g in grid:
        if rate(g) <= budget:
            return g
    return None


def calibrate(root, min_view_pairs, min_set_pairs):
    V = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(root, "*", "views.csv"))], ignore_index=True)
    R = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(root, "*", "runs.csv"))], ignore_index=True)
    R["set_deg"] = R[["rot_set_deg", "dir_set_deg"]].max(axis=1)
    # abstain (no action, never counted as an action) where the photos give too little evidence (revision 2)
    V.loc[V.n_view_pairs < min_view_pairs, "rot_view_deg"] = np.nan
    R.loc[R.n_pairs_posed < min_set_pairs, "set_deg"] = np.nan
    gv = V[(V.gt_hint == 1) & (V.view > 0) & V.rot_view_deg.notna()].drop_duplicates(["clip_id", "condition", "seed", "view"])
    gr = R[(R.gt_hint == 1) & R.set_deg.notna()].drop_duplicates(["clip_id", "condition", "seed"])
    tv = _smallest(VIEW_GRID, lambda t: (gv.rot_view_deg > t).mean(), 0.02)
    ts = _smallest(SET_GRID, lambda t: (gr.set_deg > t).mean(), 0.05)
    out = {"min_view_pairs": min_view_pairs, "min_set_pairs": min_set_pairs, "tau_view_deg": tv, "tau_set_deg": ts,
           "view_status": "calibrated" if tv is not None else "NOT CALIBRATED",
           "set_status": "calibrated" if ts is not None else "NOT CALIBRATED",
           "gt_views": int(len(gv)), "gt_hint_sets": int(len(gr)),
           "gt_view_drop_rate": float((gv.rot_view_deg > tv).mean()) if tv is not None else None,
           "gt_set_action_rate": float((gr.set_deg > ts).mean()) if ts is not None else None,
           "coverage_views_scored": float(V.rot_view_deg.notna().mean()),
           "coverage_sets_scored": float(R.set_deg.notna().mean())}
    rows = []
    for (cond, fam), g in R.groupby(["condition", "family"]):
        row = {"condition": cond, "family": fam, "n_sets": len(g), "set_scored": float(g.set_deg.notna().mean()),
               "median_set_deg": float(g.set_deg.median())}
        if ts is not None:
            row["set_action_rate"] = float((g.set_deg > ts).mean())
        if cond not in GT_CONDITIONS:
            row["auroc_set_vs_gt"] = auroc(g.set_deg.dropna(), gr.set_deg)
        gvv = V[(V.condition == cond) & (V.view > 0)]
        bad, good = gvv[gvv.was_corrupted == 1], gvv[gvv.was_corrupted == 0]
        if len(bad) and len(good):
            vals = []
            for _, run in gvv.groupby(["clip_id", "seed"]):
                x = auroc(run[run.was_corrupted == 1].rot_view_deg, run[run.was_corrupted == 0].rot_view_deg)
                if np.isfinite(x):
                    vals.append(x)
            row["auroc_bad_vs_good_view"] = float(np.mean(vals)) if vals else np.nan
            if tv is not None:
                row["bad_view_drop_rate"] = float((bad.rot_view_deg > tv).mean())
                row["good_view_drop_rate"] = float((good.rot_view_deg > tv).mean())
            row["bad_view_scored"] = float(bad.rot_view_deg.notna().mean())
        rows.append(row)
    S = pd.DataFrame(rows).sort_values(["family", "condition"])
    S.to_csv(os.path.join(root, "summary.csv"), index=False)
    json.dump(out, open(os.path.join(root, "chosen.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
    print(S.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--clips", nargs="+")
    ap.add_argument("--out")
    ap.add_argument("--calibrate", help="folder with <part>/views.csv and runs.csv: choose the cut-offs")
    ap.add_argument("--min_view_pairs", type=int, default=3, help="revision 2: score a view only with >= this many pairs")
    ap.add_argument("--min_set_pairs", type=int, default=10, help="revision 2: score a hint set only with >= this many pairs")
    a = ap.parse_args()
    if a.calibrate:
        calibrate(a.calibrate, a.min_view_pairs, a.min_set_pairs)
    else:
        run(a)


if __name__ == "__main__":
    main()
