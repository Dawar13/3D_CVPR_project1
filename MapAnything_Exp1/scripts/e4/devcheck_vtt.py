"""E4-A dev check (experiments/E4.md): what Verify-then-Trust does to the HINTS, scored against ground truth, with no
model run (CPU only). The conditions, seeds and corruptions are those of the runner (exp1/run_exp1.py:
expand_conditions, seeds_for, apply_corruption), so the hints are exactly the ones a GPU run would receive.
Dev / cal clips only; eval clips are refused.

  python scripts/e4/devcheck_vtt.py --config configs/e3_eval_synth_mapanything.json \
      --clips 'data/clips_e2/dev__*.npz' --kappa_json results/e4_vtt_cal/chosen.json --out <dir>

Per run (clip x condition x seed with pose or depth hints) it writes, before -> after VtT:
  rre_*      mean pairwise relative-rotation error of the hint rotations vs ground truth (deg; gauge free)
  rre_max_*  the worst pair
  view_max_* the worst single camera, absolute (deg; hints and ground truth share view 0's frame)
  dlog_*     mean |median log(depth hint / ground-truth depth)| over the depth views (with depth hints)
and VtT's own log (start, sigma_h, sigma_c, edges, smallest w). <out>/summary.csv: per condition means and the
share of runs made worse by more than 0.25 deg (rotation) or 0.05 (log depth).
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from exp1 import photo_check, vtt  # noqa: E402
from exp1.datasets import load_clip  # noqa: E402
from exp1.run_exp1 import apply_corruption, expand_conditions, seeds_for  # noqa: E402


def rre(Ra, Rb):
    N = len(Ra)
    e = [vtt.angle_deg(Ra[i].T @ Ra[j], Rb[i].T @ Rb[j]) for i in range(N) for j in range(i + 1, N)]
    return float(np.mean(e)), float(np.max(e))


def dlog(depth, gt, views):
    out = []
    for v in views:
        m = (depth[v] > 0) & (gt[v] > 0) & np.isfinite(depth[v]) & np.isfinite(gt[v])
        if m.sum() > 100:
            out.append(abs(float(np.median(np.log(depth[v][m] / gt[v][m])))))
    return float(np.mean(out)) if out else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kappa_json", default=None)
    ap.add_argument("--conditions", nargs="*", default=None, help="only these condition names")
    a = ap.parse_args()
    paths = sorted({p for g in a.clips for p in glob.glob(g)})
    if any(os.path.basename(p).startswith("eval__") for p in paths):
        raise SystemExit("dev / cal clips only")
    cfg = json.load(open(a.config))
    if a.kappa_json:
        k = json.load(open(a.kappa_json))
        cfg.update({x: k[x] for x in k if x.startswith("vtt_")})
    conds = [c for c in expand_conditions(cfg) if {"pose", "depth"} & set(c["inputs"])
             and (a.conditions is None or c["name"] in a.conditions)]
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for p in paths:
        clip = load_clip(p)
        t0 = time.time()
        pairs = photo_check.match_pairs(clip["images"])
        t_match = time.time() - t0
        edge_cache = {}
        Rg, gt_depth = clip["poses"][:, :3, :3], clip["depth"]
        for cond in conds:
            for seed in seeds_for(cond, cfg):
                try:
                    poses, K, depth, entry = apply_corruption(clip, cond, seed, cfg["depth_views"])
                except KeyError as e:          # e.g. AI depth not stored in this clip
                    print(f"[skip] {clip['clip_id']} {cond['name']}: missing {e}", flush=True)
                    continue
                key = np.asarray(K, np.float64).tobytes()
                t1 = time.time()
                if key not in edge_cache:
                    edge_cache[key] = vtt.pair_edges(pairs, K, cfg)
                dv = cfg["depth_views"] if "depth" in cond["inputs"] else []
                newP, newD, info = vtt.sanitize(pairs, K, poses, depth, dv, cond["inputs"], cfg, edges=edge_cache[key])
                t_vtt = time.time() - t1
                r = {"clip_id": clip["clip_id"], "condition": cond["name"], "family": cond.get("family"),
                     "seed": seed, "inputs": "+".join(cond["inputs"]), "t_match_s": t_match, "t_vtt_s": t_vtt}
                if "pose" in cond["inputs"]:
                    Rh, Rv = poses[:, :3, :3], newP[:, :3, :3]
                    r["rre_before"], r["rre_max_before"] = rre(Rh, Rg)
                    r["rre_after"], r["rre_max_after"] = rre(Rv, Rg)
                    r["view_max_before"] = max(vtt.angle_deg(Rh[i], Rg[i]) for i in range(len(Rg)))
                    r["view_max_after"] = max(vtt.angle_deg(Rv[i], Rg[i]) for i in range(len(Rg)))
                    r.update(start=info.get("start"), sigma_h_deg=info.get("sigma_h_deg"), sigma_c=info.get("sigma_c"),
                             n_edges=info.get("n_edges"), n_dir_edges=info.get("n_dir_edges"),
                             w_min=float(np.min(info["w_hint"])), max_correction_deg=float(np.max(info["correction_deg"])))
                if dv:
                    r["dlog_before"] = dlog(depth, gt_depth, dv)
                    r["dlog_after"] = dlog(newD, gt_depth, dv)
                rows.append(r)
        print(f"[ok] {clip['clip_id']}: {len(pairs)} pairs, match {t_match:.1f} s, {len(edge_cache)} K sets", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "runs.csv"), index=False)
    R = pd.DataFrame(rows)
    agg = {"n": ("clip_id", "size")}
    for c in ("rre_before", "rre_after", "rre_max_before", "rre_max_after", "view_max_before", "view_max_after",
              "dlog_before", "dlog_after", "max_correction_deg", "t_vtt_s"):
        if c in R:
            agg[c] = (c, "mean")
    S = R.groupby("condition", sort=False).agg(**agg)
    if "rre_after" in R:
        S["share_rot_worse_0.25"] = R.assign(x=R.rre_after > R.rre_before + 0.25).groupby("condition", sort=False).x.mean()
        S["worst_rot_increase"] = (R.rre_after - R.rre_before).groupby(R.condition, sort=False).max()
    if "dlog_after" in R:
        S["share_depth_worse_0.05"] = R.assign(x=R.dlog_after > R.dlog_before + 0.05).groupby("condition", sort=False).x.mean()
    S.to_csv(os.path.join(a.out, "summary.csv"))
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(S)


if __name__ == "__main__":
    main()
