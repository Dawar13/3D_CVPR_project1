"""E4-A calibration (experiments/E4.md): the photo-evidence constants of exp1/vtt.py, fitted on dev / cal clips only
(eval clips are refused). No model is run. For every image pair with RANSAC-verified matches, the refined two-view
rotation and translation direction are compared with ground truth.

  Stage 1, ACCURATE ground truth only (--accurate; ETH3D dev: laser scan; TUM fr3: motion capture): the noise of
           right edges, covariance kappa^2 x (the edge's own first-order covariance) + sys^2 I (a model-error floor:
           rolling shutter, residual distortion, ... do not shrink with more matches), for rotations and directions,
           by maximum likelihood under the edge mixture of exp1/vtt.py (right: Gaussian; wrong: uniform), by EM.
  Stage 2, all dev / cal clips (--accurate plus --noisy, e.g. SceneFun3D whose ground truth is itself registered and
           approximate), kappa fixed: the prior that an edge is right, pe(n) = sigmoid(pe_a + pe_b log n), by EM
           with a logistic M-step. The noisy sources get an extra ground-truth noise s_gt (fitted) so that their
           ground-truth error is not counted as wrong edges.

  python -m exp1.vtt_calibrate --accurate 'data/clips_e2/dev__*.npz' 'data/clips_tum/cal__*.npz' \
      --noisy 'data/clips_e2x_room/dev__*.npz' 'data/clips_e2x_room/cal__*.npz' 'data/clips_e2x_cal/cal__*.npz' \
              'data/clips_e2e/dev__*.npz' --out results/e4_vtt_cal
  Stage 3, --depth (clips with ground-truth poses and correct depth: ground truth, and LiDAR where present): the
           model-error floor of the depth evidence, vtt_depth_sys, by maximum likelihood (evidence m_v ~ N(0, se_v^2 +
           sys^2), with a uniform outlier component), by EM.
Writes <out>/edges.csv (one row per pair, with its stage-2 responsibility), <out>/depth_evidence.csv and
<out>/chosen.json; see experiments/E4.md for the exact command.
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd

from . import photo_check, vtt
from .datasets import load_clip


def clip_edges(clip, cfg=None):
    pairs = photo_check.match_pairs(clip["images"])
    edges = vtt.pair_edges(pairs, clip["K"], cfg)
    Rt, Ct = clip["poses"][:, :3, :3], clip["poses"][:, :3, 3]
    res = vtt.edge_residuals(edges, Rt, Ct)
    rows = []
    for ed, x in zip(edges, res):
        rows.append({"clip_id": clip["clip_id"], "i": ed["i"], "j": ed["j"], "n": ed["n"], "e": ed["e"],
                     "parallax_deg": np.degrees(ed["parallax"]),
                     "rot_err_deg": float(np.degrees(np.linalg.norm(x["r"]))),
                     "rot_pred_sd_deg": float(np.degrees(np.sqrt(np.trace(ed["Sr"]) / 3))),
                     "dir_err_deg": np.nan if x["a"] is None else float(np.degrees(np.linalg.norm(x["a"]))),
                     "dir_pred_sd_deg": float(np.degrees(np.sqrt(np.trace(ed["St"]) / 2))),
                     "rot_std": float(np.sqrt(max(x["r"] @ np.linalg.solve(x["Sr"], x["r"]), 0.0))),
                     "dir_std": np.nan if x["a"] is None else float(np.sqrt(max(x["a"] @ np.linalg.solve(x["St"], x["a"]), 0.0)))})
    return len(pairs), res, rows


def depth_floor_evidence(globs):
    """Stage 3 data: per clip with ground-truth poses and correct depth (ground truth; and LiDAR where the clip has
    it), the depth evidence m_v, se_v of exp1/vtt.depth_evidence for every view (ground-truth poses and K)."""
    out = []
    for p in sorted({p for g in globs for p in glob.glob(g)}):
        if os.path.basename(p).startswith("eval__"):
            raise SystemExit("eval clips are never used for calibration")
        clip = load_clip(p)
        pairs = photo_check.match_pairs(clip["images"])
        V = list(range(len(clip["images"])))
        for name in ("depth", "dev_depth"):
            if clip.get(name) is None:
                continue
            m, se = vtt.depth_evidence(pairs, clip["K"], clip["poses"], clip[name], V)
            out += [{"clip_id": clip["clip_id"], "depth": name, "view": v, "m": m[v], "se": se[v]} for v in V]
        print(f"[ok] depth {clip['clip_id']}", flush=True)
    return pd.DataFrame(out)


def fit_depth_floor(m, se, iters=200, grid=np.r_[0.0, np.geomspace(0.002, 0.5, 60)]):
    """Maximum likelihood of the floor sys in m_v ~ p N(0, se_v^2 + sys^2) + (1 - p) U(-1, 1), by EM (grid M-step)."""
    p, sys_ = 0.9, 0.02
    for _ in range(iters):
        li = np.log(p) - 0.5 * np.log(2 * np.pi * (se ** 2 + sys_ ** 2)) - 0.5 * m ** 2 / (se ** 2 + sys_ ** 2)
        v = 1.0 / (1.0 + np.exp(np.log(1 - p) + np.log(0.5) - li))
        ll = [np.sum(v * (-0.5 * np.log(2 * np.pi * (se ** 2 + g ** 2)) - 0.5 * m ** 2 / (se ** 2 + g ** 2))) for g in grid]
        new, p_new = float(grid[int(np.argmax(ll))]), float(np.clip(v.mean(), 1e-3, 1 - 1e-3))
        done = new == sys_ and abs(p_new - p) < 1e-9
        sys_, p = new, p_new
        if done:
            break
    return sys_, p


def collect(globs, label):
    paths = sorted({p for g in globs for p in glob.glob(g)})
    if any(os.path.basename(p).startswith("eval__") for p in paths):
        raise SystemExit("eval clips are never used for calibration")
    res, rows = [], []
    for p in paths:
        t0 = time.time()
        clip = load_clip(p)
        n_pairs, r, rw = clip_edges(clip)
        res += r
        rows += [dict(x, source=label, folder=os.path.basename(os.path.dirname(p))) for x in rw]
        print(f"[ok] {label} {clip['clip_id']}: {n_pairs} matched pairs, {len(r)} edges, {time.time() - t0:.1f} s",
              flush=True)
    return paths, res, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accurate", nargs="+", required=True)
    ap.add_argument("--noisy", nargs="*", default=[])
    ap.add_argument("--depth", nargs="*", default=[], help="clips with ground-truth poses and correct depth (stage 3)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pa, ra, wa = collect(a.accurate, "accurate")
    pn, rn, wn = collect(a.noisy, "noisy") if a.noisy else ([], [], [])
    nz, p1, v1 = vtt.fit_kappa_em(ra)
    kr, sr, kd, sd = nz
    res, noisy = ra + rn, [False] * len(ra) + [True] * len(rn)
    pe_a, pe_b, s_gt, v2 = vtt.fit_edge_prior_em(res, noisy, nz)
    E = pd.DataFrame(wa + wn)
    E["v_stage1"] = list(v1) + [np.nan] * len(rn)
    E["v_stage2"] = v2
    E.to_csv(os.path.join(a.out, "edges.csv"), index=False)
    Dd = depth_floor_evidence(a.depth) if a.depth else pd.DataFrame(columns=["clip_id", "depth", "view", "m", "se"])
    Dd.to_csv(os.path.join(a.out, "depth_evidence.csv"), index=False)
    Df = Dd[np.isfinite(Dd.m.astype(float))] if len(Dd) else Dd
    dsys, dp = fit_depth_floor(Df.m.values.astype(float), Df.se.values.astype(float)) if len(Df) else (0.0, 1.0)
    pe = lambda n: float(vtt._sigmoid(pe_a + pe_b * np.log(n)))
    by = {}
    for (src, folder), g in E.groupby(["source", "folder"]):
        by[f"{src}/{folder}"] = {"n_edges": int(len(g)), "median_rot_err_deg": float(g.rot_err_deg.median()),
                                 "share_rot_err_gt5deg": float((g.rot_err_deg > 5).mean()),
                                 "mean_stage2_right_prob": float(g.v_stage2.mean())}
    out = {"vtt_kappa_rot": kr, "vtt_sys_rot_deg": float(np.degrees(sr)), "vtt_kappa_dir": kd,
           "vtt_sys_dir_deg": float(np.degrees(sd)), "vtt_pe_a": pe_a, "vtt_pe_b": pe_b, "vtt_depth_sys": dsys,
           "stage3_depth_share_right": dp, "stage3_n_views": int(len(Df)),
           "stage3_by_depth": {k: {"n": int(len(g)), "median_abs_m": float(g.m.abs().median())}
                               for k, g in Df.groupby("depth")} if len(Df) else {},
           "stage1_share_right": p1, "stage2_noisy_gt_sd_deg": s_gt,
           "pe_at_n": {str(n): pe(n) for n in (15, 30, 60, 120, 250, 500)},
           "median_match_kappa_accurate": list(vtt.calibrate_kappa(E[E.source == "accurate"].rot_std,
                                                                    E[E.source == "accurate"].dir_std)),
           "n_edges": {"accurate": len(ra), "noisy": len(rn)}, "by_folder": by,
           "clips": {"accurate": [os.path.basename(p) for p in pa], "noisy": [os.path.basename(p) for p in pn]}}
    json.dump(out, open(os.path.join(a.out, "chosen.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "clips"}, indent=1))


if __name__ == "__main__":
    main()
