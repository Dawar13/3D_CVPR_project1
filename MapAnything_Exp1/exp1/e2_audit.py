"""E2-B post-hoc audit of E1 (experiments/E2.md section 2). Not part of the frozen E1 decision.

  python -m exp1.e2_audit --e1 results/e1_eval --rerun results/e2b_audit_eval \
      --clips 'data/clips/eval__*.npz' --out reports/e2_audit/ [--lit_realistic lit.json]

--rerun is configs/e2b_audit.json run with --save_extras: the E1 references and single-view /
fraction pose families with E1's exact corruptions, plus per-run poses (exp1.run_exp1.run_extras).

Writes to --out:
  copy_check.csv        check 1: E1 absorption values equal to 1; clean-hint output pose vs input pose
  practical.csv         check 2: every E1 setting vs its floor, E2 thresholds (section 0.2), Holm-corrected
  practical_family.csv  checks 2 and 5: which families still count as harmful, and at which sizes
  pairs.csv             check 3: rotation error on pairs with a bad camera vs pairs of good cameras only
  consistency.csv       check 4: how much of the injected error the bad view's pose and points take on
  parity.csv            the new VM vs E1 on identical runs
"""

import argparse
import glob
import itertools
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

from .datasets import load_clip
from .e1_analysis import NOT_TESTED, SCALE_FAMILIES, SPREAD, boot_ci, condition_table, load
from .geometry import (invert_pose, rotation_error_deg, rotation_error_deg_precise, so3_log,
                       trajectory_scale)

# E2 section 0.2 "matters" thresholds, in the metric's own units (errors are fractions)
THRESH = {"pts_relerr_scale": 0.01, "rre_deg": 0.25, "pts_relerr_metric": 0.05, "scale_off": 0.05}
LABEL = {"pts_relerr_scale": "shape", "rre_deg": "rotation", "pts_relerr_metric": "real_size",
         "scale_off": "scale_ratio"}
E1_NAME = {"scale_off": "scale_err"}  # the E1 counterpart used for its S1 flag
ALPHA = 0.05


def holm(p):
    """Holm-Bonferroni adjusted p-values (same order as p; NaN stays NaN)."""
    p = np.asarray(p, float)
    ok = np.where(np.isfinite(p))[0]
    order = ok[np.argsort(p[ok])]
    m, adj, run = len(order), np.full(len(p), np.nan), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * p[i]))
        adj[i] = run
    return adj


def signflip_p(d):
    """Exact two-sided sign-flip permutation p of the mean (all 2^n patterns)."""
    d = np.asarray(d, float)
    if len(d) > 16:
        raise ValueError("exact enumeration only for n <= 16")
    obs = abs(d.mean())
    means = [abs((d * s).mean()) for s in itertools.product((1, -1), repeat=len(d))]
    return float(np.mean(np.asarray(means) >= obs - 1e-12))


def ttest_p(d):
    d = np.asarray(d, float)
    if len(d) < 2:
        return np.nan
    if np.std(d) == 0:
        return 1.0 if d.mean() == 0 else 0.0
    return float(stats.ttest_1samp(d, 0.0).pvalue)


# ---------------------------------------------------------------- check 1
def copy_check(pv_e1, rerun, clips):
    rows = []
    for c in ("absorption_rot", "absorption_trans", "absorption_focal", "absorption_depth"):
        x = pv_e1[c].dropna()
        rows.append({"check": "E1 per_view " + c, "n": len(x), "n_round4_eq_1": int((x.round(4) == 1).sum()),
                     "n_exactly_1": int((x == 1).sum()), "n_within_1e-3": int(((x - 1).abs() < 1e-3).sum()),
                     "min": x.min(), "median": x.median(), "max": x.max()})
    rot, cen, cen_L, elem, rgb_rot, clean_gt_rot = [], [], [], [], [], []
    for cid, clip in clips.items():
        z = np.load(os.path.join(rerun, "extras", f"{cid}__clean_pose__0__none__all.npz"))
        r = np.load(os.path.join(rerun, "extras", f"{cid}__rgb_only__0__none__all.npz"))
        L = trajectory_scale(clip["poses"])
        for i in range(1, len(clip["poses"])):
            P, H, G = z["pred_pose_rel"][i], z["hint_pose_rel"][i], clip["poses"][i]
            rot.append(rotation_error_deg_precise(P[:3, :3], H[:3, :3]))  # tiny angles: no arccos floor
            cen.append(np.linalg.norm(P[:3, 3] - H[:3, 3]))
            cen_L.append(cen[-1] / L)
            elem.append(np.abs(P - H).max())
            clean_gt_rot.append(rotation_error_deg(P[:3, :3], G[:3, :3]))
            rgb_rot.append(rotation_error_deg(r["pred_pose_rel"][i][:3, :3], G[:3, :3]))
    for name, v in (("clean_pose: output vs input rotation (deg)", rot),
                    ("clean_pose: output vs input camera centre (m)", cen),
                    ("clean_pose: output vs input camera centre (/L)", cen_L),
                    ("clean_pose: max |output - input| matrix entry", elem),
                    ("context: clean_pose output vs true rotation (deg)", clean_gt_rot),
                    ("context: rgb_only output vs true rotation (deg)", rgb_rot)):
        v = np.asarray(v)
        rows.append({"check": name, "n": len(v), "n_exactly_1": np.nan,
                     "n_identical_to_input": int((v == 0).sum()) if "input" in name else np.nan,
                     "min": v.min(), "median": np.median(v), "max": v.max()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- check 2 (+5)
def practical(pr, conds, e1_dr=None, lit=None):
    ok = pr[(pr.status == "ok") & (pr.control == "none") & (pr.view_subset == "all")].copy()
    ok["scale_off"] = (ok["metric_scale_ratio"] - 1).abs()
    cm = ok.groupby(["clip_id", "condition"])[list(THRESH)].mean().reset_index()
    rows = []
    for name, c in conds.items():
        if c["family"] in NOT_TESTED:
            continue
        metrics = ["pts_relerr_scale", "rre_deg"] + (
            ["pts_relerr_metric", "scale_off"] if c["family"] in SCALE_FAMILIES else [])
        for m in metrics:
            piv = cm.pivot(index="clip_id", columns="condition", values=m)
            d = (piv[name] - piv[c["floor"]]).dropna()
            lo, hi = boot_ci(d.values)
            row = {"condition": name, "family": c["family"], "severity": c.get("severity"),
                   "e1_realistic": bool(c.get("realistic", False)), "metric": LABEL[m], "metric_col": m,
                   "floor": c["floor"], "x_mean": piv.loc[d.index, name].mean(),
                   "floor_mean": piv.loc[d.index, c["floor"]].mean(), "delta": d.mean(),
                   "ci_lo": lo, "ci_hi": hi, "n_clips": len(d), "n_worse": int((d > 0).sum()),
                   "threshold": THRESH[m], "p_t": ttest_p(d.values), "p_signflip": signflip_p(d.values)}
            row["rel_vs_floor"] = row["x_mean"] / row["floor_mean"] - 1 if row["floor_mean"] > 0 else np.nan
            if lit is not None:
                fam = lit.get(c["family"], {})
                row["lit_status"] = next((k for k in ("realistic", "edge", "beyond", "unsourced")
                                          if float(c.get("severity")) in [float(s) for s in fam.get(k, [])]),
                                         "unlabelled")
                row["lit_realistic"] = row["lit_status"] == "realistic"
            if e1_dr is not None:
                e = e1_dr[(e1_dr.condition == name) & (e1_dr.metric == E1_NAME.get(m, m))]
                row["e1_s1"] = bool(e.s1.iloc[0]) if len(e) else np.nan
            rows.append(row)
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df.p_t)
    df["p_holm_signflip"] = holm(df.p_signflip)
    df["clear"] = df.p_holm < ALPHA
    df["large_enough"] = df.delta >= df.threshold
    df["harmful"] = df.clear & df.large_enough
    df["harmful_ci_only"] = (df.ci_lo > 0) & df.large_enough  # uncorrected, for comparison with E1
    return df


def family_table(df):
    rows = []
    for fam, g in df.groupby("family", sort=False):
        def hits(mask):
            h = g[mask & g.harmful]
            return "; ".join(f"{r.severity}:{r.metric} +{r.delta:.3g}" for r in h.itertuples()) or "-"
        r = {"family": fam, "n_tests": len(g), "harmful_any_setting": bool(g.harmful.any()),
             "harmful_e1_realistic": bool((g.harmful & g.e1_realistic).any()),
             "e1_s1_realistic": bool(((g.e1_s1 == True) & g.e1_realistic).any()) if "e1_s1" in g else np.nan,
             "harmful_settings": hits(g.harmful == g.harmful)}
        if "lit_status" in g:
            r["harmful_lit_realistic"] = bool((g.harmful & (g.lit_status == "realistic")).any())
            r["harmful_lit_edge"] = bool((g.harmful & (g.lit_status == "edge")).any())
            r["lit_settings"] = "; ".join(dict.fromkeys(f"{s}:{k}" for s, k in zip(g.severity, g.lit_status)))
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- checks 3 and 4
def _runs(pv, families):
    d = pv[(pv.control == "none") & pv.family.isin(families)]
    return d.groupby(["clip_id", "condition", "family", "severity", "seed"])


def _extras(rerun, cid, cond, seed):
    return np.load(os.path.join(rerun, "extras", f"{cid}__{cond}__{seed}__none__all.npz"))


def _pair_err(P, G, bad):
    out = {"bad": [], "good": []}
    for i, j in itertools.combinations(range(len(G)), 2):
        rp, rg = invert_pose(P[i]) @ P[j], invert_pose(G[i]) @ G[j]
        out["bad" if (i in bad or j in bad) else "good"].append(rotation_error_deg(rp[:3, :3], rg[:3, :3]))
    return {k: float(np.mean(v)) if v else np.nan for k, v in out.items()}


def pairs(pv, rerun, clips):
    rows = []
    for (cid, cond, fam, sev, seed), g in _runs(pv, SPREAD):
        bad = set(g[g.was_corrupted == 1].view_idx)
        G = clips[cid]["poses"]
        e = _pair_err(_extras(rerun, cid, cond, seed)["pred_pose_rel"], G, bad)
        cl = _pair_err(_extras(rerun, cid, "clean_pose", 0)["pred_pose_rel"], G, bad)
        rg = _pair_err(_extras(rerun, cid, "rgb_only", 0)["pred_pose_rel"], G, bad)
        rows.append({"clip_id": cid, "condition": cond, "family": fam, "severity": sev, "seed": seed,
                     "n_bad": len(bad), "bad_pairs": e["bad"], "bad_pairs_clean": cl["bad"],
                     "bad_pairs_rgb": rg["bad"], "good_pairs": e["good"], "good_pairs_clean": cl["good"],
                     "good_pairs_rgb": rg["good"]})
    runs = pd.DataFrame(rows)
    per_clip = runs.groupby(["condition", "family", "severity", "clip_id"]).mean(numeric_only=True).reset_index()
    out = []
    for (cond, fam, sev), g in per_clip.groupby(["condition", "family", "severity"], sort=False):
        r = {"condition": cond, "family": fam, "severity": sev, "n_clips": len(g), "n_bad": g.n_bad.mean()}
        for k in ("bad_pairs", "good_pairs"):
            d = (g[k] - g[k + "_clean"]).values
            lo, hi = boot_ci(d)
            r.update({k: g[k].mean(), k + "_clean": g[k + "_clean"].mean(), k + "_rgb": g[k + "_rgb"].mean(),
                      k + "_delta": d.mean(), k + "_ci_lo": lo, k + "_ci_hi": hi,
                      k + "_rel_rise": g[k].mean() / g[k + "_clean"].mean() - 1,
                      k + "_n_worse_than_rgb": int((g[k] > g[k + "_rgb"]).sum())})
        out.append(r)
    return pd.DataFrame(out), runs


TRANS_ONLY = ["outlier_trans_small", "outlier_trans_large"]  # point displacement is only a check of
# the camera shift when the hint has no rotation error; rotations move every point of the view


def consistency(pv, rerun, clips):
    clean = pv[(pv.condition == "clean_pose") & (pv.control == "none")].set_index(["clip_id", "view_idx"])
    rows = []
    for (cid, cond, fam, sev, seed), g in _runs(pv, SPREAD):
        z = _extras(rerun, cid, cond, seed)
        G = clips[cid]["poses"]
        for r in g[g.was_corrupted == 1].itertuples():
            i = r.view_idx
            P, H, C = z["pred_pose_rel"][i], z["hint_pose_rel"][i], z["clean_pose_rel"][i]
            ref = clean.loc[(cid, i)]
            row = {"clip_id": cid, "condition": cond, "family": fam, "severity": sev, "seed": seed,
                   "view_idx": i, "absorption_rot_e1": r.absorption_rot, "absorption_trans_e1": r.absorption_trans,
                   # does the bad view's own depth / lens output change vs the clean-hint run?
                   "depth_ratio_vs_clean": r.pred_depth_scale / ref.pred_depth_scale,
                   "focal_ratio_vs_clean": (1 + r.pred_focal_rel_err) / (1 + ref.pred_focal_rel_err)}
            d = H[:3, 3] - G[i][:3, 3]
            if fam in TRANS_ONLY and np.linalg.norm(d) > 1e-9:
                n2 = d @ d
                axis = G[i][:3, 2]  # optical axis of the bad view (world = view-0 frame)
                dep = clips[cid]["depth"][i]
                row.update(shift_m=np.sqrt(n2),
                           shift_along_axis_frac=abs(d @ axis) / np.sqrt(n2),
                           # depth ratio the view would need for its points to stay put: 1 - axial shift / depth
                           depth_ratio_if_points_fixed=1 - (d @ axis) / np.median(dep[dep > 0]),
                           cam_along=(P[:3, 3] - C[:3, 3]) @ d / n2,
                           pts_along=z["pts_disp_along_shift_m"][i] / np.sqrt(n2),
                           pts_norm=z["pts_disp_median_m"][i] / np.sqrt(n2))
            wh = so3_log(G[i][:3, :3].T @ H[:3, :3])
            if np.linalg.norm(wh) > 1e-6:
                wp = so3_log(C[:3, :3].T @ P[:3, :3])
                row.update(rot_deg=np.degrees(np.linalg.norm(wh)), rot_along=wp @ wh / (wh @ wh),
                           rot_mag=np.linalg.norm(wp) / np.linalg.norm(wh),
                           rot_axis_cos=wp @ wh / (np.linalg.norm(wp) * np.linalg.norm(wh) + 1e-12))
            rows.append(row)
    views = pd.DataFrame(rows)
    cols = [c for c in ("absorption_trans_e1", "cam_along", "pts_along", "pts_norm",
                        "absorption_rot_e1", "rot_along", "rot_mag", "rot_axis_cos",
                        "depth_ratio_vs_clean", "focal_ratio_vs_clean") if c in views]
    summ = views.groupby(["condition", "family", "severity"], sort=False)[cols].median().reset_index()
    summ["n_bad_views"] = views.groupby(["condition", "family", "severity"], sort=False).size().values
    if "cam_along" in views:
        gap = (views.cam_along - views.pts_along).abs()
        summ["median_abs_cam_minus_pts"] = gap.groupby([views.condition]).median().reindex(summ.condition).values
        # within translation outliers: do points lag the camera more when the shift is along the view axis?
        t = views.dropna(subset=["cam_along", "pts_along"])
        summ.attrs["spearman_lag_vs_axis"] = stats.spearmanr(t.cam_along - t.pts_along,
                                                              t.shift_along_axis_frac).correlation
        # slope 1: the view's depth changes so its points stay put; 0: points move rigidly with the camera
        def slope(g):
            x, y = g.depth_ratio_if_points_fixed - 1, g.depth_ratio_vs_clean - 1
            return float((x * y).sum() / (x * x).sum())
        summ["depth_comp_slope"] = summ.condition.map(t.groupby("condition").apply(slope, include_groups=False))
        summ.attrs["depth_compensation_spearman"] = stats.spearmanr(t.depth_ratio_if_points_fixed,
                                                                     t.depth_ratio_vs_clean).correlation
    return summ, views


# ---------------------------------------------------------------- parity
def parity(pr_e1, pr_new):
    keys = ["pts_relerr_scale", "rre_deg", "pts_relerr_metric", "depth_absrel_scale", "metric_scale_ratio"]
    a = pr_e1[(pr_e1.control == "none") & (pr_e1.view_subset == "all")].set_index("run_key")
    b = pr_new[(pr_new.control == "none") & (pr_new.view_subset == "all")].set_index("run_key")
    common = a.index.intersection(b.index)
    rows = []
    for k in keys:
        x, y = a.loc[common, k].astype(float), b.loc[common, k].astype(float)
        diff = (y - x).abs()
        rows.append({"metric": k, "n_runs": len(common), "n_identical": int((diff == 0).sum()),
                     "median_abs_diff": diff.median(), "max_abs_diff": diff.max(),
                     "median_rel_diff": (diff / x.abs()).median(), "max_rel_diff": (diff / x.abs()).max()})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", required=True)
    ap.add_argument("--rerun", required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lit_realistic", help="json {family: {realistic: [severities], source: ...}}")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    pr1, pv1, cfg1 = load(a.e1)
    pr2, pv2, _ = load(a.rerun)
    clips = {os.path.splitext(os.path.basename(p))[0]: load_clip(p) for p in sorted(glob.glob(a.clips))}
    clips = {k: v for k, v in clips.items() if k in set(pr2.clip_id)}
    e1_dr = pd.read_csv(os.path.join(a.e1, "agg", "dose_response.csv"))
    lit = json.load(open(a.lit_realistic)) if a.lit_realistic else None

    cc = copy_check(pv1, a.rerun, clips)
    cc.to_csv(os.path.join(a.out, "copy_check.csv"), index=False)
    pt = practical(pr1, condition_table(cfg1), e1_dr, lit)
    pt.to_csv(os.path.join(a.out, "practical.csv"), index=False)
    ft = family_table(pt)
    ft.to_csv(os.path.join(a.out, "practical_family.csv"), index=False)
    ps, _ = pairs(pv2, a.rerun, clips)
    ps.to_csv(os.path.join(a.out, "pairs.csv"), index=False)
    cs, cv = consistency(pv2, a.rerun, clips)
    cs.to_csv(os.path.join(a.out, "consistency.csv"), index=False)
    json.dump(cs.attrs, open(os.path.join(a.out, "consistency_stats.json"), "w"), indent=1)
    cv.to_csv(os.path.join(a.out, "consistency_views.csv"), index=False)
    pa = parity(pr1, pr2)
    pa.to_csv(os.path.join(a.out, "parity.csv"), index=False)

    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 200)
    print(cc.to_string(index=False), "\n")
    print(f"practical: {len(pt)} tests, Holm over all; harmful = Holm p < {ALPHA} and delta >= threshold")
    print(ft.to_string(index=False), "\n")
    print(ps.to_string(index=False), "\n")
    print(cs.to_string(index=False), "\n")
    print(pa.to_string(index=False))


if __name__ == "__main__":
    main()
