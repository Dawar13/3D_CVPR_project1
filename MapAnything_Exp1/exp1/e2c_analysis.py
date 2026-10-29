"""E2-C analysis: how the problem behaves (experiments/E2.md section 3; preregistration in section 8).

  python -m exp1.e2c_analysis --run results/e2c_eval --clips 'data/clips_e2/eval__*.npz' \
      --views4 results/e2c_n4_eval --views20 results/e2c_n20_eval --out results/e2c_eval/agg

Statistics: the scene is the unit. Every per-run or per-view quantity is first averaged over the
seeds and clips of a scene, then over scenes; 95% CIs bootstrap over scenes. The primary tests
(PRIMARY below, fixed at tag e2-prereg-C) are Holm-corrected together; everything else is
exploratory and labelled so.

Absorption (C1, used throughout): for the bad view, with P = its predicted pose, H = its hint and
C = its predicted pose in the clean_pose run of the same clip,
    absorption = 1 - d(P, H) / d(C, H)
(d = rotation angle for rotation errors, camera-centre distance for position errors). 1 = the model
outputs the hint, 0 = it outputs what it outputs with the correct hint. Unlike E1's ratio it does not
count the model's own baseline error and stays defined at 180 deg.
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

from .datasets import load_clip
from .e1_analysis import auroc, boot_ci, condition_table, load
from .e2_audit import holm, signflip_p, ttest_p
from .geometry import invert_pose, rotation_error_deg, rotation_error_deg_precise as rot_err, so3_log, trajectory_scale

ONE_VIEW_POSE = ["c1_rot", "c1_trans", "c4_radial", "c4_orbit", "c4m_trans", "c4m_radial", "c5_rot15_v3", "c7_bad"]


# ------------------------------------------------------------------ helpers
def scene_stat(df, col, by=()):
    """Mean over seeds and clips within each scene -> one value per scene (index: by + scene)."""
    return df.groupby(list(by) + ["scene"])[col].mean()


def summarise(per_scene):
    v = per_scene.dropna().values
    lo, hi = boot_ci(v)
    return {"mean": float(np.mean(v)) if len(v) else np.nan, "ci_lo": lo, "ci_hi": hi, "n_scenes": len(v)}


def _extras(run, key):
    p = os.path.join(run, "extras", key.replace("|", "__") + ".npz")
    return np.load(p) if os.path.exists(p) else None


def bad_view_table(pv, pr, run, clips):
    """One row per corrupted view of a one-view pose corruption (control none), with absorption,
    confidence vs the same view in clean_pose, and the C3 output readouts."""
    ok = pv[pv.control == "none"]
    by_run = ok.set_index(["clip_id", "condition", "seed", "view_idx"])
    rows = []
    for (cid, cond, seed), g in ok[ok.family.isin(ONE_VIEW_POSE)].groupby(["clip_id", "condition", "seed"]):
        z = _extras(run, f"{cid}|{cond}|{seed}|none|all")
        if z is None or "hint_pose_rel" not in z.files or "clean_pose_rel" not in z.files:
            continue
        # reference = the clean-hint run with the same posed views: clean_pose, or for partial poses
        # (C7) the c7_clean run of the same fraction and seed
        frac = g.severity.iloc[0]
        ref_cond, ref_seed = ((f"c7_clean_frac={frac}", seed) if g.family.iloc[0] == "c7_bad" and float(frac) < 1
                              else ("clean_pose", 0))
        if ref_cond == "clean_pose":
            C_all = z["clean_pose_rel"]
        else:
            zr = _extras(run, f"{cid}|{ref_cond}|{ref_seed}|none|all")
            if zr is None:
                continue
            C_all = zr["pred_pose_rel"]
        clean = by_run.loc[(cid, ref_cond, ref_seed)]
        G = clips[cid]["poses"]
        L = trajectory_scale(G)
        good = g[(g.was_corrupted == 0) & (g.view_idx > 0)]
        for r in g[g.was_corrupted == 1].itertuples():
            i = r.view_idx
            P, H, C = z["pred_pose_rel"][i], z["hint_pose_rel"][i], C_all[i]
            ref = clean.loc[i]
            row = {"clip_id": cid, "scene": r.scene, "condition": cond, "family": r.family,
                   "severity": r.severity, "seed": seed, "view_idx": i, "hint_reproj_px": r.hint_reproj_px,
                   "conf_ratio": r.conf_mean / ref.conf_mean,
                   "d_shape_bad": r.view_shape_err_sim3 - ref.view_shape_err_sim3,
                   "d_rot_bad": r.pred_rot_err - ref.pred_rot_err,
                   "d_centre_bad_L": (r.pred_trans_err_m - ref.pred_trans_err_m) / L,
                   "d_logfocal_bad": np.log((1 + r.pred_focal_rel_err) / (1 + ref.pred_focal_rel_err)),
                   "d_shape_good": (good.view_shape_err_sim3.values
                                    - clean.loc[good.view_idx].view_shape_err_sim3.values).mean(),
                   "d_rot_good": (good.pred_rot_err.values - clean.loc[good.view_idx].pred_rot_err.values).mean(),
                   "auroc_conf": auroc(-g[(g.was_corrupted == 1)].conf_mean, -good.conf_mean)}
            dr_h = rot_err(C[:3, :3], H[:3, :3])
            if rot_err(G[i][:3, :3], H[:3, :3]) > 0.05:
                row["absorption"] = 1 - rot_err(P[:3, :3], H[:3, :3]) / dr_h
                wh, wp = so3_log(G[i][:3, :3].T @ H[:3, :3]), so3_log(C[:3, :3].T @ P[:3, :3])
                row["absorption_along"] = wp @ wh / (wh @ wh)
            else:
                dh = np.linalg.norm(C[:3, 3] - H[:3, 3])
                row["absorption"] = 1 - np.linalg.norm(P[:3, 3] - H[:3, 3]) / dh
                d = H[:3, 3] - G[i][:3, 3]
                row["absorption_along"] = (P[:3, 3] - C[:3, 3]) @ d / (d @ d)
            rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ C1 + C2 + C3
def trust_curves(bv):
    rows = []
    for (fam, sev), g in bv.groupby(["family", "severity"]):
        r = {"family": fam, "severity": sev, "n_views": len(g),
             "hint_reproj_px_median": g.hint_reproj_px.median()}
        for col in ("absorption", "absorption_along", "conf_ratio", "auroc_conf", "d_shape_bad", "d_rot_bad",
                    "d_centre_bad_L", "d_logfocal_bad", "d_shape_good", "d_rot_good"):
            s = summarise(scene_stat(g, col))
            r.update({col: s["mean"], col + "_ci_lo": s["ci_lo"], col + "_ci_hi": s["ci_hi"]})
        r["n_scenes"] = s["n_scenes"]
        rows.append(r)
    return pd.DataFrame(rows)


def knows_vs_obeys(bv):
    """C2: does the confidence drop track how much is copied? Spearman across bad views."""
    rows = []
    for fam in ("c1_rot", "c1_trans"):
        g = bv[bv.family == fam].dropna(subset=["absorption", "conf_ratio"])
        if len(g) < 5:
            continue
        rho = stats.spearmanr(np.log(g.conf_ratio), g.absorption).correlation
        within = [stats.spearmanr(np.log(h.conf_ratio), h.absorption).correlation
                  for _, h in g.groupby("severity") if len(h) > 4]
        rows.append({"family": fam, "spearman_pooled": rho, "spearman_within_severity_mean": np.nanmean(within),
                     "n_views": len(g)})
    return pd.DataFrame(rows)


def adoption(pr, pv):
    """C1 for shared-error channels: how much of an injected factor reaches the output (log ratio)."""
    ok_r = pr[(pr.control == "none") & (pr.view_subset == "all")]
    ok_v = pv[pv.control == "none"]
    rows = []
    msr = ok_r.set_index(["clip_id", "condition", "seed"]).metric_scale_ratio
    for fam, clean in (("c1_scale", "clean_pose"), ("c4m_scale", "clean_pose_median"), ("c1_depth", "clean_pose_depth2")):
        for (cond, sev), g in ok_r[ok_r.family == fam].groupby(["condition", "severity"]):
            if abs(float(sev) - 1) < 1e-9:
                continue
            ref = msr.xs(clean, level="condition").droplevel("seed")
            a = np.log(g.set_index("clip_id").metric_scale_ratio / ref.loc[g.clip_id].values) / np.log(float(sev))
            r = {"family": fam, "severity": sev, "readout": "scene size", **summarise(a.groupby(g.set_index("clip_id").scene).mean())}
            rows.append(r)
            if fam == "c1_depth":  # the hinted views' own depth
                v = ok_v[(ok_v.condition == cond) & (ok_v.has_depth_hint == 1)].set_index(["clip_id", "view_idx"])
                c = ok_v[(ok_v.condition == clean)].set_index(["clip_id", "view_idx"]).loc[v.index]
                a = np.log(v.pred_depth_scale.values / c.pred_depth_scale.values) / np.log(float(sev))
                rows.append({"family": fam, "severity": sev, "readout": "hinted views' depth",
                             **summarise(pd.Series(a, index=v.scene.values).groupby(level=0).mean())})
    for (cond, sev), g in ok_v[ok_v.family == "c1_focal"].groupby(["condition", "severity"]):
        if abs(float(sev) - 1) < 1e-9:
            continue
        c = ok_v[ok_v.condition == "clean_pose"].set_index(["clip_id", "view_idx"]).loc[g.set_index(["clip_id", "view_idx"]).index]
        a = np.log((1 + g.pred_focal_rel_err.values) / (1 + c.pred_focal_rel_err.values)) / np.log(float(sev))
        rows.append({"family": "c1_focal", "severity": sev, "readout": "predicted focal",
                     **summarise(pd.Series(a, index=g.scene.values).groupby(level=0).mean())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ C4
def channels(pr, pv, bv):
    """Harm per channel against the pixel size of the hint error (exploratory) and the median probe."""
    ok = pr[(pr.control == "none") & (pr.view_subset == "all")]
    clean = ok[ok.condition == "clean_pose"].set_index("clip_id")
    px = pv[(pv.control == "none") & (pv.was_corrupted == 1)].groupby("condition").hint_reproj_px.median()
    rows = []
    for (fam, cond, sev), g in ok[ok.family.str.match(r"c1_|c4_")].groupby(["family", "condition", "severity"]):
        for m in ("pts_relerr_scale", "rre_deg", "pts_relerr_metric"):
            d = g.set_index("clip_id")[m] - clean.loc[g.clip_id, m].values
            rows.append({"family": fam, "condition": cond, "severity": sev, "metric": m,
                         "hint_reproj_px_median": px.get(cond, np.nan),
                         **summarise(d.groupby(g.set_index("clip_id").scene).mean())})
    harm = pd.DataFrame(rows)
    # median probe: paired with the mean-statistic run of the same corruption
    med = []
    pairs = [("clean_pose_median", "clean_pose")]
    for c in ok[ok.family.isin(["c4m_trans", "c4m_radial", "c4m_scale"])].condition.unique():
        twin = c.replace("c4m_trans", "c1_trans").replace("c4m_radial", "c4_radial").replace("c4m_scale", "c1_scale")
        pairs.append((c, twin))
    for c, twin in pairs:
        a = ok[ok.condition == c].set_index(["clip_id", "seed"])
        b = ok[ok.condition == twin].set_index(["clip_id", "seed"])
        idx = a.index.intersection(b.index)
        for m in ("pts_relerr_scale", "rre_deg", "pts_relerr_metric"):
            d = (a.loc[idx, m] - b.loc[idx, m]).groupby(a.loc[idx, "scene"]).mean()
            gap = (b.loc[idx, m] - clean.loc[idx.get_level_values(0), m].values).groupby(a.loc[idx, "scene"]).mean()
            med.append({"median_run": c, "mean_run": twin, "metric": m, "delta_median_minus_mean": summarise(d)["mean"],
                        "ci_lo": summarise(d)["ci_lo"], "ci_hi": summarise(d)["ci_hi"],
                        "mean_run_gap_vs_clean": gap.mean(),
                        "share_of_gap_closed": -d.mean() / gap.mean() if abs(gap.mean()) > 1e-12 else np.nan,
                        "n_scenes": len(d)})
    return harm, pd.DataFrame(med)


# ------------------------------------------------------------------ C5
def evidence_scores(clips):
    """Per (clip, view): share of its true points seen by another view (depth test 5%), and texture."""
    rows = []
    for cid, c in clips.items():
        depth, K, P, img = c["depth"], c["K"], c["poses"], c["images"]
        N, H, W = depth.shape
        gray = img.astype(float).mean(-1)
        for i in range(N):
            gy, gx = np.gradient(gray[i])
            v, u = np.nonzero(depth[i] > 0)
            seen = np.zeros(len(u), bool)
            if len(u):
                z = depth[i][v, u]
                X = np.stack([(u - K[i, 0, 2]) / K[i, 0, 0] * z, (v - K[i, 1, 2]) / K[i, 1, 1] * z, z], -1)
                X = X @ P[i][:3, :3].T + P[i][:3, 3]
                for j in range(N):
                    if j == i:
                        continue
                    T = invert_pose(P[j])
                    pc = X @ T[:3, :3].T + T[:3, 3]
                    ok = pc[:, 2] > 1e-6
                    uu = np.round(pc[:, 0] / np.maximum(pc[:, 2], 1e-6) * K[j, 0, 0] + K[j, 0, 2]).astype(int)
                    vv = np.round(pc[:, 1] / np.maximum(pc[:, 2], 1e-6) * K[j, 1, 1] + K[j, 1, 2]).astype(int)
                    ok &= (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
                    dj = np.zeros(len(u))
                    dj[ok] = depth[j][vv[ok], uu[ok]]
                    seen |= ok & (dj > 0) & (np.abs(dj - pc[:, 2]) < 0.05 * pc[:, 2])
            rows.append({"clip_id": cid, "view_idx": i, "overlap": seen.mean() if len(u) else np.nan,
                         "texture": float(np.hypot(gx, gy).mean())})
    return pd.DataFrame(rows)


def evidence(bv, ev):
    d = bv.merge(ev, on=["clip_id", "view_idx"])
    rows = []
    for fam in ("c1_rot", "c1_trans"):
        for score in ("overlap", "texture"):
            g = d[d.family == fam].dropna(subset=["absorption", score])
            per_scene = g.groupby("scene").apply(
                lambda h: stats.spearmanr(h[score], h.absorption).correlation if len(h) > 4 else np.nan,
                include_groups=False)
            within = [stats.spearmanr(h[score], h.absorption).correlation for _, h in g.groupby("severity") if len(h) > 4]
            rows.append({"family": fam, "score": score, "spearman_within_severity_mean": np.nanmean(within),
                         "spearman_per_scene": summarise(per_scene)["mean"],
                         "ci_lo": summarise(per_scene)["ci_lo"], "ci_hi": summarise(per_scene)["ci_hi"]})
    return pd.DataFrame(rows), d


def view_count(runs, clips_by_n):
    """C5: absorption of the same bad camera (view 3, 15 deg) with 4, 10 and 20 photos."""
    rows = []
    for n, (run, clips) in runs.items():
        pr, pv, _ = load(run)
        bv = bad_view_table(pv, pr, run, clips)
        g = bv[bv.family == "c5_rot15_v3"]
        ok = pr[(pr.control == "none") & (pr.view_subset == "all")]
        harm = (ok[ok.family == "c5_rot15_v3"].groupby("scene").rre_deg.mean()
                - ok[ok.condition == "clean_pose"].groupby("scene").rre_deg.mean())
        rows.append({"n_views": n, **{f"absorption_{k}": v for k, v in summarise(scene_stat(g, "absorption")).items()},
                     **{f"rre_vs_clean_{k}": v for k, v in summarise(harm).items()},
                     "_absorption_by_scene": scene_stat(g, "absorption").to_dict()})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ C6, C7, C8
def correlated_vs_independent(pr, pv):
    ok = pr[pr.view_subset == "all"]
    rows = []
    pairs = [(f"c6_shared_rot_rot_deg={t}", f"c6_indep_rot_rot_deg={t}") for t in (2, 5)]
    pairs += [(f"c6_drift_level={k}", f"c6_drift_jitter_level={k}") for k in (1, 2, 3)]
    clean = ok[(ok.condition == "clean_pose") & (ok.control == "none")].set_index("clip_id")
    rgbk = ok[(ok.condition == "rgb_K") & (ok.control == "none")].set_index("clip_id")
    for corr, ind in pairs:
        r = {"correlated": corr, "independent": ind}
        for name, tag in ((corr, "corr"), (ind, "ind")):
            n = ok[(ok.condition == name) & (ok.control == "none")]
            f = ok[(ok.condition == name) & (ok.control == "filter_rgbcheck")].set_index(["clip_id", "seed"])
            nn = n.set_index(["clip_id", "seed"])
            for m in ("rre_deg", "pts_relerr_scale"):
                d = (n.set_index("clip_id")[m] - clean.loc[n.clip_id, m].values).groupby(n.set_index("clip_id").scene).mean()
                r[f"{tag}_{m}_vs_clean"] = summarise(d)["mean"]
            conf = (n.set_index("clip_id").mean_conf / clean.loc[n.clip_id].mean_conf.values).groupby(
                n.set_index("clip_id").scene).mean()
            r[f"{tag}_mean_conf_ratio"] = summarise(conf)["mean"]
            gain = (nn.rre_deg - rgbk.loc[nn.index.get_level_values(0)].rre_deg.values)
            rec = (nn.rre_deg - f.loc[nn.index].rre_deg) / gain.where(gain > 0)
            r[f"{tag}_filter_recovery_rre"] = rec.groupby(nn.scene).mean().mean()
            r[f"{tag}_filter_poses_dropped"] = f.n_pose_dropped.mean()
        a = ok[(ok.condition == corr) & (ok.control == "none")].set_index(["clip_id", "seed"])
        b = ok[(ok.condition == ind) & (ok.control == "none")].set_index(["clip_id", "seed"])
        for m in ("rre_deg", "pts_relerr_scale"):
            d = (a[m] - b.loc[a.index, m]).groupby(a.scene).mean()
            s = summarise(d)
            r.update({f"diff_{m}": s["mean"], f"diff_{m}_ci_lo": s["ci_lo"], f"diff_{m}_ci_hi": s["ci_hi"]})
        rows.append(r)
    return pd.DataFrame(rows)


def partial_hints(pr, bv):
    ok = pr[(pr.control == "none") & (pr.view_subset == "all")]
    rows = []
    for frac in (0.3, 0.6, 1.0):
        bad = ok[ok.condition == f"c7_bad_frac={frac}"].set_index(["clip_id", "seed"])
        cl = ok[ok.condition == (f"c7_clean_frac={frac}" if frac < 1 else "clean_pose")]
        cl = cl.set_index(["clip_id", "seed"]) if frac < 1 else cl.set_index("clip_id")
        r = {"pose_frac": frac}
        for m in ("rre_deg", "pts_relerr_scale"):
            ref = cl.loc[bad.index, m].values if frac < 1 else cl.loc[bad.index.get_level_values(0), m].values
            d = (bad[m] - ref).groupby(bad.scene).mean()
            s = summarise(d)
            r.update({f"{m}_bad_minus_clean": s["mean"], f"{m}_ci_lo": s["ci_lo"], f"{m}_ci_hi": s["ci_hi"]})
        a = bv[bv.condition == f"c7_bad_frac={frac}"]
        r["absorption"] = summarise(scene_stat(a, "absorption"))["mean"]
        rows.append(r)
    return pd.DataFrame(rows)


def photos_alone(pr, pv, run, clips):
    """C8: how well do photos alone (rgb_only, and rgb_K) pin down what hints would tell?"""
    rows = []
    for ref in ("rgb_only", "rgb_K"):
        pair_rot, pair_dir, logscale, view_depth = [], [], [], []
        for cid, c in clips.items():
            z = _extras(run, f"{cid}|{ref}|0|none|all")
            if z is None:
                continue
            P, G = z["pred_pose_rel"], c["poses"]
            for i in range(len(G)):
                for j in range(i + 1, len(G)):
                    rp, rg = invert_pose(P[i]) @ P[j], invert_pose(G[i]) @ G[j]
                    pair_rot.append(rotation_error_deg(rp[:3, :3], rg[:3, :3]))
                    a, b = rp[:3, 3], rg[:3, 3]
                    pair_dir.append(np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12), -1, 1))))
            r = pr[(pr.clip_id == cid) & (pr.condition == ref) & (pr.view_subset == "all")]
            logscale.append(np.log(float(r.metric_scale_ratio.iloc[0])))
            v = pv[(pv.clip_id == cid) & (pv.condition == ref) & (pv.control == "none")].pred_depth_scale.dropna()
            view_depth += list(np.log(v / v.median()))
        q = lambda x, p: float(np.percentile(np.abs(x), p)) if len(x) else np.nan
        rows.append({"reference": ref, "n_clips": len(logscale),
                     "pair_rot_err_deg_median": q(pair_rot, 50), "pair_rot_err_deg_p90": q(pair_rot, 90),
                     "pair_dir_err_deg_median": q(pair_dir, 50), "pair_dir_err_deg_p90": q(pair_dir, 90),
                     "scale_err_abs_log_median": q(logscale, 50), "scale_err_abs_log_p90": q(logscale, 90),
                     "scale_ratio_spread_sd_log": float(np.std(logscale)),
                     "view_depth_scale_abs_log_p90": q(view_depth, 90)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ primary tests (preregistered)
# (name, question, threshold, how a scene's value is computed). "Matters" thresholds for error metrics
# are the E2 section 0.2 ones; those for absorption / AUROC / correlation are fixed in E2.md section 8.
PRIMARY = [
    ("P1_rot_override", "C1", -0.10), ("P2_trans_override", "C1", -0.10), ("P3_conf_flags_bad_view", "C2", 0.15),
    ("P4_bad_view_own_shape", "C3", 0.01), ("P5_good_views_shape", "C3", 0.01), ("P6_median_changes_clean", "C4", 0.25),
    ("P7_median_fixes_gps_jump", "C4", -0.01), ("P8_absorption_vs_overlap", "C5", -0.30),
    ("P9_absorption_20_vs_4_views", "C5", -0.10), ("P10_shared_vs_indep_conf_5deg", "C6", 0.10),
    ("P11_drift_vs_jitter_conf_level2", "C6", 0.10), ("P12_partial_30_vs_100", "C7", 0.25)]
P8_SEVERITIES = (5, 10, 15, 30)  # one bad camera turned; absorption is not saturated here


def conf_ratio_diff(ok, corr, ind):
    """Per scene: mean_conf(corr)/mean_conf(clean_pose) - mean_conf(ind)/mean_conf(clean_pose), same
    clip and seed. Positive = the model's confidence drops LESS for the correlated error."""
    clean = ok[ok.condition == "clean_pose"].set_index("clip_id").mean_conf
    a = ok[ok.condition == corr].set_index(["clip_id", "seed"])
    b = ok[ok.condition == ind].set_index(["clip_id", "seed"]).loc[a.index]
    ref = clean.loc[a.index.get_level_values(0)].values
    return ((a.mean_conf - b.mean_conf) / ref).groupby(a.scene).mean()


def primary_tests(pr, bv, ev_merged, vc):
    ok = pr[(pr.control == "none") & (pr.view_subset == "all")]
    by = lambda d: d.groupby("scene").mean(numeric_only=True)

    def sev(fam, s, col):
        g = bv[(bv.family == fam) & (np.isclose(bv.severity.astype(float), s))]
        return scene_stat(g, col)

    def paired(a, b, m):
        x = ok[ok.condition == a].set_index(["clip_id", "seed"])
        y = ok[ok.condition == b]
        y = y.set_index(["clip_id", "seed"]) if (y.seed.nunique() > 1 or b != "clean_pose") else y.set_index("clip_id")
        ref = y.loc[x.index, m].values if isinstance(y.index, pd.MultiIndex) else y.loc[x.index.get_level_values(0), m].values
        return (x[m] - ref).groupby(x.scene).mean()

    v = {}
    v["P1_rot_override"] = sev("c1_rot", 30, "absorption") - sev("c1_rot", 15, "absorption")
    v["P2_trans_override"] = sev("c1_trans", 10, "absorption") - sev("c1_trans", 0.25, "absorption")
    v["P3_conf_flags_bad_view"] = sev("c1_rot", 15, "auroc_conf") - 0.5
    v["P4_bad_view_own_shape"] = sev("c1_rot", 15, "d_shape_bad")
    v["P5_good_views_shape"] = sev("c1_rot", 15, "d_shape_good")
    v["P6_median_changes_clean"] = paired("clean_pose_median", "clean_pose", "rre_deg")
    v["P7_median_fixes_gps_jump"] = paired("c4m_trans_trans_frac=10", "c1_trans_trans_frac=10", "pts_relerr_scale")
    g = ev_merged[(ev_merged.family == "c1_rot") & ev_merged.severity.astype(float).isin(P8_SEVERITIES)].copy()
    g["resid"] = g.absorption - g.groupby(["scene", "severity"]).absorption.transform("mean")
    v["P8_absorption_vs_overlap"] = g.groupby("scene").apply(
        lambda h: stats.spearmanr(h.overlap, h.resid).correlation if len(h) > 4 else np.nan, include_groups=False)
    if vc is not None and len(vc) == 3:
        a = pd.Series(vc.set_index("n_views").loc[20, "_absorption_by_scene"])
        b = pd.Series(vc.set_index("n_views").loc[4, "_absorption_by_scene"])
        v["P9_absorption_20_vs_4_views"] = (a - b.loc[a.index.intersection(b.index)]).dropna()
    # confidence of the whole run vs clean_pose: does the model notice errors that agree less?
    v["P10_shared_vs_indep_conf_5deg"] = conf_ratio_diff(ok, "c6_shared_rot_rot_deg=5", "c6_indep_rot_rot_deg=5")
    v["P11_drift_vs_jitter_conf_level2"] = conf_ratio_diff(ok, "c6_drift_level=2", "c6_drift_jitter_level=2")
    v["P12_partial_30_vs_100"] = (paired("c7_bad_frac=0.3", "c7_clean_frac=0.3", "rre_deg")
                                  - paired("c7_bad_frac=1.0", "clean_pose", "rre_deg"))
    rows = []
    for name, q, thr in PRIMARY:
        x = v.get(name, pd.Series(dtype=float)).dropna()
        s = summarise(x)
        p_sf = signflip_p(x.values) if len(x) else np.nan
        p_t = ttest_p(x.values)
        rows.append({"test": name, "question": q, **s, "p_t": p_t, "p_signflip": p_sf,
                     "p_used": p_sf if len(x) and np.std(x.values) == 0 else p_t, "threshold": thr,
                     "large_enough": (abs(s["mean"]) >= thr) if name == "P6_median_changes_clean"
                     else (s["mean"] <= thr) if thr < 0 else (s["mean"] >= thr)})
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df.p_used)
    df["clear"] = df.p_holm < 0.05
    df["finding"] = df.clear & df.large_enough
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--views4")
    ap.add_argument("--views20")
    ap.add_argument("--clips4")
    ap.add_argument("--clips20")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pr, pv, cfg = load(a.run)
    load_all = lambda g: {os.path.splitext(os.path.basename(p))[0]: load_clip(p) for p in sorted(glob.glob(g))}
    clips = {k: v for k, v in load_all(a.clips).items() if k in set(pr.clip_id)}
    out = lambda df, name: df.to_csv(os.path.join(a.out, name), index=False)

    bv = bad_view_table(pv, pr, a.run, clips)
    out(bv, "bad_views.csv")
    out(trust_curves(bv), "c1_c3_trust_curves.csv")
    out(knows_vs_obeys(bv), "c2_knows_vs_obeys.csv")
    out(adoption(pr, pv), "c1_adoption_shared_channels.csv")
    harm, med = channels(pr, pv, bv)
    out(harm, "c4_channels_harm.csv")
    out(med, "c4_median_probe.csv")
    ev = evidence_scores(clips)
    out(ev, "c5_evidence_scores.csv")
    evd, ev_merged = evidence(bv, ev)
    out(evd, "c5_evidence.csv")
    vc = None
    if a.views4 and a.views20:
        vc = view_count({4: (a.views4, load_all(a.clips4)), 10: (a.run, clips), 20: (a.views20, load_all(a.clips20))}, None)
        out(vc.drop(columns="_absorption_by_scene"), "c5_view_count.csv")
    out(correlated_vs_independent(pr, pv), "c6_correlated_vs_independent.csv")
    out(partial_hints(pr, bv), "c7_partial_hints.csv")
    out(photos_alone(pr, pv, a.run, clips), "c8_photos_alone.csv")
    pt = primary_tests(pr, bv, ev_merged, vc)
    out(pt, "primary_tests.csv")
    pd.set_option("display.width", 250, "display.max_columns", 20)
    print(pt.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
