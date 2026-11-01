"""E2-D / E2-E / E2-F analysis, exactly as preregistered at tag e2-prereg-DEF (experiments/E2.md, bottom).

  python -m exp1.e2def_analysis --out results/e2def_agg \
      --ma results/e2df_eval --ov results/e2d_eval --ma_c_primary results/e2c_eval/agg/primary_tests.csv \
      --clips 'data/clips_e2/eval__*.npz' \
      --ma_e results/e2e_ma_eval --ov_e results/e2e_ov_eval --clips_e 'data/clips_e2e/eval__*.npz'

Every part is optional (missing runs are skipped), so the same code runs on the dev runs.
Unit: the scene (mean over its clips and seeds). Reuses, unchanged: E2-B's practical test (e2_audit.practical)
for D-H, E2-C's bad-view table and primary tests for D-B (e2c_analysis), and their Holm / bootstrap helpers.
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

from .datasets import load_clip
from .e1_analysis import boot_ci, condition_table, load
from .e2_audit import ALPHA, THRESH, holm, practical, signflip_p, ttest_p
from .e2c_analysis import bad_view_table, evidence, evidence_scores, primary_tests
from .geometry import rotation_error_deg_precise as rot_err

E1_FAMILIES = ["noise_rot", "outlier_rot", "outlier_frac", "scale_all", "drift", "shared_heading_bias",
               "anchor_error", "depth_scale", "depth_one_scale"]
DEPTH_FAMILIES = ["depth_scale", "depth_one_scale"]
DB_TESTS = ["P1_rot_override", "P2_trans_override", "P3_conf_flags_bad_view", "P4_bad_view_own_shape",
            "P5_good_views_shape", "P10_shared_vs_indep_conf_5deg", "P11_drift_vs_jitter_conf_level2"]
METRIC_COL = {"shape": "pts_relerr_scale", "rotation": "rre_deg", "real_size": "pts_relerr_metric",
              "scale_ratio": "scale_off"}


def by_scene(pr):
    """Runs with clip_id replaced by the scene, so per-'clip' means in reused code are scene means."""
    d = pr.copy()
    d["clip_id"] = d["scene"]
    return d


def ok_runs(pr):
    d = pr[(pr.status == "ok") & (pr.view_subset == "all") & ~pr.control.astype(str).str.startswith("ref:")].copy()
    d["scale_off"] = (d["metric_scale_ratio"] - 1).abs()
    return d


def refresh_holm(df, p="p_t"):
    df = df.copy()
    df["p_holm"] = holm(df[p])
    df["clear"] = df.p_holm < ALPHA
    if "delta" in df and "threshold" in df:
        df["large_enough"] = df.delta >= df.threshold
        df["harmful"] = df.clear & df.large_enough
        df["better"] = df.clear & (df.delta <= -df.threshold)
    return df


# ------------------------------------------------------------------ E2-D
def d_harm(run, model):
    """D-H: E2-B's practical test on the 9 families, scene means, Holm over this model's tests."""
    pr, _, cfg = load(run)
    conds = {k: v for k, v in condition_table(cfg).items() if v["family"] in E1_FAMILIES}
    df = practical(by_scene(pr), conds)
    if model == "omnivggt":  # no metric scale in OmniVGGT's output
        df = df[~df.metric.isin(["real_size", "scale_ratio"])]
    df = refresh_holm(df)
    df.insert(0, "model", model)
    return df


def d_behaviour(run, clips_glob, model):
    """D-B: E2-C's P1-P5, P10, P11 on this model's C1/C6 runs, Holm over these 7."""
    pr, pv, _ = load(run)
    clips = {os.path.splitext(os.path.basename(p))[0]: load_clip(p) for p in sorted(glob.glob(clips_glob))}
    clips = {k: v for k, v in clips.items() if k in set(pr.clip_id)}
    bv = bad_view_table(pv, pr, run, clips)
    _, ev_merged = evidence(bv, evidence_scores(clips))
    pt = primary_tests(pr, bv, ev_merged, None)
    pt = pt[pt.test.isin(DB_TESTS)].copy()
    pt["p_holm"] = holm(pt.p_used)
    pt["clear"] = pt.p_holm < ALPHA
    pt["finding"] = pt.clear & pt.large_enough
    pt.insert(0, "model", model)
    return pt


def classify(a, b, a_rev, b_rev):
    """a, b: finding in model A / B; *_rev: Holm-clear effect beyond the threshold in the reverse direction."""
    if a and b:
        return "same"
    if (a and b_rev) or (b and a_rev):
        return "opposite"
    return "MapAnything only" if a else "OmniVGGT only" if b else "neither"


def side_by_side_harm(dh_ma, dh_ov):
    key = ["condition", "metric"]
    m = dh_ma.set_index(key)
    o = dh_ov.set_index(key)
    rows = []
    for k in m.index.union(o.index):
        a = m.loc[k] if k in m.index else None
        b = o.loc[k] if k in o.index else None
        fa = bool(a is not None and a.harmful)
        fb = bool(b is not None and b.harmful)
        ra = bool(a is not None and a.better)
        rb = bool(b is not None and b.better)
        fam = (a if a is not None else b).family
        rows.append({"family": fam, "condition": k[0], "metric": k[1],
                     "ma_delta": None if a is None else a.delta, "ma_p_holm": None if a is None else a.p_holm,
                     "ov_delta": None if b is None else b.delta, "ov_p_holm": None if b is None else b.p_holm,
                     "class": classify(fa, fb, ra, rb) if (a is not None and b is not None) else
                     ("MapAnything only (no OmniVGGT readout)" if fa else "neither")})
    df = pd.DataFrame(rows)
    fam = df.groupby("family")["class"].apply(lambda c: "same" if (c == "same").any() else
                                               "opposite" if (c == "opposite").any() else
                                               "MapAnything only" if c.str.startswith("MapAnything only").any() else
                                               "OmniVGGT only" if (c == "OmniVGGT only").any() else "neither")
    return df, fam.rename("class").reset_index()


def side_by_side_behaviour(db_ma, db_ov):
    m, o = db_ma.set_index("test"), db_ov.set_index("test")
    rows = []
    for t in DB_TESTS:
        if t not in m.index or t not in o.index:
            continue
        a, b = m.loc[t], o.loc[t]
        rev = lambda x: bool(x.clear and ((x["mean"] >= -x.threshold) if x.threshold < 0 else (x["mean"] <= -x.threshold)))
        rows.append({"test": t, "ma_mean": a["mean"], "ma_finding": bool(a.finding), "ov_mean": b["mean"],
                     "ov_finding": bool(b.finding), "class": classify(bool(a.finding), bool(b.finding), rev(a), rev(b))})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ E2-E
E_CONDS = {"mapanything": ["dev_K", "dev_pose", "dev_depth", "dev_all", "ai_depth", "dev_pose_ai_depth"],
           "omnivggt": ["dev_pose", "dev_depth", "dev_all", "ai_depth", "dev_pose_ai_depth"]}
E_METRIC_HINTS = {"dev_pose", "dev_depth", "dev_all", "ai_depth", "dev_pose_ai_depth"}
GT_OF = {"dev_K": "gt_K", "dev_pose": "gt_pose", "dev_depth": "gt_depth", "dev_all": "gt_all", "ai_depth": "gt_depth",
         "dev_pose_ai_depth": "gt_all"}


def scene_delta(ok, a, b, col, control_a="none"):
    x = ok[(ok.condition == a) & (ok.control == control_a)].groupby("scene")[col].mean()
    y = ok[(ok.condition == b) & (ok.control == "none")].groupby("scene")[col].mean()
    return (x - y.reindex(x.index)).dropna()


def e_tests(runs):
    """E-H: each real-hint condition vs rgb_only, per model; Holm over all tests of both models."""
    rows = []
    for model, run in runs.items():
        ok = ok_runs(load(run)[0])
        for c in E_CONDS[model]:
            metrics = ["shape", "rotation"] + (["real_size"] if model == "mapanything" and c in E_METRIC_HINTS else [])
            for m in metrics:
                d = scene_delta(ok, c, "rgb_only", METRIC_COL[m])
                lo, hi = boot_ci(d.values)
                rows.append({"model": model, "condition": c, "metric": m, "delta": d.mean(), "ci_lo": lo, "ci_hi": hi,
                             "n_scenes": len(d), "threshold": THRESH[METRIC_COL[m]], "p_t": ttest_p(d.values),
                             "p_signflip": signflip_p(d.values)})
    df = pd.DataFrame(rows)
    df["p_used"] = np.where(df.groupby(["model", "condition", "metric"]).delta.transform("size") > 0, df.p_t, np.nan)
    df["p_holm"] = holm(df.p_t)
    df["clear"] = df.p_holm < ALPHA
    df["harmful"] = df.clear & (df.delta >= df.threshold)
    df["helpful"] = df.clear & (df.delta <= -df.threshold)
    df["verdict"] = np.where(df.harmful, "HURTS", np.where(df.helpful, "HELPS", "no clear effect"))
    return df


def e_costs(runs):
    """Descriptive: real vs perfect hints (dev_x - gt_x), scene means with CIs."""
    rows = []
    for model, run in runs.items():
        ok = ok_runs(load(run)[0])
        for c in E_CONDS[model]:
            g = GT_OF[c] if GT_OF[c] in set(ok.condition) else None
            if g is None:
                continue
            for m in ("shape", "rotation") + (("real_size",) if model == "mapanything" else ()):
                d = scene_delta(ok, c, g, METRIC_COL[m])
                lo, hi = boot_ci(d.values)
                rows.append({"model": model, "condition": c, "vs": g, "metric": m, "delta": d.mean(), "ci_lo": lo,
                             "ci_hi": hi, "n_scenes": len(d)})
    return pd.DataFrame(rows)


def e_real_errors(clips_glob):
    """Descriptive: how wrong the real hints are, per view (poses relative to view 0) and per clip (depth)."""
    rows = []
    for p in sorted(glob.glob(clips_glob)):
        c = load_clip(p)
        P, D, N = c["poses"], c["dev_poses"], len(c["poses"])
        path = float(np.linalg.norm(np.diff(P[:, :3, 3], axis=0), axis=1).sum())
        rot = np.array([rot_err(P[i][:3, :3], D[i][:3, :3]) for i in range(1, N)])
        tr = np.array([np.linalg.norm(P[i][:3, 3] - D[i][:3, 3]) for i in range(1, N)])
        row = {"clip_id": c["clip_id"], "scene": c["scene"], "path_m": path, "rot_err_median": np.median(rot),
               "rot_err_max": rot.max(), "share_views_rot_ge5": float(np.mean(rot >= 5)),
               "share_views_rot_ge15": float(np.mean(rot >= 15)), "trans_err_max_m": tr.max(),
               "trans_err_max_over_path": tr.max() / path if path > 0 else np.nan,
               "drift_spearman_rot_vs_index": stats.spearmanr(np.arange(1, N), rot).correlation}
        g = c["depth"]
        for k in ("dev_depth", "mono_depth"):
            if k in c:
                v = (g > 0) & (c[k] > 0)
                r = c[k][v] / g[v]
                row[f"{k}_scale_ratio"] = float(np.median(r))
                row[f"{k}_absrel"] = float(np.median(np.abs(r - 1)))
                row[f"{k}_valid"] = float((c[k] > 0).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def e_absorption(run):
    """Descriptive: E1-style absorption of the real pose errors (predicted / hinted rotation error per view)."""
    _, pv, _ = load(run)
    d = pv[(pv.control == "none") & pv.condition.isin(["dev_pose", "dev_all", "dev_pose_ai_depth"]) & (pv.view_idx > 0)]
    d = d[d.hint_rot_err > 0.05]
    return d.groupby(["condition", "scene"]).absorption_rot.mean().groupby("condition").agg(["mean", "count"])


# ------------------------------------------------------------------ E2-F
R1_OF = {"depth_scale": "depth_scale_noposes_factor={}", "depth_one_scale": "depth_one_scale_noposes_factor={}"}


def f_gaps(run, dh_ma):
    """Remaining gap per remedy for every setting D-H finds harmful for MapAnything."""
    ok = ok_runs(load(run)[0])

    def cell(cond, control="none"):
        return ok[(ok.condition == cond) & (ok.control == control)]

    rows = []
    for _, h in dh_ma[dh_ma.harmful].iterrows():
        col = METRIC_COL[h.metric]
        fam, cond = h.family, h.condition
        depth = fam in DEPTH_FAMILIES
        clean = cell("clean_pose_depth2" if depth else "clean_pose")
        dmg = cell(cond)
        sev = cond.split("=")[-1]
        rem = {"R0 images only": cell("rgb_only"),
               "R1 drop poses": cell(R1_OF[fam].format(sev)) if depth else cell("rgb_K"),
               "R2 E1 filter": cell(cond, "filter_rgbcheck"), "R3 confidence drop": cell(cond, "drop_lowconf"),
               "ORACLE drop bad views": cell(cond, "drop_oracle")}
        sm = lambda d: d.groupby("scene")[col].mean()
        c_s, d_s = sm(clean), sm(dmg)
        scenes = d_s.index.intersection(c_s.index)
        for name, r in rem.items():
            r_s = sm(r).reindex(scenes)
            if r_s.isna().all():
                continue
            num, den = (r_s - c_s.reindex(scenes)), (d_s.reindex(scenes) - c_s.reindex(scenes))
            g = num.mean() / den.mean() if den.mean() != 0 else np.nan
            rng = np.random.default_rng(0)
            boots = []
            for _ in range(2000):
                i = rng.integers(0, len(scenes), len(scenes))
                dd = den.values[i].mean()
                boots.append(num.values[i].mean() / dd if dd != 0 else np.nan)
            lo, hi = np.nanpercentile(boots, [2.5, 97.5])
            rows.append({"family": fam, "condition": cond, "metric": h.metric, "remedy": name, "gap": g,
                         "ci_lo": lo, "ci_hi": hi, "n_scenes": len(scenes)})
    df = pd.DataFrame(rows)
    if len(df):
        best = (df[~df.remedy.str.startswith("ORACLE")].sort_values("gap").groupby(["condition", "metric"]).head(1)
                .rename(columns={"remedy": "best_remedy", "gap": "best_gap"})[["condition", "metric", "best_remedy", "best_gap"]])
        df = df.merge(best, on=["condition", "metric"], how="left")
        df["criterion_c_gap_ge_0.5"] = df.best_gap >= 0.5
    return df


def false_alarms(run):
    """Share of good views (not corrupted, view > 0) whose pose a deployable remedy dropped."""
    _, pv, _ = load(run)
    d = pv[pv.control.isin(["filter_rgbcheck", "drop_lowconf"]) & (pv.view_idx > 0) & (pv.was_corrupted == 0)]
    return d.groupby(["control", "family"]).has_pose_hint.apply(lambda x: float(np.mean(x == 0))).rename("false_alarm_rate").reset_index()


def f_real(run):
    """R2/R3 on the real pose hints: change vs no remedy, and how many (real-hint) views were dropped."""
    ok = ok_runs(load(run)[0])
    _, pv, _ = load(run)
    rows = []
    for c in ("dev_pose", "dev_all", "dev_pose_ai_depth"):
        for ctrl in ("filter_rgbcheck", "drop_lowconf"):
            for m in ("shape", "rotation"):
                d = scene_delta(ok, c, c, METRIC_COL[m], control_a=ctrl)
                lo, hi = boot_ci(d.values)
                drop = pv[(pv.condition == c) & (pv.control == ctrl) & (pv.view_idx > 0)].has_pose_hint
                rows.append({"condition": c, "remedy": ctrl, "metric": m, "delta_vs_no_remedy": d.mean(), "ci_lo": lo,
                             "ci_hi": hi, "share_views_dropped": float(np.mean(drop == 0)) if len(drop) else np.nan})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ma", help="MapAnything e2df run (E2-D companion + E2-F)")
    ap.add_argument("--ov", help="OmniVGGT e2d run")
    ap.add_argument("--ma_c_primary", help="E2-C primary_tests.csv (MapAnything side of D-B)")
    ap.add_argument("--clips", help="clips of the e2d run (for D-B)")
    ap.add_argument("--ma_e", help="MapAnything e2e run")
    ap.add_argument("--ov_e", help="OmniVGGT e2e run")
    ap.add_argument("--clips_e", help="E2-E clips (real error sizes)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    out = lambda df, name: (df.to_csv(os.path.join(a.out, name), index=False), print(f"wrote {name} ({len(df)} rows)"))
    have = lambda p: p and os.path.exists(os.path.join(p, "per_run.csv"))
    summary = {}
    dh_ma = dh_ov = None
    if have(a.ma):
        dh_ma = d_harm(a.ma, "mapanything")
        out(dh_ma, "d_harm_mapanything.csv")
    if have(a.ov):
        dh_ov = d_harm(a.ov, "omnivggt")
        out(dh_ov, "d_harm_omnivggt.csv")
    if dh_ma is not None and dh_ov is not None:
        sbs, fam = side_by_side_harm(dh_ma, dh_ov)
        out(sbs, "d_side_by_side_harm.csv")
        out(fam, "d_family_class.csv")
        summary["criterion_a_families_in_both_models"] = fam[fam["class"] == "same"].family.tolist()
    if have(a.ov) and a.clips:
        db_ov = d_behaviour(a.ov, a.clips, "omnivggt")
        out(db_ov, "d_behaviour_omnivggt.csv")
        if a.ma_c_primary and os.path.exists(a.ma_c_primary):
            db_ma = pd.read_csv(a.ma_c_primary)
            db_ma = db_ma[db_ma.test.isin(DB_TESTS)]
            out(side_by_side_behaviour(db_ma, db_ov), "d_side_by_side_behaviour.csv")
    runs_e = {k: v for k, v in (("mapanything", a.ma_e), ("omnivggt", a.ov_e)) if have(v)}
    if runs_e:
        eh = e_tests(runs_e)
        out(eh, "e_tests.csv")
        out(e_costs(runs_e), "e_costs_vs_perfect_hints.csv")
        summary["e_harmful"] = eh[eh.harmful][["model", "condition", "metric"]].values.tolist()
        summary["e_helpful"] = eh[eh.helpful][["model", "condition", "metric"]].values.tolist()
        for model, run in runs_e.items():
            out(e_absorption(run).reset_index(), f"e_absorption_{model}.csv")
    if a.clips_e:
        out(e_real_errors(a.clips_e), "e_real_errors.csv")
    if dh_ma is not None:
        fg = f_gaps(a.ma, dh_ma)
        out(fg, "f_remaining_gap.csv")
        out(false_alarms(a.ma), "f_false_alarms.csv")
        if len(fg):
            summary["criterion_c_settings_gap_ge_0.5"] = (
                fg.drop_duplicates(["condition", "metric"]).query("best_gap >= 0.5")[["condition", "metric", "best_remedy", "best_gap"]]
                .values.tolist())
    if have(a.ma_e):
        out(f_real(a.ma_e), "f_real_hints.csv")
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=str)
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
