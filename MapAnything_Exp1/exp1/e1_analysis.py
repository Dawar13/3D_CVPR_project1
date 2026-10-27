"""E1 analysis: signals S1-S7, the decision, tables and figures (experiments/E1.md sections 4, 6, 7).

  python -m exp1.e1_analysis --run results/e1_dev --checks     # dev practice-run checks
  python -m exp1.e1_analysis --run results/e1_eval             # signals + decision + figures

Reads <run>/per_run.csv, per_view.csv, corruption_manifest.json, run_info.json (config).
Writes <run>/agg/: dose_response.csv, contamination.csv, absorption.csv, confidence_auroc.csv,
fix.csv, scale.csv, signals.json, signals.md, fig_*.png.

Statistics: seeds are averaged within a clip; 95% CIs are percentile bootstraps over clips
(10k resamples, seed 0); all comparisons are paired by clip. benefit_lost uses means over clips:
(x - ceiling) / (floor - ceiling); > 1 = worse than the floor.

Signal rules (frozen with the e1-prereg tag; interpretation of experiments/E1.md section 6):
  S1 harm        a (family, severity, primary metric) with benefit_lost > 1, >= 70% of clips worse
                 than the floor, the paired 95% CI of (x - floor) above 0, AND a minimum effect:
                 mean(x) at least 5% above mean(floor) (benefit_lost explodes when the clean hint
                 barely helps, so tiny absolute changes must not count). Any severity.
  Reported, not gating: the same comparison against rgb_K ("photos + lens, poses dropped"),
                 the practical fallback, for every family that is given poses.
  S7 realistic   at least one S1 setting is in the family's realistic range.
  S2 spread      single/fraction outliers (P3-P6), realistic severity: the uncorrupted views'
                 per-view error rises >= 20% vs clean_pose on the same views, CI of the paired
                 difference above 0 (view_pts_err_scale, or pred_rot_err on non-reference views).
  S3 blind trust median absorption >= 0.5 over corrupted views of single-view outliers (P3-P5).
  S4 silent      within-run AUROC of the model's view confidence (clean > corrupted) <= 0.65,
                 pooled over P3-P6 and focal_one runs.
  S5 fix fails   for at least one S1 setting, filter_rgbcheck recovers < 50% of the oracle gain
                 (oracle: drop_oracle; for correlated families: rgb_K = drop all poses),
                 or the family has no pose filter / the filter drops nothing.
  S6 agrees-but-wrong  an S1 setting in a correlated family (C, K, D), or metric_scale_ratio
                 off by > 20% relative to clean_pose at a realistic setting while the filter
                 drops no pose (or the poses are clean: D1, M1).
  Decision: STOP if not S1; Strong GO if S1, S5, S7 and (S2 or S6); otherwise Weak GO.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from .metrics import LOWER_IS_BETTER

SINGLE = ["outlier_rot", "outlier_trans_small", "outlier_trans_large"]
SPREAD = SINGLE + ["outlier_frac"]
CORRELATED = ["scale_all", "drift", "shared_heading_bias", "anchor_error", "focal_all", "focal_all_noposes",
              "focal_one", "principal_shift", "depth_scale", "depth_scale_noposes", "depth_noise", "depth_one_scale"]
SCALE_FAMILIES = ["scale_all", "depth_scale", "depth_scale_noposes", "metric_flag_off"]
FILTER_ORACLE_RGBK = ["scale_all", "drift", "shared_heading_bias"]
PRIMARY = ["pts_relerr_scale", "rre_deg"]
PRIMARY_SCALE = ["pts_relerr_metric", "scale_err"]
NOT_TESTED = ("reference", "sanity")
BLUE, ORANGE, AQUA, MUTED, INK, RED = "#2a78d6", "#eb6834", "#1baf7a", "#898781", "#0b0b0b", "#e34948"


def boot_ci(x, n=10000, seed=0):
    """Percentile bootstrap 95% CI of the mean."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    return tuple(np.percentile(x[rng.integers(0, len(x), (n, len(x)))].mean(1), [2.5, 97.5]))


def load(run):
    pr = pd.read_csv(os.path.join(run, "per_run.csv"))
    pv = pd.read_csv(os.path.join(run, "per_view.csv")) if os.path.exists(os.path.join(run, "per_view.csv")) else None
    cfg = json.load(open(os.path.join(run, "run_info.json")))["config"]
    return pr, pv, cfg


def condition_table(cfg):
    from .run_exp1 import expand_conditions, seeds_for
    return {c["name"]: dict(c, seeds_used=seeds_for(c, cfg)) for c in expand_conditions(cfg)}


def clip_means(pr):
    ok = pr[pr.status == "ok"].copy()
    ok["scale_err"] = np.abs(np.log(ok["metric_scale_ratio"]))
    cols = [m for m in list(LOWER_IS_BETTER) + ["scale_err", "metric_scale_ratio", "n_pose_dropped"] if m in ok]
    keys = ["clip_id", "scene", "condition", "family", "severity", "control", "view_subset"]
    return ok.groupby(keys, dropna=False)[cols].mean().reset_index()


def values(cm, cond, metric, control="none", subset="all"):
    d = cm[(cm.condition == cond) & (cm.control == control) & (cm.view_subset == subset)]
    return d.set_index("clip_id")[metric]


def bl_stats(x, floor, ceil):
    """benefit_lost (ratio of means over shared clips) + bootstrap CI, paired x - floor."""
    df = pd.concat({"x": x, "f": floor, "c": ceil}, axis=1).dropna()
    if len(df) == 0:
        return None
    gap = (df.f - df.c).mean()
    bl = (df.x.mean() - df.c.mean()) / gap if gap > 0 else np.nan
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(df), (10000, len(df)))
    xs, fs, cs = df.x.values[idx].mean(1), df.f.values[idx].mean(1), df.c.values[idx].mean(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        bls = (xs - cs) / (fs - cs)
    d = (df.x - df.f).values
    lo, hi = boot_ci(d)
    per_clip = (df.x - df.c) / (df.f - df.c).where((df.f - df.c).abs() > 1e-9)
    return {"benefit_lost": bl, "bl_ci_lo": np.nanpercentile(bls, 2.5) if len(df) > 1 else np.nan,
            "bl_ci_hi": np.nanpercentile(bls, 97.5) if len(df) > 1 else np.nan,
            "x_mean": float(df.x.mean()), "floor_mean": float(df.f.mean()), "ceiling_mean": float(df.c.mean()),
            "rel_vs_floor": float(df.x.mean() / df.f.mean() - 1) if df.f.mean() > 0 else np.nan,
            "delta_vs_floor": float(d.mean()), "delta_ci_lo": lo, "delta_ci_hi": hi,
            "frac_worse_than_floor": float((d > 0).mean()), "n_clips": len(df),
            "per_clip_bl": per_clip.to_dict()}


MIN_EFFECT = 0.05  # S1: damaged error at least 5% above the floor (amendment 14)


def dose_response(cm, conds):
    rows = []
    for name, c in conds.items():
        if c["family"] in NOT_TESTED:
            continue
        metrics = PRIMARY + (PRIMARY_SCALE if c["family"] in SCALE_FAMILIES else [])
        for m in metrics:
            s = bl_stats(values(cm, name, m), values(cm, c["floor"], m), values(cm, c["ceiling"], m))
            if s is None:
                continue
            s1 = (s["benefit_lost"] > 1 and s["frac_worse_than_floor"] >= 0.7 and s["delta_ci_lo"] > 0
                  and s["rel_vs_floor"] >= MIN_EFFECT)
            row = {"condition": name, "family": c["family"], "severity": c.get("severity"),
                   "realistic": bool(c.get("realistic", False)), "metric": m,
                   "floor": c["floor"], "ceiling": c["ceiling"], "s1": bool(s1),
                   **{k: v for k, v in s.items() if k != "per_clip_bl"},
                   "per_clip_bl": json.dumps({k: (None if not np.isfinite(v) else round(v, 4))
                                              for k, v in s["per_clip_bl"].items()})}
            if "pose" in c["inputs"]:  # reported, not gating: would dropping all poses be better?
                k = pd.concat({"x": values(cm, name, m), "k": values(cm, "rgb_K", m)}, axis=1).dropna()
                dk = (k.x - k.k).values
                row.update(vs_rgbK_rel=float(k.x.mean() / k.k.mean() - 1) if len(k) and k.k.mean() > 0 else np.nan,
                           vs_rgbK_frac_worse=float((dk > 0).mean()) if len(k) else np.nan,
                           vs_rgbK_ci_lo=boot_ci(dk)[0], vs_rgbK_ci_hi=boot_ci(dk)[1])
            rows.append(row)
    return pd.DataFrame(rows)


def contamination(pv, conds):
    """Uncorrupted views' per-view error vs clean_pose on the same views, per clip (seed-averaged)."""
    ok = pv[pv.control == "none"]
    clean = ok[ok.condition == "clean_pose"].set_index(["clip_id", "view_idx"])
    rows = []
    for name, c in conds.items():
        if c["family"] not in SPREAD:
            continue
        d = ok[(ok.condition == name) & (ok.was_corrupted == 0)]
        if d.empty:
            continue
        for metric, sub in (("view_pts_err_scale", d), ("pred_rot_err", d[d.view_idx > 0])):
            j = sub.join(clean[metric].rename("ref"), on=["clip_id", "view_idx"])
            per_clip = j.groupby(["clip_id", "seed"])[[metric, "ref"]].mean().groupby("clip_id").mean()
            diff = per_clip[metric] - per_clip["ref"]
            lo, hi = boot_ci(diff)
            rows.append({"condition": name, "family": c["family"], "severity": c.get("severity"),
                         "realistic": bool(c.get("realistic", False)), "metric": metric,
                         "uncorrupted_err": per_clip[metric].mean(), "clean_err": per_clip["ref"].mean(),
                         "rel_rise": per_clip[metric].mean() / per_clip["ref"].mean() - 1,
                         "delta": diff.mean(), "ci_lo": lo, "ci_hi": hi, "n_clips": len(per_clip)})
    return pd.DataFrame(rows)


def auroc(pos, neg):
    """P(score_pos > score_neg) + 0.5 P(tie)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return gt + 0.5 * eq


def within_run_auroc(d, score, higher_is_corrupted):
    """Mean over runs of the AUROC between corrupted and clean non-reference views of that run."""
    vals = []
    for _, g in d[d.view_idx > 0].groupby(["clip_id", "condition", "seed"]):
        pos, neg = g[g.was_corrupted == 1][score], g[g.was_corrupted == 0][score]
        a = auroc(pos, neg) if higher_is_corrupted else auroc(-pos, -neg)
        if np.isfinite(a):
            vals.append(a)
    return (float(np.mean(vals)) if vals else np.nan), len(vals)


def detection_tables(pv, conds):
    ok = pv[pv.control == "none"]
    fam_of = {n: c["family"] for n, c in conds.items()}
    ok = ok.assign(family=ok.condition.map(fam_of))
    rows = []
    for fam in SPREAD + ["focal_one"]:
        d = ok[ok.family == fam]
        if d.empty:
            continue
        a_conf, n = within_run_auroc(d, "conf_mean", higher_is_corrupted=False)
        a_rot, _ = within_run_auroc(d, "disagree_rgbonly_rot", True)
        a_dir, _ = within_run_auroc(d, "disagree_rgbonly_dir", True)
        rows.append({"family": fam, "auroc_confidence": a_conf, "auroc_disagree_rot": a_rot,
                     "auroc_disagree_dir": a_dir, "n_runs": n})
    d = ok[ok.family.isin(SPREAD + ["focal_one"])]
    pooled, n = within_run_auroc(d, "conf_mean", higher_is_corrupted=False) if not d.empty else (np.nan, 0)
    rows.append({"family": "POOLED (S4)", "auroc_confidence": pooled,
                 "auroc_disagree_rot": within_run_auroc(d, "disagree_rgbonly_rot", True)[0] if not d.empty else np.nan,
                 "auroc_disagree_dir": within_run_auroc(d, "disagree_rgbonly_dir", True)[0] if not d.empty else np.nan,
                 "n_runs": n})
    return pd.DataFrame(rows)


def absorption_table(pv, conds):
    col = {"outlier_rot": "absorption_rot", "outlier_trans_small": "absorption_trans",
           "outlier_trans_large": "absorption_trans", "outlier_frac": "absorption_rot",
           "focal_one": "absorption_focal", "focal_all": "absorption_focal", "focal_all_noposes": "absorption_focal",
           "depth_scale": "absorption_depth", "depth_scale_noposes": "absorption_depth",
           "depth_one_scale": "absorption_depth"}
    ok = pv[(pv.control == "none") & (pv.was_corrupted == 1)]
    rows = []
    for name, c in conds.items():
        if c["family"] not in col:
            continue
        v = ok[ok.condition == name][col[c["family"]]].dropna()
        if len(v):
            rows.append({"condition": name, "family": c["family"], "severity": c.get("severity"),
                         "column": col[c["family"]], "median": v.median(), "q25": v.quantile(0.25),
                         "q75": v.quantile(0.75), "n_views": len(v)})
    return pd.DataFrame(rows)


def fix_table(cm, conds, dr):
    rows = []
    for name, c in conds.items():
        ctrls = c.get("controls", [])
        if "filter_rgbcheck" not in ctrls:
            continue
        for m in PRIMARY + (PRIMARY_SCALE if c["family"] in SCALE_FAMILIES else []):
            x = values(cm, name, m)
            oracle = values(cm, name, m, "drop_oracle") if "drop_oracle" in ctrls else values(cm, "rgb_K", m)
            filt = values(cm, name, m, "filter_rgbcheck")
            df = pd.concat({"x": x, "o": oracle, "f": filt}, axis=1).dropna()
            if df.empty:
                continue
            gain = df.x.mean() - df.o.mean()
            dropped = values(cm, name, "n_pose_dropped", "filter_rgbcheck").mean()
            rows.append({"condition": name, "family": c["family"], "severity": c.get("severity"), "metric": m,
                         "oracle": "drop_oracle" if "drop_oracle" in ctrls else "rgb_K (drop all poses)",
                         "err_damaged": df.x.mean(), "err_oracle": df.o.mean(), "err_filter": df.f.mean(),
                         "oracle_gain": gain, "filter_recovery": (df.x.mean() - df.f.mean()) / gain if gain > 0 else np.nan,
                         "filter_mean_poses_dropped": dropped})
    return pd.DataFrame(rows)


def scale_table(cm, conds):
    rows = []
    ref = values(cm, "clean_pose", "metric_scale_ratio")
    for name, c in conds.items():
        if c["family"] not in SCALE_FAMILIES and c["family"] != "sanity":
            continue
        r = values(cm, name, "metric_scale_ratio")
        if r.empty:
            continue
        df = pd.concat({"r": r, "ref": ref}, axis=1).dropna()
        factor = c.get("corruption", {}).get("factor", 1.0) if c.get("corruption") else 1.0
        rel = df.r / df.ref
        rows.append({"condition": name, "family": c["family"], "injected_factor": factor,
                     "metric": bool(c.get("metric", True)), "realistic": bool(c.get("realistic", False)),
                     "scale_ratio": df.r.mean(), "clean_scale_ratio": df.ref.mean(),
                     "rel_to_clean": rel.mean(), "rel_ci_lo": boot_ci(rel)[0], "rel_ci_hi": boot_ci(rel)[1],
                     "filter_drops_poses": float(values(cm, name, "n_pose_dropped", "filter_rgbcheck").mean())
                     if (cm.condition.eq(name) & cm.control.eq("filter_rgbcheck")).any() else 0.0})
    return pd.DataFrame(rows)


def signals(dr, cont, det, absb, fix, scale):
    s1_rows = dr[dr.s1]
    S1 = bool(len(s1_rows))
    S7 = bool(s1_rows.realistic.any()) if S1 else False
    c = cont[cont.realistic] if len(cont) else cont
    s2_rows = c[(c.rel_rise >= 0.2) & (c.ci_lo > 0)] if len(c) else c
    S2 = bool(len(s2_rows))
    single = absb[absb.family.isin(SINGLE)] if len(absb) else absb
    S3_val = float(np.nanmedian(single["median"])) if len(single) else np.nan
    S3 = bool(np.isfinite(S3_val) and S3_val >= 0.5)
    pooled = det[det.family == "POOLED (S4)"]["auroc_confidence"]
    S4_val = float(pooled.iloc[0]) if len(pooled) else np.nan
    S4 = bool(np.isfinite(S4_val) and S4_val <= 0.65)
    s5_hits = []
    for _, r in s1_rows.iterrows():
        f = fix[(fix.condition == r.condition) & (fix.metric == r.metric)] if len(fix) else fix
        if not len(f):
            s5_hits.append({"condition": r.condition, "metric": r.metric, "why": "no pose filter applies"})
        else:
            f = f.iloc[0]
            if f.filter_mean_poses_dropped == 0 or not np.isfinite(f.filter_recovery) or f.filter_recovery < 0.5:
                s5_hits.append({"condition": r.condition, "metric": r.metric,
                                "why": f"filter recovery {f.filter_recovery:.2f}, poses dropped {f.filter_mean_poses_dropped:.2f}"})
    S5 = bool(s5_hits)
    s6_corr = s1_rows[s1_rows.family.isin(CORRELATED)]
    sc = scale[(scale.realistic) & ((scale.rel_to_clean - 1).abs() > 0.2) & (scale.filter_drops_poses == 0)] if len(scale) else scale
    S6 = bool(len(s6_corr) or len(sc))
    if not S1:
        decision = "STOP"
    elif S5 and S7 and (S2 or S6):
        decision = "Strong GO"
    else:
        decision = "Weak GO"
    return {
        "S1": {"value": S1, "evidence": s1_rows[["condition", "metric", "benefit_lost", "frac_worse_than_floor",
                                                  "delta_ci_lo", "realistic"]].to_dict("records")},
        "S2": {"value": S2, "evidence": s2_rows[["condition", "metric", "rel_rise", "ci_lo"]].to_dict("records") if len(c) else []},
        "S3": {"value": S3, "median_absorption_single_outliers": S3_val},
        "S4": {"value": S4, "pooled_within_run_auroc_confidence": S4_val},
        "S5": {"value": S5, "evidence": s5_hits},
        "S6": {"value": S6, "correlated_s1": s6_corr[["condition", "metric", "benefit_lost"]].to_dict("records"),
               "scale_off": sc[["condition", "rel_to_clean"]].to_dict("records") if len(sc) else []},
        "S7": {"value": S7},
        "decision": decision,
    }


# ---------------------------------------------------------------- figures

def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=7, colors="#52514e")
    ax.grid(axis="y", color="#e1e0d9", lw=0.6)
    ax.set_axisbelow(True)


def fig_dose(dr, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fams = list(dict.fromkeys(dr.family))
    cols = 4
    rows = int(np.ceil(len(fams) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.5 * rows), squeeze=False)
    for ax, fam in zip(axes.ravel(), fams):
        d = dr[dr.family == fam]
        sev = list(dict.fromkeys(d.severity))
        x = np.arange(len(sev))
        real = [bool(d[d.severity == s].realistic.iloc[0]) for s in sev]
        for k, r in enumerate(real):
            if r:
                ax.axvspan(k - 0.5, k + 0.5, color="#f0efec", lw=0)
        for m, col in (("pts_relerr_scale", BLUE), ("rre_deg", ORANGE), ("pts_relerr_metric", AQUA)):
            e = d[d.metric == m].set_index("severity").reindex(sev)
            if e.benefit_lost.notna().any():
                y = e.benefit_lost.to_numpy(float)
                err = np.vstack([np.clip(y - e.bl_ci_lo.to_numpy(float), 0, None),
                                 np.clip(e.bl_ci_hi.to_numpy(float) - y, 0, None)])
                ax.errorbar(x, y, yerr=np.nan_to_num(err), color=col, lw=2, marker="o", ms=4,
                            capsize=0, label={"pts_relerr_scale": "3D shape", "rre_deg": "rotation",
                                              "pts_relerr_metric": "3D real size"}[m])
        ax.axhline(1, color=RED, lw=1)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(x, [f"{s:g}" if isinstance(s, (int, float)) else str(s) for s in sev])
        ax.set_title(fam, fontsize=9, loc="left")
        _style(ax)
    for ax in axes.ravel()[len(fams):]:
        ax.set_axis_off()
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("benefit lost vs severity (mean over clips, 95% bootstrap CI). 0 = clean hints, 1 = floor "
                 "(red line); shaded = realistic range", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_heatmap(dr, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    d = dr[(dr.metric == "pts_relerr_scale") & dr.realistic]
    if d.empty:
        return
    pick = d.loc[d.groupby("family").severity.apply(lambda s: s.index[-1])]
    clips = sorted({k for v in pick.per_clip_bl for k in json.loads(v)})
    mat = np.array([[json.loads(v).get(k) if json.loads(v).get(k) is not None else np.nan for k in clips]
                    for v in pick.per_clip_bl], float)
    fig, ax = plt.subplots(figsize=(3.5 + 0.75 * len(clips), 0.35 * len(pick) + 1.6))
    im = ax.imshow(np.clip(mat, -1, 3), cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=1, vmin=-1, vmax=3), aspect="auto")
    ax.set_xticks(range(len(clips)), [c.split("__")[1] for c in clips], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(pick)), [f"{f} ({s:g})" if isinstance(s, (int, float)) else f"{f} ({s})"
                                      for f, s in zip(pick.family, pick.severity)], fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.03, label="benefit lost (1 = floor)")
    ax.set_title("benefit lost per scene, 3D shape error, most severe realistic setting", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_contamination(cont, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = cont[cont.metric == "view_pts_err_scale"]
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(d) + 1.2))
    y = np.arange(len(d))[::-1]
    ax.barh(y, 100 * d.rel_rise, height=0.6, color=[BLUE if r else "#9ec5f4" for r in d.realistic])
    ax.axvline(20, color=RED, lw=1)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(y, d.condition, fontsize=7)
    ax.set_xlabel("error rise on UNcorrupted views vs clean_pose (%), dark = realistic; red = S2 threshold", fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_absorption(absb, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if absb.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(absb) + 1.2))
    y = np.arange(len(absb))[::-1]
    ax.hlines(y, absb.q25, absb.q75, color=BLUE, lw=3)
    ax.plot(absb["median"], y, "o", color=INK, ms=4)
    ax.axvline(0.5, color=RED, lw=1)
    ax.axvline(1, color=MUTED, lw=0.8)
    ax.set_yticks(y, absb.condition, fontsize=7)
    ax.set_xlabel("absorption on corrupted views (median, IQR): 1 = copies the bad hint, 0 = ignores it", fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_scale(scale, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = scale[scale.family.isin(SCALE_FAMILIES)]
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    for fam, col in zip(SCALE_FAMILIES, (BLUE, ORANGE, AQUA, INK)):
        e = d[d.family == fam]
        if len(e):
            y = e.rel_to_clean.to_numpy(float)
            err = np.nan_to_num(np.vstack([y - e.rel_ci_lo.to_numpy(float), e.rel_ci_hi.to_numpy(float) - y]))
            ax.errorbar(e.injected_factor.to_numpy(float), y, yerr=np.clip(err, 0, None),
                        fmt="o-", color=col, lw=2, ms=5, capsize=0, label=fam)
    xs = np.linspace(0.45, 2.1, 10)
    ax.plot(xs, xs, color=RED, lw=1, label="copies the hint's scale")
    ax.axhline(1, color=MUTED, lw=1, label="ignores it")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("injected scale factor", fontsize=8)
    ax.set_ylabel("predicted scene size / clean-hint size", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- dev checks

def dev_checks(run, pr, pv, cfg, conds, cm):
    out, man = [], json.load(open(os.path.join(run, "corruption_manifest.json")))
    clips = sorted(pr.clip_id.dropna().unique())
    fails = os.path.exists(os.path.join(run, "failures.jsonl"))
    expected = {(cl, n, s, ctrl) for cl in clips for n, c in conds.items() for s in c["seeds_used"]
                for ctrl in ["none"] + c.get("controls", [])}
    have = {(r.clip_id, r.condition, int(r.seed), r.control) for r in
            pr[(pr.view_subset == "all") & (pr.status == "ok")].itertuples()}
    missing = expected - have
    out.append(("no failures, every condition x clip x seed x control present",
                not fails and not missing, f"failures file: {fails}; missing runs: {len(missing)} of {len(expected)}"))

    def rel(a, b, m):
        x, y = values(cm, a, m), values(cm, b, m)
        df = pd.concat({"x": x, "y": y}, axis=1).dropna()
        return float(((df.x - df.y).abs() / df.y).max()) if len(df) else np.nan

    worst = max(rel("gauge_rigid", "clean_pose", m) for m in PRIMARY)
    out.append(("gauge_rigid ~ clean_pose (max rel. diff <= 5%)", worst <= 0.05, f"max rel. diff {worst:.4f}"))
    # Harness test: with the metric flag off, a common translation scale must be invisible
    # (t / mean|t| is scale-free and the scale feature is gated off), so x1.5 must equal
    # metric_flag_off exactly. Its distance to clean_pose is the metric-flag effect itself
    # (family M1), reported but not gating (amendment 13, experiments/E1.md).
    exact = max(rel("scale_all_nonmetric_x1.5", "metric_flag_off", m) for m in PRIMARY)
    out.append(("scale_all_nonmetric_x1.5 == metric_flag_off (exact)", exact == 0.0, f"max rel. diff {exact:.2e}"))
    flag = max(rel("scale_all_nonmetric_x1.5", "clean_pose", m) for m in PRIMARY)
    out.append(("INFO: scale_all_nonmetric_x1.5 vs clean_pose (metric-flag effect, not gating)", None,
                f"max rel. diff {flag:.4f}"))
    eq =[e.get("max_abs_diff_vs_clean_pts3d") for e in man if e["condition"] in ("zero_noise", "scale_all_x1.0")]
    out.append(("scale_all_x1.0 and zero_noise equal clean_pose", bool(eq) and all(v == 0.0 for v in eq),
                f"max |d pts3d| per clip: {eq}"))
    # Monotonicity only along one direction of damage: two-sided factor sweeps are checked per
    # side of 1 (needs >= 2 settings on that side); anchor_error's two settings are different
    # kinds of damage (15 deg turn vs 0.25 L shift), so it has no order.
    viol, checked = [], []
    two_sided = ("scale_all", "focal_all", "focal_all_noposes", "focal_one", "depth_scale", "depth_scale_noposes")
    for fam in sorted({c["family"] for c in conds.values() if c["family"] not in NOT_TESTED}):
        if fam == "anchor_error":
            continue
        names = [n for n, c in conds.items() if c["family"] == fam]
        runs = ([[n for n in sorted(names, key=lambda n: -conds[n]["severity"]) if conds[n]["severity"] < 1],
                 [n for n in sorted(names, key=lambda n: conds[n]["severity"]) if conds[n]["severity"] > 1]]
                if fam in two_sided else [names])
        for seq in runs:
            if len(seq) < 2:
                continue
            checked.append(f"{fam}[{conds[seq[0]]['severity']}->{conds[seq[-1]]['severity']}]")
            for m in PRIMARY:
                a, b = values(cm, seq[0], m).mean(), values(cm, seq[-1], m).mean()
                if b < 0.95 * a:
                    viol.append(f"{fam}/{m}: mildest {a:.4f} > most severe {b:.4f}")
    out.append(("severity sweeps roughly monotone (most severe >= 95% of mildest, per damage direction)",
                not viol, ("; ".join(viol) or "ok") + f" (checked: {', '.join(checked)})"))
    dro = pr[(pr.control == "drop_oracle") & (pr.view_subset == "uncorrupted")]
    ref = pr[(pr.control == "ref:clean_pose") & (pr.view_subset == "uncorrupted")]
    j = dro.merge(ref, on=["clip_id", "condition", "seed"], suffixes=("", "_ref"))
    r = (j.pts_relerr_scale / j.pts_relerr_scale_ref).median() if len(j) else np.nan
    out.append(("drop_oracle ~ clean_pose quality on the remaining views (median ratio <= 1.25)",
                bool(np.isfinite(r) and r <= 1.25), f"median ratio {r:.3f} over {len(j)} runs"))
    n_runs = len(pr[(pr.view_subset == "all") & (pr.status == "ok")])
    n_views = int(pr.n_views.dropna().iloc[0]) if len(pr) else 0
    # absorption needs the bad hint to be given: control "none" (the controls remove it)
    corr = pv[(pv.was_corrupted == 1) & (pv.control == "none") & pv.condition.str.startswith(tuple(SINGLE))]
    finite_abs = corr[["absorption_rot", "absorption_trans"]].notna().any(axis=1).mean() if len(corr) else 0
    ratio_ok = np.isfinite(pr.metric_scale_ratio.dropna()).all() and pr.metric_scale_ratio.notna().sum() > 0
    out.append(("per-view logs complete; absorption and metric_scale_ratio filled and finite",
                len(pv) == n_runs * n_views and finite_abs == 1.0 and bool(ratio_ok),
                f"per_view rows {len(pv)} (expected {n_runs * n_views}); absorption filled on "
                f"{finite_abs:.0%} of single-outlier corrupted views; scale ratio finite: {bool(ratio_ok)}"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out")
    ap.add_argument("--checks", action="store_true", help="dev practice-run checks (E1.md section 4)")
    a = ap.parse_args()
    out = a.out or os.path.join(a.run, "agg")
    os.makedirs(out, exist_ok=True)
    pr, pv, cfg = load(a.run)
    conds = condition_table(cfg)
    cm = clip_means(pr)

    if a.checks:
        rows = dev_checks(a.run, pr, pv, cfg, conds, cm)
        with open(os.path.join(out, "dev_checks.md"), "w") as f:
            f.write("| Check | Result | Detail |\n|---|---|---|\n")
            for name, ok, detail in rows:
                tag = "INFO" if ok is None else ("PASS" if ok else "FAIL")
                f.write(f"| {name} | {tag} | {detail} |\n")
                print(f"[{tag}] {name}: {detail}")

    dr = dose_response(cm, conds)
    cont = contamination(pv, conds)
    det = detection_tables(pv, conds)
    absb = absorption_table(pv, conds)
    fix = fix_table(cm, conds, dr)
    scale = scale_table(cm, conds)
    for name, df in (("dose_response", dr), ("contamination", cont), ("confidence_auroc", det),
                     ("absorption", absb), ("fix", fix), ("scale", scale)):
        df.to_csv(os.path.join(out, f"{name}.csv"), index=False)
    man = json.load(open(os.path.join(a.run, "corruption_manifest.json")))
    L = {}
    for e in man:
        if e.get("scene_length_L_m") is not None:
            L.setdefault(e["clip_id"], e["scene_length_L_m"])
    pd.DataFrame([{"clip_id": k, "scene": k.split("__")[1], "L_m": v} for k, v in sorted(L.items())]).to_csv(
        os.path.join(out, "scene_lengths.csv"), index=False)
    sig = signals(dr, cont, det, absb, fix, scale)
    json.dump(sig, open(os.path.join(out, "signals.json"), "w"), indent=1, default=float)
    with open(os.path.join(out, "signals.md"), "w") as f:
        f.write("| Signal | Value |\n|---|---|\n")
        for k in ("S1", "S2", "S3", "S4", "S5", "S6", "S7"):
            f.write(f"| {k} | {'YES' if sig[k]['value'] else 'NO'} |\n")
        f.write(f"\n**Decision: {sig['decision']}**\n")
    fig_dose(dr, os.path.join(out, "fig_dose_response.png"))
    fig_heatmap(dr, os.path.join(out, "fig_scene_heatmap.png"))
    fig_contamination(cont, os.path.join(out, "fig_contamination.png"))
    fig_absorption(absb, os.path.join(out, "fig_absorption.png"))
    fig_scale(scale, os.path.join(out, "fig_scale.png"))
    print("signals:", {k: sig[k]["value"] for k in ("S1", "S2", "S3", "S4", "S5", "S6", "S7")},
          "decision:", sig["decision"])


if __name__ == "__main__":
    main()
