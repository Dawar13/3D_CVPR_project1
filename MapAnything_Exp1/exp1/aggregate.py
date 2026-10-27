"""Aggregate per_run.csv -> tables + figures.

  python -m exp1.aggregate --run results/dev_pilot --metric pts_relerr_scale

Statistics: seeds are averaged within a clip first, so the unit of resampling is
the clip (scene). 95% CIs are percentile bootstrap over clips (10k resamples);
deltas vs a reference are PAIRED (same clips).

Outputs in <run>/:
  aggregate_metrics.csv   per condition x control: mean, CI, paired deltas, benefit lost
  per_scene.csv           clip-level (seed-averaged) values
  figures/*.png|pdf
"""

import argparse
import os

import numpy as np
import pandas as pd

from .metrics import LOWER_IS_BETTER

C_CORRUPT, C_RGB, C_CLEAN, C_CTRL = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
METRICS = list(LOWER_IS_BETTER)
FAMILY_LABEL = {
    "noise_rot": "Noise on all poses: rotation (deg)",
    "noise_trans": "Noise on all poses: translation (fraction of L)",
    "outlier_rot": "One outlier view: rotation (deg)",
    "outlier_trans": "One outlier view: translation (fraction of L)",
    "outlier_frac": "Outlier fraction (15 deg + 0.25 L each)",
}


def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def clip_level(df):
    keys = ["clip_id", "scene", "condition", "family", "severity", "control"]
    return df.groupby(keys, dropna=False)[METRICS + ["mean_conf"]].mean().reset_index()


def summarise(cl, metric):
    lower = LOWER_IS_BETTER[metric]
    ref = {r: cl[(cl.condition == r) & (cl.control == "none")].set_index("clip_id")[metric]
           for r in ("rgb_only", "clean_pose")}
    rows = []
    for (cond, ctrl), g in cl.groupby(["condition", "control"]):
        v = g.set_index("clip_id")[metric]
        lo, hi = boot_ci(v.values)
        row = {"condition": cond, "control": ctrl, "family": g.family.iloc[0],
               "severity": g.severity.iloc[0], "metric": metric,
               "lower_is_better": lower, "n_clips": len(v),
               "mean": v.mean(), "std": v.std(), "ci_lo": lo, "ci_hi": hi}
        for r, rv in ref.items():
            d = (v - rv).dropna()
            row[f"delta_vs_{r}"] = d.mean()
            row[f"delta_vs_{r}_ci_lo"], row[f"delta_vs_{r}_ci_hi"] = boot_ci(d.values)
            worse = d > 0 if lower else d < 0
            row[f"frac_clips_worse_than_{r}"] = worse.mean() if len(d) else np.nan
        # share of the clean-prior benefit lost; > 1 means worse than RGB-only
        gap = (ref["rgb_only"] - ref["clean_pose"]).mean()
        row["benefit_lost"] = (v.mean() - ref["clean_pose"].mean()) / gap if gap else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def subset_deltas(ok, metric):
    """Does a corrupted pose hurt only its own view, or contaminate the others?

    For outlier conditions, compares the corrupted run with RGB-only / clean-prior
    predictions scored on exactly the same views (rows with control 'ref:*').
    """
    keys = ["clip_id", "condition", "seed", "view_subset"]
    sub = ok[ok.view_subset.isin(["corrupted", "uncorrupted"])]
    cur = sub[sub.control == "none"].set_index(keys)[metric]
    rows = []
    for ref in ("rgb_only", "clean_pose"):
        r = sub[sub.control == f"ref:{ref}"].set_index(keys)[metric]
        d = (cur - r).dropna().groupby(level=["clip_id", "condition", "view_subset"]).mean()
        for (cond, vs), g in d.groupby(level=["condition", "view_subset"]):
            lo, hi = boot_ci(g.values)
            rows.append({"condition": cond, "view_subset": vs, "reference": ref,
                         "metric": metric, "n_clips": len(g), "delta": g.mean(),
                         "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def _save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200, bbox_inches="tight")


def fig_severity(summ, metric, out):
    import matplotlib.pyplot as plt
    fams = [f for f in FAMILY_LABEL if f in set(summ.family)]
    if not fams:
        return
    base = summ[summ.control == "none"].set_index("condition")
    fig, axes = plt.subplots(1, len(fams), figsize=(3.2 * len(fams), 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, fam in zip(axes, fams):
        g = summ[(summ.family == fam) & (summ.control == "none")].copy()
        g["sev"] = g.severity.astype(float)
        g = g.sort_values("sev")
        x = np.arange(len(g))
        for ref, col, lab in (("rgb_only", C_RGB, "RGB-only"), ("clean_pose", C_CLEAN, "Clean prior")):
            if ref in base.index:
                r = base.loc[ref]
                ax.axhspan(r.ci_lo, r.ci_hi, color=col, alpha=0.15, lw=0)
                ax.axhline(r["mean"], color=col, lw=2, label=lab)
        ax.errorbar(x, g["mean"].to_numpy(float), yerr=np.vstack([(g["mean"] - g.ci_lo).to_numpy(float),
                    (g.ci_hi - g["mean"]).to_numpy(float)]),
                    color=C_CORRUPT, lw=2, marker="o", ms=5, capsize=0, label="Corrupted prior")
        ax.set_xticks(x, [f"{s:g}" for s in g.sev])
        ax.set_xlabel(FAMILY_LABEL[fam], fontsize=8, color=INK2)
        _style(ax)
    arrow = "lower is better" if LOWER_IS_BETTER[metric] else "higher is better"
    axes[0].set_ylabel(f"{metric} ({arrow})", fontsize=8, color=INK)
    axes[0].legend(frameon=False, fontsize=7)
    n = int(summ.n_clips.max())
    fig.suptitle(f"Reconstruction vs corruption severity — mean, 95% bootstrap CI, n={n} clips",
                 fontsize=9, color=INK)
    _save(fig, out, "reconstruction_vs_corruption")
    plt.close(fig)


def fig_delta(summ, metric, out):
    import matplotlib.pyplot as plt
    g = summ[summ.condition != "rgb_only"].copy()
    g["label"] = g.condition + np.where(g.control == "none", "", " [" + g.control + "]")
    g = g.sort_values(["family", "condition", "control"])
    fig, ax = plt.subplots(figsize=(6.5, 0.22 * len(g) + 1))
    y = np.arange(len(g))
    cols = [C_CLEAN if c == "clean_pose" else C_CTRL if k != "none" else C_CORRUPT
            for c, k in zip(g.condition, g.control)]
    ax.errorbar(g.delta_vs_rgb_only, y, xerr=[g.delta_vs_rgb_only - g.delta_vs_rgb_only_ci_lo,
                g.delta_vs_rgb_only_ci_hi - g.delta_vs_rgb_only], fmt="none", ecolor=INK2, lw=1)
    ax.scatter(g.delta_vs_rgb_only, y, c=cols, s=28, zorder=3, edgecolors="white", linewidths=1)
    ax.axvline(0, color=C_RGB, lw=1.5)
    ax.set_yticks(y, g.label, fontsize=7)
    ax.invert_yaxis()
    worse = "right" if LOWER_IS_BETTER[metric] else "left"
    ax.set_xlabel(f"Δ {metric} vs RGB-only (paired; {worse} of 0 = worse than RGB-only)",
                  fontsize=8, color=INK)
    _style(ax)
    ax.grid(axis="x", color=GRID, lw=0.6)
    _save(fig, out, "relative_to_rgb_only")
    plt.close(fig)


def fig_per_scene(cl, summ, metric, out, top=8):
    import matplotlib.pyplot as plt
    ref = cl[(cl.condition == "rgb_only") & (cl.control == "none")].set_index("clip_id")[metric]
    conds = ["clean_pose"] + list(summ[(summ.control == "none") & (summ.family.isin(FAMILY_LABEL))]
                                   .sort_values("delta_vs_rgb_only", ascending=not LOWER_IS_BETTER[metric])
                                   .condition.head(top - 1))
    fig, ax = plt.subplots(figsize=(7, 3.2))
    clips = sorted(ref.index)
    for k, c in enumerate(conds):
        v = cl[(cl.condition == c) & (cl.control == "none")].set_index("clip_id")[metric]
        d = (v - ref).reindex(clips)
        jitter = np.linspace(-0.25, 0.25, len(clips))
        ax.scatter(k + jitter, d, s=16, color=C_CLEAN if c == "clean_pose" else C_CORRUPT,
                   edgecolors="white", linewidths=0.8, zorder=3)
        ax.plot([k - 0.3, k + 0.3], [d.mean()] * 2, color=INK, lw=2)
    ax.axhline(0, color=C_RGB, lw=1.5)
    ax.set_xticks(range(len(conds)), conds, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel(f"Δ {metric} vs RGB-only, per clip", fontsize=8, color=INK)
    ax.set_title(f"Per-clip effect (dots = clips, bar = mean, n={len(clips)}); "
                 "conditions with the largest mean harm", fontsize=8, color=INK)
    _style(ax)
    _save(fig, out, "per_scene_comparison")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--metric", default="pts_relerr_scale", choices=METRICS)
    a = ap.parse_args()

    df = pd.read_csv(os.path.join(a.run, "per_run.csv"))
    failed = df[df.status != "ok"]
    print(f"{len(df)} rows, {len(failed)} failed")
    if len(failed):
        print(failed.groupby("condition").size().to_string())
    ok = df[df.status == "ok"]
    subset_deltas(ok, a.metric).to_csv(os.path.join(a.run, "subset_deltas.csv"), index=False)
    df = ok[(ok.view_subset == "all") & ~ok.control.str.startswith("ref:")]
    cl = clip_level(df)
    cl.to_csv(os.path.join(a.run, "per_scene.csv"), index=False)
    summ = pd.concat([summarise(cl, m) for m in METRICS])
    summ.to_csv(os.path.join(a.run, "aggregate_metrics.csv"), index=False)

    s = summ[summ.metric == a.metric]
    cols = ["condition", "control", "n_clips", "mean", "ci_lo", "ci_hi", "delta_vs_rgb_only",
            "delta_vs_rgb_only_ci_lo", "delta_vs_rgb_only_ci_hi",
            "frac_clips_worse_than_rgb_only", "benefit_lost"]
    print(s[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    figs = os.path.join(a.run, "figures")
    fig_severity(s, a.metric, figs)
    fig_delta(s, a.metric, figs)
    fig_per_scene(cl, s, a.metric, figs)
    print(f"figures -> {figs}")


if __name__ == "__main__":
    main()
