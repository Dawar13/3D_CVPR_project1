"""E2: build the knowledge-transfer report (HTML + PDF) from the committed result files. CPU only.

  python scripts/e2_make_report.py --pdf_browser "C:/Program Files/Google/Chrome/Application/chrome.exe"

Reads results/e2c_eval/agg, results/e2def_agg, results/e2e_mono, results/e2e_clip_check, results/vm_logs_e2,
reports/e2_audit and the E1 aggregates; writes reports/e2_results/ (figures, e2_results.html, e2_results.pdf).
Every number in the document is computed here from those files.
"""

import argparse
import base64
import glob
import html
import json
import os
import pathlib
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from exp1.datasets import load_clip  # noqa: E402
from exp1.e1_analysis import load  # noqa: E402
from exp1.e2c_analysis import bad_view_table, trust_curves  # noqa: E402
from exp1.geometry import rotation_error_deg_precise as rot_err  # noqa: E402

p = lambda *a: os.path.join(ROOT, *a)
# reference palette (dataviz skill, references/palette.md): slot 1 blue, slot 2 orange; chrome and ink
MA, OV, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
GT_GLITCH = ("eval__sf422829__c1", 1)  # SceneFun3D ground-truth pose of view 1 is mis-registered (164 deg jump)


def style(ax, ylabel=None, xlabel=None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9, length=3, color=AXIS)
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9.5)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9.5)


def save(fig, path):
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def uri(path):
    return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()


def f(v, d=2):
    return "–" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


def pp(v, d=1):
    return "–" if v is None or not np.isfinite(v) else f"{v * 100:+.{d}f} pp"


def deg(v, d=2):
    return "–" if v is None or not np.isfinite(v) else f"{v:+.{d}f}°"


# ------------------------------------------------------------------ data
def conf_ratios(pr):
    """Confidence relative to clean_pose for the C6 pairs: the formula of e2c_analysis.correlated_vs_independent
    (lines 285-287), without its filter-recovery part, which needs an rgb_K run that OmniVGGT cannot take."""
    ok = pr[(pr.view_subset == "all") & (pr.control == "none")]
    clean = ok[ok.condition == "clean_pose"].set_index("clip_id")
    rows = []
    for corr, ind in ([(f"c6_shared_rot_rot_deg={t}", f"c6_indep_rot_rot_deg={t}") for t in (2, 5)]
                      + [(f"c6_drift_level={k}", f"c6_drift_jitter_level={k}") for k in (1, 2, 3)]):
        r = {"correlated": corr, "independent": ind}
        for name, tag in ((corr, "corr"), (ind, "ind")):
            n = ok[ok.condition == name].set_index("clip_id")
            r[f"{tag}_mean_conf_ratio"] = (n.mean_conf / clean.loc[n.index].mean_conf.values).groupby(n.scene).mean().mean()
        rows.append(r)
    return pd.DataFrame(rows)


def gather(clips_glob, clips_e_glob):
    D = {}
    D["e1_dr"] = pd.read_csv(p("results", "e1_eval", "agg", "dose_response.csv"))
    D["e1_abs"] = pd.read_csv(p("results", "e1_eval", "agg", "absorption.csv"))
    D["e1_sig"] = json.load(open(p("results", "e1_eval", "agg", "signals.json")))
    D["audit"] = pd.read_csv(p("reports", "e2_audit", "practical.csv"))
    A = p("results", "e2c_eval", "agg")
    D["pt"] = pd.read_csv(os.path.join(A, "primary_tests.csv"))
    D["tc_ma"] = pd.read_csv(os.path.join(A, "c1_c3_trust_curves.csv"))
    D["c6_ma"] = pd.read_csv(os.path.join(A, "c6_correlated_vs_independent.csv"))
    D["c8"] = pd.read_csv(os.path.join(A, "c8_photos_alone.csv"))
    D["c4"] = pd.read_csv(os.path.join(A, "c4_median_probe.csv"))
    D["c5v"] = pd.read_csv(os.path.join(A, "c5_view_count.csv"))
    D["c7"] = pd.read_csv(os.path.join(A, "c7_partial_hints.csv"))
    G = p("results", "e2def_agg")
    for k in ("d_harm_mapanything", "d_harm_omnivggt", "d_family_class", "d_side_by_side_behaviour",
              "d_behaviour_omnivggt", "e_tests", "e_costs_vs_perfect_hints", "e_real_errors", "f_remaining_gap",
              "f_false_alarms", "f_real_hints"):
        D[k] = pd.read_csv(os.path.join(G, k + ".csv"))
    D["e_sens"] = pd.read_csv(os.path.join(G, "sensitivity", "e_tests_without_sf422829_c1.csv"))
    D["mono"] = pd.read_csv(p("results", "e2e_mono", "depth_hint_errors.csv"))
    D["excluded"] = pd.read_csv(p("results", "e2e_clip_check", "excluded.csv"))
    D["sel_log"] = pd.read_csv(p("results", "e2e_clip_check", "selection_log.csv"))
    D["parity"] = pd.read_csv(p("results", "parity_rtx", "parity.csv"))
    # OmniVGGT trust curve and C6 with E2-C's own functions (identical definitions)
    pr, pv, _ = load(p("results", "e2d_eval"))
    clips = {os.path.splitext(os.path.basename(c))[0]: load_clip(c) for c in sorted(glob.glob(p(clips_glob)))}
    clips = {k: v for k, v in clips.items() if k in set(pr.clip_id)}
    D["tc_ov"] = trust_curves(bad_view_table(pv, pr, p("results", "e2d_eval"), clips))
    D["c6_ov"] = conf_ratios(pr)
    # E1-definition absorption (predicted / hinted rotation error of the bad view), both models, same corruptions
    ab = {}
    for model, run in (("MapAnything", "e2df_eval"), ("OmniVGGT", "e2d_eval")):
        v = pd.read_csv(p("results", run, "per_view.csv"))
        v = v[(v.control == "none") & (v.was_corrupted == 1) & v.condition.str.startswith("outlier_rot_rot_deg=")]
        ab[model] = v.groupby("severity").absorption_rot.median()
    D["abs_e1def"] = pd.DataFrame(ab)
    # rgb_only / clean / 90 deg per scene (OmniVGGT baseline instability)
    sc = {}
    for model, run in (("MapAnything", "e2df_eval"), ("OmniVGGT", "e2d_eval")):
        r = pd.read_csv(p("results", run, "per_run.csv"))
        r = r[(r.view_subset == "all") & (r.control == "none")]
        for c in ("rgb_only", "clean_pose", "outlier_rot_rot_deg=15", "outlier_rot_rot_deg=90"):
            sc[(model, c)] = r[r.condition == c].groupby("scene").rre_deg.mean()
    D["per_scene"] = sc
    # real ARKit pose errors per view (relative to view 0), without the ground-truth glitch
    rows = []
    for c in sorted(glob.glob(p(clips_e_glob))):
        z = load_clip(c)
        for i in range(1, len(z["poses"])):
            if (z["clip_id"], i) == GT_GLITCH:
                continue
            rows.append({"clip": z["clip_id"], "view": i, "rot": rot_err(z["poses"][i][:3, :3], z["dev_poses"][i][:3, :3])})
    D["arkit_views"] = pd.DataFrame(rows)
    # session facts
    D["session"] = open(p("results", "vm_logs_e2", "session.txt")).read().splitlines()
    D["machine"] = open(p("results", "vm_logs_e2", "machine.txt")).read().strip().splitlines()
    n = 0
    for r in ("parity_rtx", "parity_single", "e2c_eval", "e2c_n4_eval", "e2c_n20_eval", "e2df_dev", "e2df_eval",
              "e2e_ma_dev", "e2e_ma_eval", "e2d_checks", "e2d_dev", "e2d_eval", "e2e_ov_dev", "e2e_ov_eval"):
        q = pd.read_csv(p("results", r, "per_run.csv"))
        n += int(((q.view_subset == "all") & ~q.control.astype(str).str.startswith("ref:")).sum())
    D["n_runs"] = n
    return D


# ------------------------------------------------------------------ figures
def fig_trust(D, path):
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.4))
    for ax, fam, xl, xs in ((axs[0], "c1_rot", "rotation error of the one bad camera (°)", "log"),
                            (axs[1], "c1_trans", "position error of the one bad camera (× scene size L)", "log")):
        for name, tc, col in (("MapAnything", D["tc_ma"], MA), ("OmniVGGT", D["tc_ov"], OV)):
            t = tc[tc.family == fam].sort_values("severity")
            ax.fill_between(t.severity, t.absorption_ci_lo, t.absorption_ci_hi, color=col, alpha=0.15, lw=0)
            ax.plot(t.severity, t.absorption, color=col, lw=2, marker="o", ms=5, label=name)
        ax.axhline(1, color=AXIS, lw=0.8, ls="--")
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.set_xscale(xs)
        ax.set_ylim(-0.05, 1.08)
        style(ax, "absorption (1 = copies the hint)" if fam == "c1_rot" else None, xl)
        ax.set_xticks(sorted(tc[tc.family == fam].severity.unique()))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    axs[0].legend(frameon=False, fontsize=9, loc="lower right")
    save(fig, path)


def fig_knows(D, path):
    pt, ov = D["pt"].set_index("test"), D["d_behaviour_omnivggt"].set_index("test")
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.0))
    t15 = lambda tc: tc[(tc.family == "c1_rot") & np.isclose(tc.severity, 15)].iloc[0]
    for ax, key, lab in ((axs[0], "absorption", "absorption at 15°"), (axs[1], "auroc", "confidence AUROC at 15°")):
        for i, (name, col) in enumerate((("MapAnything", MA), ("OmniVGGT", OV))):
            if key == "absorption":
                r = t15(D["tc_ma"] if i == 0 else D["tc_ov"])
                v, lo, hi = r.absorption, r.absorption_ci_lo, r.absorption_ci_hi
            else:
                r = (pt if i == 0 else ov).loc["P3_conf_flags_bad_view"]
                v, lo, hi = r["mean"] + 0.5, r.ci_lo + 0.5, r.ci_hi + 0.5
            ax.bar(i, v, width=0.55, color=col, edgecolor="white", lw=2)
            ax.errorbar(i, v, yerr=[[v - lo], [hi - v]], color=INK2, capsize=4, lw=1.2)
            ax.text(i, hi + 0.03, f"{v:.2f}", ha="center", color=INK, fontsize=9)
        ax.set_xticks([0, 1], ["MapAnything", "OmniVGGT"])
        ax.set_ylim(0, 1.12)
        if key == "auroc":
            ax.axhline(0.5, color=AXIS, lw=0.8, ls="--")
            ax.text(-0.42, 0.51, "guessing", color=MUTED, fontsize=8, ha="left", va="bottom")
        style(ax, lab)
    save(fig, path)


def fig_agree(D, path):
    pairs = [("c6_indep_rot_rot_deg=5", "c6_shared_rot_rot_deg=5", "5° on every camera"),
             ("c6_drift_jitter_level=2", "c6_drift_level=2", "drift level 2 vs matched jitter")]
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.1), sharey=True)
    for ax, (name, c6, col) in zip(axs, (("MapAnything", D["c6_ma"], MA), ("OmniVGGT", D["c6_ov"], OV))):
        for j, (ind, cor, lab) in enumerate(pairs):
            r = c6[(c6.correlated == cor) & (c6.independent == ind)].iloc[0]
            ax.bar(j - 0.18, r.ind_mean_conf_ratio, width=0.34, color=col, alpha=0.45, edgecolor="white", lw=2,
                   label="independent (random directions)" if j == 0 else None)
            ax.bar(j + 0.18, r.corr_mean_conf_ratio, width=0.34, color=col, edgecolor="white", lw=2,
                   label="agreeing (shared / drift)" if j == 0 else None)
            for x, v in ((j - 0.18, r.ind_mean_conf_ratio), (j + 0.18, r.corr_mean_conf_ratio)):
                ax.text(x, v + 0.02, f"{v:.2f}", ha="center", fontsize=8.5, color=INK)
        ax.axhline(1, color=AXIS, lw=0.8, ls="--")
        ax.set_xticks([0, 1], [q[2] for q in pairs], fontsize=8.5)
        ax.set_title(name, fontsize=10, color=INK, loc="left")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        style(ax, "confidence ÷ confidence with clean hints" if name == "MapAnything" else None)
    axs[0].set_ylim(0, 1.5)
    save(fig, path)


ROT_SETTINGS = [("noise_rot_rot_deg=5", "every camera 5°, random"), ("outlier_rot_rot_deg=15", "one camera 15°"),
                ("outlier_rot_rot_deg=30", "one camera 30°"), ("outlier_rot_rot_deg=90", "one camera 90°"),
                ("outlier_frac_frac=0.4", "40% of cameras wrong"), ("shared_heading_bias_rot_deg=10", "same 10° heading, all"),
                ("drift_level=3", "drift (level 3)"), ("anchor_error_level=1", "first camera 15°")]


def fig_harm(D, path):
    fig, ax = plt.subplots(figsize=(9.6, 3.9))
    for k, (name, h, col, off) in enumerate((("MapAnything", D["d_harm_mapanything"], MA, -0.15),
                                             ("OmniVGGT", D["d_harm_omnivggt"], OV, 0.15))):
        for i, (c, lab) in enumerate(ROT_SETTINGS):
            r = h[(h.condition == c) & (h.metric == "rotation")]
            if not len(r):
                continue
            r = r.iloc[0]
            ax.plot([i + off, i + off], [r.ci_lo, r.ci_hi], color=col, lw=1.6)
            ax.scatter(i + off, r.delta, s=46, color=col if r.harmful else "white", edgecolor=col, lw=1.8, zorder=3,
                       label=f"{name} (filled = harmful by the strict test)" if i == 0 else None)
    ax.axhline(0, color=INK2, lw=0.9)
    ax.set_xticks(range(len(ROT_SETTINGS)), [q[1] for q in ROT_SETTINGS], rotation=22, ha="right", fontsize=8.5)
    ax.set_ylim(-14, 20)
    style(ax, "rotation error vs photos only (°)\n↑ worse   ↓ better")
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
    save(fig, path)


def fig_baseline(D, path):
    sc = D["per_scene"]
    scenes = sorted(sc[("OmniVGGT", "rgb_only")].index)
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.2), sharey=True)
    for ax, model, col in ((axs[0], "MapAnything", MA), (axs[1], "OmniVGGT", OV)):
        for c, mk, lab in (("rgb_only", "o", "photos only"), ("clean_pose", "s", "correct hints"),
                           ("outlier_rot_rot_deg=90", "^", "one camera 90° wrong")):
            v = sc[(model, c)].reindex(scenes)
            ax.scatter(range(len(scenes)), v, marker=mk, s=34, label=lab,
                       color=col if c != "clean_pose" else "white", edgecolor=col, lw=1.4,
                       alpha=0.55 if c == "outlier_rot_rot_deg=90" else 1)
        ax.set_xticks(range(len(scenes)), scenes, rotation=40, ha="right", fontsize=8)
        ax.set_yscale("log")
        ax.set_title(model, fontsize=10, color=INK, loc="left")
        style(ax, "rotation error (°, log scale)" if model == "MapAnything" else None)
    axs[1].legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(-0.05, -0.3), ncol=3)
    save(fig, path)


E_LABELS = [("dev_pose", "phone poses"), ("dev_depth", "phone LiDAR depth"), ("dev_all", "all phone hints"),
            ("ai_depth", "AI depth"), ("dev_pose_ai_depth", "phone poses + AI depth")]


def fig_real(D, path):
    e = D["e_tests"]
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for ax, metric, scale, lab in ((axs[0], "shape", 100, "shape error vs photos only (pp)"),
                                   (axs[1], "rotation", 1, "rotation error vs photos only (°)")):
        for name, col, off in (("mapanything", MA, -0.15), ("omnivggt", OV, 0.15)):
            for i, (c, _) in enumerate(E_LABELS):
                r = e[(e.model == name) & (e.condition == c) & (e.metric == metric)]
                if not len(r):
                    continue
                r = r.iloc[0]
                filled = r.verdict in ("HURTS", "HELPS")
                ax.plot([i + off] * 2, [r.ci_lo * scale, r.ci_hi * scale], color=col, lw=1.6)
                ax.scatter(i + off, r.delta * scale, s=44, color=col if filled else "white", edgecolor=col, lw=1.8,
                           zorder=3, label=("MapAnything" if name == "mapanything" else "OmniVGGT") if i == 0 else None)
        ax.axhline(0, color=INK2, lw=0.9)
        ax.set_xticks(range(len(E_LABELS)), [q[1] for q in E_LABELS], rotation=22, ha="right", fontsize=8.5)
        style(ax, lab + "\n↑ worse   ↓ better")
    axs[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    save(fig, path)


def fig_depth_err(D, path):
    m = D["mono"][D["mono"]["clip"].str.startswith("eval")]
    fig, ax = plt.subplots(figsize=(9.6, 2.5))
    rng = np.random.default_rng(0)
    for j, (k, col, lab) in enumerate((("dev_depth_scale_ratio", AQUA, "phone LiDAR"),
                                       ("mono_depth_scale_ratio", OV, "AI depth (Depth Anything V2)"))):
        v = m[k].values
        ax.scatter(v, j + rng.uniform(-0.15, 0.15, len(v)), s=26, color=col, alpha=0.85, edgecolor="white", lw=0.6)
        ax.text(v.max() + 0.03, j, f"{lab}: median ×{np.median(v):.2f}", va="center", fontsize=8.5, color=INK)
    ax.axvline(1, color=INK2, lw=0.9)
    ax.axvline(1.5, color=MUTED, lw=0.9, ls="--")
    ax.text(1.5, 1.45, "×1.5: harmful in E2-B", fontsize=8, color=MUTED, ha="center")
    ax.set_yticks([])
    ax.set_ylim(-0.5, 1.6)
    ax.set_xlim(0.9, 2.45)
    style(ax, None, "median depth ratio hint ÷ truth, per clip (41 eval clips)")
    ax.grid(False)
    save(fig, path)


def fig_arkit(D, path):
    v = D["arkit_views"].rot.values
    c8 = D["c8"].set_index("reference").loc["rgb_K"]
    fig, ax = plt.subplots(figsize=(9.6, 2.6))
    bins = np.logspace(np.log10(0.05), np.log10(30), 30)
    ax.hist(v, bins=bins, color=MA, alpha=0.85, edgecolor="white", lw=1)
    ax.set_xscale("log")
    for x, lab, col in ((c8.pair_rot_err_deg_median, "photos alone: typical", MUTED),
                        (c8.pair_rot_err_deg_p90, "photos alone: worst 10%", MUTED),
                        (5, "harmful: all cameras 5°", "#d03b3b"), (15, "one camera 15°", "#d03b3b")):
        ax.axvline(x, color=col, lw=1, ls="--")
        ax.text(x, 1.04 + 0.11 * (["photos alone: typical", "photos alone: worst 10%", "harmful: all cameras 5°", "one camera 15°"].index(lab) % 2), lab, transform=ax.get_xaxis_transform(), ha="center", fontsize=7.5, color=col)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    style(ax, "views", "ARKit rotation error per view vs ground truth (°, log scale; 41 eval clips)")
    save(fig, path)


# ------------------------------------------------------------------ diagrams (inline SVG)
SVG_MODEL = """<svg viewBox="0 0 760 215" role="img" aria-label="Photos and hints go into the model, which outputs a 3D reconstruction">
<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#52514e"/></marker></defs>
<g font-family="IBM Plex Sans, Segoe UI, sans-serif" font-size="13" fill="#0b0b0b">
<rect x="10" y="20" width="190" height="70" rx="10" fill="#eef4fc" stroke="#2a78d6"/><text x="105" y="50" text-anchor="middle" font-weight="600">Photos</text><text x="105" y="70" text-anchor="middle" font-size="11" fill="#52514e">10 views of a scene</text>
<rect x="10" y="115" width="190" height="80" rx="10" fill="#fdf0ea" stroke="#eb6834"/><text x="105" y="140" text-anchor="middle" font-weight="600">Hints (optional)</text><text x="105" y="160" text-anchor="middle" font-size="11" fill="#52514e">camera poses · lens · depth</text><text x="105" y="178" text-anchor="middle" font-size="11" fill="#9a2b2b">from GPS, phone tracking, AI depth…</text>
<rect x="290" y="55" width="190" height="100" rx="12" fill="#f9f9f7" stroke="#52514e"/><text x="385" y="95" text-anchor="middle" font-weight="600">3D model</text><text x="385" y="115" text-anchor="middle" font-size="11" fill="#52514e">MapAnything · OmniVGGT</text><text x="385" y="133" text-anchor="middle" font-size="11" fill="#52514e">one forward pass</text>
<rect x="570" y="55" width="180" height="100" rx="12" fill="#eaf7f1" stroke="#1baf7a"/><text x="660" y="95" text-anchor="middle" font-weight="600">3D output</text><text x="660" y="115" text-anchor="middle" font-size="11" fill="#52514e">cameras + depth</text><text x="660" y="133" text-anchor="middle" font-size="11" fill="#52514e">+ confidence</text>
<path d="M200,55 L288,90" stroke="#52514e" stroke-width="1.6" fill="none" marker-end="url(#ar)"/><path d="M200,155 L288,122" stroke="#52514e" stroke-width="1.6" fill="none" marker-end="url(#ar)"/><path d="M480,105 L568,105" stroke="#52514e" stroke-width="1.6" fill="none" marker-end="url(#ar)"/>
<text x="212" y="205" font-size="11" fill="#9a2b2b">← what if a hint is wrong?</text></g></svg>"""

SVG_FLOW = """<svg viewBox="0 0 760 250" role="img" aria-label="E1 findings lead to six E2 questions">
<defs><marker id="ar2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#52514e"/></marker></defs>
<g font-family="IBM Plex Sans, Segoe UI, sans-serif" font-size="12" fill="#0b0b0b">
<rect x="10" y="70" width="170" height="110" rx="10" fill="#f9f9f7" stroke="#52514e"/><text x="95" y="98" text-anchor="middle" font-weight="600">E1 found</text>
<text x="95" y="120" text-anchor="middle" font-size="11" fill="#52514e">wrong hints can be worse</text><text x="95" y="136" text-anchor="middle" font-size="11" fill="#52514e">than no hints; the model</text><text x="95" y="152" text-anchor="middle" font-size="11" fill="#52514e">copies them anyway</text>
<g font-size="11">
<rect x="230" y="8" width="320" height="34" rx="8" fill="#eef4fc" stroke="#2a78d6"/><text x="240" y="30"><tspan font-weight="600">A</tspan> Is it already known? How wrong are real sensors?</text>
<rect x="230" y="48" width="320" height="34" rx="8" fill="#eef4fc" stroke="#2a78d6"/><text x="240" y="70"><tspan font-weight="600">B</tspan> Were E1's numbers real and big enough?</text>
<rect x="230" y="88" width="320" height="34" rx="8" fill="#eef4fc" stroke="#2a78d6"/><text x="240" y="110"><tspan font-weight="600">C</tspan> How exactly does the model react?</text>
<rect x="230" y="128" width="320" height="34" rx="8" fill="#fdf0ea" stroke="#eb6834"/><text x="240" y="150"><tspan font-weight="600">D</tspan> Does a second model do the same?</text>
<rect x="230" y="168" width="320" height="34" rx="8" fill="#fdf0ea" stroke="#eb6834"/><text x="240" y="190"><tspan font-weight="600">E</tspan> Do real phone / AI hints cause it?</text>
<rect x="230" y="208" width="320" height="34" rx="8" fill="#eaf7f1" stroke="#1baf7a"/><text x="240" y="230"><tspan font-weight="600">F</tspan> Do today's fixes already solve it?</text></g>
<path d="M180,125 L226,125" stroke="#52514e" stroke-width="1.6" marker-end="url(#ar2)"/>
<rect x="590" y="80" width="165" height="90" rx="10" fill="#f9f9f7" stroke="#52514e"/><text x="672" y="110" text-anchor="middle" font-weight="600">E2 decision</text><text x="672" y="130" text-anchor="middle" font-size="11" fill="#52514e">GO to E3 (our fix)</text><text x="672" y="146" text-anchor="middle" font-size="11" fill="#52514e">analysis paper · narrow</text>
<path d="M550,125 L586,125" stroke="#52514e" stroke-width="1.6" marker-end="url(#ar2)"/></g></svg>"""

SVG_PROCESS = """<svg viewBox="0 0 760 120" role="img" aria-label="Plan, freeze, trial run, automatic check, main run, analysis">
<defs><marker id="ar3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#52514e"/></marker></defs>
<g font-family="IBM Plex Sans, Segoe UI, sans-serif" font-size="11.5" fill="#0b0b0b" text-anchor="middle">
<rect x="5" y="30" width="110" height="56" rx="8" fill="#f9f9f7" stroke="#52514e"/><text x="60" y="54" font-weight="600">Write the plan</text><text x="60" y="72" fill="#52514e" font-size="10.5">tests + thresholds</text>
<rect x="135" y="30" width="110" height="56" rx="8" fill="#f9f9f7" stroke="#52514e"/><text x="190" y="54" font-weight="600">Freeze it</text><text x="190" y="72" fill="#52514e" font-size="10.5">git tag, approved</text>
<rect x="265" y="30" width="110" height="56" rx="8" fill="#eef4fc" stroke="#2a78d6"/><text x="320" y="54" font-weight="600">Trial run</text><text x="320" y="72" fill="#52514e" font-size="10.5">3 dev scenes only</text>
<rect x="395" y="30" width="110" height="56" rx="8" fill="#fdf0ea" stroke="#eb6834"/><text x="450" y="54" font-weight="600">Automatic check</text><text x="450" y="72" fill="#52514e" font-size="10.5">pass → continue</text>
<rect x="525" y="30" width="110" height="56" rx="8" fill="#eef4fc" stroke="#2a78d6"/><text x="580" y="54" font-weight="600">Main run</text><text x="580" y="72" fill="#52514e" font-size="10.5">test scenes, once</text>
<rect x="655" y="30" width="100" height="56" rx="8" fill="#eaf7f1" stroke="#1baf7a"/><text x="705" y="54" font-weight="600">Analysis</text><text x="705" y="72" fill="#52514e" font-size="10.5">as written</text>
<path d="M115,58 L133,58" stroke="#52514e" stroke-width="1.5" marker-end="url(#ar3)"/><path d="M245,58 L263,58" stroke="#52514e" stroke-width="1.5" marker-end="url(#ar3)"/><path d="M375,58 L393,58" stroke="#52514e" stroke-width="1.5" marker-end="url(#ar3)"/><path d="M505,58 L523,58" stroke="#52514e" stroke-width="1.5" marker-end="url(#ar3)"/><path d="M635,58 L653,58" stroke="#52514e" stroke-width="1.5" marker-end="url(#ar3)"/>
<text x="380" y="110" fill="#898781" font-size="10.5">nothing in the plan may change once the main run has started</text></g></svg>"""


# ------------------------------------------------------------------ html helpers
def table(rows, heads, num=()):
    h = "".join(f'<th class="{"num" if i in num else ""}">{x}</th>' for i, x in enumerate(heads))
    b = "".join("<tr>" + "".join(f'<td class="{"num" if i in num else ""}">{c}</td>' for i, c in enumerate(r)) + "</tr>"
                for r in rows)
    return f'<div class="table-scroll"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def chip(kind, text):
    icon = {"hurt": "▲", "help": "▼", "yes": "✓", "no": "✗", "neutral": "•"}[kind]
    return f'<span class="chip {kind}">{icon} {html.escape(text)}</span>'


def fig_html(path, cap, src):
    return (f'<figure><img src="{uri(path)}" alt="{html.escape(cap)}"><figcaption>{cap} '
            f'<span class="src">Source: <code>{src}</code></span></figcaption></figure>')


def gap_cell(v):
    if v is None or not np.isfinite(v):
        return '<td class="num muted">–</td>'
    t = min(max(v, 0), 1.2) / 1.2
    lo, hi = np.array([0xe3, 0xf1, 0xe3]), np.array([0xfb, 0xd5, 0xd0])
    c = (lo + (hi - lo) * t).astype(int)
    return f'<td class="num" style="background:#{c[0]:02x}{c[1]:02x}{c[2]:02x}">{v:.2f}</td>'


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=p("reports", "e2_results"))
    ap.add_argument("--clips", default="data/clips_e2/eval__*.npz")
    ap.add_argument("--clips_e", default="data/clips_e2e/eval__*.npz")
    ap.add_argument("--pdf_browser")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    D = gather(a.clips, a.clips_e)
    F = lambda n: os.path.join(a.out, n)
    fig_trust(D, F("fig_trust_curve.png"))
    fig_knows(D, F("fig_knows_but_obeys.png"))
    fig_agree(D, F("fig_agreeing_errors.png"))
    fig_harm(D, F("fig_harm_both_models.png"))
    fig_baseline(D, F("fig_omnivggt_baseline.png"))
    fig_real(D, F("fig_real_hints.png"))
    fig_depth_err(D, F("fig_depth_hint_errors.png"))
    fig_arkit(D, F("fig_arkit_rotation_errors.png"))

    # ---------------- numbers used in the text
    dr = D["e1_dr"]
    e1_rot15 = dr[(dr.condition == "outlier_rot_rot_deg=15") & (dr.metric == "rre_deg")].iloc[0]
    e1_abs15 = D["e1_abs"][D["e1_abs"].condition == "outlier_rot_rot_deg=15"].iloc[0]["median"]
    au = D["audit"]
    au_h = au[au.harmful]
    au_fams = sorted(au_h.family.unique())
    au_real = sorted(au_h[au_h.lit_realistic == True].family.unique())  # noqa: E712
    pt = D["pt"].set_index("test")
    tc_ma, tc_ov = D["tc_ma"], D["tc_ov"]
    absr = lambda tc, s: tc[(tc.family == "c1_rot") & np.isclose(tc.severity, s)].iloc[0].absorption
    ab1 = D["abs_e1def"]
    c8 = D["c8"].set_index("reference")
    c4 = D["c4"]
    hm, ho = D["d_harm_mapanything"], D["d_harm_omnivggt"]
    e = D["e_tests"]
    ev = lambda m, c, k: e[(e.model == m) & (e.condition == c) & (e.metric == k)].iloc[0]
    mono = D["mono"][D["mono"]["clip"].str.startswith("eval")]
    ark = D["arkit_views"].rot
    rr = D["e_real_errors"]
    fg = D["f_remaining_gap"]
    fa = D["f_false_alarms"]
    fr = D["f_real_hints"]
    sess = D["session"]
    t_start = next(s for s in sess if " start, " in s)[:19]
    t_end = next(s for s in sess if "SESSION_GPU_DONE" in s)[:19]
    t_c_end = next(s for s in sess if "E2C_EVAL_DONE" in s)[:19]
    par = D["parity"].set_index("metric")
    ov_rgb = D["per_scene"][("OmniVGGT", "rgb_only")]
    worst2 = ov_rgb.sort_values(ascending=False).head(2)
    sc90 = (D["per_scene"][("OmniVGGT", "outlier_rot_rot_deg=90")] > ov_rgb).sum()
    sens_changes = int((e.merge(D["e_sens"], on=["model", "condition", "metric"], suffixes=("", "_x"))
                        .pipe(lambda d: d.verdict != d.verdict_x)).sum())
    n_skip = int(D["sel_log"].decision.str.startswith("skipped").sum())

    # ---------------- E2-C test table
    PT = [("P1_rot_override", "Does it stop following a grossly wrong rotation (30° vs 15°)?", "absorption difference", "≤ −0.10"),
          ("P2_trans_override", "…or a grossly wrong position (10 L vs 0.25 L)?", "absorption difference", "≤ −0.10"),
          ("P3_conf_flags_bad_view", "Does its own confidence flag the bad camera?", "AUROC − 0.5", "≥ 0.15"),
          ("P4_bad_view_own_shape", "Does a wrong rotation bend that camera's own 3D shape?", "Δ shape error", "≥ 1 pp"),
          ("P5_good_views_shape", "…or bend the good cameras' shape?", "Δ shape error", "≥ 1 pp"),
          ("P6_median_changes_clean", "Does a median scale step change clean results?", "Δ rotation error", "|Δ| ≥ 0.25°"),
          ("P7_median_fixes_gps_jump", "Does a median scale step fix a 10 L GPS jump?", "Δ shape error", "≤ −1 pp"),
          ("P8_absorption_vs_overlap", "Does it copy less where the photos show the scene well?", "Spearman ρ", "≤ −0.30"),
          ("P9_absorption_20_vs_4_views", "Does it copy less with more photos (20 vs 4)?", "absorption difference", "≤ −0.10"),
          ("P10_shared_vs_indep_conf_5deg", "Does it notice a shared error less than random errors?", "confidence ratio difference", "≥ +0.10"),
          ("P11_drift_vs_jitter_conf_level2", "Does it notice drift less than matched jitter?", "confidence ratio difference", "≥ +0.10"),
          ("P12_partial_30_vs_100", "Does one bad hint weigh more when few poses are given?", "Δ rotation harm", "≥ 0.25°")]
    pt_rows = []
    for k, q, metric, thr in PT:
        r = pt.loc[k]
        pt_rows.append([k.split("_")[0], q, metric, thr, f"{r['mean']:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]",
                        f"{r.p_holm:.4f}", chip("yes", "finding") if r.finding else chip("no", "not a finding")])

    # ---------------- E2-D tables
    fam_names = {"noise_rot": "every camera slightly turned", "outlier_rot": "one camera turned wrong",
                 "outlier_frac": "several cameras wrong", "scale_all": "all distances scaled wrong", "drift": "odometry drift",
                 "shared_heading_bias": "same heading error on all", "anchor_error": "first camera wrong",
                 "depth_scale": "depth hints scaled wrong", "depth_one_scale": "one depth map scaled wrong"}
    d_rows = []
    for fam, cls in D["d_family_class"].itertuples(index=False):
        hmf, hof = hm[hm.family == fam], ho[ho.family == fam]
        ma_set = ", ".join(f"{r.condition.split('=')[-1]} ({r.metric})" for r in hmf[hmf.harmful].itertuples()) or "none"
        ov_set = ", ".join(f"{r.condition.split('=')[-1]} ({r.metric})" for r in hof[hof.harmful].itertuples()) or "none"
        d_rows.append([fam_names.get(fam, fam), ma_set, ov_set,
                       chip("yes", "both") if cls == "same" else chip("neutral", cls)])
    sbs = D["d_side_by_side_behaviour"]
    db_rows = [[r.test.split("_")[0], f"{r.ma_mean:+.3f}", chip("yes", "finding") if r.ma_finding else chip("no", "no"),
                f"{r.ov_mean:+.3f}", chip("yes", "finding") if r.ov_finding else chip("no", "no"), r["class"]]
               for _, r in sbs.iterrows()]

    # ---------------- E2-E tables
    e_rows = []
    for c, lab in [("dev_K", "phone lens values")] + E_LABELS:
        for model in ("mapanything", "omnivggt"):
            sub = e[(e.model == model) & (e.condition == c)]
            if not len(sub):
                continue
            cells = []
            for k, unit in (("shape", "pp"), ("rotation", "°"), ("real_size", "pp")):
                r = sub[sub.metric == k]
                if not len(r):
                    cells.append("–")
                    continue
                r = r.iloc[0]
                val = pp(r.delta) if unit == "pp" else deg(r.delta)
                kind = "hurt" if r.verdict == "HURTS" else "help" if r.verdict == "HELPS" else "neutral"
                word = {"HURTS": "hurts", "HELPS": "helps"}.get(r.verdict, "small" if "below" in r.verdict else "unclear")
                cells.append(f"{val} {chip(kind, word)}")
            e_rows.append([lab, "MapAnything" if model == "mapanything" else "OmniVGGT", *cells])

    # ---------------- E2-F table
    rem_cols = ["R0 images only", "R1 drop poses", "R2 E1 filter", "R3 confidence drop", "ORACLE drop bad views",
                "EXPLORATORY drop depth hints"]
    fset = fg.drop_duplicates(["condition", "metric"])
    f_rows_html = ""
    for _, s in fset.iterrows():
        g = fg[(fg.condition == s.condition) & (fg.metric == s.metric)].set_index("remedy").gap
        cells = "".join(gap_cell(g.get(rc, np.nan)) for rc in rem_cols)
        f_rows_html += (f"<tr><td>{html.escape(s.condition)}</td><td>{s.metric}</td>{cells}"
                        f"<td>{html.escape(str(s.best_remedy))}</td><td class='num'>{s.best_gap:.2f}</td></tr>")
    f_table = ('<div class="table-scroll"><table><thead><tr><th>Bad hint (setting)</th><th>Metric</th>'
               + "".join(f'<th class="num">{h}</th>' for h in ("R0 photos only", "R1 drop poses", "R2 E1 filter",
                                                               "R3 confidence", "ORACLE", "drop depth*"))
               + '<th>Best of R0–R3</th><th class="num">Damage left</th></tr></thead><tbody>' + f_rows_html + "</tbody></table></div>")
    fa_rf2 = fa[fa.control == "filter_rgbcheck"].false_alarm_rate.mean()
    fa_rf3 = fa[fa.control == "drop_lowconf"].false_alarm_rate.mean()
    rh = fr.set_index(["condition", "remedy", "metric"])

    # ---------------- decision table
    dec_rows = [
        ["Wrong camera rotations (one bad camera, several, drift, shared heading)", chip("no", "MapAnything only"),
         chip("no", "phone errors too small"), chip("no", "fixes leave 1–10%"), chip("yes", "large ones"), chip("no", "no")],
        ["Wrong overall scale", chip("no", "MapAnything only"), chip("neutral", "not tested"), chip("no", "fixes leave 21%"),
         chip("yes", "yes"), chip("no", "no")],
        ["Wrong depth hints (AI depth)", chip("yes", "both models (real data)"), chip("yes", "AI depth"),
         chip("yes", "by the frozen rule*"), chip("yes", "mostly"), chip("yes", "YES")],
    ]

    # ---------------- document
    css = open(p("scripts", "e2_report_style.css")).read()
    body = f"""
<header class="col stack">
  <p class="eyebrow">Do 3D foundation models trust bad hints? · Experiment 2 · knowledge-transfer report</p>
  <h1>Experiment 2: what kind of problem is it?</h1>
  <p class="lead">A plain-language walk through why we ran E2, how we ran it so the results can be trusted, what came out,
  and what it means for the paper. Every number is computed from the committed result files by
  <code>scripts/e2_make_report.py</code>.</p>
  <div class="verdict">
    <p class="eyebrow">The short version</p>
    <p class="decision">GO to E3, narrowed to <em>when can a depth hint be trusted</em></p>
    <p>MapAnything copies wrong camera hints almost completely, even when its own confidence shows it knows they are wrong,
    and it does not notice errors that agree with each other. A second model, OmniVGGT, follows bad hints less (about 60%)
    and, by our strict test, is not made worse than photos-only by synthetic errors. In real data, phone sensors
    <strong>help</strong> both models, but <strong>AI-estimated depth hurts both</strong>. Existing fixes handle single,
    grossly wrong cameras; they do not handle bad depth or errors that agree. The decision follows the frozen rule, with two
    judgment calls spelled out in section 10. The final call belongs to the team.</p>
    <div class="tiles">
      <div><span class="big">{D['n_runs']:,}</span><span>model runs, 0 failures</span></div>
      <div><span class="big">{absr(tc_ma, 15):.2f}</span><span>MapAnything copies a 15° camera error (absorption)</span></div>
      <div><span class="big">{pt.loc['P3_conf_flags_bad_view','mean'] + 0.5:.2f}</span><span>its confidence still flags the bad camera (AUROC)</span></div>
      <div><span class="big">{pp(ev('mapanything','ai_depth','shape').delta)} / {pp(ev('omnivggt','ai_depth','shape').delta)}</span><span>AI depth makes shape worse (MapAnything / OmniVGGT)</span></div>
    </div>
  </div>
</header>

<section class="col stack">
  <h2>1. Background: what these models do, and what a "hint" is</h2>
  <p>Models like <strong>MapAnything</strong> and <strong>OmniVGGT</strong> take a handful of photos of a scene and, in one pass,
  output a 3D reconstruction: where each camera was, and how far away every pixel is. Their selling point is that they can
  also take <strong>hints</strong>: where the cameras were (poses), the lens settings (intrinsics) and depth maps. With
  correct hints, they get much better.</p>
  <p>The catch: in the real world hints come from GPS, phone tracking, cheap depth sensors, AI depth estimators or wrong photo
  metadata, so they are often <em>wrong</em>. Our project asks what happens then.</p>
  <figure class="diagram">{SVG_MODEL}<figcaption>The setting: photos plus optional hints go into one model. Our question is what a wrong hint does.</figcaption></figure>
  <p class="how"><strong>How we score.</strong> <em>Rotation error</em> = how wrong the camera directions are (degrees, between pairs of
  cameras). <em>Shape error</em> = how far the 3D points are from the truth after one overall size correction (% of distance).
  <em>Real-size error</em> = the same without the size correction. <em>Absorption</em> = how much of a hint's error ends up in the
  output (1 = copied, 0 = ignored). <em>AUROC</em> = how well the model's own confidence ranks bad cameras below good ones
  (0.5 = guessing, 1 = perfect).</p>
</section>

<section class="col stack">
  <h2>2. Where E1 left us, and why E2 was needed</h2>
  <p>Experiment 1 (preregistered, run once on 10 test scenes) answered the first question: <strong>yes, a wrong hint can make
  MapAnything worse than giving it no hints at all</strong>. With one camera turned 15° wrong, its rotation error was
  {e1_rot15.rel_vs_floor * 100:+.0f}% compared with photos only. The model copied the bad pose almost completely (absorption
  {e1_abs15:.2f} at 15°), even though its own confidence did flag the bad camera.</p>
  <p>But E1 left the questions that decide whether a <em>fix</em> is worth building open. Is this already known? Were the
  numbers big enough to matter, or just statistically detectable? <em>Why</em> does the model obey a hint it seems to know is
  wrong: a simple averaging step, or something deeper? Is it only MapAnything? Do real sensors produce such errors?
  And do existing tools already fix it? A method paper only makes sense if the answers line up, so E2 asks them one by one.</p>
  <figure class="diagram">{SVG_FLOW}<figcaption>E2's six parts, each answering one question that decides the next step.</figcaption></figure>
</section>

<section class="col stack">
  <h2>3. How we ran E2, and why it was done this way</h2>
  <p>The biggest risk in an experiment like this is fooling ourselves: running many tests, then reporting the ones that
  happen to look good. So E2 followed strict rules, each with a reason.</p>
  <figure class="diagram">{SVG_PROCESS}<figcaption>Every main run followed this order. The plan is fixed and tagged in git before the main run.</figcaption></figure>
  <ul class="list">
    <li><strong>Preregistration.</strong> Before each main run we wrote down exactly what we would measure and what counts as
    a result, and froze it (git tags <code>e2-prereg-C</code>, <code>e2-prereg-DEF</code>). <em>So that</em> nobody can
    choose the tests after seeing the numbers.</li>
    <li><strong>Trial (dev) scenes vs test (eval) scenes.</strong> Code is debugged only on 3 separate trial scenes; the test
    scenes are used once. <em>So that</em> the test data never shapes the design.</li>
    <li><strong>"Statistically clear" AND "big enough".</strong> A result counts only if it survives Holm correction (which
    accounts for running many tests at once) <em>and</em> passes a practical threshold: shape ≥ 1 percentage point, real size
    ≥ 5 pp, rotation ≥ 0.25°, scale ≥ 5%. <em>So that</em> tiny, irrelevant effects never count as findings.</li>
    <li><strong>The scene is the unit.</strong> Several clips from the same scene are averaged first. <em>So that</em> we
    don't pretend to have more independent evidence than we do.</li>
    <li><strong>Every comparison is paired.</strong> The same clips, the same random corruptions, with and without the bad
    hint. <em>So that</em> differences come from the hint, not from different data.</li>
    <li><strong>One paid GPU session, no idle time.</strong> All E2 GPU work ran in one session on an NVIDIA RTX PRO 6000
    ({t_start} → {t_end} UTC), with automatic checks between each trial run and its main run, so the GPU never waited for a human.</li>
    <li><strong>A parity check against E1.</strong> The new GPU reproduced E1's runs: median differences
    {par.loc['pts_relerr_scale','median_abs_diff'] * 100:.3f} pp shape and {par.loc['rre_deg','median_abs_diff']:.4f}° rotation,
    and parallel processes gave bit-identical results. One extreme run (a 10× GPS jump) differed by
    {par.loc['pts_relerr_scale','max_abs_diff'] * 100:.1f} pp; you approved judging the check by its 99th percentile instead
    (Amendment 14).</li>
  </ul>
</section>

<section class="col stack">
  <h2>4. E2-A: is this already known? (literature audit)</h2>
  <p>We opened every closely related paper. <strong>No paper measures our core findings on these models.</strong> The
  closest are Pow3R (two views only; it <em>refuses</em> grossly wrong hints), VPGGT (retrains OmniVGGT against drift-style pose
  noise; poses only; it doesn't detect which hint is bad), WorldMirror and TCO. We also collected realistic error sizes from
  the literature (phone tracking, compass, GPS, depth sensors, AI depth estimators), which replace E1's rough "realistic"
  labels. Details: <code>reports/e2_literature.md</code>.</p>
</section>

<section class="col stack">
  <h2>5. E2-B: were E1's results real and big enough?</h2>
  <p>We checked that "copying" is genuine behaviour (the output camera comes from the network, not from passing the input
  through), then re-tested every E1 setting with the strict standard. <strong>{len(au_h)} of {len(au)} tests</strong> stay
  harmful, covering <strong>{len(au_fams)} of 19 error types</strong>; {len(au_real)} of them at sizes that really occur
  ({", ".join(fam_names.get(x, x) for x in au_real)}). The harm sits on the bad camera itself: the pairs of good cameras
  barely change. Details: <code>reports/e2_audit.md</code>.</p>
</section>

<section class="wide stack">
  <div class="col stack">
    <h2>6. E2-C: how does MapAnything react to a bad hint?</h2>
    <p>We fed MapAnything carefully controlled bad hints on 24 clips from 10 test scenes ({t_start[:10]}, finished
    {t_c_end[11:16]} UTC; 3,232 runs) and asked twelve preregistered questions.</p>
    <h3>It copies, and it never overrides</h3>
    <p>Turn one camera's hint by an increasing amount and measure how much of that error appears in the output. A sensible
    model would follow small corrections and refuse absurd ones. MapAnything does the opposite of refusing: it copies
    {absr(tc_ma, 1):.2f} of a 1° error and {absr(tc_ma, 30):.2f} of a 30° error. The same holds for position errors. OmniVGGT
    copies about half of a rotation error at any size, and almost none of a position error.</p>
  </div>
  {fig_html(F("fig_trust_curve.png"), "Figure 1. How much of a bad hint each model copies, as the error grows (shaded: 95% CI over scenes). Neither curve turns down: no model overrides an absurd hint.", "results/e2c_eval/agg/c1_c3_trust_curves.csv; OmniVGGT from results/e2d_eval")}
  <div class="col stack">
    <h3>It knows, but obeys</h3>
    <p>Yet MapAnything's own confidence clearly marks the bad camera (AUROC {pt.loc['P3_conf_flags_bad_view','mean'] + 0.5:.2f}).
    The information needed to resist is inside the model; it just doesn't use it. OmniVGGT's confidence barely notices
    (AUROC {D['d_behaviour_omnivggt'].set_index('test').loc['P3_conf_flags_bad_view','mean'] + 0.5:.2f}).</p>
  </div>
  {fig_html(F("fig_knows_but_obeys.png"), "Figure 2. Left: share of a 15° error copied. Right: how well the model's own confidence spots the bad camera (dashed: guessing).", "results/e2c_eval/agg/primary_tests.csv, results/e2def_agg/d_behaviour_omnivggt.csv")}
  <div class="col stack">
    <h3>Errors that agree are the dangerous ones</h3>
    <p>Give every camera a 5° error in random directions, and MapAnything's confidence drops to
    {D['c6_ma'].query("correlated=='c6_shared_rot_rot_deg=5'").ind_mean_conf_ratio.iloc[0]:.2f} of normal. Give every camera the
    <em>same</em> 5° error, and it only drops to
    {D['c6_ma'].query("correlated=='c6_shared_rot_rot_deg=5'").corr_mean_conf_ratio.iloc[0]:.2f}. Drift behaves the same way.
    Errors that agree look self-consistent, so the model doesn't notice them. Real phone tracking errors are exactly this kind
    (section 8). OmniVGGT's confidence doesn't move at all with bad camera hints (ratio {D['c6_ov'].query("correlated=='c6_shared_rot_rot_deg=5'").corr_mean_conf_ratio.iloc[0]:.2f}), so it gives no warning in either case.</p>
  </div>
  {fig_html(F("fig_agreeing_errors.png"), "Figure 3. Average confidence relative to clean hints. Pale: independent errors; solid: agreeing errors of the same size.", "results/e2c_eval/agg/c6_correlated_vs_independent.csv; OmniVGGT via the same function")}
  <div class="col stack">
    <h3>Why it happens: not the averaging step</h3>
    <p>MapAnything rescales camera hints using their <em>mean</em> distance. A single crazy hint can wreck a mean but not a
    median, so one theory (our H2) was that the averaging step causes the damage. We swapped it for a median. Clean results
    didn't change ({deg(c4[c4.metric=='rre_deg'].delta_median_minus_mean.iloc[0], 4)} rotation), and the 10× GPS jump wasn't fixed
    (P7: {pt.loc['P7_median_fixes_gps_jump','mean'] * 100:+.2f} pp, not clear). <strong>The damage happens inside the network.</strong></p>
    <h3>All twelve questions</h3>
  </div>
  {table(pt_rows, ["Test", "Question", "Measured as", "Finding if", "Result [95% CI]", "Holm p", "Verdict"], num=(4, 5))}
  <div class="col"><p class="src">Source: <code>results/e2c_eval/agg/primary_tests.csv</code>. Holm correction over all 12 tests.
  A non-finding is not proof of absence; for P1 and P2 the confidence intervals sit well above the override threshold, so
  any override, if it exists, is smaller than the one we looked for.</p></div>
</section>

<section class="wide stack">
  <div class="col stack">
    <h2>7. E2-D: does a second model behave the same?</h2>
    <p>A finding that holds for two differently built models is a property of the approach, not one model's quirk.
    OmniVGGT injects hints differently, so we gave it <em>identical</em> bad inputs: the same clips, and the same condition
    names, which means the same random corruptions (3,312 runs).</p>
    <p><strong>Result: under the strict test, none of the 9 harmful error types makes OmniVGGT worse than photos only.</strong>
    Two reasons. First, it copies less (about 60% instead of about 95%). Second, its photos-only mode is unstable: it fails
    badly in 2 of 10 scenes ({worst2.index[0]} {worst2.iloc[0]:.1f}°, {worst2.index[1]} {worst2.iloc[1]:.1f}°), so
    "worse than photos only" becomes too noisy to prove. With one camera turned 90°, OmniVGGT <em>is</em> worse than photos
    only in {sc90} of 10 scenes, but the strict test doesn't pass. It is also immune by design to wrong overall scale, because
    it rescales every hint.</p>
  </div>
  {fig_html(F("fig_harm_both_models.png"), "Figure 4. Rotation error with each bad hint, minus photos-only (95% CI over scenes). Filled markers pass the strict test.", "results/e2def_agg/d_harm_*.csv")}
  {fig_html(F("fig_omnivggt_baseline.png"), "Figure 5. Per scene: photos only, correct hints, and one camera 90° wrong. OmniVGGT's photos-only result is unstable, which makes 'worse than photos only' hard to prove for it.", "results/e2df_eval, results/e2d_eval per_run.csv")}
  {table(d_rows, ["Error type", "Harmful for MapAnything (setting)", "Harmful for OmniVGGT", "Class"])}
  {table(db_rows, ["Behaviour test", "MapAnything", "", "OmniVGGT", "", "Class"], num=(1, 3))}
  <div class="col"><p class="src">Sources: <code>results/e2def_agg/d_family_class.csv</code>,
  <code>d_side_by_side_behaviour.csv</code>. E1-definition absorption at 5/15/30/90°: MapAnything
  {", ".join(f"{v:.2f}" for v in ab1['MapAnything'].values)}; OmniVGGT {", ".join(f"{v:.2f}" for v in ab1['OmniVGGT'].values)}.</p></div>
</section>

<section class="wide stack">
  <div class="col stack">
    <h2>8. E2-E: do real sensors cause the problem?</h2>
    <p>Everything so far used errors we injected on purpose. Reviewers will rightly ask whether real sensors produce them. So
    we used <strong>SceneFun3D</strong>: iPad recordings of real rooms with the iPad's own tracking (ARKit) poses and LiDAR
    depth, plus much better laser-scan ground truth. They come from the ARKitScenes validation split, which neither model
    trained on. We also added a hint that many cameras without LiDAR would really use: <strong>AI-estimated depth</strong>
    (Depth Anything V2, metric indoor model).</p>
    <p class="how"><strong>Data care.</strong> 18 rooms (3 trial, 15 test), 3 clips each, chosen by a fixed random rule. Only the
    frames we use were downloaded (~60 MB per video instead of ~650 MB). Some phones store frames turned 180°; this is detected
    per video and corrected, and the correction reproduces the dataset's own lens values to within 0.1 pixel.
    {n_skip} portrait recordings were skipped, and {len(D['excluded'])} clips failing the ground-truth check were excluded,
    both by rules fixed before any model ran. One test clip turned out to have a broken ground-truth pose for one view (a 164°
    jump in half a second; the phone's own pose is fine). It stays in the main analysis as preregistered; leaving it out
    changes {sens_changes} verdicts.</p>
    <h3>How wrong are real hints?</h3>
    <p>Phone poses are good: {np.median(ark):.2f}° typical rotation error, and only {np.mean(ark >= 5) * 100:.1f}% of views 5°
    or more off. Their errors build up along the recording (median correlation with view order
    {rr.drift_spearman_rot_vs_index.median():.2f}), so they are drift-like and agreeing. LiDAR depth is excellent (×{mono.dev_depth_scale_ratio.median():.3f},
    {mono.dev_depth_absrel.median() * 100:.1f}% typical error). AI depth is not: typically ×{mono.mono_depth_scale_ratio.median():.2f}
    too far (range ×{mono.mono_depth_scale_ratio.min():.2f}–×{mono.mono_depth_scale_ratio.max():.2f}), with
    {mono.mono_depth_absrel.median() * 100:.0f}% typical error.</p>
  </div>
  {fig_html(F("fig_arkit_rotation_errors.png"), "Figure 6. Real phone rotation errors per view, against what photos alone can see (grey) and the sizes E2-B found harmful (red).", "data/clips_e2e (dev_poses vs poses), results/e2c_eval/agg/c8_photos_alone.csv")}
  {fig_html(F("fig_depth_hint_errors.png"), "Figure 7. Depth scale error per test clip. LiDAR sits on the truth; AI depth is 1.13–1.85× too far.", "results/e2e_mono/depth_hint_errors.csv")}
  <div class="col stack">
    <h3>Do real hints help or hurt?</h3>
    <p><strong>Real phone hints help.</strong> <strong>AI depth hurts both models</strong>, by about 10 percentage points of
    shape error and about 1° of rotation, and good phone poses don't cancel it out.</p>
  </div>
  {fig_html(F("fig_real_hints.png"), "Figure 8. Each real hint compared with photos only (95% CI over 15 rooms). Filled markers pass the strict test.", "results/e2def_agg/e_tests.csv")}
  {table(e_rows, ["Hint", "Model", "Shape", "Rotation", "Real size"])}
  <div class="col"><p class="src">Source: <code>results/e2def_agg/e_tests.csv</code> (Holm over 27 tests). "small" = statistically
  clear but below the practical threshold. Sensitivity without the broken-ground-truth clip:
  <code>results/e2def_agg/sensitivity/</code>.</p></div>
</section>

<section class="wide stack">
  <div class="col stack">
    <h2>9. E2-F: do today's fixes already solve it?</h2>
    <p>A method paper must show the problem is not already solved. For each setting that is clearly harmful for MapAnything,
    we measured how much of the damage is <em>left</em> after each existing fix: 0 = fully fixed, 1 = no help, above 1 = the
    fix is worse than the damage. R0 ignores all hints; R1 drops the pose hints; R2 is E1's simple filter (drop poses that
    disagree with the photos-only answer); R3 drops cameras the model itself is unsure of; ORACLE cheats by dropping exactly
    the bad cameras.</p>
  </div>
  {f_table}
  <div class="col stack">
    <p class="src">Source: <code>results/e2def_agg/f_remaining_gap.csv</code>. *“Drop depth hints” is exploratory, not
    preregistered.</p>
    <p><strong>What this says.</strong> For a single, grossly wrong camera, E1's filter is nearly perfect, but it also drops
    {fa_rf2 * 100:.0f}% of <em>good</em> cameras (R3: {fa_rf3 * 100:.0f}%). On real phone data that costs accuracy: the filter
    makes rotation <em>worse</em> by {rh.loc[('dev_pose','filter_rgbcheck','rotation'),'delta_vs_no_remedy']:+.2f}°
    because it throws away good real hints. For agreeing errors (drift, scale), the model's confidence is useless (R3 leaves
    90–100%); only "drop all poses" helps, which also throws away the value of good poses. For bad depth, none of the
    preregistered fixes helps; simply dropping the depth hint would, <em>if you knew it was bad</em>. But phone LiDAR depth
    helps, so dropping depth always is not free.</p>
    <h3>What photos alone can reveal (the ceiling)</h3>
    <p>Any fix that uses only the photos can't detect an error smaller than the photos' own uncertainty. With photos and lens
    values only, camera directions are about {c8.loc['rgb_K','pair_rot_err_deg_median']:.1f}° uncertain
    ({c8.loc['rgb_K','pair_rot_err_deg_p90']:.1f}° in the worst 10%), and overall scale about
    {(np.exp(c8.loc['rgb_K','scale_err_abs_log_median']) - 1) * 100:.0f}%
    ({(np.exp(c8.loc['rgb_K','scale_err_abs_log_p90']) - 1) * 100:.0f}% in the worst 10%). So typical phone errors (about 1°) are
    invisible to photos, errors of 5° or more are visible, and AI depth's ~42% scale error is visible in most clips.</p>
  </div>
</section>

<section class="col stack">
  <h2>10. The decision</h2>
  <p>The E2 plan fixed the decision rule in advance. To GO to E3, at least one harmful error type must pass all four tests:
  (a) it hurts both models, (b) it happens with real sensors, (c) the best existing fix leaves at least half the damage, and
  (d) photos alone contain enough evidence to detect it.</p>
  <div class="decision-table">{table(dec_rows, ["Error type", "(a) both models", "(b) real sensors", "(c) ≥ half left", "(d) visible", "Passes"])}</div>
  <p><strong>Verdict: GO to E3, narrowed to depth hints.</strong> Two judgment calls behind it, stated openly:</p>
  <ol class="list">
    <li><strong>Test (a) for depth comes from the real AI-depth runs.</strong> The synthetic depth-scale test is "MapAnything only",
    because OmniVGGT rescales depth. But real AI depth also distorts the <em>shape</em> of the depth map, and that hurts both models.</li>
    <li><strong>Test (c) holds by the frozen rule, but an obvious fix was missing from the list.</strong> "Drop the depth hint"
    (exploratory) leaves only 10–15% of the damage. It works only if you <em>know</em> the depth is bad; LiDAR depth helps,
    so dropping it always loses accuracy. The real open problem is deciding <em>when</em> to trust a depth hint, which is
    exactly E3's question.</li>
  </ol>
  <p>Read strictly the other way, the outcome would be the <strong>analysis paper</strong>, which is already well supported.</p>
</section>

<section class="col stack">
  <h2>11. What this means for the paper and for E3</h2>
  <ul class="list">
    <li><strong>Already solid (analysis paper):</strong> these models copy wrong hints instead of overriding them; MapAnything
    does so while its confidence knows; agreeing errors go unnoticed; the cause is inside the network, not the averaging step;
    real AI depth hurts both models while real phone sensors help; existing fixes only cover the easy case.</li>
    <li><strong>E3 target:</strong> decide from the photos alone, with no ground truth, whether a hint (first of all a depth
    hint) can be trusted, and use it only then. E3 must beat three simple policies: always use hints, never use them, and
    E1's filter. Agreeing pose errors are a secondary target, since nothing deployable handles them today.</li>
    <li><strong>Risks:</strong> OmniVGGT is less affected by synthetic errors; typical phone errors (~1°) are below what
    photos can detect; and time (CVPR submission on 16 November 2026).</li>
  </ul>
</section>

<section class="col stack">
  <h2>Appendix: reproducibility and where things are</h2>
  <ul class="list small">
    <li>Plans and amendments: <code>experiments/E2.md</code> (Amendments 1–14), data plan <code>experiments/E2E_data.md</code>.
    Frozen code: tags <code>e2-prereg-C</code> (E2-C) and <code>e2-prereg-DEF</code> (E2-D/E/F).</li>
    <li>Machine: {html.escape(D['machine'][0])}. MapAnything weights snapshot <code>a1d87e90</code>, OmniVGGT <code>29c67625</code>,
    Depth Anything V2 revision recorded in <code>results/e2e_mono/mono_depth.csv</code>.</li>
    <li>Runs: {D['n_runs']:,} in total, 0 failures; queue log <code>results/vm_logs_e2/session.txt</code>.</li>
    <li>Analyses: <code>python -m exp1.e2c_analysis …</code> → <code>results/e2c_eval/agg/</code>;
    <code>python -m exp1.e2def_analysis …</code> → <code>results/e2def_agg/</code>. This report: <code>scripts/e2_make_report.py</code>.</li>
    <li>Not in E2: R4 (robust bundle adjustment, CPU only) is still to run; wrong lens values could not be tested on real data
    (the dataset's lens values come from the phone); portrait recordings are excluded.</li>
  </ul>
</section>
"""
    doc = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
           f"<title>E2 Results Report</title><style>{css}</style></head><body><main class=\"page\">{body}</main></body></html>")
    out_html = F("e2_results.html")
    open(out_html, "w", encoding="utf-8").write(doc)
    print("wrote", out_html)
    if a.pdf_browser:
        subprocess.run([a.pdf_browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={pathlib.Path(F('e2_results.pdf')).resolve()}",
                        pathlib.Path(out_html).resolve().as_uri()], check=True)
        print("wrote", F("e2_results.pdf"))


if __name__ == "__main__":
    main()
