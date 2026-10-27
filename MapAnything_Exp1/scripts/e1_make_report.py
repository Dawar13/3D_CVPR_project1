"""E1: build the report from the frozen-run outputs (no model, CPU only).

  python scripts/e1_make_report.py --run results/e1_eval \
      --demo results/e1_demo_pose results/e1_demo_other [--pdf_browser CHROME]

Reads <run>/agg/* (written by exp1.e1_analysis), <run>/per_run.csv, run_info.json and the
dev-clip demo folders. Writes:
  reports/e1_results.md            technical report (experiments/E1.md section 7)
  reports/e1_results/              figures, tables (csv), e1_results.html (offline), e1_results.pdf
Every number comes from those files.
"""

import argparse
import base64
import glob
import html
import json
import os
import pathlib
import shutil
import subprocess

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

NAMES = {
    "noise_rot": ("All cameras slightly turned", "jittery tracking", "°"),
    "noise_trans": ("All cameras slightly moved", "jittery tracking", "×L"),
    "outlier_rot": ("One camera turned wrong", "bad compass reading / flipped photo", "°"),
    "outlier_trans_small": ("One camera moved a little", "small GPS glitch", "×L"),
    "outlier_trans_large": ("One camera moved far", "GPS jump", "×L"),
    "outlier_frac": ("Several cameras wrong (15° + 0.25 L each)", "bad stretch of sensor data", "of views"),
    "scale_all": ("All distances scaled wrong", "wrong units / scale drift", "×"),
    "drift": ("Odometry drift along the path", "visual-inertial drift", "level"),
    "shared_heading_bias": ("Same heading error on every camera", "compass / gravity misalignment", "°"),
    "anchor_error": ("First (reference) camera wrong", "bad first frame", "level"),
    "focal_all": ("Wrong focal length, all photos (with poses)", "wrong EXIF / calibration", "×"),
    "focal_all_noposes": ("Wrong focal length, all photos (no poses)", "wrong EXIF / calibration", "×"),
    "focal_one": ("Wrong focal length, one photo", "one photo with wrong metadata", "×"),
    "principal_shift": ("Image centre shifted", "crop / resize metadata bug", "×width"),
    "depth_scale": ("Depth hints scaled wrong (with poses)", "depth sensor / mono-depth scale", "×"),
    "depth_scale_noposes": ("Depth hints scaled wrong (no poses)", "depth sensor / mono-depth scale", "×"),
    "depth_noise": ("Noisy depth hints", "noisy depth sensor", "σ"),
    "depth_one_scale": ("One depth hint scaled wrong", "one bad depth map", "×"),
    "metric_flag_off": ("Metric flag forgotten", "user forgets the flag", ""),
}
GROUP = {f: g for g, fs in {
    "P: independent pose errors": ["noise_rot", "noise_trans", "outlier_rot", "outlier_trans_small",
                                   "outlier_trans_large", "outlier_frac"],
    "C: correlated pose errors": ["scale_all", "drift", "shared_heading_bias", "anchor_error"],
    "K: lens errors": ["focal_all", "focal_all_noposes", "focal_one", "principal_shift"],
    "D: depth-hint errors": ["depth_scale", "depth_scale_noposes", "depth_noise", "depth_one_scale"],
    "M: metric flag": ["metric_flag_off"]}.items() for f in fs}
SIGNALS = {
    "S1": ("Harm", "Wrong hints make it worse than no hints"),
    "S2": ("Spread", "One bad hint damages the good views"),
    "S3": ("Blind trust", "The model copies bad hints"),
    "S4": ("Silent", "The model's confidence gives no warning"),
    "S5": ("Simple fix fails", "The easy filter is not enough"),
    "S6": ("Agrees-but-wrong", "Correlated / lens / depth errors hurt, or scale goes wrong unnoticed"),
    "S7": ("Realistic", "The harm happens at realistic error sizes"),
}


def p(*a):
    return os.path.join(ROOT, *a)


def fmt(v, d=2, pct=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    return f"{100 * v:+.{d - 1}f}%" if pct else f"{v:.{d}f}"


def sev_str(fam, s):
    unit = NAMES.get(fam, ("", "", ""))[2]
    if fam == "outlier_frac":
        return f"{float(s):.0%}"
    if unit in ("level", ""):
        return f"{unit} {s}".strip()
    return f"{float(s):g}{unit}" if unit in ("°", "×L", "×width") else f"{unit}{float(s):g}"


def family_summary(dr):
    """One row per family: the realistic setting with the largest shape-or-rotation harm."""
    rows = []
    for fam in NAMES:
        d = dr[dr.family == fam]
        if d.empty:
            continue
        real = d[d.realistic]
        s1_any, s1_real = bool(d.s1.any()), bool(real.s1.any())
        pr = real[real.metric.isin(["pts_relerr_scale", "rre_deg"])]
        if pr.empty:
            continue
        worst = pr.loc[pr.rel_vs_floor.idxmax()]
        sh = real[(real.metric == "pts_relerr_scale") & (real.condition == worst.condition)]
        ro = real[(real.metric == "rre_deg") & (real.condition == worst.condition)]
        sh, ro = (sh.iloc[0] if len(sh) else None), (ro.iloc[0] if len(ro) else None)
        rows.append({
            "group": GROUP.get(fam, ""), "family": fam, "name": NAMES[fam][0], "cause": NAMES[fam][1],
            "harm_realistic": s1_real, "harm_any": s1_any, "setting": sev_str(fam, worst.severity),
            "floor": worst.floor,
            "shape_bl": sh.benefit_lost if sh is not None else np.nan,
            "shape_rel": sh.rel_vs_floor if sh is not None else np.nan,
            "rot_bl": ro.benefit_lost if ro is not None else np.nan,
            "rot_rel": ro.rel_vs_floor if ro is not None else np.nan,
            "rot_abs": ro.delta_vs_floor if ro is not None else np.nan,
            "worse_frac": max(sh.frac_worse_than_floor if sh is not None else 0,
                              ro.frac_worse_than_floor if ro is not None else 0),
            "vs_rgbK": (worst.get("vs_rgbK_rel", np.nan) if "vs_rgbK_rel" in worst else np.nan),
        })
    return pd.DataFrame(rows)


PLOT = {"pts_relerr_scale": ("3D shape", "#2a78d6"), "rre_deg": ("camera rotation", "#eb6834"),
        "pts_relerr_metric": ("3D at real size", "#1baf7a")}


def fig_harm(dr, path):
    """Headline: % change vs the matching 'no hint' floor (stable where benefit_lost is not)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fams = [f for f in NAMES if f in set(dr.family)]
    cols, rows = 4, int(np.ceil(len(fams) / 4))
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 2.55 * rows), squeeze=False)
    for ax, fam in zip(axes.ravel(), fams):
        d = dr[dr.family == fam]
        sev = list(dict.fromkeys(d.severity))
        x = np.arange(len(sev))
        for k, s in enumerate(sev):
            if bool(d[d.severity == s].realistic.iloc[0]):
                ax.axvspan(k - 0.5, k + 0.5, color="#f0efec", lw=0)
        for m, (lab, col) in PLOT.items():
            e = d[d.metric == m].set_index("severity").reindex(sev)
            if not e.rel_vs_floor.notna().any():
                continue
            y = 100 * e.rel_vs_floor.to_numpy(float)
            fm = e.floor_mean.to_numpy(float)
            lo, hi = 100 * e.delta_ci_lo.to_numpy(float) / fm, 100 * e.delta_ci_hi.to_numpy(float) / fm
            ax.errorbar(x, y, yerr=np.nan_to_num(np.clip(np.vstack([y - lo, hi - y]), 0, None)), color=col, lw=2,
                        marker="o", ms=4, capsize=0, label=lab)
        ax.axhline(0, color="#e34948", lw=1)
        ax.set_xticks(x, [sev_str(fam, s) for s in sev], fontsize=7)
        ax.set_title(NAMES[fam][0], fontsize=8, loc="left")
        ax.tick_params(labelsize=7, colors="#52514e")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", color="#e1e0d9", lw=0.6)
        ax.set_axisbelow(True)
    for ax in axes.ravel()[len(fams):]:
        ax.set_axis_off()
    axes[0, 0].legend(fontsize=7, frameon=False)
    for r in range(rows):
        axes[r, 0].set_ylabel("% vs without that hint", fontsize=7)
    fig.suptitle("How much worse than WITHOUT that hint (%, mean over 10 test scenes, 95% CI). Red line = same as "
                 "without the hint; above = the wrong hint hurts. Shaded = realistic error sizes.", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_scene_harm(run, dr, path):
    """Per scene: % change vs the floor at each family's most severe realistic setting."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    import sys
    sys.path.insert(0, ROOT)
    from exp1.e1_analysis import clip_means, values
    cm = clip_means(pd.read_csv(os.path.join(run, "per_run.csv")))
    pick = []
    for fam in NAMES:
        d = dr[(dr.family == fam) & dr.realistic & (dr.metric == "pts_relerr_scale")]
        if len(d):
            pick.append((fam, d.iloc[-1]))
    clips = sorted(cm.clip_id.unique())
    fig, axes = plt.subplots(1, 2, figsize=(15, 0.36 * len(pick) + 1.8), sharey=True)
    for ax, m in zip(axes, ("pts_relerr_scale", "rre_deg")):
        mat = np.array([[100 * (values(cm, r.condition, m).get(c, np.nan) / values(cm, r.floor, m).get(c, np.nan) - 1)
                         for c in clips] for _, r in pick])
        im = ax.imshow(np.clip(mat, -100, 300), cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0, vmin=-100, vmax=300),
                       aspect="auto")
        ax.set_xticks(range(len(clips)), [c.split("__")[1] for c in clips], rotation=45, ha="right", fontsize=7)
        ax.set_title(f"{PLOT[m][0]}: % vs without that hint", fontsize=9)
    axes[0].set_yticks(range(len(pick)), [f"{NAMES[f][0]} ({sev_str(f, r.severity)})" for f, r in pick], fontsize=7)
    fig.colorbar(im, ax=axes, fraction=0.025, label="% worse than without the hint (clipped -100..300)")
    fig.suptitle("Per test scene, at each kind's most severe realistic setting. Red = the wrong hint is worse than "
                 "no hint in that scene; blue = still better.", fontsize=9)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def md_table(df, cols, heads):
    out = "| " + " | ".join(heads) + " |\n|" + "---|" * len(heads) + "\n"
    for _, r in df.iterrows():
        out += "| " + " | ".join(str(c(r)) if callable(c) else str(r[c]) for c in cols) + " |\n"
    return out


def html_table(df, cols, heads, num=()):
    th = "".join(f'<th class="{"num" if h in num else ""}">{html.escape(h)}</th>' for h in heads)
    body = ""
    for _, r in df.iterrows():
        cells = []
        for c, h in zip(cols, heads):
            v = c(r) if callable(c) else r[c]
            cls = "num" if h in num else ""
            cells.append(f'<td class="{cls}">{v}</td>')
        body += "<tr>" + "".join(cells) + "</tr>"
    return f'<div class="table-scroll"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def chip(ok, yes="YES", no="NO"):
    return f'<span class="chip {"yes" if ok else "no"}">{yes if ok else no}</span>'


def data_uri(path):
    return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=p("results", "e1_eval"))
    ap.add_argument("--demo", nargs="*", default=[p("results", "e1_demo_pose"), p("results", "e1_demo_other")])
    ap.add_argument("--out", default=p("reports", "e1_results"))
    ap.add_argument("--md", default=p("reports", "e1_results.md"))
    ap.add_argument("--commit", default="71582c1")
    ap.add_argument("--pdf_browser")
    ap.add_argument("--takeaways", help="HTML snippet (a <section>) with the plain-language takeaways")
    a = ap.parse_args()
    agg = os.path.join(a.run, "agg")
    os.makedirs(a.out, exist_ok=True)

    sig = json.load(open(os.path.join(agg, "signals.json")))
    dr = pd.read_csv(os.path.join(agg, "dose_response.csv"))
    cont = pd.read_csv(os.path.join(agg, "contamination.csv"))
    absb = pd.read_csv(os.path.join(agg, "absorption.csv"))
    auc = pd.read_csv(os.path.join(agg, "confidence_auroc.csv"))
    fix = pd.read_csv(os.path.join(agg, "fix.csv"))
    scale = pd.read_csv(os.path.join(agg, "scale.csv"))
    L = pd.read_csv(os.path.join(agg, "scene_lengths.csv"))
    pr = pd.read_csv(os.path.join(a.run, "per_run.csv"))
    info = json.load(open(os.path.join(a.run, "run_info.json")))
    checks = open(os.path.join(agg, "dev_checks.md")).read() if os.path.exists(os.path.join(agg, "dev_checks.md")) else ""
    for f in glob.glob(os.path.join(agg, "*.csv")) + glob.glob(os.path.join(agg, "*.png")) + \
            [os.path.join(agg, "signals.json"), os.path.join(agg, "signals.md")]:
        shutil.copy(f, a.out)
    demos = []
    for dd in a.demo:
        for f in sorted(glob.glob(os.path.join(dd, "*__overview.png"))):
            dst = os.path.join(a.out, f"demo_{os.path.basename(dd)}.png")
            shutil.copy(f, dst)
            demos.append(dst)

    fig_harm(dr, os.path.join(a.out, "fig_harm_vs_size.png"))
    fig_scene_harm(a.run, dr, os.path.join(a.out, "fig_scene_harm.png"))
    runs = pr[pr.view_subset == "all"]
    n_ok, n_fail = int((runs.status == "ok").sum()), int((runs.status != "ok").sum())
    n_clips = runs.clip_id.nunique()
    fs = family_summary(dr)
    fs.to_csv(os.path.join(a.out, "family_summary.csv"), index=False)
    s1 = dr[dr.s1]
    dec = sig["decision"]

    # ---------------------------------------------------------------- markdown report
    md = [f"# E1 results: do wrong hints make MapAnything worse than no hints?\n",
          f"Eval split, run once from the frozen commit `{args_commit(a)}` (tag `e1-prereg`). Plan and amendments: "
          "`experiments/E1.md`. Analysis: `python -m exp1.e1_analysis --run results/e1_eval`. All numbers below come "
          f"from `results/e1_eval/agg/` (copied to `reports/e1_results/`).\n",
          f"## Decision: **{dec}**\n",
          md_table(pd.DataFrame([{"k": k, **dict(zip(("name", "meaning"), SIGNALS[k])), "v": sig[k]["value"]}
                                 for k in SIGNALS]), ["k", "name", "meaning", lambda r: "YES" if r.v else "NO"],
                   ["Signal", "Name", "Plain meaning", "Eval result"]),
          "\nDecision rule (frozen): STOP if not S1; Strong GO if S1, S5, S7 and (S2 or S6); otherwise Weak GO.\n",
          "## Setup\n",
          f"- {n_clips} eval clips × 10 views (518×336; relief / relief_2 turned upright, 336×518), "
          f"{n_ok} runs ok, {n_fail} failed (`results/e1_eval/per_run.csv`).\n",
          f"- MapAnything `{info['mapanything_commit'][:10]}`, checkpoint `{info['checkpoint']}`, torch {info['torch']}, "
          f"GPU {info['gpu']}.\n",
          "- Scene size L per clip (metres): " + ", ".join(f"{r.scene} {r.L_m:.2f}" for r in L.itertuples()) + ".\n",
          "- Statistics: seeds averaged within a clip; 95% CIs = bootstrap over clips; paired by clip. "
          "benefit_lost = (x − ceiling)/(floor − ceiling), ratio of means; 1 = as bad as the floor.\n",
          "\n## 1. Per-family summary (worst realistic setting)\n",
          md_table(fs, ["group", "name", lambda r: "YES" if r.harm_realistic else ("only extreme" if r.harm_any else "NO"),
                        "setting", "floor", lambda r: fmt(r.shape_bl), lambda r: fmt(r.shape_rel, 2, True),
                        lambda r: fmt(r.rot_bl), lambda r: fmt(r.rot_rel, 2, True),
                        lambda r: f"{r.rot_abs:+.3f}°" if np.isfinite(r.rot_abs) else "–",
                        lambda r: f"{r.worse_frac:.0%}", lambda r: fmt(r.vs_rgbK, 2, True)],
                   ["Group", "Damage", "S1 harm (realistic)", "Setting", "Floor", "Shape BL", "Shape vs floor",
                    "Rotation BL", "Rotation vs floor", "Rotation Δ", "Scenes worse", "vs photos+lens"]),
          "\n`Shape` = `pts_relerr_scale`, `Rotation` = `rre_deg`. \"vs photos+lens\" = the same run against "
          "`rgb_K` (all poses dropped); negative = the damaged hints are still better than dropping them.\n",
          "\n## 2. All S1 settings\n",
          md_table(s1, ["condition", "metric", lambda r: "yes" if r.realistic else "no", lambda r: fmt(r.benefit_lost),
                        lambda r: fmt(r.rel_vs_floor, 2, True), lambda r: f"{r.delta_vs_floor:.4f}",
                        lambda r: f"[{r.delta_ci_lo:.4f}, {r.delta_ci_hi:.4f}]", lambda r: f"{r.frac_worse_than_floor:.0%}"],
                   ["Condition", "Metric", "Realistic", "benefit_lost", "vs floor", "Δ vs floor", "95% CI", "Scenes worse"])
          if len(s1) else "No setting meets S1.\n",
          "\n## 3. Spread to good views (S2), realistic settings\n",
          md_table(cont[cont.realistic], ["condition", "metric", lambda r: fmt(r.rel_rise, 2, True),
                                          lambda r: f"[{r.ci_lo:.4f}, {r.ci_hi:.4f}]"],
                   ["Condition", "Per-view metric", "Rise on uncorrupted views", "95% CI of Δ"]),
          "\n## 4. Blind trust (S3) and confidence (S4)\n",
          md_table(absb, ["condition", lambda r: f"{r['median']:.2f}", lambda r: f"[{r.q25:.2f}, {r.q75:.2f}]", "n_views"],
                   ["Condition", "Median absorption", "IQR", "Corrupted views"]),
          "\nAbsorption = model error ÷ hint error on the corrupted view (1 = copies the bad hint, 0 = ignores it).\n\n",
          md_table(auc, ["family", lambda r: fmt(r.auroc_confidence), lambda r: fmt(r.auroc_disagree_rot),
                         lambda r: fmt(r.auroc_disagree_dir), "n_runs"],
                   ["Family", "AUROC confidence", "AUROC disagree (rot)", "AUROC disagree (dir)", "Runs"]),
          "\nWithin-run AUROC: chance a clean view scores \"cleaner\" than the corrupted view of the same run "
          "(confidence: higher; disagreement with images-only: lower). S4 is YES when the pooled confidence AUROC ≤ 0.65.\n",
          "\n## 5. Scale (C1, D1, M1)\n",
          md_table(scale, ["condition", lambda r: f"{r.injected_factor:g}", lambda r: "on" if r.metric else "off",
                           lambda r: f"{r.rel_to_clean:.3f}", lambda r: f"[{r.rel_ci_lo:.3f}, {r.rel_ci_hi:.3f}]",
                           lambda r: f"{r.filter_drops_poses:.2f}"],
                   ["Condition", "Injected factor", "Metric flag", "Predicted size ÷ clean-hint size", "95% CI",
                    "Filter: poses dropped"]),
          "\n## 6. Fixes: oracle vs simple filter (S5)\n",
          md_table(fix[fix.metric.isin(["pts_relerr_scale", "rre_deg", "pts_relerr_metric"])],
                   ["condition", "metric", "oracle", lambda r: f"{r.err_damaged:.4f}", lambda r: f"{r.err_oracle:.4f}",
                    lambda r: f"{r.err_filter:.4f}", lambda r: fmt(r.filter_recovery),
                    lambda r: f"{r.filter_mean_poses_dropped:.2f}"],
                   ["Condition", "Metric", "Oracle", "Damaged", "Oracle", "Filter", "Filter recovery", "Poses dropped"]),
          "\n## 7. Signal evidence\n```json\n" + json.dumps(sig, indent=1, default=float)[:6000] + "\n```\n",
          "\n## Checks on the eval run\n" + checks,
          "\n## Figures\n`reports/e1_results/`: fig_dose_response.png (headline), fig_scene_heatmap.png, "
          "fig_contamination.png, fig_absorption.png, fig_scale.png; demo_*.png = dev-clip visual examples (qualitative).\n",
          "\n## Limitations\n- One clip of 10 views per scene, 10 eval scenes; CIs are over scenes.\n"
          "- Realistic ranges follow the preregistered rule (amendment 2), not yet literature-sourced (E1.md §6).\n"
          "- ETH3D GT depth is sparse (≈ 6-35% of pixels per clip); depth hints are that sparse GT.\n"
          "- Single model (MapAnything); OmniVGGT replication is E4.\n",
          "\n## Reproduce\n```\ngit checkout e1-prereg\npython -m exp1.run_exp1 --config configs/e1_full.json "
          "--clips 'data/clips/eval__*.npz' --out results/e1_eval\npython -m exp1.e1_analysis --run results/e1_eval --checks\n"
          "python scripts/e1_make_report.py\n```\n"]
    open(a.md, "w", encoding="utf-8").write("\n".join(s.strip("\n") + "\n" for s in md))

    # ---------------------------------------------------------------- HTML page
    tpl = open(p("scripts", "e1_report_template.html"), encoding="utf-8").read()
    sig_rows = "".join(f"<tr><td><strong>{k}</strong> {html.escape(SIGNALS[k][0])}</td><td>{html.escape(SIGNALS[k][1])}</td>"
                       f"<td>{chip(sig[k]['value'])}</td></tr>" for k in SIGNALS)
    fam_html = html_table(
        fs, [lambda r: html.escape(r["name"]) + f'<div class="sub">{html.escape(r.cause)}</div>',
             lambda r: chip(r.harm_realistic, "YES", "only extreme" if r.harm_any else "NO"), "setting",
             lambda r: fmt(r.shape_rel, 2, True), lambda r: fmt(r.rot_rel, 2, True),
             lambda r: f"{r.rot_abs:+.3f}°" if np.isfinite(r.rot_abs) else "–",
             lambda r: f"{r.worse_frac:.0%}", lambda r: fmt(r.vs_rgbK, 2, True)],
        ["Wrong hint", "Worse than no hint?", "Setting", "3D shape vs no-hint", "Rotation vs no-hint", "Rotation Δ",
         "Scenes worse", "vs dropping poses"],
        num=("3D shape vs no-hint", "Rotation vs no-hint", "Rotation Δ", "Scenes worse", "vs dropping poses"))
    figs = {os.path.basename(f): data_uri(f) for f in glob.glob(os.path.join(a.out, "*.png"))}
    fill = {
        "DECISION": html.escape(dec), "SIG_ROWS": sig_rows, "FAMILY_TABLE": fam_html,
        "N_OK": str(n_ok), "N_FAIL": str(n_fail), "N_CLIPS": str(n_clips), "COMMIT": html.escape(args_commit(a)),
        "S3_VAL": fmt(sig["S3"]["median_absorption_single_outliers"]),
        "S4_VAL": fmt(sig["S4"]["pooled_within_run_auroc_confidence"]),
        "N_S1": str(len(s1)), "N_S1_REAL": str(int(s1.realistic.sum())) if len(s1) else "0",
        "N_FAM_HARM": str(int(fs.harm_realistic.sum())), "N_FAM": str(len(fs)),
        "L_LIST": ", ".join(f"{r.scene} {r.L_m:.1f} m" for r in L.itertuples()),
    }
    for k, v in figs.items():
        fill["IMG:" + k] = v
    page = tpl
    for k, v in fill.items():
        page = page.replace("{{" + k + "}}", v)
    demo_html = "".join(f'<figure><img src="{figs[os.path.basename(d)]}" alt="Dev-clip reconstruction examples">'
                        f'<figcaption>{html.escape(os.path.basename(d))}: dev clip delivery_area, qualitative; '
                        'grey = laser truth, colour = MapAnything, black = true cameras, red = predicted, '
                        'orange = the wrong hint.</figcaption></figure>' for d in demos)
    page = page.replace("{{DEMOS}}", demo_html)
    page = page.replace("{{TAKEAWAYS}}", open(a.takeaways, encoding="utf-8").read() if a.takeaways else "")
    left = [t for t in page.split("{{")[1:] if "}}" in t]
    assert not left, f"unfilled placeholders: {[t.split('}}')[0] for t in left]}"
    out_html = os.path.join(a.out, "e1_results.html")
    open(out_html, "w", encoding="utf-8").write(page)
    if a.pdf_browser:
        subprocess.run([a.pdf_browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={pathlib.Path(a.out, 'e1_results.pdf').resolve()}",
                        pathlib.Path(out_html).resolve().as_uri()], check=True, capture_output=True, timeout=180)
    print(f"wrote {a.md} and {a.out}")


def args_commit(a):
    return a.commit


if __name__ == "__main__":
    main()
