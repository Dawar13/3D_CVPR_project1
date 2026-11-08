"""E2X results report (experiments/E2X.md). Run on the laptop from MapAnything_Exp1/ after the session fetch, the
E3 calibration (exp1.e3_calibrate, one folder per model under results/e2x_e3cal_agg/) and the analysis
(scripts/e2x/analyse_e2x.py --out results/e2x_agg):

  python scripts/e2x/make_report.py [--pdf_browser "C:/Program Files/Google/Chrome/Application/chrome.exe"]

Every number comes from a results file; the Files section lists them. A part whose inputs are missing is reported
as missing, never estimated. Writes reports/e2x_results/ (figures, e2x_results.html, e2x_results.pdf).
"""

import argparse
import base64
import glob
import html
import json
import os
import pathlib
import re
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
P = lambda *a: os.path.join(ROOT, *a)
OUT = P("reports", "e2x_results")
AGG, E3, LOGS = P("results", "e2x_agg"), P("results", "e2x_e3cal_agg"), P("results", "vm_logs_e2x")
# reference palette (dataviz skill, references/palette.md; validated: scripts/validate_palette.js, light): fixed per
# model, never by rank. Aqua is below 3:1 on white, so every figure has direct labels and a table beside it.
COL = {"mapanything": "#2a78d6", "omnivggt": "#eb6834", "pi3x": "#1baf7a"}
NAME = {"mapanything": "MapAnything", "omnivggt": "OmniVGGT", "pi3x": "Pi3X"}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
USED = []   # every input file read, for the Files section


def rd(path):
    if not os.path.exists(path):
        return None
    USED.append(os.path.relpath(path, ROOT).replace(os.sep, "/"))
    if path.endswith(".csv"):
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    return json.load(open(path)) if path.endswith(".json") else open(path).read()


def style(ax, ylabel=None, xlabel=None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=8.5)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=8.5)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def uri(path):
    return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()


def f(v, d=2):
    try:
        return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{float(v):.{d}f}"
    except (TypeError, ValueError):
        return html.escape(str(v))


def table(df, cols, heads=None, num=()):
    if df is None or not len(df):
        return "<p class='missing'>no rows</p>"
    heads = heads or cols
    h = "".join(f"<th>{html.escape(x)}</th>" for x in heads)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r.get(c)
            cells.append(f"<td class='num'>{f(v, num[c] if isinstance(num, dict) and c in num else 2)}</td>"
                         if c in num else f"<td>{html.escape(str(v))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{h}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def missing(what):
    return f"<p class='missing'>Not available: {html.escape(what)} (input file missing; nothing is estimated).</p>"


# ---------------------------------------------------------------------------------------------------------- figures
def fig_break_even(be):
    fams = [("c1_rot", "One wrong camera: rotation error Δ vs images only (°)", "camera turned by (°)"),
            ("c1_depth", "Depth hint scale: shape error Δ vs correct poses", "depth scale factor")]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    for ax, (fam, title, xl) in zip(axes, fams):
        d = be[be.family == fam]
        for m in ("mapanything", "omnivggt", "pi3x"):
            g = d[d.model == m].sort_values("size")
            if not len(g):
                continue
            ax.fill_between(g["size"], g.ci_lo, g.ci_hi, color=COL[m], alpha=0.12, linewidth=0)
            ax.plot(g["size"], g.delta_vs_floor, color=COL[m], linewidth=2, marker="o", markersize=4.5)
            last = g.iloc[-1]
            ax.annotate(NAME[m], (last["size"], last.delta_vs_floor), xytext=(4, 0), textcoords="offset points",
                        fontsize=8, color=INK2, va="center")
        ax.axhline(0, color=INK2, linewidth=0.9)
        if fam == "c1_rot":
            ax.set_xscale("log")
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        style(ax, "above 0 = the hint makes it worse", xl)
    fig.subplots_adjust(wspace=0.38)
    return save(fig, "fig_break_even.png")


def fig_safety(sf):
    d = sf[sf.metric == "rre_deg"]
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    lab = {"none": "use all hints", "filter_rgbcheck": "R2 photo filter", "drop_lowconf": "R3 confidence",
           "witness": "E3 repair", "witness_drop": "E3 drop", "R0 images only": "R0 images only"}
    for m in ("mapanything", "omnivggt", "pi3x"):
        g = d[d.model == m]
        if not len(g):
            continue
        ax.scatter(g.benefit_kept, g.damage_removed, s=46, color=COL[m], edgecolor="white", linewidth=1.5, zorder=3,
                   label=NAME[m])
        for _, r in g.iterrows():
            ax.annotate(lab.get(r.remedy, r.remedy), (r.benefit_kept, r.damage_removed), xytext=(5, 3),
                        textcoords="offset points", fontsize=7, color=INK2)
    ax.set_xlim(-0.15, 1.25)
    ax.set_ylim(-0.4, 1.15)
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.axvline(0, color=AXIS, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    style(ax, "share of bad-hint damage removed", "share of good-hint benefit kept")
    ax.set_title("Safety vs benefit (rotation; calibration clips, DESCRIPTIVE)", fontsize=9, color=INK, loc="left")
    return save(fig, "fig_safety.png")


def fig_real_errors(re_):
    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    x = np.sort(re_.rot_err_median.dropna().values)
    ax.hist(x, bins=min(20, max(5, len(x) // 2)), color=COL["mapanything"], edgecolor="white", linewidth=2)
    ax.axvline(np.median(x), color=INK2, linewidth=1, linestyle="--")
    ax.annotate(f"median {np.median(x):.2f}°", (np.median(x), ax.get_ylim()[1] * 0.9), xytext=(5, 0),
                textcoords="offset points", fontsize=8, color=INK2)
    style(ax, "room clips", "iPad (ARKit) camera rotation error per clip, median over views (°)")
    ax.set_title("How wrong real whole-room phone poses are (vs SceneFun3D ground truth)", fontsize=9, color=INK,
                 loc="left")
    return save(fig, "fig_real_errors.png")


# -------------------------------------------------------------------------------------------------------- sections
def session_section():
    s = rd(os.path.join(LOGS, "session.txt"))
    machine = rd(os.path.join(LOGS, "machine.txt"))
    mono = rd(os.path.join(LOGS, "mono_verify.txt"))
    if s is None:
        return missing("results/vm_logs_e2x/session.txt"), {}
    lines = s.splitlines()
    gates = [l for l in lines if " GATE " in f" {l} " or "GATE_" in l]
    marks = [l for l in lines if re.search(r"_DONE|JOB_FAILED|!!|remain|skipped|measured", l)]
    runs = []
    for d in sorted(glob.glob(P("results", "e2x_*"))):
        pr = os.path.join(d, "per_run.csv")
        if not os.path.exists(pr) or "agg" in d:
            continue
        try:
            df = pd.read_csv(pr)
        except pd.errors.EmptyDataError:
            continue
        if "status" not in df:
            continue
        fl = os.path.join(d, "failures.jsonl")
        oom = os.path.join(d, "oom_retries.jsonl")
        runs.append({"run": os.path.basename(d), "rows": len(df), "ok": int((df.status == "ok").sum()),
                     "failed rows": int((df.status != "ok").sum()),
                     "failures.jsonl lines": sum(1 for _ in open(fl)) if os.path.exists(fl) else 0,
                     "out-of-memory resets": sum(len(json.loads(x)["runs_reset"]) for x in open(oom)) if os.path.exists(oom) else 0})
        USED.append(os.path.relpath(pr, ROOT).replace(os.sep, "/"))
    R = pd.DataFrame(runs)
    out = ["<h2 id='session'>1. The session</h2>"]
    if machine:
        out.append(f"<pre>{html.escape(machine.strip())}</pre>")
    out.append(f"<p>First and last queue lines:</p><pre>{html.escape(lines[0])}\n{html.escape(lines[-1])}</pre>")
    if mono:
        out.append(f"<p>AI depth check against E2-E's record: <code>{html.escape(mono.strip())}</code></p>")
    out.append("<h3>Runs per job</h3>" + table(R, list(R.columns), num={c: 0 for c in R.columns if c != "run"}))
    out.append("<h3>Dev gates and job markers</h3><pre>" + html.escape("\n".join(gates + marks) or "none") + "</pre>")
    return "\n".join(out), {"gates": gates, "runs": R}


def xa_section():
    out = ["<h2 id='xa'>2. X-A: Pi3X, the third model (ETH3D)</h2>"]
    harm = rd(os.path.join(AGG, "xa_harm_pi3x.csv"))
    beh = rd(os.path.join(AGG, "xa_behaviour_pi3x.csv"))
    fam = rd(os.path.join(AGG, "xa_family_ma_vs_pi3x.csv"))
    sbsb = rd(os.path.join(AGG, "xa_behaviour_ma_vs_pi3x.csv"))
    if harm is None:
        return "\n".join(out + [missing("results/e2x_agg/xa_harm_pi3x.csv (Pi3X ETH3D eval)")]), {}
    h = harm[harm.harmful]
    out.append(f"<p>Preregistered D-H test (E2-D verbatim, Holm within the model): <b>{len(h)}</b> of {len(harm)} "
               f"tests harmful, families: {html.escape(', '.join(sorted(h.family.unique())) or 'none')}.</p>")
    out.append(table(h.sort_values("p_holm"), ["condition", "metric", "x_mean", "floor_mean", "delta", "ci_lo", "ci_hi",
                                               "n_worse", "p_holm"],
                     num={"x_mean": 3, "floor_mean": 3, "delta": 3, "ci_lo": 3, "ci_hi": 3, "n_worse": 0, "p_holm": 4}))
    if beh is not None:
        out.append("<h3>Behaviour (P1–P5, P10, P11)</h3>" + table(
            beh, ["test", "question", "mean", "ci_lo", "ci_hi", "n_scenes", "p_holm", "finding"],
            num={"mean": 3, "ci_lo": 3, "ci_hi": 3, "n_scenes": 0, "p_holm": 4}))
    if fam is not None:
        out.append("<h3>Per family: MapAnything vs Pi3X</h3>" + table(fam, ["family", "class"]))
    if sbsb is not None:
        out.append("<h3>Behaviour side by side</h3>" + table(sbsb, list(sbsb.columns),
                                                              num={c: 3 for c in sbsb.columns if c.endswith("_mean")}))
    p1 = beh[beh.test.str.startswith("P1")] if beh is not None else None
    return "\n".join(out), {"pi3x_copies": bool(p1 is not None and len(p1) and p1.finding.any())}


def xb_section():
    out = ["<h2 id='xb'>3. X-B: real hints over whole rooms</h2>"]
    info = {"hurts": []}
    re_ = rd(os.path.join(AGG, "xb_room_real_errors.csv"))
    if re_ is not None and len(re_):
        out.append(f"<img src='{uri(fig_real_errors(re_))}' alt='Histogram of per-clip ARKit rotation errors'>")
        out.append(f"<p>{len(re_)} room clips: median rotation error {f(re_.rot_err_median.median())}° "
                   f"(clip medians), share of views ≥ 5° {f(re_.share_views_rot_ge5.mean())}, "
                   f"depth scale ratio LiDAR/GT median {f(re_.dev_depth_scale_ratio.median(), 3)} "
                   "(<code>results/e2x_agg/xb_room_real_errors.csv</code>).</p>")
    for m in ("mapanything", "omnivggt", "pi3x"):
        t = rd(os.path.join(AGG, f"xb_room_tests_{m}.csv"))
        if t is None:
            out.append(f"<h3>{NAME[m]}</h3>" + missing(f"results/e2x_agg/xb_room_tests_{m}.csv"))
            continue
        out.append(f"<h3>{NAME[m]}: each real hint set vs images only (room eval clips)</h3>" + table(
            t, ["condition", "metric", "delta", "ci_lo", "ci_hi", "n_scenes", "p_holm", "verdict"],
            num={"delta": 3, "ci_lo": 3, "ci_hi": 3, "n_scenes": 0, "p_holm": 4}))
        hurts = t[(t.condition == "dev_pose") & t.metric.isin(["rotation", "shape"]) & (t.verdict == "HURTS")]
        if len(hurts):
            info["hurts"].append(m)
        rem = rd(os.path.join(AGG, f"xb_room_remedies_{m}.csv"))
        if rem is not None and len(rem):
            out.append("<p>Remedies on the real hints (Δ vs no remedy):</p>" + table(
                rem, list(rem.columns), num={c: 3 for c in rem.columns if c not in ("condition", "remedy", "metric")}))
    e2e = rd(os.path.join(AGG, "xb_e2e_tests_pi3x.csv"))
    if e2e is not None:
        out.append("<h3>Pi3X on the E2-E short clips</h3>" + table(
            e2e, ["condition", "metric", "delta", "ci_lo", "ci_hi", "n_scenes", "p_holm", "verdict"],
            num={"delta": 3, "ci_lo": 3, "ci_hi": 3, "n_scenes": 0, "p_holm": 4}))
    return "\n".join(out), info


def e3_section():
    out = ["<h2 id='e3'>4. E3 calibration (calibration and dev clips only; nothing here is a result)</h2>",
           "<p>Rule fixed before the session (E2X.md §5): the smallest <code>e3_z</code> whose false-action rate on "
           "ground-truth hints is ≤ 5% for any group action and ≤ 2% of views for single-view drops; the same for the "
           "depth cut-offs; otherwise NOT CALIBRATED, never loosened.</p>"]
    info = {}
    for m in ("mapanything", "pi3x", "omnivggt"):
        ch = rd(os.path.join(E3, m, "chosen.json"))
        if ch is None:
            out.append(f"<h3>{NAME[m]}</h3>" + missing(f"results/e2x_e3cal_agg/{m}/chosen.json"))
            continue
        info[m] = ch
        fa = rd(os.path.join(E3, m, "false_action_rates.csv"))
        det = rd(os.path.join(E3, m, "detection_rates.csv"))
        out.append(f"<h3>{NAME[m]}</h3><p>Chosen: <code>{html.escape(json.dumps(ch))}</code></p>")
        if fa is not None:
            out.append(table(fa, ["part", "setting", "n", "group_action_rate", "view_drop_rate", "meets_rule"],
                             num={"n": 0, "group_action_rate": 3, "view_drop_rate": 3}))
        if det is not None and len(det):
            out.append("<p>Detection at the chosen settings (reported, not used to choose):</p>" + table(
                det, list(det.columns), num={c: 3 for c in det.columns if c != "condition"}))
    sf = rd(os.path.join(AGG, "angle_safety_preview.csv"))
    if sf is not None and len(sf):
        out.append(f"<img src='{uri(fig_safety(sf))}' alt='Safety vs benefit scatter'>")
        out.append(table(sf, ["model", "metric", "remedy", "benefit_kept", "damage_removed", "n_bad_conditions"],
                         num={"benefit_kept": 3, "damage_removed": 3, "n_bad_conditions": 0}))
    info["safety"] = sf
    return "\n".join(out), info


def angles_section():
    out = ["<h2 id='angles'>5. Angles (DESCRIPTIVE)</h2>"]
    be = rd(os.path.join(AGG, "angle_break_even.csv"))
    if be is None:
        return "\n".join(out + [missing("results/e2x_agg/angle_break_even.csv")])
    out.append(f"<img src='{uri(fig_break_even(be))}' alt='Break-even curves per model'>")
    first = (be[be.hint_worse_than_floor].sort_values("size").groupby(["model", "family"]).first().reset_index()
             [["model", "family", "condition", "size", "delta_vs_floor", "ci_lo", "ci_hi"]])
    out.append("<p>Smallest error size whose 95% CI lies above 0 (the hint is then worse than going without it):</p>"
               + table(first, list(first.columns), num={"size": 2, "delta_vs_floor": 3, "ci_lo": 3, "ci_hi": 3}))
    out.append("<details><summary>All break-even rows</summary>" + table(
        be, ["model", "family", "condition", "metric", "delta_vs_floor", "ci_lo", "ci_hi", "n_scenes"],
        num={"delta_vs_floor": 3, "ci_lo": 3, "ci_hi": 3, "n_scenes": 0}) + "</details>")
    return "\n".join(out)


def referee_section():
    out = ["<h2 id='referee'>6. Geometry referee (L2, planned baseline R5; dev/cal clips)</h2>"]
    ch = rd(P("results", "e2x_geometry_referee_deg", "chosen.json"))
    eth = rd(P("results", "e2x_geometry_referee_deg", "eth3d_dev_descriptive.csv"))
    if ch is None:
        return "\n".join(out + [missing("results/e2x_geometry_referee_deg/chosen.json")])
    out.append(f"<p>Calibration (rule fixed in the script): view part <b>{html.escape(str(ch['view_status']))}</b>, "
               f"set part <b>{html.escape(str(ch['set_status']))}</b>; {f(100 * ch['coverage_views_scored'], 0)}% of "
               f"views scored. Details: <code>results/e2x_geometry_referee_deg/README.md</code>.</p>")
    if eth is not None:
        e = eth[eth.family == "outlier_rot"]
        out.append("<p>ETH3D dev, one wrong camera: the referee measures the error itself.</p>" + table(
            e, ["condition", "bad_view_median_deg", "good_view_median_deg"],
            num={"bad_view_median_deg": 2, "good_view_median_deg": 2}))
    return "\n".join(out)


def decision_section(xb, e3, e3v=None):
    out = ["<h2 id='decision'>Decisions (rules fixed in advance)</h2>",
           "<h3>E2X continue / stop rule (E2X.md §5; primary calibration)</h3>"]
    ma = e3.get("mapanything")
    calibrated = ma is not None and isinstance(ma.get("e3_z"), (int, float))
    sf = e3.get("safety")
    better = []
    if sf is not None and len(sf):
        for metric in ("rre_deg", "pts_relerr_scale"):
            g = sf[(sf.model == "mapanything") & (sf.metric == metric)]
            e3r = g[g.remedy.isin(["witness", "witness_drop"])]
            base = g[g.remedy.isin(["filter_rgbcheck", "drop_lowconf", "R0 images only"])]
            if len(e3r) and len(base):
                b = base.loc[base.damage_removed.idxmax()]
                ok = e3r[(e3r.damage_removed > b.damage_removed) & (e3r.benefit_kept >= b.benefit_kept)]
                better.append((metric, len(ok) > 0, b.remedy))
    sig_a = bool(xb.get("hurts"))
    sig_b = any(x[1] for x in better)
    rows = [
        ("(i-a) real whole-room pose hints HURT for ≥ 1 model (X-B, dev_pose, rotation or shape)",
         "YES: " + ", ".join(NAME[m] for m in xb["hurts"]) if sig_a else ("NO" if xb is not None else "missing")),
        ("(i-b) E3 removes more damage than the best of R0/R2/R3 at ≥ its benefit kept (MapAnything, cal clips; "
         "R1 'drop all poses' is not in the E3-cal grid, R0 images only stands in)",
         "; ".join(f"{m}: {'YES' if ok else 'NO'} (vs {b})" for m, ok, b in better) or "missing"),
        ("(ii) E3 calibrated for MapAnything", ("YES, e3_z = " + str(ma["e3_z"])) if calibrated else
         ("NOT CALIBRATED" if ma is not None else "missing")),
    ]
    out.append("<table><thead><tr><th>Signal</th><th>Reading</th></tr></thead><tbody>" + "".join(
        f"<tr><td>{html.escape(a)}</td><td>{html.escape(b)}</td></tr>" for a, b in rows) + "</tbody></table>")
    if ma is None:
        verdict = "Not decidable yet: the MapAnything E3 calibration is missing."
    elif not calibrated:
        verdict = ("ANALYSIS PAPER (no method claim): E3 is NOT CALIBRATED for MapAnything, so the photo witness "
                   "cannot be made safe on correct hints under the fixed rule. E3 is reported as a negative result.")
    elif sig_a or sig_b:
        verdict = "GO for the method paper: both conditions hold."
    else:
        verdict = ("Condition (i) does not hold: neither harmful real whole-room poses nor an E3 advantage over the "
                   "existing remedies. Not GO for the method paper under the fixed rule.")
    out.append(f"<p class='verdict'>{html.escape(verdict)}</p>")
    out.append("<h3>E3 method evaluation (E3.md; thresholds from the accurate-ground-truth calibration)</h3>")
    if e3v is None:
        out.append(missing("results/e3_eval_agg/e3_verdict.json"))
    else:
        out.append(f"<p class='verdict'>E3 method claim holds: {html.escape(str(e3v.get('method_claim_holds')))} "
                   f"(SAFE: {html.escape(str(e3v.get('safe')))}). Both readings are reported: the primary rule above and "
                   "this evaluation.</p>")
    out.append("<p>Inside-the-network direction: Shubh's G1 oracle dial gave no reading in session A (its own γ = 1 "
               "check failed). Session A2 measured the same dial (M1, section 6).</p>")
    return "\n".join(out), verdict


XM, E3EVAL = P("results", "e2x_xm_agg"), P("results", "e3_eval_agg")
CASE = {"one_cam_15": "one camera 15°", "shared_heading_5": "shared heading 5°", "drift_level2": "drift level 2",
        "depth_x1.5": "depth ×1.5 (views 3, 7)"}


def fig_m1(curve):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2))
    for ax, (case, ylab) in zip(axes, (("one_cam_15", "bad camera rotation error (°)"),
                                       ("depth_x1.5", "shape error (relative)"))):
        g = curve[(curve.case == case) & (curve.model == "mapanything")].sort_values("gamma")
        ax.plot(g.gamma, g.err, color=COL["mapanything"], linewidth=2, marker="o", markersize=4.5)
        ax.annotate("MapAnything", (g.gamma.iloc[-1], g.err.iloc[-1]), xytext=(-60, 6), textcoords="offset points",
                    fontsize=8, color=INK2)
        ax.set_title(f"Hint dial, {CASE[case]}", fontsize=9, color=INK, loc="left")
        style(ax, ylab, "γ (0 = hint off, 1 = hint as given)")
    fig.subplots_adjust(wspace=0.32)
    return save(fig, "fig_m1_dial.png")


def fig_m3(patch):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for m in ("mapanything", "pi3x"):
        g = patch[(patch.model == m) & (patch.case == "one_cam_15") & (patch.source == "clean")].sort_values("layer")
        if not len(g):
            continue
        x = g.layer / g.layer.max()
        ax.plot(x, g.share_removed, color=COL[m], linewidth=2, marker="o", markersize=4)
        ax.annotate(NAME[m], (x.iloc[len(x) // 2], g.share_removed.iloc[len(x) // 2]), xytext=(0, 8),
                    textcoords="offset points", fontsize=8, color=INK2)
    for k in (1, 9, 17, 25, 33):   # Pi3X camera-hint injections (pi3x.py:499)
        ax.axvline(k / 33, color=GRID, linewidth=1, linestyle=":")
    ax.set_ylim(-0.1, 1.1)
    style(ax, "share of the damage removed", "depth in the network (layer / last patched layer)")
    ax.set_title("Swap the bad camera's state for its correct-hint state (one camera 15°)", fontsize=9, color=INK,
                 loc="left")
    return save(fig, "fig_m3_patch.png")


def xm_section():
    out = ["<h2 id='xm'>6. Inside the network (session A2, DESCRIPTIVE; ETH3D eval, 24 clips, 10 scenes)</h2>",
           "<p>Frozen at <code>e2x-a2-prereg</code> (amendments 1–2: plumbing fixes before any eval run). Pi3X injects "
           "the camera hint again after blocks 1, 9, 17, 25, 33, so a swapped state can be overwritten later "
           "(stated before the run); the dotted lines mark those blocks.</p>"]
    d1 = rd(os.path.join(XM, "m1_dial.csv"))
    c1 = rd(os.path.join(XM, "m1_curve.csv"))
    p3 = rd(os.path.join(XM, "m3_patch.csv"))
    if d1 is None or p3 is None:
        return "\n".join(out + [missing("results/e2x_xm_agg/")])
    if c1 is not None:
        out.append(f"<img src='{uri(fig_m1(c1))}' alt='Hint dial curves'>")
    out.append("<h3>M1 hint dial</h3>" + table(d1, ["model", "case", "r2_mean", "r2_ci_lo", "r2_ci_hi", "n_scenes_linear",
                                                   "switch_index_mean", "n_scenes_switch", "n_scenes", "err_gamma0",
                                                   "err_gamma1"],
                                              num={"r2_mean": 3, "r2_ci_lo": 3, "r2_ci_hi": 3, "n_scenes_linear": 0,
                                                   "switch_index_mean": 3, "n_scenes_switch": 0, "n_scenes": 0,
                                                   "err_gamma0": 3, "err_gamma1": 3}))
    out.append(f"<img src='{uri(fig_m3(p3))}' alt='Share of damage removed by patching per layer'>")
    pn = rd(os.path.join(XM, "m3_point_of_no_return.csv"))
    if pn is not None:
        out.append("<h3>M3 point of no return (first layer whose swap removes < 50% of the damage)</h3>" +
                   table(pn, ["model", "case", "source", "point_of_no_return_layer"]))
    piv = p3[p3.source == "clean"].pivot_table(index=["model", "case"], columns="layer", values="share_removed").round(2)
    out.append("<details><summary>M3 share removed per layer (source: correct-hint run)</summary>" +
               piv.to_html(classes="t", border=0) + "</details>")
    m5 = rd(os.path.join(XM, "m5_spread.csv"))
    if m5 is not None:
        out.append("<details><summary>M5: swapping the OTHER cameras' states</summary>" +
                   m5.pivot_table(index=["model", "case"], columns="layer", values="share_removed").round(2)
                   .to_html(classes="t", border=0) + "</details>")
    st = rd(os.path.join(XM, "m2_m4_state.csv"))
    if st is not None:
        out.append("<details><summary>M2/M4: size of the bad − correct state difference and its cosine with the first "
                   "layer</summary>" + st.pivot_table(index=["model", "case"], columns="layer",
                                                       values="cos_with_first_layer").round(2).to_html(classes="t", border=0)
                   + "</details>")
    m8 = rd(os.path.join(XM, "m8_scale.csv"))
    if m8 is not None:
        out.append("<h3>M8: two hints disagree (output scale vs ground truth)</h3>" + table(
            m8, ["model", "condition", "pose_hint_scale", "depth_hint_scale", "output_scale_vs_gt", "ci_lo", "ci_hi",
                 "shape_err", "n_scenes"],
            num={"pose_hint_scale": 2, "depth_hint_scale": 2, "output_scale_vs_gt": 3, "ci_lo": 3, "ci_hi": 3,
                 "shape_err": 3, "n_scenes": 0}))
    return "\n".join(out)


def secondary_section():
    out = ["<h2 id='secondary'>5b. E3 calibration on accurate ground truth (E3-cal amendment 1, SECONDARY)</h2>",
           "<p>Written before any TUM run (tag <code>e3cal-amend1</code>): the same rule on ETH3D dev (laser scan) and "
           "TUM RGB-D fr3 (motion capture) clips; TUM clips failing the E0 ground-truth check were excluded first "
           "(<code>results/e2x_tum_clip_check/excluded.csv</code>). The primary verdict (section 4) stands.</p>"]
    rows = []
    for m in ("mapanything", "pi3x"):
        for kind in ("primary", "secondary"):
            ch = rd(os.path.join(E3, m if kind == "primary" else f"{m}_secondary", "chosen.json"))
            if ch is None:
                continue
            rows.append({"model": NAME[m], "calibration": kind, "e3_z": ch["e3_z"],
                         "tau_depth_shared": ch["e3_tau_depth_shared"], "tau_depth_view": ch["e3_tau_depth_view"],
                         "n_gt_pose_decisions": ch["n_gt_pose_decisions"],
                         "n_gt_depth_decisions": ch["n_gt_depth_decisions"]})
    if not rows:
        return "\n".join(out + [missing("results/e2x_e3cal_agg/<model>_secondary/chosen.json")])
    out.append(table(pd.DataFrame(rows), ["model", "calibration", "e3_z", "tau_depth_shared", "tau_depth_view",
                                          "n_gt_pose_decisions", "n_gt_depth_decisions"],
                     num={"n_gt_pose_decisions": 0, "n_gt_depth_decisions": 0}))
    for m in ("mapanything", "pi3x"):
        fa = rd(os.path.join(E3, f"{m}_secondary", "false_action_rates.csv"))
        if fa is not None:
            out.append(f"<details><summary>{NAME[m]}: secondary false-action rates</summary>" + table(
                fa, ["part", "setting", "n", "group_action_rate", "view_drop_rate", "meets_rule"],
                num={"n": 0, "group_action_rate": 3, "view_drop_rate": 3}) + "</details>")
    return "\n".join(out)


def fig_e3_safety(sb):
    d = sb[sb.metric == "rre_deg"]
    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    lab = {"none": "use all hints", "witness": "E3", "witness_drop": "E3-drop", "filter_rgbcheck": "R2",
           "drop_lowconf": "R3", "rgb_only": "R0 images only", "drop_oracle": "ORACLE"}
    for data, mk in (("synth", "o"), ("real_room", "s"), ("real_e2e", "^")):
        g = d[d.data == data]
        if not len(g):
            continue
        ax.scatter(g.benefit_kept, g.damage_removed, s=44, marker=mk, color=COL["mapanything"], edgecolor="white",
                   linewidth=1.5, zorder=3, label=data)
        for _, r in g.iterrows():
            if np.isfinite(r.benefit_kept) and np.isfinite(r.damage_removed):
                ax.annotate(lab.get(r.remedy, r.remedy), (r.benefit_kept, r.damage_removed), xytext=(5, 3),
                            textcoords="offset points", fontsize=7, color=INK2)
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.axvline(0, color=AXIS, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    style(ax, "share of bad-hint damage removed", "share of correct-hint benefit kept")
    ax.set_title("E3 evaluation: safety vs benefit (rotation, MapAnything)", fontsize=9, color=INK, loc="left")
    return save(fig, "fig_e3_safety.png")


def e3eval_section():
    out = ["<h2 id='e3eval'>7. E3 method evaluation (experiments/E3.md, eval splits)</h2>"]
    v = rd(os.path.join(E3EVAL, "e3_verdict.json"))
    t = rd(os.path.join(E3EVAL, "e3_tests.csv"))
    if t is None:
        return "\n".join(out + [missing("results/e3_eval_agg/e3_tests.csv")]), None
    if v is not None:
        out.append(f"<p class='verdict'>E3 decision (fixed before the run): method claim holds = "
                   f"{html.escape(str(v.get('method_claim_holds')))}; SAFE = {html.escape(str(v.get('safe')))}; "
                   f"HELP on agreeing / AI-depth: {html.escape(', '.join(v.get('help_on_agreeing_or_ai_depth') or []) or 'none')}; "
                   f"BEST on agreeing / AI-depth: {html.escape(', '.join(v.get('best_on_agreeing_or_ai_depth') or []) or 'none')}.</p>")
    warn = t[(t.test == "SAFE") & (t.ci_lo > 0) & (t.delta >= t.threshold) & ~t.harmful]
    if len(warn):
        items = "; ".join(f"{r.data} {r.condition} {r.metric}: +{r.delta:.3f} (CI {r.ci_lo:.3f} to {r.ci_hi:.3f})"
                          for _, r in warn.iterrows())
        out.append("<p class='verdict' style='border-left-color:#e34948;background:#fdf0ef'>Caution: SAFE passes under "
                   f"the strict rule, but {len(warn)} SAFE tests show harm whose 95% CI excludes 0 and whose estimate "
                   f"exceeds the practical threshold, without surviving Holm: {html.escape(items)}. E3 is therefore not "
                   "shown to be safe on these data.</p>")
    for test in ("SAFE", "HELP", "BEST"):
        g = t[t.test == test]
        out.append(f"<h3>{test}</h3>" + table(g, ["data", "condition", "metric", "vs", "delta", "ci_lo", "ci_hi",
                                                    "n_scenes", "p_holm", "harmful" if test == "SAFE" else "better"],
                                               num={"delta": 3, "ci_lo": 3, "ci_hi": 3, "n_scenes": 0, "p_holm": 4}))
    sb = rd(os.path.join(E3EVAL, "e3_safety_benefit.csv"))
    if sb is not None and len(sb):
        out.append(f"<img src='{uri(fig_e3_safety(sb))}' alt='E3 safety vs benefit'>")
        out.append(table(sb, ["data", "metric", "remedy", "benefit_kept", "damage_removed", "n_bad_conditions"],
                         num={"benefit_kept": 3, "damage_removed": 3, "n_bad_conditions": 0}))
    return "\n".join(out), v


CSS = """body{font-family:Inter,Segoe UI,Arial,sans-serif;color:#0b0b0b;background:#fcfcfb;max-width:980px;margin:0 auto;
padding:24px 16px;font-size:14px;line-height:1.5}h1{font-size:24px}h2{font-size:18px;margin-top:32px;border-top:1px solid
#e1e0d9;padding-top:16px}h3{font-size:15px}table{border-collapse:collapse;font-size:12px;margin:8px 0;display:block;
overflow-x:auto}th,td{border-bottom:1px solid #e1e0d9;padding:3px 8px;text-align:left;white-space:nowrap}td.num{text-align:
right;font-variant-numeric:tabular-nums}img{max-width:100%}pre{background:#f1f0ec;padding:8px;font-size:11px;overflow-x:
auto;white-space:pre-wrap}.missing{color:#898781;font-style:italic}.verdict{font-weight:600;background:#eef4fc;padding:
10px;border-left:4px solid #2a78d6}code{font-size:12px}@media print{h2{break-before:auto}table,img{break-inside:avoid}}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_browser")
    ap.add_argument("--agg", help="override results/e2x_agg (previews only)")
    ap.add_argument("--e3", help="override results/e2x_e3cal_agg (previews only)")
    ap.add_argument("--logs", help="override results/vm_logs_e2x (previews only)")
    ap.add_argument("--out", help="override reports/e2x_results (previews only)")
    ap.add_argument("--xm", help="override results/e2x_xm_agg (previews only)")
    ap.add_argument("--e3eval", help="override results/e3_eval_agg (previews only)")
    a = ap.parse_args()
    global AGG, E3, LOGS, OUT, XM, E3EVAL
    AGG, E3, LOGS, OUT = a.agg or AGG, a.e3 or E3, a.logs or LOGS, a.out or OUT
    XM, E3EVAL = a.xm or XM, a.e3eval or E3EVAL
    os.makedirs(OUT, exist_ok=True)
    s_html, _ = session_section()
    xa_html, xa = xa_section()
    xb_html, xb = xb_section()
    e3_html, e3 = e3_section()
    an_html = angles_section()
    rf_html = referee_section()
    sc_html = secondary_section()
    xm_html = xm_section()
    ev_html, e3v = e3eval_section()
    dc_html, verdict = decision_section(xb, e3, e3v)
    files = "".join(f"<li><code>{html.escape(u)}</code></li>" for u in sorted(set(USED)))
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport"
content="width=device-width,initial-scale=1"><title>E2X Results</title><style>{CSS}</style></head><body>
<h1>Extended E2 (E2X): results</h1>
<p>Built by <code>scripts/e2x/make_report.py</code> from the files listed at the end. Preregistration: tag
<code>e2x-prereg</code> (<code>experiments/E2X.md</code> §5). DESCRIPTIVE parts are labelled; calibration-clip
numbers are not results.</p>
<p class='verdict'>{html.escape(verdict)}</p>
{dc_html}{s_html}{xa_html}{xb_html}{e3_html}{sc_html}{xm_html}{ev_html}{an_html}{rf_html}
<h2 id='files'>8. Files</h2><ul>{files}</ul></body></html>"""
    out = os.path.join(OUT, "e2x_results.html")
    open(out, "w", encoding="utf-8").write(doc)
    print("wrote", out)
    if a.pdf_browser:
        pdf = pathlib.Path(OUT, "e2x_results.pdf").resolve()
        subprocess.run([a.pdf_browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={pdf}", pathlib.Path(out).resolve().as_uri()], check=False)
        print("wrote", pdf)


if __name__ == "__main__":
    main()
