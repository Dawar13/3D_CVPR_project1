"""Plain-language project status and E2X/E3 plan (2026-11-04) as HTML + PDF. Every number is copied from the result
file or report named next to it; nothing is computed here except diagram geometry. Run from MapAnything_Exp1/:

  python scripts/e2x/make_plan_pdf.py --pdf_browser "C:/Program Files/Google/Chrome/Application/chrome.exe"

Writes reports/e2x_plan/e2x_plan.html and e2x_plan.pdf.
"""

import argparse
import math
import os
import pathlib
import subprocess

OUT = os.path.join("reports", "e2x_plan")
CSS = open(os.path.join("scripts", "e2_report_style.css"), encoding="utf-8").read()
EXTRA_CSS = """
.box { padding: 16px 18px; border: 1px solid var(--line); border-radius: 10px; background: var(--surface); display: grid; gap: 10px; }
.box.key { border-left: 4px solid var(--accent); }
.box.warn { border-left: 4px solid var(--hurt); }
.box.ok { border-left: 4px solid var(--help); }
.qa h3 { color: var(--accent); }
ol.steps { margin: 0; padding-left: 1.3rem; display: grid; gap: 8px; }
.small { font-size: 0.88rem; color: var(--ink-2); }
.two { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
@media (max-width: 700px) { .two { grid-template-columns: 1fr; } }
td b { font-weight: 600; }
.status { font-family: var(--font-data); font-size: 0.72rem; padding: 1px 7px; border-radius: 999px; white-space: nowrap; }
.status.done { background: var(--help-soft); color: var(--help); }
.status.next { background: #e3ecf8; color: #1f5fae; }
.status.gap { background: var(--hurt-soft); color: var(--hurt); }
@media print { .two { display: grid; } .box { break-inside: avoid; } }
"""

INK, MUTED, LINE, HURT, HELP, ACC = "#0b0b0b", "#898781", "#c9c7bf", "#9a2b2b", "#2f6b2f", "#2a78d6"


def svg_text(x, y, s, size=12, fill=INK, anchor="start", weight=400):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}" font-family="IBM Plex Sans, Segoe UI, sans-serif">{s}</text>')


def arrow(x, y, deg, length, color, width=2.2, dash=None):
    a = math.radians(deg)
    x2, y2 = x + length * math.sin(a), y - length * math.cos(a)
    hx, hy = math.sin(a), -math.cos(a)
    px, py = -hy, hx
    h1 = (x2 - 7 * hx + 4 * px, y2 - 7 * hy + 4 * py)
    h2 = (x2 - 7 * hx - 4 * px, y2 - 7 * hy - 4 * py)
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"{d}/>'
            f'<polygon points="{x2:.1f},{y2:.1f} {h1[0]:.1f},{h1[1]:.1f} {h2[0]:.1f},{h2[1]:.1f}" fill="{color}"/>')


def fig_error_types():
    """Three panels, 10 cameras each: true direction (dashed grey) and hinted direction (red)."""
    panels = [("1. One shared tilt", "every camera off by the same angle", lambda i: 12),
              ("2. Drift", "the error grows along the recording", lambda i: 2.5 * i),
              ("3. One bad camera", "one camera far off, the rest correct", lambda i: 35 if i == 4 else 0)]
    W, H, pw = 805, 215, 250
    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Three ways hints go wrong">']
    for k, (title, sub, err) in enumerate(panels):
        x0 = 15 + k * (pw + 12)
        parts.append(f'<rect x="{x0}" y="8" width="{pw}" height="{H - 16}" rx="8" fill="#fbfbf9" stroke="{LINE}"/>')
        parts.append(svg_text(x0 + 12, 30, title, 13, INK, weight=600))
        parts.append(svg_text(x0 + 12, 47, sub, 11, MUTED))
        for i in range(10):
            x = x0 + 22 + i * 23
            base = 168
            parts.append(arrow(x, base, 0, 58, "#a9a69e", 2.0, "4 3"))       # true direction (drawn upright)
            parts.append(arrow(x, base, err(i), 58, HURT, 2.2))
            parts.append(f'<circle cx="{x}" cy="{base}" r="4" fill="{INK}"/>')
            parts.append(svg_text(x, base + 18, str(i + 1), 9.5, MUTED, "middle"))
    parts.append(svg_text(15, H - 0, "", 1))
    parts.append("</svg>")
    legend = (f'<span style="color:#7d7a72">- - - true direction</span> &nbsp; '
              f'<span style="color:{HURT}">&#8212; direction the hint gives</span> &nbsp; (all cameras drawn pointing the same way; angles exaggerated so they are visible)')
    return "".join(parts), legend


def box(x, y, w, h, title, lines, fill="#fff", stroke=LINE, tcolor=INK):
    s = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}"/>',
         svg_text(x + w / 2, y + 22, title, 12.5, tcolor, "middle", 600)]
    for k, ln in enumerate(lines):
        s.append(svg_text(x + w / 2, y + 41 + 15 * k, ln, 11, "#52514e", "middle"))
    return "".join(s)


def harrow(x1, y, x2, color=MUTED):
    return (f'<line x1="{x1}" y1="{y}" x2="{x2 - 6}" y2="{y}" stroke="{color}" stroke-width="1.6"/>'
            f'<polygon points="{x2},{y} {x2 - 8},{y - 4} {x2 - 8},{y + 4}" fill="{color}"/>')


def fig_two_roles():
    W, H = 780, 272
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Two roles of the photos-only run">']
    s.append(svg_text(15, 22, "A. Photos only as a BASELINE (R0): its answer is the final answer; all hints are thrown away", 12.5, INK, weight=600))
    s.append(box(15, 34, 150, 62, "10 photos", ["hints ignored"]))
    s.append(harrow(165, 65, 215))
    s.append(box(215, 34, 150, 62, "3D program", ["photos only"]))
    s.append(harrow(365, 65, 415))
    s.append(box(415, 34, 345, 62, "Final 3D model", ["safe, but loses everything good hints add"], "#efeee9"))
    s.append(svg_text(15, 128, "B. Photos only INSIDE THE FIX: its answer is only a reference to check each hint against", 12.5, ACC, weight=600))
    s.append(box(15, 140, 150, 96, "10 photos", ["+ the hints", "(camera directions,", "depth, lens)"]))
    s.append(harrow(165, 165, 215, ACC))
    s.append(box(215, 140, 150, 50, "3D program", ["photos only (x3)"], "#e3ecf8", ACC))
    s.append(harrow(365, 165, 415, ACC))
    s.append(box(415, 140, 160, 50, "Check the hints", ["repair / drop / keep"], "#e3ecf8", ACC))
    s.append(harrow(575, 165, 615, ACC))
    s.append(box(615, 140, 150, 96, "3D program", ["photos + checked", "hints: the final", "3D model"], "#e3f1e3", HELP))
    s.append(f'<path d="M90,242 L90,250 L690,250 L690,242" fill="none" stroke="{HELP}" stroke-width="1.2"/>')
    s.append(svg_text(390, 266, "the good (or repaired) hints still reach the final run", 11, HELP, "middle"))
    s.append("</svg>")
    return "".join(s)


def fig_fix_steps():
    W, H = 780, 150
    steps = [("1. Ask the photos", ["photos-only run,", "3 photo orders"]),
             ("2. How sure?", ["do the 3 answers", "agree?"]),
             ("3. Compare", ["fit: shared tilt,", "drift, one bad"]),
             ("4. Decide", ["repair / drop,", "or keep if unsure"]),
             ("5. Depth check", ["rescale depth that", "is too far / near"]),
             ("6. Final run", ["photos + checked", "hints"])]
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Steps of the fix">']
    w, gap = 116, 16
    for k, (t, ln) in enumerate(steps):
        x = 10 + k * (w + gap)
        fill = "#e3f1e3" if k == 5 else "#fff"
        s.append(box(x, 20, w, 80, t, ln, fill, HELP if k == 5 else LINE))
        if k < 5:
            s.append(harrow(x + w, 60, x + w + gap))
    s.append(svg_text(10, 130, "Extra cost: 3 photos-only runs per clip (once), plus 1-2 runs per hint set. One run takes about 0.26 s.", 11, MUTED))
    s.append("</svg>")
    return "".join(s)


def table(header, rows, cls=""):
    th = "".join(f"<th>{h}</th>" for h in header)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="table-scroll {cls}"><table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def src(*paths):
    return '<p class="src">Source: ' + "; ".join(f"<code>{p}</code>" for p in paths) + "</p>"


def build():
    err_svg, err_legend = fig_error_types()
    S = []
    S.append(f"""
<header class="col stack">
  <p class="eyebrow">Project status &middot; 4 November 2026 &middot; plain-language version</p>
  <h1>Do 3D models trust bad hints? Where we stand, and what comes next</h1>
  <p class="lead">What has been found, how the experiments are set up, what the planned fix does, why bad hints
  matter in the real world, and how the next GPU sessions decide the final paper.</p>
</header>

<section class="col stack">
  <div class="box key">
    <h3 style="margin:0">The whole paper in three sentences</h3>
    <ol class="steps">
      <li><b>The problem.</b> Programs that build 3D models from photos accept extra <i>hints</i> (where each camera was,
      lens settings, depth maps) and <b>obey them, even when they are wrong, and even when the program itself can tell</b>.</li>
      <li><b>It matters.</b> Real hint sources (AI depth estimates, phone compasses, GPS, long phone recordings) make
      errors of the size that hurts.</li>
      <li><b>A fix.</b> A cheap check that compares the hints with what the photos alone say can repair or drop bad hints
      without retraining, and leave good hints alone.</li>
    </ol>
    {table(["Part", "Status"], [
        ["1. The problem", '<span class="status done">shown</span> for MapAnything; partly for OmniVGGT; <span class="status next">next</span> &pi;3-X'],
        ["2. It matters in practice", '<span class="status done">shown</span> for AI depth; <span class="status next">next</span> camera directions over whole rooms; error sizes from published studies'],
        ["3. The fix", '<span class="status next">built and tested on trial data</span>; final test after the next GPU session']])}
  </div>
</section>

<section class="col stack">
  <h2>1. The question</h2>
  <p>Programs like MapAnything turn a handful of photos into a 3D model. They can also take <b>hints</b>: where each
  camera was (position and direction), lens settings, and depth maps. Correct hints make them much better. Real hints
  come from phones, GPS, drones or AI depth estimators, and these are sometimes wrong. We ask three things:</p>
  <ol class="list">
    <li>Do wrong hints make the result <b>worse than using no hints at all</b>?</li>
    <li><b>Why</b> does it happen?</li>
    <li>Can we <b>catch and fix</b> bad hints automatically, using only the photos, with no retraining?</li>
  </ol>
</section>

<section class="col stack">
  <h2>2. What is done so far</h2>
  {table(["Step", "Question", "Answer"], [
    ["<b>E0</b>", "Does our pipeline work?", "Yes. The setup checks passed, runs repeat bit for bit, and one 10-photo run takes 0.26 s."],
    ["<b>E1</b>", "Do wrong hints hurt?", "<b>Yes, badly, for camera-direction errors.</b> One camera turned 15&deg; more than doubles the program's direction error compared with photos only (+114%, all 10 scenes). Slow drift: +142%. All cameras sharing one 5&deg; compass error: +108%. Wrong camera <i>positions</i> are mostly shrugged off."],
    ["<b>E2</b>", "What exactly happens, and does it happen elsewhere?", "MapAnything <b>copies</b> wrong directions (77% of a 1&deg; error, 96% of a 30&deg; error) and never overrules them. It <b>knows</b> something is off: in 93 of 100 random pairs of one bad and one good camera, the bad one gets lower confidence. It obeys anyway. The damage happens inside the network, not in a simple averaging step. A second program (OmniVGGT) copies about half of a direction error. Real iPad hints <b>helped</b> both programs; <b>AI-guessed depth hurt both</b>. E1's simple filter catches one very wrong camera but throws away 12% of good cameras."],
    ["<b>Now</b>", "Extended E2 and the first version of the fix (E3)", "Laptop work done; GPU session prepared. Sections 3&ndash;8."]])}
  {src("reports/e0_logs/8_det_*.txt", "reports/e1_results.md", "results/e1_eval/agg/signals.json", "reports/e2_results/e2_results.pdf")}
</section>

<section class="col stack">
  <h2>3. The setup</h2>
  <h3>3.1 The programs (three)</h3>
  {table(["Program", "Role", "Why it is included"], [
    ["<b>MapAnything</b> (Meta)", "Main subject", "Takes every kind of hint, camera by camera. This is where we found the problem."],
    ["<b>OmniVGGT</b>", "Second program", "Built differently (on VGGT). Shows whether the problem belongs to one program only."],
    ["<b>&pi;3-X</b> (new)", "Third program", "Also takes all kinds of hints, camera by camera. A benchmark paper (SpatialBench) reports that it <i>partly overrules</i> wrong camera hints, the opposite of MapAnything, so it is a real test."],
    ["Depth Anything V2", "Not tested itself", "Only produces the <b>AI depth hints</b>, the way a phone without a depth sensor would."]])}

  <h3>3.2 The datasets (two, for different jobs)</h3>
  {table(["Dataset", "What it is", "Why we use it", "How"], [
    ["<b>ETH3D</b>", "13 scenes, high-quality photos, laser-scanned exact ground truth", "Exact truth lets us <b>break hints on purpose</b> by known amounts (turn one camera 15&deg;, add drift, scale depth &times;1.5) and measure the damage precisely", "3 trial scenes, 10 final scenes; 10 photos per clip"],
    ["<b>SceneFun3D / ARKitScenes</b> (iPad room scans)", "Real iPad recordings with the iPad's <b>own</b> camera tracking (ARKit) and depth sensor (LiDAR), plus separately computed ground truth", "Real-world hints with real mistakes", "<b>Short</b> clips (E2-E, about 0.8 m of walking) and new <b>whole-room</b> clips (10 photos spread over the whole recording)"]])}
  <p><b>Why the whole-room clips?</b> Over a short walk the iPad's tracking is almost perfect, so its hints helped. Across a
  whole room it drifts. Measured on 150 recordings:</p>
  {table(["10 photos spread over&hellip;", "Typical size of the area", "Worst camera's direction error (typical)", "1 clip in 10 is worse than", "Clips over 5&deg;"], [
    ["a short stretch (E2-E style)", "0.80 m", "1.23&deg;", "2.71&deg;", "4.6%"],
    ["the whole recording", "3.36 m", "2.88&deg;", "13.46&deg;", "20.7%"]])}
  <p>Photos alone get camera directions right to about 0.8&deg;. So across a room, <b>the iPad's hint is often worse than the
  photos</b>, which is exactly where obeying should hurt.</p>
  {src("results/e2x_survey/sf_device_survey.log", "results/e2x_survey/sf_device_survey_clean.csv", "reports/e2_results/e2_results.pdf (photos-only uncertainty)")}

  <div class="box">
    <p><b>Data checked and set aside:</b></p>
    <ul class="list small">
      <li><b>ADVIO</b> (phone): its "truth" comes from the same phone's motion sensor, so it cannot judge the phone.</li>
      <li><b>Zurich drone:</b> GPS is typically 4.69 m off (worst 42.67 m) while the drone moves about 0.71 m per second.
      But the logged orientation is the drone's, not the camera's, which turns on its own. Kept as an option for GPS tests.</li>
      <li><b>ScanNet++</b>: access not granted. <b>LaMAR</b>: access must be requested.</li>
    </ul>
    {src("results/e2x_survey/advio_probe.log", "results/e2x_survey/zurich_probe.log")}
  </div>
  <div class="box warn">
    <p><b>Quality control.</b> The iPad dataset's "truth" is itself sometimes wrong: 493 of 73,512 frames jump away from
    their neighbours by more than 5&deg; (80 by more than 30&deg;), while the iPad's own tracking never jumps more than
    30&deg;. Every new clip therefore went through two checks (depth consistency, and whether the "true" poses fit the
    photos). <b>21 of 114 new clips failed and were set aside before any program saw them.</b></p>
    {src("results/e2x_survey/sf_gt_spikes.log", "results/e2x_clip_check/")}
  </div>

  <h3>3.3 How a test works</h3>
  <p>Each clip is 10 photos plus hints. Each program runs several ways on the <b>same</b> clip:</p>
  <ol class="list">
    <li><b>Photos only</b>, no hints.</li>
    <li><b>Correct hints</b>, the best case.</li>
    <li><b>Wrong hints</b>: errors put in on purpose (ETH3D) or real iPad / AI hints (SceneFun3D).</li>
    <li><b>Wrong hints plus a remedy</b>: the existing remedies (3.4), and later our fix.</li>
  </ol>
  <p>Errors are compared clip by clip, with the <b>room or scene</b> as the unit, so ten clips of one room cannot count as
  ten pieces of evidence. With many comparisons a statistical correction is applied, and an effect only counts if it is
  also <b>big enough to matter</b>: 1 percentage point for shape, 0.25&deg; for direction, 5% for size.</p>
  <p><b>What is measured:</b> shape error of the 3D points; camera-direction error between camera pairs and the share of
  pairs within 5&deg;; the standard pose score other papers report (AUC@30, now computed for 11,938 finished runs);
  depth error; real-world size error. For the fix also: how much damage it repairs, how often it wrongly touches good
  hints, and the extra running time.</p>
  {src("results/e2c_eval/auc30.csv", "results/e2df_eval/auc30.csv", "results/e2d_eval/auc30.csv", "results/e2e_ma_eval/auc30.csv", "results/e2e_ov_eval/auc30.csv")}

  <h3>3.4 The existing remedies (what the fix must beat)</h3>
  {table(["Name", "What it does", "What we know"], [
    ["Use all hints", "Trust everything", "This is the problem itself"],
    ["<b>R0</b> photos only", "Ignore all hints", "Safe, but throws away everything good hints add"],
    ["<b>R1</b> drop camera hints", "Keep lens and depth, drop camera positions/directions", ""],
    ["<b>R2</b> E1's filter", "Compare each camera hint with the photos-only answer; drop it if it differs by more than 10&deg; in direction or 15&deg; in movement direction", "Catches one very wrong camera, but drops 12% of good cameras and misses errors shared by all cameras"],
    ["<b>R3</b> confidence drop", "Drop cameras the program itself is unsure about, then rerun", "Same weakness on shared errors"],
    ["<b>R4</b> classic geometry check", "The traditional non-AI method (robust bundle adjustment)", "Planned (CPU only)"],
    ["<b>ORACLE</b>", "Drop exactly the bad hints, using the truth", "Cheating upper bound, for reference only"],
    ["VPGGT, TCO", "Methods from other papers", "VPGGT: no public code. TCO: about 160 s per scene and no licence, so not run"]])}
</section>

<section class="col stack">
  <h2>4. Do bad hints really happen? (the question a reviewer will ask first)</h2>
  <p>If bad hints were rare, the paper would be about an exotic problem. So here is what is known about each real hint
  source: the size of its errors, and whether errors of that size hurt in our tests.</p>
  {table(["Hint source", "Typical real error", "Where the number comes from", "Does that size hurt?"], [
    ["<b>AI depth</b> (cameras and phones without a depth sensor)", "&times;1.42 too far, range &times;1.13&ndash;&times;1.85, on <b>every</b> iPad clip", "Measured by us (E2-E)", '<span class="status gap">yes, measured</span> shape +10.0 / +9.1 pp worse (MapAnything / OmniVGGT)'],
    ["AI depth networks in general", "34&ndash;55% average depth error on ETH3D", "Metric3D v2 paper (TPAMI 2024), Table 3", "depth &times;1.5 hurts (E2-B)"],
    ["<b>Phone compass</b> (camera direction)", "indoors: 12.45&deg; average, single readings 13&ndash;25&deg;; outdoors after calibration 1&ndash;3&deg;, 2&ndash;3&times; worse later; mean deviations 5.5&ndash;6.9&deg;", "Abadi et al. (IPIN 2013); Tartachynska &amp; Pylypyak (2026)", "one camera 15&deg;, shared 5&ndash;10&deg; and first camera 15&deg; all hurt (E1, E2-B)"],
    ["<b>Phone GPS</b> (camera position)", "4.9 m open sky; 7&ndash;13 m in cities; up to 99.7 m. Our drone data: 4.69 m typical, 42.67 m worst", "GPS.gov; Merry &amp; Bettinger (PLOS ONE 2019); our Zurich survey", "positions are mostly shrugged off; only big jumps hurt (E1)"],
    ["<b>Phone tracking</b> over a whole room (ARKit)", "worst camera 2.88&deg; typical, over 5&deg; in 20.7% of clips", "Measured by us (150 recordings)", '<span class="status next">being tested</span> (Part X-B)'],
    ["Single-camera 3D tracking (monocular SLAM)", "scale changed by 1 : 0.37 over one loop", "Strasdat et al. (RSS 2010)", "scale &times;2 hurts (E2-B)"],
    ["Lens data in photo files (EXIF)", "\"usually good, but can occasionally be off by more than a factor of two\"", "Snavely et al. (TOG 2006)", "lens errors not yet shown to hurt"],
    ["3D reconstruction software (COLMAP), often used to make hints", "493 of 73,512 frames jump by more than 5&deg; in a curated dataset", "Measured by us (SceneFun3D)", "same as the camera-direction rows"]])}
  {src("reports/e2_literature.md (literature sizes, with page numbers)", "reports/e2_audit.md (E2-B: 7 error families harmful at literature-realistic sizes)", "results/e2e_mono/", "results/e2x_survey/")}
  <div class="box key">
    <p><b>Our answer to "isn't this rare?"</b></p>
    <ol class="steps">
      <li><b>Not for AI depth.</b> Every iPad clip had a scale error between 13% and 85%, and it hurt both programs.</li>
      <li><b>For camera directions it depends on the source</b>, and we say so openly: phone tracking over a short walk is
      good (it helped), while compass-based directions, GPS and long recordings are not.</li>
      <li><b>The fix is a seatbelt.</b> If it costs almost nothing on good hints and never makes them worse, it is worth
      having even when failures are uncommon. So the paper must show, and report, how rarely it touches good hints.</li>
    </ol>
    <p class="small"><b>Honest gap:</b> the only real-world harm we have <i>measured ourselves</i> so far is AI depth. Measured
    real-world harm from wrong camera directions depends on Part X-B (whole-room iPad clips).</p>
  </div>
</section>

<section class="col stack">
  <h2>5. The fix (E3), explained step by step</h2>
  <h3>5.1 Why does the fix run "photos only", if that is a baseline?</h3>
  <p>The photos-only run has <b>two different jobs</b>, and only one of them is the baseline:</p>
  <figure class="diagram">{fig_two_roles()}</figure>
  <p>The reason the fix needs it: <b>no fixed choice wins everywhere.</b> With good hints, using them is much better
  (MapAnything's own paper, two-photo benchmark: relative point error 0.08 with photos only versus 0.02 with all hints). With bad hints,
  ignoring them is better (E1). A real system does not know which case it is in. So the fix decides <b>clip by clip and
  hint by hint</b>, and the photos-only answer is the <b>referee</b> it uses to decide, not the final answer.</p>
  {table(["", "Use all hints", "Photos only (R0)", "The fix (goal)"], [
    ["<b>Hints are good</b>", "best", "clearly worse", "as good as using all hints"],
    ["<b>Hints are bad</b>", "much worse (E1, E2)", "safe", "as good as photos only, or better (it keeps the good hints)"]])}
  {src("context.md section 3 (MapAnything paper, Table 2)", "reports/e1_results.md")}

  <h3>5.2 The three ways hints go wrong</h3>
  <p>Picture 10 cameras along a walk. Each hint says which way a camera was pointing. Hints can be wrong in three typical ways:</p>
  <figure class="diagram">{err_svg}<figcaption>{err_legend}</figcaption></figure>
  {table(["Kind", "Plain example", "What the photos-only answer shows"], [
    ["<b>1. One shared tilt</b>", "The phone's compass is off by 5&deg;, the camera is mounted slightly crooked, or the first camera's direction is wrong (every other camera is measured relative to it)", "Every hint differs from the photos by the <b>same</b> angle. Each one looks only a little off; together they are consistently off."],
    ["<b>2. Drift</b>", "Tracking adds a tiny mistake at every step: camera 2 is 0.5&deg; off, camera 3 1&deg;, &hellip; camera 10 4.5&deg;", "The difference <b>grows steadily</b> from camera to camera."],
    ["<b>3. One bad camera</b>", "A GPS jump or a compass glitch next to metal", "<b>One</b> camera disagrees strongly; the others agree."]])}

  <h3>5.3 What the fix does in each case</h3>
  {table(["Kind", "What the fix does", "Why this way"], [
    ["One shared tilt", "Measures the common angle and <b>turns all hints back</b> by it", "The hints are still correct <i>relative to each other</i>, so all of that information is kept"],
    ["Drift", "Measures how fast the error grows and <b>undoes</b> it", "Same: the rest of each hint is kept"],
    ["One bad camera", "<b>Drops that one camera's hint</b>", "Its hint has nothing worth keeping"],
    ["Depth too far or too near", "Predicts depth from the camera hints alone, compares, and <b>rescales</b> the depth hints (e.g. divides AI depth by about 1.4)", "AI depth is usually wrong in overall scale, not in shape"]])}
  <div class="box">
    <p><b>Worked example (shared tilt).</b> The photos-only answer says camera 2 turned 10&deg; and camera 3 turned 20&deg;
    from camera 1. The hints say 15&deg; and 25&deg;. Every hint is 5&deg; more than the photos, so the fix subtracts 5&deg;
    from all hints. They now agree with the photos, and the final run uses <b>all 10 corrected hints</b>. E1's filter would
    have kept all of them unchanged, because each one is only 5&deg; off, under its 10&deg; limit, and the damage stays.</p>
  </div>

  <h3>5.4 "Act only when the photos are sure"</h3>
  <p>The photos-only answer is not perfect either: typically 0.8&deg; off (2.0&deg; in the worst 10% of clips), and on some
  iPad trial clips 2&ndash;4.5&deg; off overall, or 15&deg; on a single camera, even when the hints were correct. If hints
  and photos disagree by 1&deg;, we cannot tell who is wrong.</p>
  <p>So the fix first measures <b>how sure the photos are on this clip</b>: it runs photos-only three times with the photos
  in a different order. If the three answers agree, the photos are a reliable witness here; if they wobble, they are not.
  A hint is repaired or dropped <b>only if it disagrees with the photos by several times the photos' own wobble</b>.
  Otherwise it is left alone.</p>
  <p class="small">Think of two witnesses. If one of them is unsure of what they saw, you do not use their story to overrule the other.</p>
  <figure class="diagram">{fig_fix_steps()}</figure>

  <h3>5.5 First trial numbers (trial clips only, not results)</h3>
  <p>Rows are the error put in on purpose. "Fix measured" is what the fix's comparison found, without being told.</p>
  {table(["Error put in", "Fix measured", "Same check with correct hints"], [
    ["One camera 15&deg;", "15.0&deg; on that camera", "under 1&deg; on the others"],
    ["First camera 15&deg;", "14.8&deg; shared", "0.6&deg;"],
    ["Shared 5&deg; compass error", "4.7&deg;", "0.6&deg;"],
    ["Drift levels 1 / 2 / 3", "4.2 / 7.7 / 15.1&deg;", "0.9&deg;"]])}
  {src("results/e2x_survey/e3_witness_dev.csv (3 ETH3D trial clips and 8 iPad trial clips; first prototype)")}
  <div class="box warn"><p><b>The open risk.</b> On the iPad trial clips the photos themselves were sometimes unreliable
  (above). If the "how sure" measurement does not catch those cases, the fix would damage good hints. This is exactly what
  the calibration in the next GPU session checks first.</p></div>
</section>

<section class="col stack">
  <h2>6. Extended E2: what the next GPU session does</h2>
  <div class="two">
    <div class="box"><h3 style="margin:0">Part X-A: is obeying common?</h3>
    <p class="small">&pi;3-X gets exactly the same broken ETH3D hints as MapAnything and OmniVGGT (identical down to the
    random numbers). Does it copy wrong hints, or overrule them as SpatialBench suggests? Either answer is informative.</p></div>
    <div class="box"><h3 style="margin:0">Part X-B: do real hints hurt when the error is bigger?</h3>
    <p class="small">All three programs on the whole-room iPad clips (&pi;3-X also on the short clips): photos only, true
    hints, iPad poses, iPad depth sensor, AI depth, and remedies R2/R3. Do real iPad poses still help across a whole room?
    Does AI depth hurt again?</p></div>
  </div>
  <div class="box key">
    <h3 style="margin:0">Part E3-calibration: setting the fix's cut-offs without peeking</h3>
    <ul class="list small">
      <li>The fix runs on <b>calibration rooms</b> (10 rooms never used for final results) and the ETH3D <b>trial</b> clips,
      with every kind of broken hint and with real hints. Every decision it makes is logged.</li>
      <li><b>Rule, fixed in advance:</b> choose the cut-off so that, <b>on correct hints</b>, the fix acts on at most 5% of
      clips (shared/drift repairs) and drops at most 2% of cameras. "Do no harm" comes first.</li>
      <li>How well it catches errors is recorded but <b>not</b> used to choose the cut-off, so the choice cannot be tuned
      toward a nice-looking result.</li>
    </ul>
  </div>
  <p>Jobs run most important first, so an early stop only loses the least important ones. Each final run starts only if
  its trial run passes automatic checks. Estimate (not measured): about 2.5&ndash;3.5 GPU-hours; &pi;3-X's speed is
  unknown until its trial run.</p>
  {src("experiments/E2X.md (sections 4-6)", "configs/e2x_*.json", "scripts/session_queue_e2x.sh")}
</section>

<section class="col stack">
  <h2>7. How E3 gets decided</h2>
  <ol class="steps">
    <li><b>After the GPU session</b> (laptop): analyse X-A, X-B and the calibration.</li>
    <li><b>Lock the E3 plan</b> before the final E3 run: the cut-offs from calibration, the final clips, the remedies to
    beat, and the success rules.</li>
    <li><b>Second GPU session:</b> the fix against all remedies on the <b>final</b> clips: ETH3D final scenes with every
    kind of broken hint, whole-room iPad clips with real iPad hints, iPad clips with AI depth.</li>
  </ol>
  <p><b>Proposed success rules</b> (fixed in step 2, so not final yet):</p>
  <ul class="list">
    <li><b>Repairs damage:</b> on errors shared by all cameras (compass, drift, first camera, depth scale), where the existing
    remedies leave at least half of the damage, the fix removes clearly more. The original target was about 70% of what
    the cheating ORACLE removes.</li>
    <li><b>Does no harm:</b> on correct hints and on good real iPad hints, any change stays below the "big enough to matter"
    thresholds.</li>
    <li><b>Beats the best existing remedy</b> on those error types.</li>
  </ul>
  {table(["Outcome", "Paper"], [
    ["&pi;3-X also obeys; whole-room iPad hints or AI depth hurt; the fix passes", "<b>Strongest workshop paper:</b> a general problem, real-world evidence and a working fix"],
    ["The fix passes for some error types only (e.g. depth scale and shared errors)", "Paper scoped to those types, stated honestly"],
    ["The fix touches good hints too often, even at strict cut-offs", "Analysis paper (problem + mechanism + why simple fixes fail); the fix becomes future work"]])}
</section>

<section class="col stack">
  <h2>8. One story, not a mix of everything</h2>
  <p>Every experiment answers one of three questions; anything that does not goes to the appendix or is dropped.</p>
  {table(["The paper's question", "Main paper", "Appendix", "Not in the paper"], [
    ["<b>1. Do the programs obey wrong hints?</b>", "Trust curves: how much of a wrong hint is copied; \"knows but obeys\" (confidence); three programs side by side", "The E1 audit; the averaging (mean vs median) check showing the damage is inside the network; 4- and 20-photo clips; partial hints", "&mdash;"],
    ["<b>2. Does it matter in practice?</b>", "Real error sizes of each hint source (section 4); AI depth harm; whole-room iPad hints", "Ground-truth quality checks and excluded clips", "ADVIO; Zurich drone (unless a GPS test is added)"],
    ["<b>3. Can it be fixed without retraining?</b>", "The fix vs all remedies; harm on good hints; extra time", "Settings of the fix; the drop-only variant", "&mdash;"]])}
  <div class="box ok"><p><b>Figure 1 of the paper (planned):</b> three approaches (use all hints, photos only, the fix) on
  good hints and on bad hints. Using all hints wins on good hints and loses badly on bad ones; photos only is the reverse;
  the fix aims to be close to the best in both. That one picture is the whole paper.</p></div>
</section>

<section class="col stack">
  <h2>9. What is needed next</h2>
  <ol class="list">
    <li>Approval of the plan in <code>experiments/E2X.md</code> (section 5), so it can be locked.</li>
    <li>The RTX PRO 6000 VM for one session (estimate 2.5&ndash;3.5 GPU-hours).</li>
    <li>A LaMAR access request (lamar.ethz.ch), which only the project owner can submit.</li>
  </ol>
</section>
""")
    body = "\n".join(S)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Project status and E2X plan</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>{CSS}{EXTRA_CSS}</style></head>
<body><main class="page">{body}</main></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_browser")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    html = os.path.join(OUT, "e2x_plan.html")
    with open(html, "w", encoding="utf-8", newline="\n") as f:
        f.write(build())
    print("wrote", html)
    if a.pdf_browser:
        pdf = pathlib.Path(OUT, "e2x_plan.pdf").resolve()
        subprocess.run([a.pdf_browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={pdf}", pathlib.Path(html).resolve().as_uri()], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("wrote", pdf)


if __name__ == "__main__":
    main()
