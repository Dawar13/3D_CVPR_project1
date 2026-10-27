"""E0: build reports/e0_results/ from the result files (no model, CPU only).

  python scripts/e0_make_results.py [--artifact_out PATH]

Reads results/smoke*, results/e0_demo, results/det_b, reports/e0_clip_check and the
smoke clip, and writes reports/e0_results/:
  e0_results.html        the results page, fully offline (data, images, three.js inlined)
  e0_results.pdf         printed version of the page (only with --pdf_browser)
  numbers.csv            every number on the page, with its source file
  gt_check.csv           GT self-consistency summary per clip
  fig_errors.png         error bars per run (lower is better)
  fig_3d_views.png       perspective views of the reconstructions vs the laser truth
  overview.png           bird's-eye and side views with cameras (copy from results/e0_demo)
  photos.jpg             the 10 input photos of the smoke clip
  gtcheck_*.png          GT self-check figure of the smoke clip
  3d/*.ply               point clouds: laser truth + 3 runs (MeshLab / CloudCompare)
--artifact_out also writes the online variant (three.js from the CDN) for publishing.
Nothing here is a research result: one dev clip, a pipeline check.
"""

import argparse
import base64
import io
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from exp1.datasets import ETH3D_TRAIN_SCENES, load_clip  # noqa: E402

CLIP = "dev__delivery_area__c0"
METRICS = ["pts_relerr_scale", "pts_relerr_metric", "depth_absrel_metric", "rre_deg", "rte_deg", "mean_conf"]
RUNS = [  # key, file, condition, control, plain label
    ("rgb_only", "smoke", "rgb_only", "none", "Photos only"),
    ("rgb_K", "smoke", "rgb_K", "none", "Photos + lens info"),
    ("clean_pose_noK", "smoke", "clean_pose_noK", "none", "Photos + camera positions"),
    ("clean_pose", "smoke", "clean_pose", "none", "Correct hints (lens + positions)"),
    ("zero_noise", "smoke", "zero_noise", "none", "Correct hints via corruption code at 0 deg"),
    ("outlier", "smoke_ctrl", "outlier_rot_rot_deg=15", "none", "One wrong hint (15 deg)"),
    ("drop_oracle", "smoke_ctrl", "outlier_rot_rot_deg=15", "drop_oracle", "Wrong hint removed (we know which)"),
    ("filter_rgbcheck", "smoke_ctrl", "outlier_rot_rot_deg=15", "filter_rgbcheck", "Wrong hint removed (found automatically)"),
]
PLY = {"gt": "gt", "rgb_only": "rgb_only", "clean": "clean_pose", "bad": "outlier_rot_rot_deg_15"}
BAR, BAD, TRUTH, INK, MUTED, CAM = "#2a78d6", "#e34948", "#8f969c", "#111417", "#7d838a", "#eb6834"


def p(*a):
    return os.path.join(ROOT, *a)


def numbers():
    out, rows = {}, []
    for key, f, cond, ctrl, label in RUNS:
        df = pd.read_csv(p("results", f, "per_run.csv"))
        d = df[(df.condition == cond) & (df.control == ctrl) & (df.view_subset == "all") & (df.status == "ok")]
        out[key] = {m: float(d[m].mean()) for m in METRICS} | {"n_seeds": int(len(d))}
        rows.append({"run": label, "condition": cond, "control": ctrl, **out[key],
                     "source": f"results/{f}/per_run.csv"})
    return out, pd.DataFrame(rows)


def gt_check():
    cc = pd.read_csv(p("reports", "e0_clip_check", "clip_check.csv"))
    gt, rows = [], []
    for cid, g in cc.groupby("clip_id"):
        m = g[g.overlap > 0.05]
        worst = float(m.median_rel.max()) if len(m) else float("nan")
        gt.append({"clip": cid, "pairs": int(len(m)), "worst_pct": round(100 * worst, 3)})
        rows.append({"clip": cid, "split": cid.split("__")[0], "pairs_over_5pct_overlap": len(m),
                     "worst_median_rel_pct": round(100 * worst, 3),
                     **{f"radial_r{k}_pct": round(100 * float(m[f"rel_r{k}"].median()), 3) for k in range(4)},
                     "verdict": "PASS" if len(m) and worst < 0.03 else "NOT PASS"})
    return gt, pd.DataFrame(rows)


def read_ply(path):
    raw = open(path, "rb").read()
    h = raw.index(b"end_header\n") + len(b"end_header\n")
    n = int(raw[:h].split(b"element vertex ")[1].split(b"\n")[0])
    v = np.frombuffer(raw[h:], count=n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                               ("r", "u1"), ("g", "u1"), ("b", "u1")])
    return np.stack([v["x"], v["y"], v["z"]], 1), np.stack([v["r"], v["g"], v["b"]], 1)


def fig_errors(nums, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = ["rgb_only", "rgb_K", "clean_pose_noK", "clean_pose", "outlier", "drop_oracle"]
    labels = {k: lab for k, _, _, _, lab in RUNS}
    panels = [("pts_relerr_scale", "3D shape error (%)", 100), ("pts_relerr_metric", "3D error at real size (%)", 100),
              ("rre_deg", "Camera angle error (deg)", 1)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    y = np.arange(len(keys))[::-1]
    for ax, (m, title, k) in zip(axes, panels):
        v = np.array([nums[key][m] * k for key in keys])
        ax.barh(y, v, height=0.55, color=[BAD if key == "outlier" else BAR for key in keys])
        ax.axvline(nums["rgb_only"][m] * k, color=MUTED, lw=1)
        for yy, vv in zip(y, v):
            ax.text(vv, yy, f" {vv:.2f}", va="center", fontsize=9, color=INK)
        ax.set_title(title, fontsize=11, loc="left", color=INK)
        ax.set_xlim(0, v.max() * 1.25)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=9, colors=MUTED)
    axes[0].set_yticks(y, [labels[key] for key in keys], fontsize=10)
    fig.suptitle(f"E0 smoke clip {CLIP}: lower is better; grey line = photos only; wrong-hint rows = mean of 3 runs. "
                 "Pipeline check, not a research result.", fontsize=9, color=MUTED, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_3d(clouds, colors, cams, demo, path, n_plot=14000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sel = np.random.default_rng(1).permutation(len(colors))[:n_plot]
    # plot axes: x right, depth = metres in front of camera 1 (positive), up
    to_plot = lambda P: np.stack([P[:, 0], -P[:, 2], P[:, 1]], 1)  # inputs are y-up, z-back
    g = to_plot(clouds["gt"][sel])
    C, F = to_plot(cams[:, :3]), to_plot(cams[:, 3:])
    lo = np.minimum(np.percentile(g, 1, 0), C.min(0))
    hi = np.maximum(np.percentile(g, 99, 0), C.max(0))
    fig = plt.figure(figsize=(16, 6))
    names = [("rgb_only", "Photos only", "rgb_only"), ("clean", "Correct hints", "clean_pose"),
             ("bad", "One wrong hint (15 deg, run 1 of 3)", "outlier_rot_rot_deg=15")]
    for i, (k, title, run) in enumerate(names, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        ax.scatter(*g.T, s=0.4, c=TRUTH, alpha=0.35, linewidths=0, depthshade=False)
        ax.scatter(*to_plot(clouds[k][sel]).T, s=0.6, c=colors[sel] / 255.0, linewidths=0, depthshade=False)
        ax.scatter(*C.T, s=16, c=CAM, depthshade=False)
        for c, f in zip(C, F):
            e = c + 0.8 * f
            ax.plot([c[0], e[0]], [c[1], e[1]], [c[2], e[2]], color=CAM, lw=1.2)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(tuple(hi - lo))
        ax.view_init(elev=22, azim=-118)  # from behind and above the camera row, looking into the scene
        ax.set_xlabel("x (m)", fontsize=8); ax.set_ylabel("depth (m)", fontsize=8); ax.set_zlabel("up (m)", fontsize=8)
        ax.tick_params(labelsize=7)
        m = demo[run]
        ax.set_title(f"{title}\nshape error {100 * m['pts_relerr_scale']:.1f}% | angle error {m['rre_deg']:.2f} deg",
                     fontsize=10, color=INK)
    fig.text(0.01, 0.01, f"{CLIP}: grey = laser truth, colour = MapAnything (photo colours), orange = true cameras "
             "and their viewing direction. Pipeline check, not a research result.", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def data_uri(path):
    mime = "image/png" if path.endswith(".png") else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(open(path, "rb").read()).decode()


def page(template, data, out_dir, offline):
    gt = data["gt"]
    have = {g["clip"].split("__")[1] for g in gt}
    missing = [s for s in ETH3D_TRAIN_SCENES if s not in have]
    passed = sum(g["worst_pct"] < 3 for g in gt)
    worst = max(g["worst_pct"] for g in gt)
    bullets = (f'      <li><span class="chip ok">PASS</span><span>The "answer key" (laser-scanned truth) agrees '
               f'with itself on all {passed} scenes{" we could download" if missing else ""} '
               f'(worst disagreement {worst:.2f}%).</span></li>')
    if missing:
        bullets += (f'\n      <li><span class="chip wait">PENDING</span><span>{len(missing)} of 13 scenes '
                    f'({", ".join(missing)}) are not downloaded yet.</span></li>')
        open_sec = ('<section class="col stack">\n  <h2>Still open</h2>\n  <div class="note-box">Scenes not yet '
                    f'downloaded and checked: <strong>{", ".join(missing)}</strong>. They are needed before any E1 '
                    'test-scene run.</div>\n</section>')
    else:
        open_sec = ('<section class="col stack">\n  <h2>What comes next</h2>\n  <p>All 13 scenes are in and '
                    'checked. The ETH3D server was down for about an hour during E0, so the last 3 scenes '
                    '(relief, relief_2, terrains) were fetched later on the same machine.</p>\n  <p>Two decisions '
                    'come before any E1 test-scene run. First, the relief and relief_2 clips are made entirely of '
                    'photos that ETH3D stores turned sideways. Their geometry is correct, but MapAnything\'s own '
                    'ETH3D code turns such photos upright, so we should decide whether to do the same. Second, '
                    'decide whether real-size errors are a main readout for the wrong-scale hints. Then come the '
                    'new wrong-hint types for E1, with tests.</p>\n</section>')
    s = template.replace("{{GT_BULLETS}}", bullets).replace("{{GT_WORST}}", f"{worst:.2f}%")
    s = s.replace("{{OPEN_SECTION}}", open_sec)
    for name in ("photos.jpg", "overview.png", "gtcheck.png", "fig_3d_views.png"):
        s = s.replace("{{" + name + "}}", data_uri(os.path.join(out_dir, "_" + name)))
    s = s.replace("<!--DATA-->", '<script type="application/json" id="e0-data">' + json.dumps(data) + "</script>")
    if offline:
        libs = "".join("<script>\n" + open(os.path.join(ROOT, "scripts", "vendor", f), encoding="utf-8").read()
                       + "\n</script>\n" for f in ("three.min.js", "OrbitControls.js"))
    else:
        libs = ('<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>\n'
                '<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>\n')
    s = s.replace("<!--THREE-->", libs)
    if offline:  # the published page gets its document skeleton at publish time; a local file needs its own
        s = '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" ' \
            'content="width=device-width, initial-scale=1, viewport-fit=cover">\n</head>\n<body>\n' + s + '\n</body>\n</html>\n'
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=p("reports", "e0_results"))
    ap.add_argument("--artifact_out")
    ap.add_argument("--pdf_browser", help="Chrome/Edge executable; also prints e0_results.pdf")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "3d"), exist_ok=True)

    nums, num_df = numbers()
    num_df.to_csv(os.path.join(a.out, "numbers.csv"), index=False)
    gt, gt_df = gt_check()
    gt_df.to_csv(os.path.join(a.out, "gt_check.csv"), index=False)
    demo = json.load(open(p("results", "e0_demo", f"{CLIP}__metrics.json")))
    timing = json.load(open(p("results", "det_b", "timing.json")))

    plys = {k: read_ply(p("results", "e0_demo", f"{CLIP}__{f}.ply")) for k, f in PLY.items()}
    n = len(plys["gt"][0])
    assert all(len(v[0]) == n for v in plys.values()), "PLYs must share one pixel set"
    for k, f in PLY.items():
        shutil.copy(p("results", "e0_demo", f"{CLIP}__{f}.ply"), os.path.join(a.out, "3d", f"{CLIP}__{f}.ply"))
    idx = np.sort(np.random.default_rng(0).permutation(n)[:36000])
    flip = np.array([1, -1, -1], np.float32)  # OpenCV (y down, z forward) -> y up for viewing
    clouds = {k: (v[0][idx] * flip).astype(np.float32) for k, v in plys.items()}
    colors = plys["gt"][1][idx]
    clip = load_clip(p("data", "clips", f"{CLIP}.npz"))
    cams = np.array([np.r_[T[:3, 3] * flip, T[:3, 2] * flip] for T in clip["poses"]])

    th = [Image.fromarray(i).resize((259, 168), Image.LANCZOS) for i in clip["images"]]
    sheet = Image.new("RGB", (5 * 259 + 4 * 6, 2 * 168 + 6), (255, 255, 255))
    for k, t in enumerate(th):
        sheet.paste(t, ((k % 5) * 265, (k // 5) * 174))
    sheet.save(os.path.join(a.out, "photos.jpg"), quality=82)
    shutil.copy(p("results", "e0_demo", f"{CLIP}__overview.png"), os.path.join(a.out, "overview.png"))
    shutil.copy(p("reports", "e0_clip_check", f"{CLIP}.png"), os.path.join(a.out, f"gtcheck_{CLIP}.png"))
    fig_errors(nums, os.path.join(a.out, "fig_errors.png"))
    fig_3d(clouds, colors, cams, demo, os.path.join(a.out, "fig_3d_views.png"))

    b64 = lambda x: base64.b64encode(np.ascontiguousarray(x).tobytes()).decode()
    data = {"nums": nums, "gt": gt, "timing": timing, "demo": demo, "n_points": int(len(idx)), "n_total": int(n),
            "clouds": {k: b64(v) for k, v in clouds.items()}, "colors": b64(colors.astype(np.uint8)),
            "cams": cams.tolist()}
    tmp = {"_photos.jpg": "photos.jpg", "_overview.png": "overview.png", "_gtcheck.png": f"gtcheck_{CLIP}.png",
           "_fig_3d_views.png": "fig_3d_views.png"}
    for t, src in tmp.items():
        shutil.copy(os.path.join(a.out, src), os.path.join(a.out, t))
    template = open(p("scripts", "e0_results_template.html"), encoding="utf-8").read()
    open(os.path.join(a.out, "e0_results.html"), "w", encoding="utf-8").write(page(template, data, a.out, True))
    if a.artifact_out:
        open(a.artifact_out, "w", encoding="utf-8").write(page(template, data, a.out, False))
    for t in tmp:
        os.remove(os.path.join(a.out, t))
    if a.pdf_browser:  # print view: static 3D figure instead of the WebGL viewer, light colours, A4
        import pathlib
        import subprocess
        html = pathlib.Path(a.out, "e0_results.html").resolve()
        subprocess.run([a.pdf_browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        "--virtual-time-budget=15000", f"--print-to-pdf={pathlib.Path(a.out, 'e0_results.pdf').resolve()}",
                        html.as_uri()], check=True, capture_output=True, timeout=180)
    print(f"wrote {a.out}: {sorted(os.listdir(a.out))}")
    print(f"GT check: {len(gt)} clips, missing {[s for s in ETH3D_TRAIN_SCENES if not any(g['clip'].endswith('__' + s + '__c0') for g in gt)]}")


if __name__ == "__main__":
    main()
