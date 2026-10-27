"""E0 Step 5: ground-truth self-consistency of the prepared clips. CPU only, no model.

For every clip and every view pair (0, j): back-project view-0 GT depth with K0 and
pose0, project it into view j with Kj and pose_j, and compare the projected z with
view j's GT depth at the landing pixel (nearest). Per pair it reports:
  overlap    fraction of valid view-0 pixels that land on a valid view-j pixel
  median_rel median |z_proj - d_j| / d_j over those pixels (robust to occlusion)
  rel_rK     median_rel in 4 radial bins (distance from the image centre, 0 = centre,
             1 = corner). Error that grows outwards means depth and images disagree
             on lens distortion.

  python scripts/e0_check_clips.py --clips 'data/clips/dev__*.npz' --out reports/e0_clip_check

PASS: median_rel < 3% for every pair with overlap > 5%. Exit code 1 on FAIL.
Writes <out>/clip_check.csv and <out>/<clip_id>.png.
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from exp1.datasets import ETH3D_ROTATED, load_clip  # noqa: E402
from exp1.geometry import invert_pose  # noqa: E402

PASS_REL, MIN_OVERLAP = 0.03, 0.05
RADIAL_BINS = [0.0, 0.25, 0.5, 0.75, 1.01]


def reproject(depth_i, K_i, T_i, K_j, T_j):
    """Valid view-i pixels -> (u_j, v_j, z_j) in view j. Poses are cam2world."""
    v, u = np.nonzero(depth_i > 0)
    z = depth_i[v, u].astype(np.float64)
    pc = np.stack([(u - K_i[0, 2]) / K_i[0, 0] * z, (v - K_i[1, 2]) / K_i[1, 1] * z, z], -1)
    T = invert_pose(T_j) @ T_i
    pj = pc @ T[:3, :3].T + T[:3, 3]
    zj = pj[:, 2]
    zs = np.where(zj > 1e-6, zj, np.nan)
    return K_j[0, 0] * pj[:, 0] / zs + K_j[0, 2], K_j[1, 1] * pj[:, 1] / zs + K_j[1, 2], zj


def pair_stats(clip, i, j):
    depth = clip["depth"]
    H, W = depth.shape[1:]
    uj, vj, zj = reproject(depth[i], clip["K"][i], clip["poses"][i], clip["K"][j], clip["poses"][j])
    n_valid = len(zj)
    iu, iv = np.round(uj), np.round(vj)
    ok = (zj > 1e-6) & (iu >= 0) & (iu < W) & (iv >= 0) & (iv < H)
    iu, iv, zj, uj, vj = iu[ok].astype(int), iv[ok].astype(int), zj[ok], uj[ok], vj[ok]
    dj = depth[j][iv, iu]
    hit = dj > 0
    rel = np.abs(zj[hit] - dj[hit]) / dj[hit]
    r = (np.hypot(iu[hit] - (W - 1) / 2, iv[hit] - (H - 1) / 2)
         / np.hypot((W - 1) / 2, (H - 1) / 2))
    row = {"pair": f"{i}-{j}", "n_px": int(hit.sum()),
           "overlap": float(hit.sum() / max(n_valid, 1)),
           "median_rel": float(np.median(rel)) if len(rel) else np.nan,
           "p90_rel": float(np.percentile(rel, 90)) if len(rel) else np.nan}
    for k in range(len(RADIAL_BINS) - 1):
        sel = (r >= RADIAL_BINS[k]) & (r < RADIAL_BINS[k + 1])
        row[f"rel_r{k}"] = float(np.median(rel[sel])) if sel.sum() >= 20 else np.nan
    return row, (uj[hit], vj[hit], rel)


def save_figure(clip, pts, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d0 = np.where(clip["depth"][0] > 0, clip["depth"][0], np.nan)
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
    ax[0].imshow(clip["images"][0])
    ax[0].set_title("view 0")
    im = ax[1].imshow(d0, cmap="turbo")
    ax[1].set_title("view 0 GT depth (m), blank = invalid", fontsize=9)
    fig.colorbar(im, ax=ax[1], fraction=0.035)
    ax[2].imshow(clip["images"][1])
    u, v, rel = pts
    sub = np.random.default_rng(0).permutation(len(u))[:4000]
    sc = ax[2].scatter(u[sub], v[sub], c=np.clip(rel[sub], 0, 0.1), s=1.5, cmap="RdYlGn_r",
                       vmin=0, vmax=0.1)
    ax[2].set_title("view 1 + view-0 points (rel. depth diff, 0-10%)", fontsize=9)
    fig.colorbar(sc, ax=ax[2], fraction=0.035)
    for a in ax:
        a.set_axis_off()
    fig.suptitle(f"{clip['clip_id']}  depth_undistorted={clip.get('depth_undistorted', 'n/a')}",
                 fontsize=9)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def check_clip(clip, out_dir=None):
    rows, pts1 = [], None
    for j in range(1, len(clip["depth"])):
        row, pts = pair_stats(clip, 0, j)
        rows.append(row)
        if j == 1:
            pts1 = pts
    meaningful = [r for r in rows if r["overlap"] > MIN_OVERLAP]
    if not meaningful:
        verdict = "NO_OVERLAP"
    else:
        verdict = "PASS" if all(r["median_rel"] < PASS_REL for r in meaningful) else "FAIL"
    if out_dir is not None:
        save_figure(clip, pts1, os.path.join(out_dir, f"{clip['clip_id']}.png"))
    return rows, verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True, help="glob of clip .npz files")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    paths = sorted(glob.glob(a.clips))
    if not paths:
        raise SystemExit(f"no clips match {a.clips}")
    os.makedirs(a.out, exist_ok=True)
    fields = ["clip_id", "pair", "n_px", "overlap", "median_rel", "p90_rel",
              *[f"rel_r{k}" for k in range(len(RADIAL_BINS) - 1)]]
    verdicts = {}
    with open(os.path.join(a.out, "clip_check.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for p in paths:
            clip = load_clip(p)
            rows, verdict = check_clip(clip, a.out)
            verdicts[clip["clip_id"]] = verdict
            for r in rows:
                w.writerow({"clip_id": clip["clip_id"], **r})
            names = [os.path.basename(str(n)) for n in clip["names"]]
            rotated = sorted(set(names) & set(ETH3D_ROTATED.get(clip["scene"], [])))
            m = [r for r in rows if r["overlap"] > MIN_OVERLAP]
            radial = np.nanmedian([[r[f"rel_r{k}"] for k in range(4)] for r in m], 0) if m else []
            print(f"[{verdict}] {clip['clip_id']}  depth_undistorted="
                  f"{clip.get('depth_undistorted', 'n/a')}  pairs>{MIN_OVERLAP:.0%}overlap="
                  f"{len(m)}/{len(rows)}  worst median_rel="
                  f"{max((r['median_rel'] for r in m), default=float('nan')):.4f}  "
                  f"radial(centre->corner)={np.round(radial, 4).tolist()}"
                  + (f"  ROTATED images: {rotated}" if rotated else ""))
    bad = {k: v for k, v in verdicts.items() if v != "PASS"}
    print("OVERALL:", "PASS" if not bad else f"NOT PASS {bad}")
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
