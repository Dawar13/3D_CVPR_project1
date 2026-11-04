"""Exclusion rule of the extended-E2 SceneFun3D clips (experiments/E2X.md section 3.2), applied before any model run.
A clip is excluded if it fails the E0 ground-truth check (scripts/e0_check_clips.py: median_rel >= 3% on a pair with
> 5% overlap) or if a GT view fails the photo check (exp1/photo_check.py): its score is > MAX_PX AND more than
RATIO x the score of the device pose of the same view, which is scored on the same matches. The device poses (ARKit)
are independent of the GT (COLMAP), so "both fit badly" means the matches are unreliable (seen on wide-baseline room
pairs, where both sources agree within a few degrees), while "GT fits badly, device fits" means the GT is wrong.
Excluded clips are moved to <folder>/excluded/, outside every run glob. Run from MapAnything_Exp1/, CPU only:

  python scripts/e2x/check_clips.py data/clips_e2x_room data/clips_e2x_cal --out results/e2x_clip_check
"""

import argparse
import csv
import glob
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, ".."))
from e0_check_clips import MIN_OVERLAP, PASS_REL, check_clip  # noqa: E402
from exp1 import photo_check  # noqa: E402
from exp1.datasets import load_clip  # noqa: E402

MAX_PX, RATIO = 2.0, 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    gt_rows, ph_rows = [], []
    for folder in a.folders:
        excluded = []
        for p in sorted(glob.glob(os.path.join(folder, "*.npz"))):
            clip = load_clip(p)
            cid = clip["clip_id"]
            rows, verdict = check_clip(clip, a.out)
            gt_rows += [{"clip_id": cid, **r} for r in rows]
            ph, _ = photo_check.check_clip(dict(np.load(p)), ("poses", "dev_poses"))
            ph_rows += [{"clip_id": cid, **r} for r in ph]
            gt = {r["view"]: r["sampson_px"] for r in ph if r["key"] == "poses"}
            dv = {r["view"]: r["sampson_px"] for r in ph if r["key"] == "dev_poses"}
            bad_views = [v for v, x in gt.items() if np.isfinite(x) and x > MAX_PX and x > RATIO * dv.get(v, np.inf)]
            worst_px = max((x for x in gt.values() if np.isfinite(x)), default=float("nan"))
            m = [r for r in rows if r["overlap"] > MIN_OVERLAP]
            worst_rel = max((r["median_rel"] for r in m), default=float("nan"))
            why = []
            if verdict == "FAIL":
                why.append(f"failed the E0 ground-truth check (worst median_rel {worst_rel:.4f} >= {PASS_REL} on a "
                           f"pair with >{MIN_OVERLAP:.0%} overlap)")
            if bad_views:
                why.append("GT view(s) fail the photo check: " + ", ".join(
                    f"view {v} {gt[v]:.2f} px (device {dv.get(v, np.nan):.2f} px)" for v in bad_views))
            print(f"[{'EXCLUDED' if why else 'ok'}] {cid}  E0 {verdict} (worst {worst_rel:.4f})  "
                  f"photo worst GT view {worst_px:.2f} px", flush=True)
            if why:
                excluded.append((cid, "; ".join(why), p))
        with open(os.path.join(folder, "excluded.csv"), "w", newline="") as f:
            csv.writer(f).writerows([("clip_id", "reason"), *[(c, r) for c, r, _ in excluded]])
        shutil.copy(os.path.join(folder, "excluded.csv"),
                    os.path.join(a.out, f"excluded_{os.path.basename(os.path.normpath(folder))}.csv"))
        if excluded:
            os.makedirs(os.path.join(folder, "excluded"), exist_ok=True)
            for _, _, p in excluded:
                shutil.move(p, os.path.join(folder, "excluded", os.path.basename(p)))
        print(f"{folder}: {len(excluded)} excluded", flush=True)
    for name, rows in (("gt_check.csv", gt_rows), ("photo_check.csv", ph_rows)):
        if rows:
            with open(os.path.join(a.out, name), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)


if __name__ == "__main__":
    main()
