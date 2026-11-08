"""E4-A dev diagnostic (experiments/E4.md §1.4 item 10): for every matched pair of the given dev / cal clips, how well
the matches fit (a) the refined photo estimate, (b) the ground-truth relative pose and (c) the device (ARKit)
relative pose, as median Sampson error in pixels, next to the rotation error of the photo estimate. A photo estimate
far from the ground truth whose matches fit the ground truth just as well is an ambiguous pair, not a wrong ground
truth. Dev / cal clips only.
  python scripts/e4/gt_epipolar_check.py --clips data/clips_e2x_cal/cal__sf483632__c2.npz ... --out results/e4_dev/gt_epipolar_check.csv
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from exp1 import photo_check, vtt  # noqa: E402
from exp1.datasets import load_clip  # noqa: E402


def rel(P, i, j):
    R = P[j][:3, :3].T @ P[i][:3, :3]
    d = P[i][:3, 3] - P[j][:3, 3]
    return R, P[j][:3, :3].T @ d / np.linalg.norm(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--cal", default="results/e4_vtt_cal/chosen.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = {k: v for k, v in json.load(open(a.cal)).items() if k.startswith("vtt_")}
    rows = []
    for p in sorted({p for g in a.clips for p in glob.glob(g)}):
        if os.path.basename(p).startswith("eval__"):
            raise SystemExit("dev / cal clips only")
        clip = load_clip(p)
        pairs = photo_check.match_pairs(clip["images"])
        for ed in vtt.pair_edges(pairs, clip["K"], cfg):
            i, j, f = ed["i"], ed["j"], clip["K"][ed["i"]][0, 0]
            r = {"clip_id": clip["clip_id"], "i": i, "j": j, "n": ed["n"],
                 "photo_px": float(np.median(np.abs(vtt._sampson(ed["R"], ed["t"], ed["a"], ed["b"]))) * f)}
            Rg, tg = rel(clip["poses"], i, j)
            r["gt_px"] = float(np.median(np.abs(vtt._sampson(Rg, tg, ed["a"], ed["b"]))) * f)
            r["photo_rot_err_deg"] = vtt.angle_deg(ed["R"], Rg)
            if clip.get("dev_poses") is not None:
                Rd, td = rel(clip["dev_poses"], i, j)
                r["device_px"] = float(np.median(np.abs(vtt._sampson(Rd, td, ed["a"], ed["b"]))) * f)
            rows.append(r)
        print(f"[ok] {clip['clip_id']}", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(a.out, index=False)
    with pd.option_context("display.width", 200):
        print(D.round(3).to_string())


if __name__ == "__main__":
    main()
