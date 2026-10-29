"""E2-D step 1: summarise the OmniVGGT checks run with configs/e2d_checks.json.

  python scripts/e2d_checks.py --run results/e2d_checks
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

M = ["pts_relerr_scale", "rre_deg", "depth_absrel_scale"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    pr = pd.read_csv(os.path.join(a.run, "per_run.csv"))
    pv = pd.read_csv(os.path.join(a.run, "per_view.csv"))
    man = json.load(open(os.path.join(a.run, "corruption_manifest.json")))
    runs = pr[(pr.view_subset == "all") & (pr.control == "none")]
    fails = os.path.exists(os.path.join(a.run, "failures.jsonl"))
    print(f"runs {len(runs)}  all ok {bool((runs.status == 'ok').all())}  failures file {fails}")
    print(f"runtime median {runs.runtime_s.median():.2f} s  peak GPU memory {runs.peak_mem_gb.max():.2f} GB")
    print("\nexact output checks (max |pts3d - clean_pose pts3d| over all pixels; 0 = identical):")
    for e in man:
        for k, v in e.items():
            if k.startswith("max_abs_diff_vs_"):
                print(f"  {e['clip_id']:24s} {e['condition']:22s} {v:.3g}")
    m = runs.set_index(["clip_id", "condition"])[M]
    print("\nper clip (lower is better):")
    print(m.unstack("condition").round(4).T.to_string())
    d = (m.xs("inv_depth2_x1.5", level="condition") - m.xs("rgb_depth2", level="condition")).abs().max().max()
    print(f"\ninvariance, both depth hints x1.5 vs unscaled: max metric difference {d:.3g}")
    o = pv[(pv.condition == "outlier_rot_rot_deg=15") & (pv.was_corrupted == 1)].absorption_rot
    print(f"one camera 15 deg: median absorption (E1 definition) {o.median():.3f} over {len(o)} bad views")


if __name__ == "__main__":
    main()
