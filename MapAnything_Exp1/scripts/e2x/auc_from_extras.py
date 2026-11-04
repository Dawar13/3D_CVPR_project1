"""Pose AUC@30 (exp1.metrics.auc) for finished runs, from their saved predicted poses (extras/*.npz, key
pred_pose_rel) and the clips' ground-truth poses. CPU only. Writes <run>/auc30.csv (run_key, auc30, rra5, rta5);
rra5/rta5 are recomputed as a check against per_run.csv. Run from MapAnything_Exp1/:

  python scripts/e2x/auc_from_extras.py results/e2c_eval data/clips_e2  [more run/clip-folder pairs ...]
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from exp1.metrics import pose_metrics  # noqa: E402


def one(run, clip_dir):
    pr = pd.read_csv(os.path.join(run, "per_run.csv"))
    pr = pr[pr.view_subset == "all"].set_index("run_key")
    gt, rows = {}, []
    for f in sorted(glob.glob(os.path.join(run, "extras", "*.npz"))):
        key = os.path.basename(f)[:-4].replace("__", "|")
        parts = key.split("|")
        clip_id = "__".join(parts[:3])
        run_key = "|".join([clip_id] + parts[3:])
        if clip_id not in gt:
            p = os.path.join(clip_dir, clip_id + ".npz")
            if not os.path.exists(p):
                continue
            gt[clip_id] = np.load(p)["poses"]
        m = pose_metrics(np.load(f)["pred_pose_rel"], gt[clip_id])
        row = {"run_key": run_key, "auc30": m["auc30"], "rra5": m["rra5"], "rta5": m["rta5"]}
        if run_key in pr.index:
            row["rra5_check"] = float(abs(pr.loc[run_key, "rra5"] - m["rra5"]))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(run, "auc30.csv"), index=False)
    worst = out["rra5_check"].max() if "rra5_check" in out else float("nan")
    print(f"{run}: {len(out)} runs, {len(gt)} clips; max |rra5 - per_run.csv| = {worst:.2e}")


if __name__ == "__main__":
    a = sys.argv[1:]
    for run, clip_dir in zip(a[::2], a[1::2]):
        one(run, clip_dir)
