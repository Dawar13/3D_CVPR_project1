"""Progress of the GPU queue on the VM: one bar per job, counted from the per-run files written so far
(merged folder and shard folders). Run from the HEAD tree:  python scripts/session_progress.py
Jobs: name | code tree | config | clip glob | out folder (relative to the tree)."""

import csv
import glob
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import json  # noqa: E402

from exp1.run_exp1 import expand_conditions, seeds_for  # noqa: E402

F = os.path.expanduser("~/3D-CVPR_Exp/MapAnything_Exp1")
H = os.path.expanduser("~/e2d/MapAnything_Exp1")
JOBS = [("parity vs E1", F, "configs/e2b_audit.json", "data/clips/eval__*.npz", "results/parity_rtx"),
        ("E2-C main grid", F, "configs/e2c_full.json", "data/clips_e2/eval__*.npz", "results/e2c_eval"),
        ("E2-C 4-view", F, "configs/e2c_views.json", "data/clips_n4/eval__*.npz", "results/e2c_n4_eval"),
        ("E2-C 20-view", F, "configs/e2c_views.json", "data/clips_n20/eval__*.npz", "results/e2c_n20_eval"),
        ("E2-D checks", H, "configs/e2d_checks.json", "data/clips/dev__*.npz", "results/e2d_checks"),
        ("E2-D dev (OmniVGGT)", H, "configs/e2d_full.json", "data/clips_e2/dev__*__c0.npz", "results/e2d_dev"),
        ("E2-D dev (MapAnything)", H, "configs/e2d_mapanything.json", "data/clips_e2/dev__*__c0.npz",
         "results/e2d_ma_dev")]


def expected(tree, config, clips):
    cfg = json.load(open(os.path.join(tree, config)))
    per_clip = sum(len(seeds_for(c, cfg)) * (1 + len(c.get("controls", []))) for c in expand_conditions(cfg))
    return per_clip * len(glob.glob(os.path.join(tree, clips)))


def done(tree, out):
    keys, fails = set(), 0
    for p in glob.glob(os.path.join(tree, out, "per_run.csv")) + glob.glob(os.path.join(tree, out, "_shards", "*", "run", "per_run.csv")):
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                if r["view_subset"] == "all" and not r["control"].startswith("ref:"):
                    keys.add(r["run_key"])
                    fails += r["status"] != "ok"
    return len(keys), fails


def bar(name, d, t, w=30):
    f = d * w // max(t, 1)
    return f"{name:<24} {'#' * f}{'.' * (w - f)} {d:5d} / {t:<5d} {100 * d // max(t, 1):3d}%"


if __name__ == "__main__":
    for name, tree, cfg, clips, out in JOBS:
        if not os.path.exists(os.path.join(tree, cfg)):
            continue
        d, nf = done(tree, out)
        print(bar(name, d, expected(tree, cfg, clips)) + (f"   FAILED: {nf}" if nf else ""))
    print(subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip())
