"""Automatic dev gates of the E2-D/E/F preregistration (experiments/E2.md, e2-prereg-DEF). Run on the VM by
scripts/session_queue.sh between a part's dev run and its eval run; exit code 0 = PASS (start the eval run).

  python scripts/dev_gates.py ma_e2df --run results/e2df_dev --config configs/e2df_mapanything.json
  python scripts/dev_gates.py ov_e2d  --run results/e2d_dev  --config configs/e2d_full.json --checks results/e2d_checks
  python scripts/dev_gates.py e2e     --run results/e2e_ma_dev --config configs/e2e_mapanything.json

Gates:
  every part : no failures.jsonl, every run status ok, and every expected run present
  ov_e2d     : repeat_clean within 1/10 of the practical thresholds of clean_pose on every dev clip
               (shape 0.1 pp, rotation 0.025 deg), and clean_pose rotation error below rgb_only on >= 2 of 3 clips
  e2e        : gt_pose rotation error below rgb_only in >= 2 of 3 dev visits (scene = mean over its clips)
"""

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from exp1.run_exp1 import expand_conditions, seeds_for  # noqa: E402


def runs(run_dir):
    pr = pd.read_csv(os.path.join(run_dir, "per_run.csv"))
    return pr[(pr.view_subset == "all") & ~pr.control.astype(str).str.startswith("ref:")]


def complete(run_dir, config):
    """Reasons the run is incomplete or failed (empty list = fine)."""
    why = []
    if os.path.exists(os.path.join(run_dir, "failures.jsonl")):
        why.append(f"{run_dir}: failures.jsonl exists")
    r = runs(run_dir)
    if (r.status != "ok").any():
        why.append(f"{run_dir}: {(r.status != 'ok').sum()} runs not ok")
    cfg = json.load(open(config))
    per_clip = sum(len(seeds_for(c, cfg)) * (1 + len(c.get("controls", []))) for c in expand_conditions(cfg))
    n_clips = r.clip_id.nunique()
    if len(r) != per_clip * n_clips or n_clips == 0:
        why.append(f"{run_dir}: {len(r)} runs, expected {per_clip} x {n_clips} clips")
    return why


def by_clip(r, cond, metric):
    return r[(r.condition == cond) & (r.control == "none")].groupby("clip_id")[metric].mean()


def hints_help(r, hinted, unit="clip_id", need=2):
    """hinted rotation error below rgb_only in >= need units (clips or scenes)."""
    a = r[(r.condition == hinted) & (r.control == "none")].groupby(unit).rre_deg.mean()
    b = r[(r.condition == "rgb_only") & (r.control == "none")].groupby(unit).rre_deg.mean()
    better = int((a < b.reindex(a.index)).sum())
    return [] if better >= need else [f"{hinted} beats rgb_only on rotation in only {better} of {len(a)} {unit}s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["ma_e2df", "ov_e2d", "e2e"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checks", help="ov_e2d: the configs/e2d_checks.json run")
    a = ap.parse_args()
    why = complete(a.run, a.config)
    r = runs(a.run) if not why else None
    if a.part == "ov_e2d":
        why += [] if a.checks and os.path.exists(os.path.join(a.checks, "per_run.csv")) else ["no checks run"]
        if a.checks and not why:
            if os.path.exists(os.path.join(a.checks, "failures.jsonl")):
                why.append("checks: failures.jsonl exists")
            c = runs(a.checks)
            for m, tol in (("pts_relerr_scale", 0.001), ("rre_deg", 0.025)):
                d = (by_clip(c, "repeat_clean", m) - by_clip(c, "clean_pose", m)).abs()
                if d.isna().any() or len(d) == 0 or (d > tol).any():
                    why.append(f"repeat_clean vs clean_pose {m}: max |diff| {d.max():.5f} > {tol}")
        if r is not None:
            why += hints_help(r, "clean_pose")
    elif a.part == "e2e" and r is not None:
        why += hints_help(r, "gt_pose", unit="scene")
    print(f"GATE {a.part} {a.run}: " + ("PASS" if not why else "FAIL\n  " + "\n  ".join(why)))
    sys.exit(1 if why else 0)


if __name__ == "__main__":
    main()
