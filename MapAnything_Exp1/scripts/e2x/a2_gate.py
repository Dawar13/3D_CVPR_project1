"""Session A2 dev gates (experiments/E2X.md, X-M "A2 grid and readouts"). Exit 0 = PASS, 1 = FAIL; prints one line.

  python scripts/e2x/a2_gate.py xm --run results/xm_ma_dev [--backend mapanything]
      0 failures; every case complete (its ref/bad row exists); MapAnything only: the dial at gamma = 1 equals the
      ungated bad run (rotation error and shape error within 1e-3); ref/clean has a lower rotation error than
      ref/rgb_only on >= 2 of 3 dev clips (>= 2/3 of the clips if there are more).
  python scripts/e2x/a2_gate.py m8 --run results/m8_ma_dev
      0 failures; m8_pose1.0_depth1.0 equals clean_pose_depth2 (same inputs) within 1e-6 on rotation and shape.
"""

import argparse
import os
import sys

import pandas as pd


def fail(name, why):
    print(f"GATE a2 {name}: FAIL ({why})")
    sys.exit(1)


def xm(run, backend):
    name = f"xm {run}"
    if os.path.exists(os.path.join(run, "failures.jsonl")):
        fail(name, "failures.jsonl exists")
    R = pd.read_csv(os.path.join(run, "per_run.csv"))
    done = R[(R.experiment == "ref") & (R.source == "bad")]
    cases = R.groupby(["clip_id", "case"]).size().index
    if len(done) != len(cases):
        fail(name, f"{len(cases) - len(done)} incomplete cases")
    if backend == "mapanything":
        one = R[(R.experiment == "m1_dial") & (R.gamma.astype(float) == 1.0)].set_index(["clip_id", "case"])
        bad = done.set_index(["clip_id", "case"])
        for col in ("rre_deg", "pts_relerr_scale"):
            d = (one[col] - bad.loc[one.index, col]).abs().max()
            if not d <= 1e-3:
                fail(name, f"gamma = 1 differs from no gate on {col} by {d}")
    ref = R[R.experiment == "ref"].pivot_table(index="clip_id", columns="source", values="rre_deg", aggfunc="first")
    wins = int((ref["clean"] < ref["rgb_only"]).sum())
    need = min(len(ref), max(2, -(-2 * len(ref) // 3)))
    if wins < need:
        fail(name, f"clean hints beat images only on {wins} of {len(ref)} clips (need {need})")
    print(f"GATE a2 {name}: PASS ({len(cases)} cases, clean beats images only on {wins}/{len(ref)} clips)")


def m8(run):
    name = f"m8 {run}"
    if os.path.exists(os.path.join(run, "failures.jsonl")):
        fail(name, "failures.jsonl exists")
    R = pd.read_csv(os.path.join(run, "per_run.csv"))
    R = R[(R.control == "none") & (R.view_subset == "all")].set_index(["clip_id", "condition"])
    a = R.xs("m8_pose1.0_depth1.0", level="condition")
    b = R.xs("clean_pose_depth2", level="condition").loc[a.index]
    for col in ("rre_deg", "pts_relerr_scale"):
        d = (a[col] - b[col]).abs().max()
        if not d <= 1e-6:
            fail(name, f"pose x1 depth x1 differs from clean_pose_depth2 on {col} by {d}")
    print(f"GATE a2 {name}: PASS ({len(a)} clips)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["xm", "m8"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--backend", default="mapanything")
    a = ap.parse_args()
    xm(a.run, a.backend) if a.kind == "xm" else m8(a.run)
