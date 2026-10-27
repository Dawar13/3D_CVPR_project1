"""E0 Step 8: determinism noise floor and timing for one condition on one clip.

Run twice (two processes, two folders), the second time comparing to the first:
  python scripts/e0_determinism.py --clip data/clips/dev__delivery_area__c0.npz --out results/det_a
  python scripts/e0_determinism.py --clip data/clips/dev__delivery_area__c0.npz --out results/det_b \
      --compare results/det_a

Each process runs --repeats inferences (the first is warm-up) and saves the last
prediction to <out>/pred.npz plus timings to <out>/timing.json. Reports max |delta|
of pts3d within the process and, with --compare, across processes, also as a
fraction of the clip's scene length L.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from exp1 import model_io  # noqa: E402
from exp1.datasets import load_clip  # noqa: E402
from exp1.geometry import trajectory_scale  # noqa: E402
from exp1.run_exp1 import expand_conditions  # noqa: E402

KEYS = ("pts3d", "depth_z", "pose", "conf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/pilot.json")
    ap.add_argument("--condition", default="clean_pose")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compare", help="output folder of an earlier run")
    a = ap.parse_args()

    cfg = json.load(open(a.config))
    cond = {c["name"]: c for c in expand_conditions(cfg)}[a.condition]
    if cond.get("corruption"):
        raise SystemExit("use an uncorrupted reference condition")
    clip = load_clip(a.clip)
    L = trajectory_scale(clip["poses"])
    model = model_io.load_model(cfg["checkpoint"], a.device)
    views = model_io.build_views(clip["images"], clip["K"], clip["poses"], cond["inputs"])

    preds = [model_io.run(model, views, a.device) for _ in range(a.repeats)]
    rt = [p["runtime_s"] for p in preds]
    within = max(float(np.abs(p["pts3d"] - preds[0]["pts3d"]).max()) for p in preds)
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, "pred.npz"), **{k: preds[-1][k] for k in KEYS})
    timing = {"clip": a.clip, "condition": a.condition, "n_views": len(clip["images"]),
              "runtime_s": rt, "runtime_s_after_warmup_median": float(np.median(rt[1:] or rt)),
              "peak_mem_gb": max(p["peak_mem_gb"] for p in preds), "L_m": L,
              "within_process_max_abs_pts3d": within}
    json.dump(timing, open(os.path.join(a.out, "timing.json"), "w"), indent=1)
    print(f"runtime per inference: {['%.3f' % t for t in rt]} s (first = warm-up); "
          f"peak {timing['peak_mem_gb']:.2f} GB")
    print(f"within-process max |d pts3d| over {a.repeats} repeats: {within:.3g} m = {within / L:.3g} L")

    if a.compare:
        other = np.load(os.path.join(a.compare, "pred.npz"))
        for k in KEYS:
            d = float(np.abs(preds[-1][k] - other[k]).max())
            extra = f" = {d / L:.3g} L" if k == "pts3d" else ""
            print(f"across processes max |d {k}|: {d:.3g}{extra}")


if __name__ == "__main__":
    main()
