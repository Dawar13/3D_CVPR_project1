"""Experiment 1 runner: every condition x clip x seed, one inference each.

  CUDA_VISIBLE_DEVICES=4 python -m exp1.run_exp1 --config configs/pilot.json \
      --clips 'data/clips/dev__*.npz' --out results/dev_pilot

Writes (append-only, resumable):
  <out>/per_run.csv              one row per (clip, condition, seed, control, view_subset)
  <out>/corruption_manifest.json every corrupted input actually used
  <out>/failures.jsonl           any run that raised, with traceback
  <out>/run_info.json            config, checkpoint, git commit, torch/CUDA versions

Controls (only on outlier families):
  drop_oracle     : ORACLE diagnostic — the truly corrupted views get no pose input
  filter_rgbcheck : DEPLOYABLE — compare each supplied pose (relative to view 0)
                    with the RGB-only prediction; views whose relative rotation
                    differs by > filter_rot_tau_deg, or whose camera-centre
                    direction differs by > filter_dir_tau_deg, get no pose input.
"""

import argparse
import csv
import glob
import hashlib
import json
import os
import subprocess
import traceback

import numpy as np

from . import model_io
from .corrupt_geometry import corrupt_poses
from .datasets import load_clip
from .controls import rgbcheck_mask
from .metrics import LOWER_IS_BETTER, evaluate


def expand_conditions(cfg):
    out = []
    for c in cfg["conditions"]:
        if "sweep" not in c:
            out.append(dict(c))
            continue
        for v in c["values"]:
            spec = dict(c["corruption"])
            spec[c["sweep"]] = v
            out.append({"name": f"{c['family']}_{c['sweep']}={v}", "family": c["family"],
                        "inputs": c["inputs"], "corruption": spec,
                        "severity": v, "controls": c.get("controls", [])})
    names = [c["name"] for c in out]
    assert len(names) == len(set(names)), "duplicate condition names"
    return out


def image_hash(images):
    return hashlib.sha256(np.ascontiguousarray(images).tobytes()).hexdigest()[:16]


def git_commit(path):
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


FIELDS = ["run_key", "clip_id", "scene", "split", "condition", "family", "severity",
          "seed", "control", "view_subset", "n_views", "n_affected", "frac_affected",
          "n_pose_dropped", "dropped_are_corrupted", "image_hash", "status",
          "runtime_s", "peak_mem_gb", "n_valid_px", "scale_factor",
          *LOWER_IS_BETTER, "mean_conf"]
CACHED = ("rgb_only", "clean_pose")  # re-run on resume: later conditions need them


class Writer:
    def __init__(self, path):
        self.path = path
        self.done = set()
        if os.path.exists(path):
            with open(path) as f:
                for r in csv.DictReader(f):
                    self.done.add(r["run_key"])

    def write(self, row):
        if row["run_key"] in self.done:
            return
        self.done.add(row["run_key"])
        new = not os.path.exists(self.path)
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--clips", required=True, help="glob of clip .npz files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mapanything_repo", default="third_party/map-anything")
    ap.add_argument("--only", nargs="*", help="run only these condition names (smoke tests)")
    a = ap.parse_args()

    cfg = json.load(open(a.config))
    conds = expand_conditions(cfg)
    if a.only:
        conds = [c for c in conds if c["name"] in a.only]
    clips = sorted(glob.glob(a.clips))
    if not clips:
        raise SystemExit(f"no clips match {a.clips}")
    os.makedirs(a.out, exist_ok=True)

    import torch
    info = {"config": cfg, "clips": clips, "checkpoint": cfg["checkpoint"],
            "mapanything_commit": git_commit(a.mapanything_repo),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "infer_kwargs": model_io.INFER_KW}
    json.dump(info, open(os.path.join(a.out, "run_info.json"), "w"), indent=1)

    model = model_io.load_model(cfg["checkpoint"], a.device)
    writer = Writer(os.path.join(a.out, "per_run.csv"))
    man_path = os.path.join(a.out, "corruption_manifest.json")
    manifest = json.load(open(man_path)) if os.path.exists(man_path) else []
    fail_path = os.path.join(a.out, "failures.jsonl")

    for path in clips:
        clip = load_clip(path)
        cid, N = clip["clip_id"], len(clip["images"])
        h_img = image_hash(clip["images"])
        cache = {}  # condition name -> predictions needed by later conditions

        def infer(views):
            return model_io.run(model, views, a.device)

        def record(cond, seed, control, pred, entry, pose_mask):
            affected = entry["affected_views"] if entry else []
            clean_views = [i for i in range(N) if i not in affected]
            subsets = {"all": None}
            if affected and len(affected) < N:
                subsets["corrupted"] = affected
                subsets["uncorrupted"] = clean_views
            for sub, views in subsets.items():
                key = f"{cid}|{cond['name']}|{seed}|{control}|{sub}"
                m = evaluate(pred, clip, views)
                writer.write({
                    "run_key": key, "clip_id": cid, "scene": clip["scene"],
                    "split": cid.split("__")[0], "condition": cond["name"],
                    "family": cond["family"], "severity": cond.get("severity", ""),
                    "seed": seed, "control": control, "view_subset": sub,
                    "n_views": N, "n_affected": len(affected),
                    "frac_affected": len(affected) / N,
                    "n_pose_dropped": 0 if pose_mask is None else int((~pose_mask).sum()),
                    "dropped_are_corrupted": "" if pose_mask is None else
                        int(set(np.where(~pose_mask)[0]) <= set(affected)),
                    "image_hash": h_img, "status": "ok",
                    "runtime_s": pred["runtime_s"], "peak_mem_gb": pred["peak_mem_gb"],
                    **m})
                # references scored on the SAME view subset, for contamination analysis
                if sub != "all" and control == "none":
                    for ref in CACHED:
                        if ref in cache:
                            writer.write({"run_key": f"{key}|{ref}", "clip_id": cid,
                                          "scene": clip["scene"], "split": cid.split("__")[0],
                                          "condition": cond["name"], "family": cond["family"],
                                          "severity": cond.get("severity", ""), "seed": seed,
                                          "control": f"ref:{ref}", "view_subset": sub,
                                          "n_views": N, "n_affected": len(affected),
                                          "image_hash": h_img, "status": "ok",
                                          **evaluate(cache[ref], clip, views)})

        for cond in conds:
            spec = cond.get("corruption")
            randomized = spec is not None and (
                spec["mode"] == "subset" or spec.get("rot_deg", 0) or spec.get("trans_frac", 0))
            for seed in (cfg["seeds"] if randomized else [0]):
                base_key = f"{cid}|{cond['name']}|{seed}"
                if f"{base_key}|none|all" in writer.done and cond["name"] not in CACHED:
                    continue
                try:
                    poses, entry = clip["poses"], None
                    if spec:
                        poses, entry = corrupt_poses(clip["poses"], spec, seed, cid, cond["name"])
                        manifest.append(entry)
                    assert image_hash(clip["images"]) == h_img, "images changed!"
                    views = model_io.build_views(clip["images"], clip["K"], poses, cond["inputs"])
                    pred = infer(views)
                    if not all(np.isfinite(pred[k]).all() for k in ("pts3d", "depth_z", "pose")):
                        raise FloatingPointError("non-finite model output")
                    if cond["name"] in CACHED:
                        cache[cond["name"]] = pred
                    if cond["name"] == "zero_noise":
                        assert np.array_equal(poses, clip["poses"]), "zero-noise poses differ"
                        if "clean_pose" in cache:
                            entry["max_abs_diff_vs_clean_pts3d"] = float(
                                np.abs(pred["pts3d"] - cache["clean_pose"]["pts3d"]).max())
                    record(cond, seed, "none", pred, entry, None)

                    for ctrl in cond.get("controls", []):
                        if ctrl == "drop_oracle":
                            mask = np.ones(N, bool)
                            mask[entry["affected_views"]] = False
                        elif ctrl == "filter_rgbcheck":
                            if "rgb_only" not in cache:
                                raise RuntimeError("filter_rgbcheck needs rgb_only run first")
                            mask = rgbcheck_mask(poses, cache["rgb_only"]["pose"],
                                                 cfg["filter_rot_tau_deg"], cfg["filter_dir_tau_deg"])
                        else:
                            raise ValueError(ctrl)
                        v2 = model_io.build_views(clip["images"], clip["K"], poses,
                                                  cond["inputs"], pose_mask=mask)
                        p2 = infer(v2)
                        record(cond, seed, ctrl, p2, entry, mask)
                except Exception as e:
                    with open(fail_path, "a") as f:
                        f.write(json.dumps({"key": base_key, "error": repr(e),
                                            "traceback": traceback.format_exc()}) + "\n")
                    writer.write({"run_key": f"{base_key}|none|all", "clip_id": cid,
                                  "scene": clip["scene"], "condition": cond["name"],
                                  "family": cond["family"], "seed": seed,
                                  "control": "none", "view_subset": "all",
                                  "status": f"FAILED: {type(e).__name__}"})
                    print(f"[FAIL] {base_key}: {e!r}")
        json.dump(manifest, open(man_path, "w"), indent=1)
        print(f"[done] {cid}")


if __name__ == "__main__":
    main()
