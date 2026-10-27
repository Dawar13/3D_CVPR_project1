"""Experiment 1 runner: every condition x clip x seed, one inference each.

  CUDA_VISIBLE_DEVICES=4 python -m exp1.run_exp1 --config configs/pilot.json \
      --clips 'data/clips/dev__*.npz' --out results/dev_pilot

Writes (append-only, resumable):
  <out>/per_run.csv              one row per (clip, condition, seed, control, view_subset)
  <out>/per_view.csv             one row per (run, view): hint error, model error, confidence,
                                 absorption, disagreement with the images-only prediction
  <out>/corruption_manifest.json every corrupted input actually used
  <out>/failures.jsonl           any run that raised, with traceback
  <out>/run_info.json            config, checkpoint, git commit, torch/CUDA versions

Condition keys (config): inputs (subset of intrinsics/pose/depth), metric (is_metric_scale,
default true), corruption (target pose/K/depth + type, see corrupt_geometry / corrupt_inputs),
sweep+values or levels (severity sweep), seeds (how many of cfg["seeds"]; default: all for
random pose corruption, else 1), controls, expect_equal (a reference the output must match
exactly), floor / ceiling / realistic (used by the analysis only). depth_views (top level):
the views that get a depth hint when "depth" is in inputs.

Controls (pose corruption only):
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
from .controls import rgbcheck_mask
from .corrupt_geometry import corrupt_poses
from .corrupt_inputs import corrupt_depth, corrupt_intrinsics
from .datasets import load_clip
from .geometry import invert_pose, rotation_error_deg, trajectory_scale
from .metrics import LOWER_IS_BETTER, evaluate, per_view

COND_KEYS = ("inputs", "metric", "seeds", "controls", "floor", "ceiling", "expect_equal", "label")


def expand_conditions(cfg):
    out = []
    for c in cfg["conditions"]:
        if "sweep" not in c and "levels" not in c:
            out.append(dict(c))
            continue
        if "levels" in c:
            key, items = "level", [(k + 1, dict(lv)) for k, lv in enumerate(c["levels"])]
        else:
            key, items = c["sweep"], [(v, {c["sweep"]: v}) for v in c["values"]]
        for sev, upd in items:
            spec = dict(c["corruption"])
            spec.update(upd)
            d = {k: c[k] for k in COND_KEYS if k in c}
            d.update(name=f"{c['family']}_{key}={sev}", family=c["family"], corruption=spec,
                     severity=sev, controls=c.get("controls", []))
            if "realistic" in c:
                d["realistic"] = sev in c["realistic"]
            out.append(d)
    names = [c["name"] for c in out]
    assert len(names) == len(set(names)), "duplicate condition names"
    return out


def seeds_for(cond, cfg):
    if "seeds" in cond:
        return cfg["seeds"][:int(cond["seeds"])]
    spec = cond.get("corruption")
    randomized = spec is not None and spec.get("target", "pose") == "pose" and (
        spec.get("mode") == "subset" or spec.get("rot_deg", 0) or spec.get("trans_frac", 0))
    return cfg["seeds"] if randomized else [0]


def apply_corruption(clip, cond, seed, depth_views):
    poses, K, depth, entry = clip["poses"], clip["K"], clip["depth"], None
    spec = cond.get("corruption")
    if spec:
        cid, name = clip["clip_id"], cond["name"]
        target = spec.get("target", "pose")
        if target == "pose":
            poses, entry = corrupt_poses(poses, spec, seed, cid, name)
        elif target == "K":
            K, entry = corrupt_intrinsics(K, spec, seed, cid, name, clip["images"].shape[2])
        elif target == "depth":
            depth, entry = corrupt_depth(depth, depth_views, spec, seed, cid, name)
        else:
            raise ValueError(f"unknown corruption target {target!r}")
    return poses, K, depth, entry


def image_hash(images):
    return hashlib.sha256(np.ascontiguousarray(images).tobytes()).hexdigest()[:16]


def git_commit(path):
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


FIELDS = ["run_key", "clip_id", "scene", "split", "condition", "family", "severity", "realistic",
          "seed", "control", "view_subset", "metric_flag", "n_views", "n_affected", "frac_affected",
          "n_pose_dropped", "dropped_are_corrupted", "image_hash", "status",
          "runtime_s", "peak_mem_gb", "n_valid_px", "scale_factor", "metric_scale_ratio",
          "model_scale_factor", *LOWER_IS_BETTER, "mean_conf"]
VIEW_FIELDS = ["run_key", "clip_id", "scene", "split", "condition", "family", "severity", "realistic",
               "seed", "control", "view_idx", "was_corrupted", "has_pose_hint", "has_K_hint",
               "has_depth_hint", "conf_mean", "conf_median", "hint_rot_err", "hint_trans_err_m",
               "pred_rot_err", "pred_trans_err_m", "absorption_rot", "absorption_trans",
               "hint_focal_rel_err", "pred_focal_rel_err", "absorption_focal", "hint_depth_scale",
               "pred_depth_scale", "absorption_depth", "disagree_rgbonly_rot", "disagree_rgbonly_dir",
               "view_pts_err_scale", "view_pts_err_metric", "view_depth_absrel_scale",
               "view_depth_absrel_metric"]
CACHED = ("rgb_only", "clean_pose")  # re-run on resume: later conditions need them


class Writer:
    def __init__(self, path, fields=FIELDS, key="run_key"):
        self.path, self.fields, self.key = path, fields, key
        self.done = set()
        if os.path.exists(path):
            with open(path) as f:
                for r in csv.DictReader(f):
                    self.done.add(r[key])

    def write(self, row):
        if row[self.key] in self.done:
            return
        self.done.add(row[self.key])
        new = not os.path.exists(self.path)
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fields, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)


def _dir_err(a, b):
    c = a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def view_rows(base, pred, clip, s, inputs, hints, pose_mask, affected, rgb_pose, depth_views):
    """Per-view hint error vs truth, model error vs truth, absorption = model error / hint error."""
    pv = per_view(pred, clip, s)
    gt, Kgt, dgt = clip["poses"], clip["K"], clip["depth"]
    hint_poses, hint_K, hint_depth = hints
    N, L = len(gt), trajectory_scale(gt)
    has_pose = [("pose" in inputs) and (pose_mask is None or bool(pose_mask[i])) for i in range(N)]
    hint_rel = [invert_pose(hint_poses[0]) @ T for T in hint_poses] if any(has_pose) else None
    rgb_rel = None if rgb_pose is None else [invert_pose(rgb_pose[0]) @ T for T in rgb_pose]
    f_gt = 0.5 * (Kgt[:, 0, 0] + Kgt[:, 1, 1])
    rows = []
    for i in range(N):
        r = dict(base, view_idx=i, was_corrupted=int(i in affected), has_pose_hint=int(has_pose[i]),
                 has_K_hint=int("intrinsics" in inputs),
                 has_depth_hint=int("depth" in inputs and i in depth_views))
        for k in ("conf_mean", "conf_median", "view_pts_err_scale", "view_pts_err_metric",
                  "view_depth_absrel_scale", "view_depth_absrel_metric", "pred_depth_scale"):
            r[k] = pv[k][i]
        P = pv["pred_poses_rel"][i]
        r["pred_rot_err"] = rotation_error_deg(P[:3, :3], gt[i][:3, :3])
        r["pred_trans_err_m"] = float(np.linalg.norm(P[:3, 3] - gt[i][:3, 3]))
        r["pred_focal_rel_err"] = float(pv["pred_focal"][i] / f_gt[i] - 1)
        if has_pose[i]:
            H = hint_rel[i]
            r["hint_rot_err"] = rotation_error_deg(H[:3, :3], gt[i][:3, :3])
            r["hint_trans_err_m"] = float(np.linalg.norm(H[:3, 3] - gt[i][:3, 3]))
            if r["hint_rot_err"] > 0.05:
                r["absorption_rot"] = r["pred_rot_err"] / r["hint_rot_err"]
            if r["hint_trans_err_m"] > 0.01 * L:
                r["absorption_trans"] = r["pred_trans_err_m"] / r["hint_trans_err_m"]
            if rgb_rel is not None and i > 0:
                G = rgb_rel[i]
                r["disagree_rgbonly_rot"] = rotation_error_deg(H[:3, :3], G[:3, :3])
                r["disagree_rgbonly_dir"] = _dir_err(H[:3, 3], G[:3, 3])
        if "intrinsics" in inputs:
            r["hint_focal_rel_err"] = float(0.5 * (hint_K[i, 0, 0] + hint_K[i, 1, 1]) / f_gt[i] - 1)
            if abs(r["hint_focal_rel_err"]) > 0.01:
                r["absorption_focal"] = r["pred_focal_rel_err"] / r["hint_focal_rel_err"]
        if r["has_depth_hint"]:
            v = dgt[i] > 0
            r["hint_depth_scale"] = float(np.median(hint_depth[i][v] / dgt[i][v])) if v.any() else np.nan
            if abs(r["hint_depth_scale"] - 1) > 0.01:
                r["absorption_depth"] = (r["pred_depth_scale"] - 1) / (r["hint_depth_scale"] - 1)
        rows.append(r)
    return rows


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
    depth_views = tuple(cfg.get("depth_views", ()))
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
    vwriter = Writer(os.path.join(a.out, "per_view.csv"), VIEW_FIELDS)
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

        def record(cond, seed, control, pred, entry, pose_mask, hints):
            affected = entry["affected_views"] if entry else []
            clean_views = [i for i in range(N) if i not in affected]
            subsets = {"all": None}
            if affected and len(affected) < N - 1:
                subsets["corrupted"] = affected
                subsets["uncorrupted"] = clean_views
            common = {"clip_id": cid, "scene": clip["scene"], "split": cid.split("__")[0],
                      "condition": cond["name"], "family": cond["family"],
                      "severity": cond.get("severity", ""), "realistic": cond.get("realistic", ""),
                      "seed": seed, "control": control}
            for sub, views in subsets.items():
                key = f"{cid}|{cond['name']}|{seed}|{control}|{sub}"
                m = evaluate(pred, clip, views)
                writer.write({
                    "run_key": key, **common, "view_subset": sub,
                    "metric_flag": int(cond.get("metric", True)),
                    "n_views": N, "n_affected": len(affected),
                    "frac_affected": len(affected) / N,
                    "n_pose_dropped": 0 if pose_mask is None else int((~pose_mask).sum()),
                    "dropped_are_corrupted": "" if pose_mask is None else
                        int(set(np.where(~pose_mask)[0]) <= set(affected)),
                    "image_hash": h_img, "status": "ok",
                    "runtime_s": pred["runtime_s"], "peak_mem_gb": pred["peak_mem_gb"],
                    "model_scale_factor": pred["metric_scaling_factor"], **m})
                if sub == "all":
                    rgb_pose = cache["rgb_only"]["pose"] if "rgb_only" in cache else None
                    for r in view_rows({"run_key": key, **common}, pred, clip, m["scale_factor"],
                                       cond["inputs"], hints, pose_mask, affected, rgb_pose, depth_views):
                        r["run_key"] = f"{key}|v{r['view_idx']}"
                        vwriter.write(r)
                # references scored on the SAME view subset, for contamination analysis
                if sub != "all" and control == "none":
                    for ref in CACHED:
                        if ref in cache:
                            writer.write({"run_key": f"{key}|{ref}", **common, "control": f"ref:{ref}",
                                          "view_subset": sub, "n_views": N, "n_affected": len(affected),
                                          "image_hash": h_img, "status": "ok",
                                          **evaluate(cache[ref], clip, views)})

        for cond in conds:
            metric = bool(cond.get("metric", True))
            for seed in seeds_for(cond, cfg):
                base_key = f"{cid}|{cond['name']}|{seed}"
                if f"{base_key}|none|all" in writer.done and cond["name"] not in CACHED:
                    continue
                try:
                    poses, K, depth, entry = apply_corruption(clip, cond, seed, depth_views)
                    if entry:
                        manifest.append(entry)
                    assert image_hash(clip["images"]) == h_img, "images changed!"

                    def run_views(pose_mask):
                        views = model_io.build_views(clip["images"], K, poses, cond["inputs"],
                                                     pose_mask, metric, depth, depth_views)
                        p = infer(views)
                        if not all(np.isfinite(p[k]).all() for k in ("pts3d", "depth_z", "pose")):
                            raise FloatingPointError("non-finite model output")
                        return p

                    pred = run_views(None)
                    if cond["name"] in CACHED:
                        cache[cond["name"]] = pred
                    ref = cond.get("expect_equal", "clean_pose" if cond["name"] == "zero_noise" else None)
                    if cond["name"] == "zero_noise":
                        assert np.array_equal(poses, clip["poses"]), "zero-noise poses differ"
                    if ref and ref in cache and entry is not None:
                        k = "max_abs_diff_vs_clean_pts3d" if ref == "clean_pose" else f"max_abs_diff_vs_{ref}_pts3d"
                        entry[k] = float(np.abs(pred["pts3d"] - cache[ref]["pts3d"]).max())
                    hints = (poses, K, depth)
                    record(cond, seed, "none", pred, entry, None, hints)

                    for ctrl in cond.get("controls", []):
                        if entry is None or entry["target"] != "pose":
                            raise ValueError(f"{ctrl} needs a pose corruption")
                        if ctrl == "drop_oracle":
                            if 0 in entry["affected_views"]:
                                raise ValueError("drop_oracle cannot drop view 0")
                            mask = np.ones(N, bool)
                            mask[entry["affected_views"]] = False
                        elif ctrl == "filter_rgbcheck":
                            if "rgb_only" not in cache:
                                raise RuntimeError("filter_rgbcheck needs rgb_only run first")
                            mask = rgbcheck_mask(poses, cache["rgb_only"]["pose"],
                                                 cfg["filter_rot_tau_deg"], cfg["filter_dir_tau_deg"])
                        else:
                            raise ValueError(ctrl)
                        record(cond, seed, ctrl, run_views(mask), entry, mask, hints)
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
