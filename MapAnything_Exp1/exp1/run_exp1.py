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
  <out>/extras/<run_key>.npz     only with --save_extras (E2): poses relative to view 0
                                 (predicted, hinted, clean_pose run) and how far each view's
                                 points moved vs the clean_pose run; see run_extras()
  <out>/gpu_log.txt              start/end time, runs done and peak GPU memory per invocation

Condition keys (config): inputs (subset of intrinsics/pose/depth), metric (is_metric_scale,
default true), corruption (target pose/K/depth + type, see corrupt_geometry / corrupt_inputs),
sweep+values or levels (severity sweep), pair_with (E2: reuse the random corruption of another
family's setting of the same size), pose_frac (E2: only that share of views gets a pose, view 0
always), scale_stat ("median": E2-C4 pose-scale probe), seeds (how many of cfg["seeds"]; default: all for
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
import time
import traceback

import numpy as np

from . import model_io
from .controls import rgbcheck_mask
from .corrupt_geometry import _seed_for, corrupt_poses
from .corrupt_inputs import corrupt_depth, corrupt_intrinsics
from .datasets import load_clip
from .geometry import invert_pose, rotation_error_deg, trajectory_scale
from .metrics import LOWER_IS_BETTER, evaluate, per_view, reproj_shift_px

COND_KEYS = ("inputs", "metric", "seeds", "controls", "floor", "ceiling", "expect_equal", "label",
             "pose_frac", "scale_stat")


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
            if "pair_with" in c:  # same random draw as that family's setting of the same size (E2-C4)
                spec["seed_as"] = f"{c['pair_with']}_{key}={sev}"
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


def posed_mask(n_views, frac, seed, clip_id):
    """Partial poses (E2-C7): view 0 plus round(frac * N) - 1 other views, chosen by a seeded draw
    that depends on (seed, clip, frac) only, so clean and corrupted runs share the posed set."""
    rng = np.random.default_rng(_seed_for(seed, clip_id, f"posed|{frac}"))
    k = max(2, int(round(frac * n_views)))
    mask = np.zeros(n_views, bool)
    mask[0] = True
    mask[rng.choice(np.arange(1, n_views), size=k - 1, replace=False)] = True
    return mask


def apply_corruption(clip, cond, seed, depth_views, base_mask=None):
    poses, K, depth, entry = clip["poses"], clip["K"], clip["depth"], None
    spec = cond.get("corruption")
    if spec and base_mask is not None:  # corrupt only views that receive a pose
        spec = dict(spec, eligible=[int(i) for i in np.where(base_mask)[0] if i > 0])
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
               "view_depth_absrel_metric", "view_shape_err_sim3", "hint_reproj_px"]
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


def run_extras(pred, clip, hints, inputs, pose_mask, clean):
    """Compact per-run arrays for post-hoc analysis (E2). All poses are relative to their own view 0.
    pts_disp_*: per view, over pixels valid in both runs, the median distance (m, model scale)
    between this run's points and the clean_pose run's points, and the median component of that
    displacement along the view's injected camera shift (hint centre - true centre)."""
    N = len(clip["images"])
    T0 = invert_pose(pred["pose"][0])
    out = {"pred_pose_rel": np.stack([T0 @ T for T in pred["pose"]]),
           "pred_focal": 0.5 * (pred["intrinsics"][:, 0, 0] + pred["intrinsics"][:, 1, 1]),
           "metric_scaling_factor": np.float64(pred["metric_scaling_factor"])}
    if "pose" in inputs:
        H0 = invert_pose(hints[0][0])
        out["hint_pose_rel"] = np.stack([H0 @ T for T in hints[0]])
    if pose_mask is not None:
        out["pose_mask"] = np.asarray(pose_mask, bool)
    if clean is not None:
        C0 = invert_pose(clean["pose"][0])
        out["clean_pose_rel"] = np.stack([C0 @ T for T in clean["pose"]])
        p = pred["pts3d"] @ T0[:3, :3].T + T0[:3, 3]
        q = clean["pts3d"] @ C0[:3, :3].T + C0[:3, 3]
        G0 = invert_pose(clip["poses"][0])
        med, along = np.full(N, np.nan), np.full(N, np.nan)
        for i in range(N):
            m = pred["mask"][i] & clean["mask"][i]
            if not m.any():
                continue
            d = p[i][m] - q[i][m]
            med[i] = np.median(np.linalg.norm(d, axis=-1))
            if "hint_pose_rel" in out:
                shift = out["hint_pose_rel"][i][:3, 3] - (G0 @ clip["poses"][i])[:3, 3]
                if np.linalg.norm(shift) > 1e-9:
                    along[i] = np.median(d @ (shift / np.linalg.norm(shift)))
        out["pts_disp_median_m"], out["pts_disp_along_shift_m"] = med, along
    return out


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
                  "view_depth_absrel_scale", "view_depth_absrel_metric", "pred_depth_scale",
                  "view_shape_err_sim3"):
            r[k] = pv[k][i]
        if i in affected:  # size of this view's hint error in pixels (E2-C4 matched sizes)
            hp = hint_rel if has_pose[i] and not np.allclose(hint_rel[i], gt[i]) else None
            hk = hint_K if "intrinsics" in inputs and not np.allclose(hint_K[i], Kgt[i]) else None
            hd = hint_depth if r["has_depth_hint"] and not np.array_equal(hint_depth[i], dgt[i]) else None
            if hp is not None or hk is not None or hd is not None:
                r["hint_reproj_px"] = reproj_shift_px(i, dgt, Kgt, gt, hk, None if hp is None else np.stack(hp), hd)
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
    ap.add_argument("--backend", default="mapanything", choices=["mapanything", "omnivggt"],
                    help="omnivggt: exp1/model_io_omnivggt.py (E2-D); checkpoint and repo from the config")
    ap.add_argument("--only", nargs="*", help="run only these condition names (smoke tests)")
    ap.add_argument("--save_extras", action="store_true", help="write <out>/extras/<run_key>.npz")
    ap.add_argument("--mem_fraction", type=float, default=0.25,
                    help="cap this process's share of GPU memory (shared GPU; E2 plan 0.3)")
    a = ap.parse_args()
    t_start = time.strftime("%Y-%m-%d %H:%M:%S")
    global model_io
    if a.backend == "omnivggt":
        from . import model_io_omnivggt as model_io

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
    if a.device.startswith("cuda") and torch.cuda.is_available() and a.mem_fraction:
        torch.cuda.set_per_process_memory_fraction(a.mem_fraction)
    extras_dir = os.path.join(a.out, "extras")
    if a.save_extras:
        os.makedirs(extras_dir, exist_ok=True)
    peak, n_new = 0.0, 0
    info = {"config": cfg, "clips": clips, "checkpoint": cfg["checkpoint"],
            "backend": a.backend,
            "mapanything_commit": git_commit(a.mapanything_repo),
            "model_repo_commit": git_commit(model_io.REPO) if a.backend == "omnivggt" else None,
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
        ref_scores = {}  # (reference, views) -> evaluate() of that cached reference

        def infer(views):
            return model_io.run(model, views, a.device)

        def record(cond, seed, control, pred, entry, pose_mask, hints):
            nonlocal peak, n_new
            peak, n_new = max(peak, pred["peak_mem_gb"]), n_new + 1
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
                if sub == "all" and a.save_extras:
                    np.savez_compressed(os.path.join(extras_dir, key.replace("|", "__") + ".npz"),
                                        **run_extras(pred, clip, hints, cond["inputs"], pose_mask,
                                                     cache.get("clean_pose")))
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
                            mk = (ref, tuple(views))
                            if mk not in ref_scores:  # a reference on the same views scores the same
                                ref_scores[mk] = evaluate(cache[ref], clip, views)
                            writer.write({"run_key": f"{key}|{ref}", **common, "control": f"ref:{ref}",
                                          "view_subset": sub, "n_views": N, "n_affected": len(affected),
                                          "image_hash": h_img, "status": "ok", **ref_scores[mk]})

        for cond in conds:
            metric = bool(cond.get("metric", True))
            for seed in seeds_for(cond, cfg):
                base_key = f"{cid}|{cond['name']}|{seed}"
                if f"{base_key}|none|all" in writer.done and cond["name"] not in CACHED:
                    continue
                try:
                    base = posed_mask(N, cond["pose_frac"], seed, cid) if "pose_frac" in cond else None
                    poses, K, depth, entry = apply_corruption(clip, cond, seed, depth_views, base)
                    if entry:
                        manifest.append(entry)
                    assert image_hash(clip["images"]) == h_img, "images changed!"

                    def run_views(pose_mask):
                        if base is not None:
                            pose_mask = base if pose_mask is None else pose_mask & base
                        views = model_io.build_views(clip["images"], K, poses, cond["inputs"],
                                                     pose_mask, metric, depth, depth_views)
                        stat = cond.get("scale_stat", "mean")
                        if stat != "mean":
                            model_io.set_pose_scale_statistic(stat)
                        try:
                            p = infer(views)
                        finally:
                            if stat != "mean":
                                model_io.set_pose_scale_statistic("mean")
                        if not all(np.isfinite(p[k]).all() for k in ("pts3d", "depth_z", "pose")):
                            raise FloatingPointError("non-finite model output")
                        return p

                    pred = run_views(None)
                    if cond["name"] in CACHED:
                        cache[cond["name"]] = pred
                        ref_scores = {k: v for k, v in ref_scores.items() if k[0] != cond["name"]}
                    ref = cond.get("expect_equal", "clean_pose" if cond["name"] == "zero_noise" else None)
                    if cond["name"] == "zero_noise":
                        assert np.array_equal(poses, clip["poses"]), "zero-noise poses differ"
                    if ref and ref in cache and entry is not None:
                        k = "max_abs_diff_vs_clean_pts3d" if ref == "clean_pose" else f"max_abs_diff_vs_{ref}_pts3d"
                        entry[k] = float(np.abs(pred["pts3d"] - cache[ref]["pts3d"]).max())
                    hints = (poses, K, depth)
                    record(cond, seed, "none", pred, entry, base, hints)

                    for ctrl in cond.get("controls", []):
                        if entry is None or entry["target"] != "pose":
                            raise ValueError(f"{ctrl} needs a pose corruption")
                        if ctrl == "drop_oracle":
                            if 0 in entry["affected_views"]:
                                raise ValueError("drop_oracle cannot drop view 0")
                            mask = np.ones(N, bool) if base is None else base.copy()
                            mask[entry["affected_views"]] = False
                        elif ctrl == "filter_rgbcheck":
                            if "rgb_only" not in cache:
                                raise RuntimeError("filter_rgbcheck needs rgb_only run first")
                            mask = rgbcheck_mask(poses, cache["rgb_only"]["pose"],
                                                 cfg["filter_rot_tau_deg"], cfg["filter_dir_tau_deg"])
                        else:
                            raise ValueError(ctrl)
                        record(cond, seed, ctrl, run_views(mask), entry,
                               mask if base is None else mask & base, hints)
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
    with open(os.path.join(a.out, "gpu_log.txt"), "a") as f:
        f.write(f"start={t_start} end={time.strftime('%Y-%m-%d %H:%M:%S')} config={a.config} "
                f"clips={a.clips} runs_recorded={n_new} peak_mem_gb={peak:.2f} "
                f"mem_fraction={a.mem_fraction}\n")


if __name__ == "__main__":
    main()
