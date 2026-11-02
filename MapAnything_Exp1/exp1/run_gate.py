"""Gate-track runner (experiments/G.md): oracle hint gates (G1) and token signals (G2). Inference only.

  python -m exp1.run_gate --config configs/g1_oracle.json --clips 'data/clips_e2e/dev__*.npz' --out results/g1_rooms_ma

Dev clips only unless --allow_eval is passed: eval runs happen once, from a frozen tag.

Writes (resumable):
  <out>/per_run.csv               one row per (clip, condition, seed)
  <out>/corruption_manifest.json  every hint used
  <out>/checks.csv                gate sanity checks on the first clip of each dataset
  <out>/failures.jsonl            any run that raised
  <out>/run_info.json             config, versions
  <out>/tokens/<key>.npz          only when the config has token_layers (G2)

A condition: name, inputs (subset of intrinsics/pose/depth), plus
  dataset    "rooms" (clips with phone hints) or "eth"; other clips skip it
  scenario   shared random draw and analysis group; role = always/never/clean/oracle_*
  depth      depth hint spec for exp1/depth_hints.build_hint
  pose       "gt" (default), "dev" (phone), or a corrupt_geometry spec
  gate       {"depth": "zero"|"oracle_view"|"oracle_patch", "pose": "zero"|"oracle_view"}: in-network gamma, ORACLE
  drop       {"depth": "oracle_view"|"oracle_patch", "pose": "oracle_view"}: remove the hint at the input, ORACLE
  seeds      how many draws (default 1)
  signal     G2 labels: {"task": name, "kind": "view"|"group"} for bad hints, {"neg_for": [names]} for good ones
"""

import argparse
import csv
import glob
import json
import os
import subprocess
import time
import traceback

import numpy as np

from .corrupt_geometry import corrupt_poses
from .datasets import load_clip
from .depth_hints import build_hint, patch_gamma, view_gamma
from .geometry import invert_pose, rotation_error_deg
from .metrics import LOWER_IS_BETTER, evaluate
from .run_exp1 import Writer, _dir_err

FIELDS = ["run_key", "clip_id", "scene", "split", "dataset", "condition", "scenario", "role", "seed", "backend",
          "n_bad_views", "bad_px_share", "gate", "drop", "status", "runtime_s", "peak_mem_gb", "n_valid_px",
          "scale_factor", "metric_scale_ratio", "model_scale_factor", *LOWER_IS_BETTER, "mean_conf"]
TOKEN_REF = "rgb_only"


def get_backend(name):
    if name == "omnivggt":
        from . import model_io_omnivggt as m
    else:
        from . import model_io as m
    return m


def adapt_inputs(backend, inputs):
    """OmniVGGT takes a camera hint as pose AND intrinsics and cannot take intrinsics alone (model_io_omnivggt.py)."""
    s = set(inputs)
    if backend == "omnivggt" and "pose" not in s:
        s.discard("intrinsics")
    return sorted(s)


def dataset_of(clip):
    return "rooms" if "dev_depth" in clip else "eth"


def check_split(clips, allow_eval):
    names = [os.path.basename(c) for c in clips]
    bad = [n for n in names if not n.startswith("dev__")]
    if bad and not allow_eval:
        raise SystemExit(f"{len(bad)} non-dev clips (e.g. {bad[0]}). Eval runs once, from a frozen tag: pass --allow_eval "
                         "only then (experiments/G.md).")


def build_inputs(clip, cond, seed):
    """-> poses, K, depth, depth_views, bad depth views, bad depth mask, bad pose views, manifest entries."""
    N = len(clip["images"])
    rooms = dataset_of(clip) == "rooms"
    scenario = cond.get("scenario", cond["name"])
    K = clip["dev_K"] if rooms else clip["K"]
    poses, entries, bad_pose = clip["poses"], [], []
    p = cond.get("pose", "gt")
    if p == "dev":
        poses = clip["dev_poses"]
    elif isinstance(p, dict):
        poses, e = corrupt_poses(clip["poses"], p, seed, clip["clip_id"], scenario)
        entries.append(e)
        bad_pose = list(e["affected_views"])
    depth, bad_depth, mask = None, [], None
    if "depth" in cond:
        depth, bad_depth, mask, e = build_hint(clip, cond["depth"], seed, scenario)
        entries.append(e)
    views = list(range(N))
    drop = cond.get("drop", {})
    if drop.get("depth") == "oracle_view":
        views = [i for i in views if i not in bad_depth]
    elif drop.get("depth") == "oracle_patch":
        depth = depth.copy()
        depth[mask] = 0
    elif "depth" in drop:
        raise ValueError(f"unknown depth drop {drop['depth']!r}")
    return poses, K, depth, views, bad_depth, mask, bad_pose, entries


def gammas(cond, N, bad_depth, mask, bad_pose):
    g = cond.get("gate", {})
    gd = {None: None, "zero": np.zeros(N, np.float32), "oracle_view": view_gamma(N, bad_depth)}.get(g.get("depth"), "patch")
    if isinstance(gd, str):
        if g["depth"] != "oracle_patch":
            raise ValueError(f"unknown depth gate {g['depth']!r}")
        gd = patch_gamma(mask)
    pg = g.get("pose")
    if pg not in (None, "zero", "oracle_view"):
        raise ValueError(f"unknown pose gate {pg!r}")
    gp = None if pg is None else (np.zeros(N, np.float32) if pg == "zero" else view_gamma(N, bad_pose))
    return gd, gp


def infer(io, model, device, views, gd=None, gp=None, layers=None, N=None):
    cap = io.TokenCapture(model, layers, N) if layers else None
    if gd is not None or gp is not None:
        io.set_hint_gate(model, views, gd, gp)
    try:
        pred = io.run(model, views, device)
    finally:
        io.clear_hint_gate(model)
        if cap:
            cap.remove()
    if not all(np.isfinite(pred[k]).all() for k in ("pts3d", "depth_z", "pose")):
        raise FloatingPointError("non-finite model output")
    return pred, (cap.tokens if cap else None)


def run_checks(io, model, device, clip, path):
    """gamma = 1 must leave the output unchanged; gamma = 0 on every view should match the hint being absent."""
    N = len(clip["images"])
    rooms = dataset_of(clip) == "rooms"
    K = clip["dev_K"] if rooms else clip["K"]
    depth = clip["dev_depth"] if rooms else clip["depth"]
    rows = []
    for name, inputs, kw_one, kw_zero, absent in (
        ("depth", ["intrinsics", "depth"], {"gd": np.ones(N, np.float32)}, {"gd": np.zeros(N, np.float32)}, ["intrinsics"]),
        ("pose", ["intrinsics", "pose"], {"gp": np.ones(N, np.float32)}, {"gp": np.zeros(N, np.float32)}, ["intrinsics"]),
    ):
        ins = adapt_inputs(getattr(io, "__name__", "").rsplit("_", 1)[-1], inputs)
        ab = adapt_inputs(getattr(io, "__name__", "").rsplit("_", 1)[-1], absent)
        v = io.build_views(clip["images"], K, clip["poses"], ins, None, True, depth, tuple(range(N)))
        base = infer(io, model, device, v)[0]["pts3d"]
        one = infer(io, model, device, v, **kw_one)[0]["pts3d"]
        zero = infer(io, model, device, v, **kw_zero)[0]["pts3d"]
        none = infer(io, model, device, io.build_views(clip["images"], K, clip["poses"], ab, None, True, depth, ()))[0]["pts3d"]
        rows.append({"clip": clip["clip_id"], "hint": name, "gamma1_vs_no_gate": float(np.abs(one - base).max()),
                     "gamma0_vs_absent": float(np.abs(zero - none).max()),
                     "hint_vs_absent": float(np.abs(base - none).max())})
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"[check] {r}", flush=True)
        if r["gamma1_vs_no_gate"] > 1e-3:
            raise SystemExit(f"gate with gamma = 1 changed the output: {r}")
        if r["hint"] == "depth" and r["gamma0_vs_absent"] > 0.05 * max(r["hint_vs_absent"], 1e-6) and r["gamma0_vs_absent"] > 1e-3:
            print(f"[check] WARNING: gamma = 0 differs from an absent depth hint: {r}", flush=True)


def token_record(pred, tokens, ref_pred, ref_tokens, layers, clip, poses, depth, inputs, depth_views):
    """Scores available without ground truth, per view. d_* = mean / p90 over patches of 1 - cos(hinted token, photos-only
    token) at each layer. Also the baselines: the confidence of the model and hint-vs-photos disagreement."""
    import torch
    N = len(clip["images"])
    out = {"layers": np.array(layers), "conf": pred["conf"].reshape(N, -1).mean(1).astype(np.float32),
           "mean_token": np.stack([tokens[k].mean(1).numpy() for k in layers], 1).astype(np.float16)}
    d_mean, d_p90 = [], []
    for k in layers:
        c = torch.nn.functional.cosine_similarity(tokens[k], ref_tokens[k], dim=-1).numpy()
        d_mean.append((1 - c).mean(1))
        d_p90.append(np.percentile(1 - c, 90, axis=1))
    out["d_mean"], out["d_p90"] = np.stack(d_mean, 1).astype(np.float32), np.stack(d_p90, 1).astype(np.float32)
    rot, dirn = np.full(N, np.nan, np.float32), np.full(N, np.nan, np.float32)
    if "pose" in inputs:
        H0, G0 = invert_pose(poses[0]), invert_pose(ref_pred["pose"][0])
        for i in range(1, N):
            H, G = H0 @ poses[i], G0 @ ref_pred["pose"][i]
            rot[i] = rotation_error_deg(H[:3, :3], G[:3, :3])
            dirn[i] = _dir_err(H[:3, 3], G[:3, 3])
    dd = np.full(N, np.nan, np.float32)
    if "depth" in inputs:
        for i in depth_views:
            v = (depth[i] > 0) & (ref_pred["depth_z"][i] > 1e-6)
            if v.any():
                dd[i] = abs(np.log(np.median(depth[i][v] / ref_pred["depth_z"][i][v])))
    out.update(dis_rot=rot, dis_dir=dirn, dis_depth=dd)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backend", default="mapanything", choices=["mapanything", "omnivggt"])
    ap.add_argument("--allow_eval", action="store_true", help="only from a frozen tag; see experiments/G.md")
    ap.add_argument("--no_checks", action="store_true")
    a = ap.parse_args()

    cfg = json.load(open(a.config))
    clips = sorted(glob.glob(a.clips))
    if not clips:
        raise SystemExit(f"no clips match {a.clips}")
    check_split(clips, a.allow_eval)
    io = get_backend(a.backend)
    layers = cfg.get("token_layers", {}).get(a.backend)
    os.makedirs(a.out, exist_ok=True)
    tok_dir = os.path.join(a.out, "tokens")
    if layers:
        os.makedirs(tok_dir, exist_ok=True)

    import torch
    info = {"config": cfg, "config_path": a.config, "clips": clips, "backend": a.backend, "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        info["repo_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        info["repo_commit"] = "unknown"
    json.dump(info, open(os.path.join(a.out, "run_info.json"), "w"), indent=1)

    model = io.load_model(cfg[f"checkpoint_{a.backend}"], a.device)
    writer = Writer(os.path.join(a.out, "per_run.csv"), FIELDS)
    man_path = os.path.join(a.out, "corruption_manifest.json")
    manifest = json.load(open(man_path)) if os.path.exists(man_path) else []
    checked = set()

    for path in clips:
        clip = load_clip(path)
        cid, N, ds = clip["clip_id"], len(clip["images"]), dataset_of(clip)
        if ds not in checked and not a.no_checks:
            checked.add(ds)
            run_checks(io, model, a.device, clip, os.path.join(a.out, "checks.csv"))
        ref_pred, ref_tokens = None, None
        conds = [c for c in cfg["conditions"] if c.get("dataset", ds) == ds]
        if layers:   # the photos-only reference comes first: every token score is relative to it
            conds = sorted(conds, key=lambda c: c["name"] != TOKEN_REF)
        for cond in conds:
            for seed in range(int(cond.get("seeds", 1))):
                key = f"{cid}|{cond['name']}|{seed}"
                tok_path = os.path.join(tok_dir, key.replace("|", "__") + ".npz")
                need_ref = layers and cond["name"] == TOKEN_REF
                if key in writer.done and not need_ref:
                    continue
                try:
                    poses, K, depth, dviews, bad_d, mask, bad_p, entries = build_inputs(clip, cond, seed)
                    inputs = adapt_inputs(a.backend, cond["inputs"])
                    pose_mask = None
                    if cond.get("drop", {}).get("pose") == "oracle_view":
                        pose_mask = view_gamma(N, bad_p) > 0
                    views = io.build_views(clip["images"], K, poses, inputs, pose_mask, True, depth, tuple(dviews))
                    gd, gp = gammas(cond, N, bad_d, mask, bad_p)
                    pred, tokens = infer(io, model, a.device, views, gd, gp, layers, N)
                    if need_ref:
                        ref_pred, ref_tokens = pred, tokens
                    m = evaluate(pred, clip)
                    writer.write({"run_key": key, "clip_id": cid, "scene": clip["scene"], "split": cid.split("__")[0],
                                  "dataset": ds, "condition": cond["name"], "scenario": cond.get("scenario", ""),
                                  "role": cond.get("role", ""), "seed": seed, "backend": a.backend,
                                  "n_bad_views": len(set(bad_d) | set(bad_p)),
                                  "bad_px_share": "" if mask is None else float(mask.mean()),
                                  "gate": json.dumps(cond.get("gate", {})), "drop": json.dumps(cond.get("drop", {})),
                                  "status": "ok", "runtime_s": pred["runtime_s"], "peak_mem_gb": pred["peak_mem_gb"],
                                  "model_scale_factor": pred["metric_scaling_factor"], **m})
                    manifest.extend(dict(e, condition=cond["name"]) for e in entries)
                    if layers and "signal" in cond:
                        rec = token_record(pred, tokens, ref_pred, ref_tokens, layers, clip, poses, depth, inputs, dviews)
                        sig = cond["signal"]
                        np.savez_compressed(
                            tok_path, **rec, bad_views=np.array(sorted(set(bad_d) | set(bad_p)), int),
                            meta=np.array(json.dumps({"clip": cid, "scene": clip["scene"], "dataset": ds,
                                                      "condition": cond["name"], "seed": seed, "signal": sig})))
                    if need_ref:
                        np.savez_compressed(tok_path, mean_token=np.stack(
                            [tokens[k].mean(1).numpy() for k in layers], 1).astype(np.float16), layers=np.array(layers),
                            meta=np.array(json.dumps({"clip": cid, "scene": clip["scene"], "dataset": ds,
                                                      "condition": TOKEN_REF, "seed": seed})))
                except Exception as e:
                    with open(os.path.join(a.out, "failures.jsonl"), "a") as f:
                        f.write(json.dumps({"key": key, "error": repr(e), "traceback": traceback.format_exc()}) + "\n")
                    writer.write({"run_key": key, "clip_id": cid, "scene": clip["scene"], "condition": cond["name"],
                                  "scenario": cond.get("scenario", ""), "role": cond.get("role", ""), "seed": seed,
                                  "backend": a.backend, "status": f"FAILED: {type(e).__name__}"})
                    print(f"[FAIL] {key}: {e!r}", flush=True)
        json.dump(manifest, open(man_path, "w"), indent=1)
        print(f"[done] {cid}", flush=True)


if __name__ == "__main__":
    main()
