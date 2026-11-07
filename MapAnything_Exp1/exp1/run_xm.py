"""E2X X-M runner (experiments/E2X.md, X-M, session A2): inside the model, how does it decide to obey? Inference only.

  python -m exp1.run_xm --config configs/e2x_xm_mapanything.json --clips 'data/clips_e2/dev__*__c0.npz' --out results/xm_ma_dev

Dev clips only unless --allow_eval is passed: eval runs happen once, from a frozen tag (e2x-a2-prereg).

For every clip and case (a hint corruption: e.g. one camera 15 deg, shared 5 deg, drift level 2, depth x1.5):
  references   rgb_only (no hint), clean (correct hints), bad (the case's hints), drop (bad views without the hint);
               the internal state of clean / bad / drop is captured at every listed layer (io.TokenCapture, full).
  M1 dial      the bad hint's contribution scaled by gamma (io.set_hint_gate at the fusion step) on the bad views only
               (pose cases) or the depth views (depth cases), gamma in config["dial"].
  M3 patch     the bad run with the bad views' state at layer k replaced by their state in the clean run ("clean")
               and in the drop run ("drop"), one layer at a time (io.TokenPatch).
  M5 spread    the bad run with all OTHER views' state at layer k replaced by the clean run's.
  M2/M4 (stats, no extra pass) per layer and view: relative size of (bad - clean) state, and its cosine with the
               same difference at the first listed layer (does the hint's signal travel unchanged, a "copy path"?).
Every pass is scored with metrics.evaluate (all views) plus the bad views' rotation error vs ground truth and vs the
hint. Writes (resumable): <out>/per_run.csv, <out>/state_stats.csv, <out>/failures.jsonl, <out>/run_info.json.

Config: checkpoint, backend, inputs (hint groups of the clean/bad runs), depth_views, layers, dial,
cases: [{name, corruption (run_exp1 spec, "also" allowed), inputs (optional, overrides)}], seeds (default [0]).
"""

import argparse
import glob
import json
import os
import time
import traceback

import numpy as np

from .datasets import load_clip
from .geometry import invert_pose, rotation_error_deg
from .metrics import LOWER_IS_BETTER, evaluate
from .run_exp1 import Writer, apply_corruption

FIELDS = ["run_key", "clip_id", "scene", "split", "case", "seed", "experiment", "source", "layer", "gamma",
          "patched_views", "bad_views", "status", "runtime_s", "peak_mem_gb", "bad_rot_err_gt_deg",
          "bad_rot_err_hint_deg", "good_rot_err_gt_deg", *LOWER_IS_BETTER]
STAT_FIELDS = ["run_key", "clip_id", "case", "seed", "layer", "view", "is_bad", "rel_diff_bad_vs_clean",
               "cos_with_first_layer", "rel_diff_drop_vs_clean"]


def get_backend(name):
    if name == "omnivggt":
        from . import model_io_omnivggt as m
    elif name == "pi3x":
        from . import model_io_pi3x as m
    else:
        from . import model_io as m
    return m


def _rel(poses):
    T0 = invert_pose(poses[0])
    return np.stack([T0 @ T for T in poses])


def rot_errs(pred, clip, hint_poses, bad):
    """Mean rotation error (deg) of the bad views vs GT and vs the hint, and of the good views (i > 0) vs GT."""
    P, G, H = _rel(pred["pose"]), _rel(clip["poses"]), _rel(hint_poses)
    N = len(P)
    good = [i for i in range(1, N) if i not in bad]
    e = lambda A, B, idx: float(np.mean([rotation_error_deg(A[i][:3, :3], B[i][:3, :3]) for i in idx])) if idx else np.nan
    return e(P, G, bad), e(P, H, bad), e(P, G, good)


def state_stats(clean, bad, drop, layers, bad_views):
    rows = []
    first = layers[0]
    for k in layers:
        d = (bad[k] - clean[k]).reshape(len(clean[k]), -1)
        d0 = (bad[first] - clean[first]).reshape(len(clean[first]), -1)
        dd = (drop[k] - clean[k]).reshape(len(clean[k]), -1)
        nc = np.linalg.norm(clean[k].reshape(len(clean[k]), -1), axis=1) + 1e-12
        for v in range(len(d)):
            cos = float(d[v] @ d0[v] / (np.linalg.norm(d[v]) * np.linalg.norm(d0[v]) + 1e-12))
            rows.append({"layer": k, "view": v, "is_bad": int(v in bad_views),
                         "rel_diff_bad_vs_clean": float(np.linalg.norm(d[v]) / nc[v]),
                         "cos_with_first_layer": cos, "rel_diff_drop_vs_clean": float(np.linalg.norm(dd[v]) / nc[v])})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--allow_eval", action="store_true")
    ap.add_argument("--mem_fraction", type=float, default=0.0)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    clips = sorted(glob.glob(a.clips))
    if not clips:
        raise SystemExit(f"no clips match {a.clips}")
    if not a.allow_eval and any(os.path.basename(c).startswith("eval__") for c in clips):
        raise SystemExit("eval clips need --allow_eval (run once, from the frozen tag)")
    io = get_backend(cfg.get("backend", "mapanything"))
    os.makedirs(a.out, exist_ok=True)
    import torch
    if a.device.startswith("cuda") and torch.cuda.is_available() and a.mem_fraction:
        torch.cuda.set_per_process_memory_fraction(a.mem_fraction)
    json.dump({"config": cfg, "clips": clips, "backend": cfg.get("backend", "mapanything"),
               "torch": torch.__version__, "cuda": torch.version.cuda,
               "infer_kwargs": getattr(io, "INFER_KW", None)}, open(os.path.join(a.out, "run_info.json"), "w"), indent=1)
    model = io.load_model(cfg["checkpoint"], a.device)
    W = Writer(os.path.join(a.out, "per_run.csv"), FIELDS)
    S = Writer(os.path.join(a.out, "state_stats.csv"), STAT_FIELDS)
    fail_path = os.path.join(a.out, "failures.jsonl")
    layers = list(cfg["layers"])
    depth_views = tuple(cfg.get("depth_views", ()))
    for path in clips:
        clip = load_clip(path)
        cid, N = clip["clip_id"], len(clip["images"])
        for case in cfg["cases"]:
            inputs = case.get("inputs", cfg["inputs"])
            for seed in cfg.get("seeds", [0]):
                base = f"{cid}|{case['name']}|{seed}"
                if f"{base}|ref|bad||" in W.done:
                    continue
                try:
                    cond = {"name": case["name"], "corruption": case["corruption"]}
                    poses, K, depth, entry = apply_corruption(clip, cond, seed, depth_views)
                    depth_case = case["corruption"].get("target") == "depth"
                    bad = sorted(set(entry["affected_views"]) - ({0} if not depth_case else set()))
                    dv = depth_views if "depth" in inputs else ()

                    def views_for(P, D, pose_mask=None, inp=inputs):
                        return io.build_views(clip["images"], K if "intrinsics" in inp else clip["K"], P, inp,
                                              pose_mask, True, D, dv)

                    def run(views, gd=None, gp=None, patch=None, capture=False):
                        cap = io.TokenCapture(model, layers, N, full=True) if capture else None
                        pt = io.TokenPatch(model, *patch) if patch else None
                        if gd is not None or gp is not None:
                            io.set_hint_gate(model, views, gd, gp)
                        try:
                            pred = io.run(model, views, a.device)
                        finally:
                            io.clear_hint_gate(model)
                            if pt:
                                pt.remove()
                            if cap:
                                cap.remove()
                        if not all(np.isfinite(pred[k]).all() for k in ("pts3d", "depth_z", "pose")):
                            raise FloatingPointError("non-finite model output")
                        return pred, (cap.tokens if cap else None)

                    def record(experiment, source, pred, layer="", gamma="", patched=()):
                        m = evaluate(pred, clip)
                        b_gt, b_hint, g_gt = rot_errs(pred, clip, poses, bad)
                        W.write({"run_key": f"{base}|{experiment}|{source}|{layer}|{gamma}", "clip_id": cid,
                                 "scene": clip["scene"], "split": cid.split("__")[0], "case": case["name"], "seed": seed,
                                 "experiment": experiment, "source": source, "layer": layer, "gamma": gamma,
                                 "patched_views": " ".join(map(str, patched)), "bad_views": " ".join(map(str, bad)),
                                 "status": "ok", "runtime_s": pred["runtime_s"], "peak_mem_gb": pred["peak_mem_gb"],
                                 "bad_rot_err_gt_deg": b_gt, "bad_rot_err_hint_deg": b_hint,
                                 "good_rot_err_gt_deg": g_gt, **{k: m.get(k) for k in LOWER_IS_BETTER}})

                    clean_in = views_for(clip["poses"], clip["depth"])
                    bad_in = views_for(poses, depth)
                    if depth_case:
                        drop_in = views_for(poses, depth, inp=[x for x in inputs if x != "depth"])
                    else:
                        mask = np.ones(N, bool)
                        mask[bad] = False
                        drop_in = views_for(poses, depth, pose_mask=mask)
                    rgb = views_for(clip["poses"], clip["depth"], inp=[])
                    record("ref", "rgb_only", run(rgb)[0])
                    p_clean, t_clean = run(clean_in, capture=True)
                    record("ref", "clean", p_clean)
                    p_drop, t_drop = run(drop_in, capture=True)
                    record("ref", "drop", p_drop)
                    p_bad, t_bad = run(bad_in, capture=True)
                    for r in state_stats(t_clean, t_bad, t_drop, layers, bad):   # M2 / M4, from the captured states
                        S.write({"run_key": f"{base}|{r['layer']}|{r['view']}", "clip_id": cid, "case": case["name"],
                                 "seed": seed, **r})
                    dial = cfg.get("dial", []) if hasattr(io, "set_hint_gate") else []   # Pi3X: no fusion gate
                    for g in dial:                           # M1
                        gam = np.ones(N, np.float32)
                        if depth_case:
                            pred = run(bad_in, gd=np.where(np.isin(np.arange(N), list(dv)), g, 1.0).astype(np.float32))[0]
                        else:
                            gam[bad] = g
                            pred = run(bad_in, gp=gam)[0]
                        record("m1_dial", "bad", pred, gamma=g)
                    good = [v for v in range(N) if v not in bad]
                    for k in layers:                         # M3 and M5
                        for src, tok in (("clean", t_clean), ("drop", t_drop)):
                            record("m3_patch_bad_views", src, run(bad_in, patch=(k, bad, tok[k]))[0], layer=k, patched=bad)
                        record("m5_patch_other_views", "clean", run(bad_in, patch=(k, good, t_clean[k]))[0], layer=k,
                               patched=good)
                    record("ref", "bad", p_bad)   # written last: marks the case as complete for resume
                except Exception as e:
                    with open(fail_path, "a") as f:
                        f.write(json.dumps({"key": base, "error": repr(e), "traceback": traceback.format_exc()}) + "\n")
                    print(f"[FAIL] {base}: {e!r}", flush=True)
        print(f"[done] {cid}", flush=True)


if __name__ == "__main__":
    main()
