"""Gate track (experiments/G.md): depth hints with one error isolated, and hints that are bad only in places.
Pure numpy, no model. Every builder returns (depth, bad_views, bad_mask, entry), with the manifest entry the repo rules
require. Types marked ORACLE use ground truth to build the hint: they are diagnostics, never deployable.

  gt, lidar, ai                    the depth of the clip, dev_depth, mono_depth
  ai_fix_global     ORACLE         AI depth with its overall scale error removed
  ai_fix_view       ORACLE         AI depth with the scale of each view error removed = distortion only
  gt_ai_global      ORACLE         true depth x the overall AI scale = overall scale error only
  gt_ai_view        ORACLE         true depth x the per-view AI scales = per-view scale error only
  mix_ai                           frac of the views get AI depth, the others LiDAR
  view_bad                         frac of the views of `source` scaled by factor
  patch_bad                        one rectangle covering area of every view of `source` scaled by factor
"""

import numpy as np

from .corrupt_geometry import _seed_for

PATCH = 14
SOURCES = {"gt": "depth", "lidar": "dev_depth", "ai": "mono_depth"}


def _need(clip, key):
    if key not in clip:
        raise KeyError(f"{clip.get('clip_id')}: no {key!r} (mono_depth comes from exp1/mono_depth.py)")
    return np.asarray(clip[key], np.float32)


def global_ratio(hint, gt):
    v = (hint > 0) & (gt > 0)
    return float(np.median(hint[v] / gt[v]))


def view_ratios(hint, gt):
    out = []
    for h, g in zip(hint, gt):
        v = (h > 0) & (g > 0)
        out.append(float(np.median(h[v] / g[v])) if v.any() else 1.0)
    return np.array(out, np.float32)


def _pick(n, frac, rng):
    k = max(1, int(round(frac * n)))
    return sorted(int(i) for i in rng.choice(n, size=k, replace=False))


def _rect(H, W, area, rng):
    side = np.sqrt(area)
    h, w = max(1, int(round(H * side))), max(1, int(round(W * side)))
    top, left = int(rng.integers(0, H - h + 1)), int(rng.integers(0, W - w + 1))
    m = np.zeros((H, W), bool)
    m[top:top + h, left:left + w] = True
    return m


def build_hint(clip, spec, seed, scenario):
    """spec: {"type": ..., plus frac / factor / area / source}. The random draw depends on (seed, clip, scenario) only,
    so every condition of a scenario sees the same bad views and rectangles."""
    kind = spec["type"]
    cid = clip.get("clip_id", "")
    rng = np.random.default_rng(_seed_for(seed, cid, scenario))
    gt = np.asarray(clip["depth"], np.float32)
    N = len(gt)
    bad_views, bad_mask, info = [], np.zeros(gt.shape, bool), {}
    if kind in SOURCES:
        depth = _need(clip, SOURCES[kind]).copy()
    elif kind in ("ai_fix_global", "ai_fix_view", "gt_ai_global", "gt_ai_view"):
        ai = _need(clip, "mono_depth")
        if kind.endswith("global"):
            r = global_ratio(ai, gt)
            info["global_ratio"] = r
            depth = ai / r if kind.startswith("ai") else gt * r
        else:
            r = view_ratios(ai, gt)
            info["view_ratios"] = [float(x) for x in r]
            depth = ai / r[:, None, None] if kind.startswith("ai") else gt * r[:, None, None]
    elif kind == "mix_ai":
        depth = _need(clip, "dev_depth").copy()
        ai = _need(clip, "mono_depth")
        bad_views = _pick(N, spec["frac"], rng)
        depth[bad_views] = ai[bad_views]
        bad_mask[bad_views] = True
    elif kind in ("view_bad", "patch_bad"):
        depth = _need(clip, SOURCES[spec.get("source", "gt")]).copy()
        f = np.float32(spec["factor"])
        if kind == "view_bad":
            bad_views = _pick(N, spec["frac"], rng)
            bad_mask[bad_views] = True
        else:
            bad_views = list(range(N))
            for i in range(N):
                bad_mask[i] = _rect(*gt.shape[1:], spec["area"], rng)
        depth[bad_mask] *= f
    else:
        raise ValueError(f"unknown depth hint type {kind!r}")
    entry = {"clip_id": cid, "scenario": scenario, "target": "depth_hint", "type": kind, "spec": spec, "seed": seed,
             "affected_views": bad_views, "bad_px_share": float(bad_mask.mean()), "oracle": kind.startswith(("ai_fix", "gt_ai")),
             **info}
    return depth, bad_views, bad_mask, entry


def patch_gamma(bad_mask, patch=PATCH):
    """(V, H, W) bool -> (V, h, w) float: 1 - share of bad pixels in each model patch."""
    V, H, W = bad_mask.shape
    if H % patch or W % patch:
        raise ValueError(f"{W}x{H} is not a multiple of {patch}")
    h, w = H // patch, W // patch
    return 1.0 - bad_mask.reshape(V, h, patch, w, patch).mean(axis=(2, 4), dtype=np.float32)


def view_gamma(n_views, bad_views):
    g = np.ones(n_views, np.float32)
    g[list(bad_views)] = 0.0
    return g


def decompose(hint, gt):
    """Experiment G0. Error of a depth hint (N,H,W) vs truth after removing nothing, its overall scale, the scale of each view.
    What is left after the per-view rescale is distortion inside the maps."""
    v = (hint > 0) & (gt > 0)
    r_all = global_ratio(hint, gt)
    r_view = view_ratios(hint, gt)
    rel = lambda a: float(np.mean(np.abs(a[v] - gt[v]) / gt[v]))
    return {"overall_ratio": r_all, "view_ratio_min": float(r_view.min()), "view_ratio_max": float(r_view.max()),
            "view_ratio_sd_log": float(np.std(np.log(r_view))), "absrel_raw": rel(hint),
            "absrel_after_overall": rel(hint / r_all), "absrel_after_per_view": rel(hint / r_view[:, None, None]),
            "valid_share": float(v.mean())}
