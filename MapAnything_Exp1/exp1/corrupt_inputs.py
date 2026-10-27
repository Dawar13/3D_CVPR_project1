"""Deterministic, logged corruption of lens (intrinsics) and depth hints.

Intrinsics (spec["target"] = "K"), applied to the 3x3 K of the chosen views:
  "focal"     fx, fy  times `factor`
  "principal" cx, cy  plus `shift` * image width (both in the +x / +y direction)
  views: spec["views"] = "all" (default), "one" (one random non-reference view), or a list.

Depth (spec["target"] = "depth"), applied to the depth hints of the chosen views
(a subset of the config's depth_views; spec["views"] defaults to all depth views):
  "scale" every valid depth times `factor`
  "noise" every valid depth times (1 + sigma * n), n ~ N(0, 1) per pixel, kept >= 5%
Invalid pixels (0) stay 0. factor 1 / shift 0 / sigma 0 return the input exactly.
"""

import numpy as np

from .corrupt_geometry import _seed_for


def _views(spec, n_views, default, rng):
    v = spec.get("views", default)
    if v == "all":
        return list(range(n_views))
    if v == "one":
        return [int(rng.integers(1, n_views))]
    return sorted(int(i) for i in v)


def corrupt_intrinsics(K, spec, base_seed, clip_id, cond_name, width):
    """Return (corrupted K (N,3,3), manifest_entry). K is never modified."""
    clean = np.asarray(K, dtype=np.float64)
    seed = _seed_for(base_seed, clip_id, cond_name)
    rng = np.random.default_rng(seed)
    views = _views(spec, len(clean), "all", rng)
    out = clean.copy()
    kind = spec["type"]
    for i in views:
        if kind == "focal":
            out[i, 0, 0] *= float(spec["factor"])
            out[i, 1, 1] *= float(spec["factor"])
        elif kind == "principal":
            out[i, 0, 2] += float(spec["shift"]) * width
            out[i, 1, 2] += float(spec["shift"]) * width
        else:
            raise ValueError(f"unknown intrinsics corruption type {kind!r}")
    entry = {"clip_id": clip_id, "condition": cond_name, "target": "K", "type": kind,
             "base_seed": base_seed, "rng_seed": seed, "spec": spec, "width_px": int(width),
             "affected_views": views,
             "per_view": {int(i): {"K_before": clean[i].tolist(), "K_after": out[i].tolist()}
                          for i in views}}
    return out, entry


def corrupt_depth(depth, depth_views, spec, base_seed, clip_id, cond_name):
    """Return (depth with corrupted hint views (N,H,W), manifest_entry). depth is never modified."""
    clean = np.asarray(depth, dtype=np.float32)
    seed = _seed_for(base_seed, clip_id, cond_name)
    rng = np.random.default_rng(seed)
    views = _views(spec, len(clean), list(depth_views), rng)
    if not set(views) <= set(depth_views):
        raise ValueError(f"depth corruption on {views}, but depth hints are only on {list(depth_views)}")
    out = clean.copy()
    kind = spec["type"]
    per_view = {}
    for i in views:
        valid = clean[i] > 0
        if kind == "scale":
            out[i][valid] = clean[i][valid] * np.float32(spec["factor"])
        elif kind == "noise":
            if float(spec["sigma"]) > 0:
                m = 1.0 + float(spec["sigma"]) * rng.normal(size=int(valid.sum()))
                out[i][valid] = clean[i][valid] * np.maximum(m, 0.05).astype(np.float32)
        else:
            raise ValueError(f"unknown depth corruption type {kind!r}")
        r = out[i][valid] / clean[i][valid]
        per_view[int(i)] = {"n_valid": int(valid.sum()), "ratio_median": float(np.median(r)) if r.size else 1.0,
                            "ratio_p10": float(np.percentile(r, 10)) if r.size else 1.0,
                            "ratio_p90": float(np.percentile(r, 90)) if r.size else 1.0}
    entry = {"clip_id": clip_id, "condition": cond_name, "target": "depth", "type": kind,
             "base_seed": base_seed, "rng_seed": seed, "spec": spec, "depth_views": list(depth_views),
             "affected_views": views, "per_view": per_view}
    return out, entry
