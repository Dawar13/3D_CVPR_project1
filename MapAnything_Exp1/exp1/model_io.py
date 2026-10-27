"""The ONLY place that touches the MapAnything API.

Calls follow the official README (facebookresearch/map-anything):
  model = MapAnything.from_pretrained("facebook/map-anything").to(device)
  views = preprocess_inputs([{"img": ..., "intrinsics": ..., "camera_poses": ...,
                              "is_metric_scale": ...}, ...])
  preds = model.infer(views, ...)
RGB-only is obtained the official way: optional keys are simply absent.

Before the first real run, check README.md "Things to verify" against the
installed commit; any mismatch should be fixed here and nowhere else.

Statically verified against map-anything 3d10cf7a (2026-08-07) by reading the
source: every call, key and output index below matches. Evidence with file:line
is in reports/e0_interface.md. Runtime confirmation is E0 Steps 6-8.
"""

import time

import numpy as np
import torch

INFER_KW = dict(
    memory_efficient_inference=False,
    use_amp=True,
    amp_dtype="bf16",
    # keep every predicted pixel: masking would zero points and change the set of
    # evaluated pixels between conditions. We filter with GT validity only.
    apply_mask=False,
    mask_edges=False,
    apply_confidence_mask=False,
)

# input groups a condition may switch on
INPUT_GROUPS = {"intrinsics", "pose", "depth"}


def load_model(checkpoint, device):
    from mapanything.models import MapAnything
    model = MapAnything.from_pretrained(checkpoint).to(device)
    model.eval()
    return model


def build_views(images, K, poses, inputs, pose_mask=None, metric=True, depth=None, depth_views=()):
    """images (N,H,W,3) uint8; K (N,3,3); poses (N,4,4) cam2world; depth (N,H,W) z-depth, 0 = invalid.

    inputs: subset of INPUT_GROUPS. pose_mask: optional bool (N,) — when given,
    only views with True receive a pose (used by the prior-removal controls).
    depth_views: views that receive a `depth_z` hint when "depth" is in inputs
    (MapAnything requires intrinsics with depth, inference.py:166-176).
    metric: is_metric_scale. MapAnything defaults an absent flag to True
    (inference.py:265-272), so metric=False is set on every view; it only gates
    the pose/depth scale features (model.py:1043-1045, 1162-1165).
    """
    unknown = set(inputs) - INPUT_GROUPS
    if unknown:
        raise ValueError(f"unknown input groups {unknown}")
    if "depth" in inputs and "intrinsics" not in inputs:
        raise ValueError("depth hints need intrinsics")
    views = []
    for i in range(len(images)):
        v = {"img": torch.from_numpy(images[i].copy())}
        if "intrinsics" in inputs:
            v["intrinsics"] = torch.from_numpy(K[i]).float()
        if "pose" in inputs and (pose_mask is None or pose_mask[i]):
            v["camera_poses"] = torch.from_numpy(poses[i]).float()
        if "depth" in inputs and i in depth_views:
            v["depth_z"] = torch.from_numpy(np.ascontiguousarray(depth[i])).float()
        if "camera_poses" in v or not metric:
            v["is_metric_scale"] = torch.tensor([bool(metric)])
        views.append(v)
    return views


def _np(x):
    return x.detach().float().cpu().numpy()


@torch.no_grad()
def run(model, views, device):
    from mapanything.utils.image import preprocess_inputs
    processed = preprocess_inputs(views)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.time()
    preds = model.infer(processed, **INFER_KW)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    runtime = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30 if device.startswith("cuda") else 0.0

    out = {"pts3d": [], "depth_z": [], "pose": [], "conf": [], "mask": [], "intrinsics": []}
    for p in preds:
        out["pts3d"].append(_np(p["pts3d"][0]))               # (H,W,3) world
        out["depth_z"].append(_np(p["depth_z"][0]).squeeze(-1))  # (H,W)
        out["pose"].append(_np(p["camera_poses"][0]))         # (4,4) cam2world
        out["conf"].append(_np(p["conf"][0]))                 # (H,W)
        out["intrinsics"].append(_np(p["intrinsics"][0]))     # (3,3) recovered from rays
        out["mask"].append(_np(p["mask"][0]).squeeze(-1) > 0.5 if "mask" in p
                           else np.ones(out["conf"][-1].shape, bool))
    out = {k: np.stack(v) for k, v in out.items()}
    # one scale per scene, shape (1, 1) at runtime (reports/e0_interface.md)
    out["metric_scaling_factor"] = float(_np(preds[0]["metric_scaling_factor"]).ravel()[0])
    out["runtime_s"] = runtime
    out["peak_mem_gb"] = peak
    return out


@torch.no_grad()
def official_example(model, image_paths, device):
    """README image-only example, kwargs copied verbatim (E0 Step 6). Raw predictions."""
    from mapanything.utils.image import load_images
    views = load_images(image_paths)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.time()
    preds = model.infer(
        views,
        memory_efficient_inference=True,
        minibatch_size=None,
        use_amp=True,
        amp_dtype="bf16",
        apply_mask=True,
        mask_edges=True,
        apply_confidence_mask=False,
        confidence_percentile=10,
        use_multiview_confidence=False,
    )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    runtime = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30 if device.startswith("cuda") else 0.0
    return preds, runtime, peak
