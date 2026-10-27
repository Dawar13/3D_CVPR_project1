"""The ONLY place that touches the MapAnything API.

Calls follow the official README (facebookresearch/map-anything):
  model = MapAnything.from_pretrained("facebook/map-anything").to(device)
  views = preprocess_inputs([{"img": ..., "intrinsics": ..., "camera_poses": ...,
                              "is_metric_scale": ...}, ...])
  preds = model.infer(views, ...)
RGB-only is obtained the official way: optional keys are simply absent.

Before the first real run, check README.md "Things to verify" against the
installed commit; any mismatch should be fixed here and nowhere else.
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
INPUT_GROUPS = {"intrinsics", "pose"}


def load_model(checkpoint, device):
    from mapanything.models import MapAnything
    model = MapAnything.from_pretrained(checkpoint).to(device)
    model.eval()
    return model


def build_views(images, K, poses, inputs, pose_mask=None, metric=True):
    """images (N,H,W,3) uint8; K (N,3,3); poses (N,4,4) cam2world.

    inputs: subset of INPUT_GROUPS. pose_mask: optional bool (N,) — when given,
    only views with True receive a pose (used by the prior-removal controls).
    """
    unknown = set(inputs) - INPUT_GROUPS
    if unknown:
        raise ValueError(f"unknown input groups {unknown}")
    views = []
    for i in range(len(images)):
        v = {"img": torch.from_numpy(images[i].copy())}
        if "intrinsics" in inputs:
            v["intrinsics"] = torch.from_numpy(K[i]).float()
        if "pose" in inputs and (pose_mask is None or pose_mask[i]):
            v["camera_poses"] = torch.from_numpy(poses[i]).float()
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

    out = {"pts3d": [], "depth_z": [], "pose": [], "conf": [], "mask": []}
    for p in preds:
        out["pts3d"].append(_np(p["pts3d"][0]))               # (H,W,3) world
        out["depth_z"].append(_np(p["depth_z"][0]).squeeze(-1))  # (H,W)
        out["pose"].append(_np(p["camera_poses"][0]))         # (4,4) cam2world
        out["conf"].append(_np(p["conf"][0]))                 # (H,W)
        out["mask"].append(_np(p["mask"][0]).squeeze(-1) > 0.5 if "mask" in p
                           else np.ones(out["conf"][-1].shape, bool))
    out = {k: np.stack(v) for k, v in out.items()}
    out["runtime_s"] = runtime
    out["peak_mem_gb"] = peak
    return out
