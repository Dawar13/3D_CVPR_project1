"""The ONLY place that touches the OmniVGGT API (E2-D). Same interface as model_io.py, so the runner,
corruptions and metrics are shared: `python -m exp1.run_exp1 --backend omnivggt ...`.

Read from OmniVGGT-official @ c8cbfc6 (third_party/OmniVGGT-official), file:line as cited:
  - Model: OmniVGGT() with the safetensors state dict, strict (inference.py:323-324). Called through
    model.inference(...) (omnivggt/models/omnivggt.py:69-77), as inference.py:356 does. fp32, no autocast
    (inference.py:354-356). The README quick start calls model(**inputs) with *_gt_index keys, which
    forward() does not accept (omnivggt.py:21-28).
  - Images: RGB in [0, 1], shape (S, 3, H, W), H and W multiples of 14 (visual_util.py:731-750;
    utils/image.py:26 ToTensor).
  - Camera hint = extrinsics (world-to-camera, 3x4) AND intrinsics together, for the views listed in
    camera_gt_index. A hint cannot be pose-only or intrinsics-only (visual_util.py:803-822).
    - Hinted cameras are made relative to the first hinted camera, and translations are divided by
      their MEAN distance (omnivggt_aggregator.py:90-110), so no metric scale enters.
    - Intrinsics enter only as the field of view (pose_enc.py:49-58); the principal point is unused.
    - View 0 must be hinted when any view is (README "Input Description").
  - Depth hint: depth (1, S, H, W, 1) with a validity mask (1, S, H, W) for depth_gt_index, divided by
    ONE mean over all hinted views together (omnivggt_aggregator.py:112-133).
  - Missing hints: a missing camera is a zero token through a zero-initialised adapter; a missing depth
    is a learned placeholder (omnivggt_aggregator.py:344-346, 366-376, 438-450).
  - Outputs:
    - pose_enc -> world-to-camera extrinsics and intrinsics with the principal point at the image
      centre (pose_enc.py:65-104).
    - depth + depth_conf.
    - World points: the depth unprojected with the predicted cameras, the official viewer's default
      (inference.py:83-89).
    - There is no metric scale: metric_scaling_factor is NaN and the *_metric errors are meaningless.
"""

import os
import sys
import time

import numpy as np
import torch

INFER_KW = dict(dtype="fp32", autocast=False, points="depth_unprojection")
INPUT_GROUPS = {"intrinsics", "pose", "depth"}
REPO = os.environ.get("OMNIVGGT_REPO", os.path.join(os.path.dirname(__file__), "..", "third_party",
                                                     "OmniVGGT-official"))


def _import():
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    from omnivggt.models.omnivggt import OmniVGGT
    from omnivggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
    from omnivggt.utils.pose_enc import pose_encoding_to_extri_intri
    return OmniVGGT, closed_form_inverse_se3, unproject_depth_map_to_point_map, pose_encoding_to_extri_intri


def load_model(checkpoint, device):
    """checkpoint: a .safetensors path or a Hugging Face repo id (e.g. Livioni/OmniVGGT)."""
    from safetensors.torch import load_file
    OmniVGGT = _import()[0]
    path = checkpoint
    if not os.path.exists(path):
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(checkpoint, "OmniVGGT.safetensors")
    model = OmniVGGT()
    model.load_state_dict(load_file(path), strict=True)
    return model.to(device).eval()


def build_views(images, K, poses, inputs, pose_mask=None, metric=True, depth=None, depth_views=()):
    """Same arguments as model_io.build_views. `metric` has no counterpart (scale is normalised away).
    Raises for hint combinations OmniVGGT cannot take (pose without intrinsics, intrinsics alone)."""
    unknown = set(inputs) - INPUT_GROUPS
    if unknown:
        raise ValueError(f"unknown input groups {unknown}")
    if ("pose" in inputs) != ("intrinsics" in inputs):
        raise ValueError("OmniVGGT takes camera hints as pose AND intrinsics together")
    N, H, W = images.shape[:3]
    if H % 14 or W % 14:
        raise ValueError(f"image size {W}x{H} is not a multiple of 14")
    cam = [i for i in range(N) if "pose" in inputs and (pose_mask is None or pose_mask[i])]
    if cam and cam[0] != 0:
        raise ValueError("view 0 must have a camera hint when any view has one")
    ext = np.zeros((N, 3, 4), np.float32)
    intr = np.zeros((N, 3, 3), np.float32)
    for i in cam:
        T = np.linalg.inv(np.asarray(poses[i], np.float64))  # cam2world -> world2cam
        ext[i], intr[i] = T[:3], K[i]
    d = np.zeros((N, H, W), np.float32)
    dv = [i for i in sorted(depth_views) if "depth" in inputs]
    for i in dv:
        d[i] = depth[i]
    return {"images": torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float() / 255.0,
            "extrinsics": torch.from_numpy(ext)[None], "intrinsics": torch.from_numpy(intr)[None],
            "depth": torch.from_numpy(d)[None, ..., None], "mask": torch.from_numpy((d > 0).astype(np.float32))[None],
            "camera_gt_index": cam, "depth_gt_index": dv}


def _np(x):
    return x.detach().float().cpu().numpy()


@torch.no_grad()
def run(model, views, device):
    _, inv_se3, unproject, decode = _import()
    inp = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in views.items()}
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.time()
    pred = model.inference(**inp)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    runtime = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30 if device.startswith("cuda") else 0.0
    H, W = inp["images"].shape[-2:]
    ext, intr = decode(pred["pose_enc"], (H, W))
    ext, intr = _np(ext[0]), _np(intr[0])                      # (S,3,4) world2cam, (S,3,3)
    depth = _np(pred["depth"][0])                               # (S,H,W,1)
    pts = unproject(depth, ext, intr)                           # (S,H,W,3) in the predicted world
    c2w = inv_se3(np.concatenate([ext, np.tile([[[0, 0, 0, 1]]], (len(ext), 1, 1))], 1))
    return {"pts3d": pts.astype(np.float32), "depth_z": depth[..., 0], "pose": np.asarray(c2w, np.float64),
            "conf": _np(pred["depth_conf"][0]), "mask": np.ones(depth.shape[:3], bool), "intrinsics": intr,
            "metric_scaling_factor": float("nan"), "runtime_s": runtime, "peak_mem_gb": peak}


def set_pose_scale_statistic(stat):
    if stat != "mean":
        raise NotImplementedError("the median pose-scale probe exists for MapAnything only (E2-C4)")
