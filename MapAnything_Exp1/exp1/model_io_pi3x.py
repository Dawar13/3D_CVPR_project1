"""The ONLY place that touches the Pi3X API (extended E2, third model). Same interface as model_io.py, so the runner,
corruptions and metrics are shared: `python -m exp1.run_exp1 --backend pi3x ...`.

Read from github.com/yyfz/Pi3 @ 9fa3ddb (third_party/Pi3), file:line as cited:
  - Model: Pi3X.from_pretrained("yyfz233/Pi3X") (example_mm.py:74), multimodal branch kept (use_multimodal=True,
    pi3x.py:20-24). bf16 autocast on GPUs with compute capability >= 8 (example_mm.py:108-111, README.md:170).
  - Images: RGB in [0, 1], (B, N, 3, H, W), H and W multiples of 14 (pi3x.py:250 patch grid; utils/basic.py:159-165).
  - Hints (pi3x.py:205-216, 283-357), each with its own per-view boolean mask of shape (B, N) (used as (B, N) at
    :322-326 and :373-374; the docstring's (B, N, N) is wrong):
    - poses: (B, N, 4, 4) camera-to-world, OpenCV axes. Made relative to view 0 (:339), translations divided by the
      mean valid depth over ALL views (:334, :340) or, without any depth hint, by the mean distance of views 1..N
      from view 0 (:343-356; unposed views count with distance 0). So the metric scale of the hints never reaches
      the network; only the scale of the pose hints relative to the depth hints does. A single posed view is
      switched off (:328-331); the run is then images-only for poses.
    - intrinsics: (B, N, 3, 3) in pixels, turned into unnormalised rays K^-1 [u v 1] (:311-312). Poses without
      intrinsics are accepted (rays are then zeros with the ray mask off, :307-309).
    - depths: (B, N, H, W), 0 = invalid (:300-302; normalize_depth :515).
  - Masks MUST be passed: a mask left as None is drawn at random (:322-326).
  - Missing hint = the hint feature multiplied by 0 (:373-374 depth/ray; :502 pose), as in MapAnything.
  - Outputs (:443-450): camera_poses (B, N, 4, 4) camera-to-world, metric; local_points (camera frame, metric, z =
    depth); points (world, metric); conf (logits; the official mask uses sigmoid > 0.1, example_mm.py:125);
    metric (one scale per scene). Intrinsics are recovered from the ray directions of local_points with the
    principal point at the image centre, as the example does (example_mm.py:118-119, utils/geometry.py:518).
"""

import os
import sys
import time

import numpy as np
import torch

INFER_KW = dict(dtype="bf16 autocast (fp32 below sm80: fp16)", masks="explicit per view", conf="sigmoid(logit)")
INPUT_GROUPS = {"intrinsics", "pose", "depth"}
REPO = os.environ.get("PI3_REPO", os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "Pi3"))


def _import():
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    from pi3.models.pi3x import Pi3X
    from pi3.utils.geometry import recover_intrinsic_from_rays_d
    return Pi3X, recover_intrinsic_from_rays_d


def load_model(checkpoint, device):
    """checkpoint: a .safetensors path or a Hugging Face repo id (yyfz233/Pi3X)."""
    Pi3X = _import()[0]
    if os.path.exists(checkpoint):
        from safetensors.torch import load_file
        model = Pi3X(use_multimodal=True)
        model.load_state_dict(load_file(checkpoint), strict=True)
    else:
        model = Pi3X.from_pretrained(checkpoint)
    return model.to(device).eval()


def build_views(images, K, poses, inputs, pose_mask=None, metric=True, depth=None, depth_views=()):
    """Same arguments as model_io.build_views. `metric` has no counterpart (hint scale is normalised away)."""
    unknown = set(inputs) - INPUT_GROUPS
    if unknown:
        raise ValueError(f"unknown input groups {unknown}")
    N, H, W = images.shape[:3]
    if H % 14 or W % 14:
        raise ValueError(f"image size {W}x{H} is not a multiple of 14")
    m_pose = np.array([("pose" in inputs) and (pose_mask is None or bool(pose_mask[i])) for i in range(N)])
    if m_pose.any() and not m_pose[0]:
        raise ValueError("view 0 must have a pose hint when any view has one (Pi3X makes poses relative to view 0)")
    m_ray = np.full(N, "intrinsics" in inputs)
    dv = [i for i in sorted(depth_views) if "depth" in inputs]
    m_depth = np.zeros(N, bool)
    m_depth[dv] = True
    P = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    for i in np.flatnonzero(m_pose):
        P[i] = np.asarray(poses[i], np.float32)
    d = np.zeros((N, H, W), np.float32)
    for i in dv:
        d[i] = depth[i]
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a))[None]
    return {"imgs": torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float()[None] / 255.0,
            "poses": t(P) if m_pose.any() else None,
            "intrinsics": t(np.asarray(K, np.float32)) if m_ray.any() else None,
            "depths": t(d) if m_depth.any() else None,
            "mask_add_pose": t(m_pose), "mask_add_ray": t(m_ray), "mask_add_depth": t(m_depth)}


def _np(x):
    return x.detach().float().cpu().numpy()


@torch.no_grad()
def run(model, views, device):
    _, recover_K = _import()
    inp = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in views.items()}
    use_amp = device.startswith("cuda")
    dtype = torch.bfloat16 if use_amp and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    if use_amp:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.amp.autocast("cuda", dtype=dtype, enabled=use_amp):
        res = model(with_prior=True, **inp)
    if use_amp:
        torch.cuda.synchronize()
    runtime = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30 if use_amp else 0.0
    lp = res["local_points"].float()
    K = recover_K(torch.nn.functional.normalize(lp, dim=-1), force_center_principal_point=True)
    depth = _np(lp[0, ..., 2])
    return {"pts3d": _np(res["points"][0]), "depth_z": depth, "pose": _np(res["camera_poses"][0]).astype(np.float64),
            "conf": _np(torch.sigmoid(res["conf"][0, ..., 0].float())), "mask": np.ones(depth.shape, bool),
            "intrinsics": _np(K[0]), "metric_scaling_factor": float(_np(res["metric"]).ravel()[0]),
            "runtime_s": runtime, "peak_mem_gb": peak}


def set_pose_scale_statistic(stat):
    if stat != "mean":
        raise NotImplementedError("the median pose-scale probe exists for MapAnything only (E2-C4)")
