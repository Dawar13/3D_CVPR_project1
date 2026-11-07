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


def median_pose_translations(pose_translations, return_norm_factor=False):
    """Drop-in for mapanything.utils.geometry.normalize_pose_translations (geometry.py:1558-1595)
    with the MEDIAN instead of the mean distance of the non-zero translations as the pose scale.
    Diagnostic only (E2-C C4). torch.quantile(0.5) averages the two middle values for even counts."""
    assert pose_translations.ndim == 3 and pose_translations.shape[2] == 3
    dis = pose_translations.norm(dim=-1)  # [B, V]
    factors = []
    for row in dis:
        nz = row[row > 0]
        factors.append(torch.quantile(nz.float(), 0.5).to(row.dtype) if nz.numel() else row.new_zeros(()))
    norm_factor = torch.stack(factors).clip(min=1e-8)
    normalized = pose_translations / norm_factor.unsqueeze(-1).unsqueeze(-1)
    return (normalized, norm_factor) if return_norm_factor else normalized


def set_pose_scale_statistic(stat):
    """'mean' (MapAnything as shipped) or 'median'. model.py imports the function by name
    (model.py:22-26) and calls it once per forward (model.py:1122-1126), so swapping the module
    attribute changes only the pose-scale statistic; third_party is not edited."""
    import mapanything.models.mapanything.model as mm
    from mapanything.utils import geometry
    if stat == "mean":
        mm.normalize_pose_translations = geometry.normalize_pose_translations
    elif stat == "median":
        mm.normalize_pose_translations = median_pose_translations
    else:
        raise ValueError(f"unknown pose-scale statistic {stat!r}")


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


# ---------------------------------------------------------------------------------------------------------------------
# Gate track (experiments/G.md). Additive: nothing above changes, and with no gate set the model is untouched.
# ---------------------------------------------------------------------------------------------------------------------
def _gated_fuse(orig, gamma):
    """F_in + gamma * (F_out - F_in) around a fuse method, whose 4th positional argument is F_in (B*V, C, h, w):
    _encode_and_fuse_depths (model.py:871-878, sum at 1048-1052), _encode_and_fuse_cam_quats_and_trans (1056-1065,
    sum at 1168-1173). gamma: (V,) per view or (V, h, w) per patch. gamma = 0 removes exactly what this fuse added."""
    def wrapped(*args, **kwargs):
        feats = args[3]
        fused = orig(*args, **kwargs)
        g = torch.as_tensor(np.asarray(gamma), device=fused.device, dtype=fused.dtype)
        if fused.shape[0] != g.shape[0]:
            raise ValueError(f"gate has {g.shape[0]} views, features have {fused.shape[0]} (batch size must be 1)")
        if g.ndim == 1:
            g = g[:, None, None, None]
        elif g.ndim == 3:
            if g.shape[1:] != fused.shape[2:]:
                raise ValueError(f"gate grid {tuple(g.shape[1:])} != feature grid {tuple(fused.shape[2:])}")
            g = g[:, None]
        else:
            raise ValueError("gamma must be (V,) or (V, h, w)")
        # in fp32, then back: in bf16, feats + 1 * (fused - feats) is not fused (rounding of the difference); the
        # E2X session-A gate check (gamma = 1 must change nothing) failed by up to 0.23 on ETH3D, likely for this
        # reason (seen in the lost VM log). With the difference taken in fp32, gamma = 1 returns fused and
        # gamma = 0 returns feats exactly.
        out = feats.float() + g.float() * (fused.float() - feats.float())
        return out.to(fused.dtype)
    return wrapped


def set_hint_gate(model, views=None, gamma_depth=None, gamma_pose=None):
    """Install the hint gate for the next run; `views` is unused here (same signature as model_io_omnivggt).
    Call clear_hint_gate afterwards. Instance attributes shadow the methods; third_party is not edited."""
    clear_hint_gate(model)
    if gamma_depth is not None:
        model._encode_and_fuse_depths = _gated_fuse(type(model)._encode_and_fuse_depths.__get__(model), gamma_depth)
    if gamma_pose is not None:
        model._encode_and_fuse_cam_quats_and_trans = _gated_fuse(
            type(model)._encode_and_fuse_cam_quats_and_trans.__get__(model), gamma_pose)


def clear_hint_gate(model):
    for name in ("_encode_and_fuse_depths", "_encode_and_fuse_cam_quats_and_trans"):
        model.__dict__.pop(name, None)


class TokenCapture:
    """Patch tokens after chosen GLOBAL attention blocks, as {layer: (V, T, C) float32 on CPU}.
    MapAnything: even indices of info_sharing.self_attention_blocks are global (N, V*T + extra, C)
    (uniception 0.1.7 alternating_attention_transformer.py:397-401); the scale token sits at the end."""

    def __init__(self, model, layers, n_views, full=False):
        # full: accepted for the same signature as the other backends; MapAnything views have patch tokens only
        self.tokens, self.n_views, self.handles = {}, n_views, []
        blocks = model.info_sharing.self_attention_blocks
        for k in layers:
            if k % 2:
                raise ValueError(f"block {k} is frame attention; only even (global) blocks are captured")
            self.handles.append(blocks[k].register_forward_hook(self._hook(k)))

    def _hook(self, k):
        def fn(_module, _inputs, out):
            V = self.n_views
            T = (out.shape[1] - 1) // V    # one extra token: the scale token (model.py:1584-1596)
            self.tokens[k] = out[0, :V * T].reshape(V, T, -1).detach().float().cpu()
        return fn

    def remove(self):
        for h in self.handles:
            h.remove()


class TokenPatch:
    """E2X X-M (activation patching): after GLOBAL block `layer`, replace the patch tokens of `views` with stored
    ones, e.g. TokenCapture(...).tokens[layer] from another run of the same clip (same images, other hints).
    tokens: (V, T, C). The scale token (last) and the other views are left as they are. Use as a context manager
    or call remove(). Same block layout as TokenCapture."""

    def __init__(self, model, layer, views, tokens):
        if layer % 2:
            raise ValueError(f"block {layer} is frame attention; only even (global) blocks are patched")
        self.views, self.tokens = [int(v) for v in views], tokens
        blocks = model.info_sharing.self_attention_blocks
        self.handles = [blocks[layer].register_forward_hook(self._hook)]

    def _hook(self, _module, _inputs, out):
        V, T = self.tokens.shape[0], self.tokens.shape[1]
        if out.shape[1] != V * T + 1:
            raise ValueError(f"stored tokens ({V} views x {T}) do not match the block output {tuple(out.shape)}")
        out = out.clone()
        for v in self.views:
            out[0, v * T:(v + 1) * T] = self.tokens[v].to(out.device, out.dtype)
        return out

    def remove(self):
        for h in self.handles:
            h.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()
