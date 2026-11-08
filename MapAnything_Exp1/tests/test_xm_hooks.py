"""E2X X-M tools on CPU with small fake models that have the real block layouts (no weights, no GPU):
  - TokenPatch for MapAnything, OmniVGGT and Pi3X: the patched views carry the stored state at that layer, every
    other view and the scale token are untouched, patching a run with its own state changes nothing;
  - the hint gate is exact at gamma = 1 and gamma = 0 in bf16 (the session-A gate check);
  - M8: two corruptions in one condition (pose scale x depth scale).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from exp1 import model_io, model_io_omnivggt, model_io_pi3x, run_exp1  # noqa: E402

V, T, C = 3, 4, 8


class Mix(nn.Module):
    """A block that mixes all tokens (so a patched view changes the others later)."""

    def __init__(self, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.w = nn.Parameter(torch.randn(C, C, generator=g) * 0.3)

    def forward(self, x, **_):
        return x + torch.tanh(x @ self.w) + 0.1 * x.mean(dim=-2, keepdim=True)


# ----------------------------------------------------------------------------------------------- MapAnything layout
class FakeMA(nn.Module):
    """info_sharing.self_attention_blocks; tokens (1, V*T + 1, C), scale token last (even blocks = global)."""

    def __init__(self):
        super().__init__()
        self.info_sharing = nn.Module()
        self.info_sharing.self_attention_blocks = nn.ModuleList([Mix(i) for i in range(6)])

    def forward(self, x):
        for b in self.info_sharing.self_attention_blocks:
            x = b(x)
        return x


def test_mapanything_patch():
    m = FakeMA()
    xa, xb = torch.randn(1, V * T + 1, C), torch.randn(1, V * T + 1, C)
    cap = model_io.TokenCapture(m, [2], V)
    out_a = m(xa)
    ta = cap.tokens[2].clone()
    m(xb)
    tb = cap.tokens[2].clone()
    cap.remove()
    with model_io.TokenPatch(m, 2, [1], ta):
        cap2 = model_io.TokenCapture(m, [2], V)
        out_p = m(xb)
        tp = cap2.tokens[2]
        cap2.remove()
    assert torch.equal(tp[1], ta[1]) and torch.equal(tp[0], tb[0]) and torch.equal(tp[2], tb[2])
    assert not torch.allclose(out_p, m(xb))           # the patch reached the output
    with model_io.TokenPatch(m, 2, [0, 1, 2], ta):    # every view from run A: run A's output, scale token aside
        out_all = m(xb)
    assert not torch.allclose(out_all, out_a)        # the scale token still comes from run B
    plain_b = m(xb)
    with model_io.TokenPatch(m, 2, [1], tb):         # own state: identity
        assert torch.allclose(m(xb), plain_b, atol=1e-6)
    with pytest.raises(ValueError):
        model_io.TokenPatch(m, 1, [1], ta)           # frame block


def test_gate_exact_in_fp32_and_per_patch():
    """A2 amendment 2: exact pass-through at gamma = 1 / 0 also in fp32 and with per-patch gates."""
    feats = torch.randn(V, C, 2, 2) * 3
    fused = feats * 1.37 + 0.11 * torch.randn(V, C, 2, 2)
    orig = lambda a, b, c, f: fused
    assert torch.equal(model_io._gated_fuse(orig, np.ones(V, np.float32))(None, None, None, feats), fused)
    assert torch.equal(model_io._gated_fuse(orig, np.zeros(V, np.float32))(None, None, None, feats), feats)
    g = np.ones((V, 2, 2), np.float32)
    g[1, 0, 0] = 0
    out = model_io._gated_fuse(orig, g)(None, None, None, feats)
    assert torch.equal(out[1, :, 0, 0], feats[1, :, 0, 0]) and torch.equal(out[0], fused[0])


def test_gate_exact_in_bf16():
    feats = torch.randn(V, C, 2, 2).to(torch.bfloat16)
    fused = (feats * 1.37 + 0.11 * torch.randn(V, C, 2, 2)).to(torch.bfloat16)
    orig = lambda a, b, c, f: fused
    one = model_io._gated_fuse(orig, np.ones(V, np.float32))(None, None, None, feats)
    zero = model_io._gated_fuse(orig, np.zeros(V, np.float32))(None, None, None, feats)
    assert one.dtype == torch.bfloat16 and torch.equal(one, fused)
    assert torch.equal(zero, feats)
    half = model_io._gated_fuse(orig, np.full(V, 0.5, np.float32))(None, None, None, feats)
    assert torch.allclose(half.float(), (feats.float() + fused.float()) / 2, atol=1e-2)


# --------------------------------------------------------------------------------------------------- OmniVGGT layout
class FakeOV(nn.Module):
    def __init__(self, start=2):
        super().__init__()
        self.aggregator = nn.Module()
        self.aggregator.patch_start_idx = start
        self.aggregator.global_blocks = nn.ModuleList([Mix(10 + i) for i in range(3)])

    def forward(self, x):          # (1, V*P, C), camera + register tokens first in each view
        for b in self.aggregator.global_blocks:
            x = b(x)
        return x


@pytest.mark.parametrize("full", [True, False])
def test_omnivggt_patch(full):
    m = FakeOV()
    P = T + m.aggregator.patch_start_idx
    xa, xb = torch.randn(1, V * P, C), torch.randn(1, V * P, C)
    cap = model_io_omnivggt.TokenCapture(m, [1], V, full=full)
    m(xa)
    ta = cap.tokens[1].clone()
    m(xb)
    tb = cap.tokens[1].clone()
    cap.remove()
    with model_io_omnivggt.TokenPatch(m, 1, [2], ta):
        cap2 = model_io_omnivggt.TokenCapture(m, [1], V, full=full)
        m(xb)
        tp = cap2.tokens[1]
        cap2.remove()
    assert torch.equal(tp[2], ta[2]) and torch.equal(tp[0], tb[0]) and torch.equal(tp[1], tb[1])
    if not full:   # camera and register tokens of the patched view kept from run B
        capf = model_io_omnivggt.TokenCapture(m, [1], V, full=True)
        with model_io_omnivggt.TokenPatch(m, 1, [2], ta):
            m(xb)
        mb = model_io_omnivggt.TokenCapture(m, [1], V, full=True)
        m(xb)
        assert torch.equal(capf.tokens[1][2, :2], mb.tokens[1][2, :2])
        capf.remove()
        mb.remove()


# ------------------------------------------------------------------------------------------------------- Pi3X layout
class FakePi3X(nn.Module):
    """decoder blocks alternating frame (even, (N, hw, C)) / global (odd, (1, N*hw, C)); a camera-hint injection
    after block 1 (as pi3x.py:499-505); 5 register tokens first in each view."""

    def __init__(self):
        super().__init__()
        self.patch_start_idx = 5
        self.decoder = nn.ModuleList([Mix(20 + i) for i in range(6)])
        self.inject = nn.Parameter(torch.randn(C) * 0.5)

    def forward(self, x, pose_hint=True):   # x (N, hw, C)
        N, hw, _ = x.shape
        for i, blk in enumerate(self.decoder):
            x = x.reshape(N, hw, -1) if i % 2 == 0 else x.reshape(1, N * hw, -1)
            x = blk(x)
            if i == 1 and pose_hint:
                x = x.reshape(1, N, hw, -1)
                x[..., self.patch_start_idx:, :] += self.inject
                x = x.reshape(1, N * hw, -1)
        return x.reshape(N, hw, -1)


def test_pi3x_capture_after_injection_and_patch():
    m = FakePi3X()
    hw = T + m.patch_start_idx
    x = torch.randn(V, hw, C)
    c1 = model_io_pi3x.TokenCapture(m, [1], V, full=True)
    m(x, pose_hint=True)
    a = c1.tokens[1].clone()
    m(x, pose_hint=False)
    b = c1.tokens[1].clone()
    c1.remove()
    d = (a - b)[:, m.patch_start_idx:]                 # the injection is inside the captured state
    assert torch.allclose(d, m.inject.detach().expand_as(d), atol=1e-6)
    assert torch.equal(a[:, :m.patch_start_idx], b[:, :m.patch_start_idx])
    with model_io_pi3x.TokenPatch(m, 1, [1], b):       # remove the hint's effect on view 1 at that point
        c2 = model_io_pi3x.TokenCapture(m, [1], V, full=True)
        m(x, pose_hint=True)
        p = c2.tokens[1]
        c2.remove()
    assert torch.equal(p[1], b[1]) and torch.equal(p[0], a[0]) and torch.equal(p[2], a[2])
    for bad in (0, 4, 5):
        with pytest.raises(ValueError):
            model_io_pi3x.TokenPatch(m, bad, [1], b)


# ---------------------------------------------------------------------------------------------------------------- M8
def _clip(n=4, h=6, w=8):
    rng = np.random.default_rng(0)
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[1:, :3, 3] = rng.normal(size=(n - 1, 3))
    return {"clip_id": "dev__x__c0", "poses": poses, "K": np.tile(np.eye(3), (n, 1, 1)),
            "depth": rng.uniform(1, 3, size=(n, h, w)).astype(np.float32),
            "images": np.zeros((n, h, w, 3), np.uint8)}


def test_m8_pose_and_depth_scale_in_one_condition():
    clip = _clip()
    cond = {"name": "m8_p1.5_d0.67", "family": "m8", "inputs": ["intrinsics", "pose", "depth"],
            "corruption": {"target": "pose", "type": "scale", "factor": 1.5,
                           "also": {"target": "depth", "type": "scale", "factor": 0.67}}}
    poses, K, depth, entry = run_exp1.apply_corruption(clip, cond, 0, (0, 2))
    assert np.allclose(poses[1:, :3, 3], clip["poses"][1:, :3, 3] * 1.5)
    assert np.allclose(depth[[0, 2]], clip["depth"][[0, 2]] * np.float32(0.67))
    assert np.array_equal(depth[[1, 3]], clip["depth"][[1, 3]])
    assert entry["target"] == "pose" and entry["also"]["target"] == "depth"
    assert entry["affected_views"] == [0, 1, 2, 3]
    assert "also" in cond["corruption"]                  # the config itself is not modified
    p1, _, d1, e1 = run_exp1.apply_corruption(clip, {**cond, "corruption": {k: v for k, v in cond["corruption"].items()
                                                                           if k != "also"}}, 0, (0, 2))
    assert np.allclose(p1, poses) and np.array_equal(d1, clip["depth"]) and "also" not in e1
