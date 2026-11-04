"""Gate track (experiments/G.md): depth hint builders, gate arithmetic, the runner and analyses with fake models, and the
OmniVGGT hooks on the real aggregator class with tiny random weights. No weights downloaded, no GPU."""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

from exp1 import depth_hints as dh

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")


def synth_clip(N=10, H=98, W=168, rooms=True, seed=0):
    rng = np.random.default_rng(seed)
    gt = rng.uniform(1, 5, (N, H, W)).astype(np.float32)
    gt[:, :5, :5] = 0
    poses = np.stack([np.eye(4) for _ in range(N)])
    poses[:, 0, 3] = np.arange(N) * 0.2
    K = np.tile(np.array([[100., 0, W / 2], [0, 100., H / 2], [0, 0, 1]]), (N, 1, 1)).astype(np.float32)
    c = {"clip_id": "dev__synth__c0", "scene": "synth", "depth": gt, "poses": poses, "K": K,
         "images": rng.integers(0, 255, (N, H, W, 3)).astype(np.uint8)}
    if rooms:
        c.update(dev_depth=gt * 0.99, dev_K=K, dev_poses=poses,
                 mono_depth=gt * 1.4 * rng.uniform(0.9, 1.1, (N, 1, 1)).astype(np.float32) + 0.1)
    return c


def test_isolated_errors():
    c = synth_clip()
    ai = c["mono_depth"]
    fixed, *_ = dh.build_hint(c, {"type": "ai_fix_view"}, 0, "s")
    assert np.allclose(dh.view_ratios(fixed, c["depth"]), 1, atol=1e-3)
    g, *_ = dh.build_hint(c, {"type": "ai_fix_global"}, 0, "s")
    assert abs(dh.global_ratio(g, c["depth"]) - 1) < 1e-3
    only, *_ = dh.build_hint(c, {"type": "gt_ai_view"}, 0, "s")
    assert np.allclose(dh.view_ratios(only, c["depth"]), dh.view_ratios(ai, c["depth"]), atol=1e-3)
    assert np.array_equal(only == 0, c["depth"] == 0), "scaling truth must keep its holes"
    assert np.array_equal(c["mono_depth"], ai), "builders must not modify the clip"


def test_entry_marks_oracle():
    c = synth_clip()
    assert dh.build_hint(c, {"type": "ai_fix_view"}, 0, "s")[3]["oracle"] is True
    assert dh.build_hint(c, {"type": "ai"}, 0, "s")[3]["oracle"] is False


def test_bad_views_and_rectangles():
    c = synth_clip()
    d, bad, mask, e = dh.build_hint(c, {"type": "view_bad", "source": "lidar", "frac": 0.3, "factor": 1.5}, 0, "s")
    assert len(bad) == 3 and e["affected_views"] == bad
    assert np.allclose(d[bad], c["dev_depth"][bad] * 1.5)
    good = [i for i in range(10) if i not in bad]
    assert np.array_equal(d[good], c["dev_depth"][good])
    d, bad, mask, e = dh.build_hint(c, {"type": "patch_bad", "source": "lidar", "area": 0.3, "factor": 1.5}, 0, "s")
    assert bad == list(range(10)) and 0.25 < mask.mean() < 0.35
    assert np.allclose(d[mask], c["dev_depth"][mask] * 1.5) and np.array_equal(d[~mask], c["dev_depth"][~mask])
    d, bad, mask, _ = dh.build_hint(c, {"type": "mix_ai", "frac": 0.3}, 0, "s")
    assert np.array_equal(d[bad], c["mono_depth"][bad])


def test_same_scenario_same_draw():
    c = synth_clip()
    s = {"type": "view_bad", "source": "lidar", "frac": 0.3, "factor": 1.5}
    assert dh.build_hint(c, s, 0, "scen")[1] == dh.build_hint(c, s, 0, "scen")[1]
    draws = {tuple(dh.build_hint(c, s, k, "scen")[1]) for k in range(8)}
    assert len(draws) > 1


def test_patch_gamma():
    m = np.zeros((2, 28, 28), bool)
    m[0, :14, :14] = True
    m[1, 0, :7] = True
    g = dh.patch_gamma(m)
    assert g.shape == (2, 2, 2) and g[0, 0, 0] == 0 and g[0, 1, 1] == 1
    assert g[1, 0, 0] == pytest.approx(1 - 7 / 196)
    with pytest.raises(ValueError):
        dh.patch_gamma(np.zeros((1, 15, 14), bool))


def test_decompose_separates_scale_from_distortion():
    gt = np.full((4, 10, 10), 2.0, np.float32)
    scale_only = gt * np.array([1.2, 1.4, 1.6, 1.8], np.float32)[:, None, None]
    r = dh.decompose(scale_only, gt)
    assert r["absrel_after_per_view"] < 1e-5 and r["absrel_after_overall"] > 0.05
    distorted = gt * (1 + 0.3 * np.sign(np.arange(10) - 4.5))[None, None, :].astype(np.float32)
    assert dh.decompose(distorted, gt)["absrel_after_per_view"] > 0.1


torch = pytest.importorskip("torch")
from exp1 import g_analysis, model_io, model_io_omnivggt, run_gate  # noqa: E402


class FakeMA:
    """Same call shape as MapAnything fuse methods: the 4th positional argument is F_in."""

    def __init__(self):
        self.calls = 0

    def _encode_and_fuse_depths(self, views, n, b, feats, mask):
        self.calls += 1
        return feats + 2.0

    def _encode_and_fuse_cam_quats_and_trans(self, views, n, b, feats, q, t, mask):
        return feats + 3.0


def test_mapanything_gate_arithmetic():
    m, f = FakeMA(), torch.zeros(4, 8, 2, 3)
    model_io.set_hint_gate(m, None, np.array([1, 0, 0.5, 1], np.float32), None)
    out = m._encode_and_fuse_depths(None, 4, 1, f, None)
    assert out[0].eq(2).all() and out[1].eq(0).all() and out[2].eq(1).all()
    assert m._encode_and_fuse_cam_quats_and_trans(None, 4, 1, f, 0, 0, None).eq(3).all(), "pose must stay ungated"
    model_io.clear_hint_gate(m)
    assert m._encode_and_fuse_depths(None, 4, 1, f, None).eq(2).all()

    g = np.ones((4, 2, 3), np.float32)
    g[0, 1, 2] = 0
    model_io.set_hint_gate(m, None, g, np.zeros(4, np.float32))
    out = m._encode_and_fuse_depths(None, 4, 1, f, None)
    assert out[0, :, 1, 2].eq(0).all() and out[0, :, 0, 0].eq(2).all()
    assert m._encode_and_fuse_cam_quats_and_trans(None, 4, 1, f, 0, 0, None).eq(0).all()
    model_io.clear_hint_gate(m)
    assert "_encode_and_fuse_depths" not in m.__dict__

    model_io.set_hint_gate(m, None, np.ones((4, 5, 5), np.float32))
    with pytest.raises(ValueError, match="grid"):
        m._encode_and_fuse_depths(None, 4, 1, f, None)


def test_mapanything_token_capture_shapes():
    class B(torch.nn.Module):
        def forward(self, x):
            return x

    class Info(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attention_blocks = torch.nn.ModuleList([B(), B(), B()])

    class M:
        info_sharing = Info()
    cap = model_io.TokenCapture(M(), [2], 3)
    x = torch.arange(3 * 4 + 1, dtype=torch.float32)[None, :, None].repeat(1, 1, 5)
    M.info_sharing.self_attention_blocks[2](x)
    t = cap.tokens[2]
    assert t.shape == (3, 4, 5) and t[1, 0, 0] == 4 and t[2, 3, 0] == 11, "views in order, scale token left out"
    cap.remove()
    with pytest.raises(ValueError, match="frame"):
        model_io.TokenCapture(M(), [1], 3)


def tiny_omnivggt():
    os.environ.setdefault("OMNIVGGT_REPO", os.path.join(ROOT, "third_party", "OmniVGGT-official"))
    if not os.path.isdir(model_io_omnivggt.REPO):
        pytest.skip("third_party/OmniVGGT-official not present")
    if model_io_omnivggt.REPO not in sys.path:
        sys.path.insert(0, model_io_omnivggt.REPO)
    try:
        from omnivggt.models.omnivggt_aggregator import ZeroAggregator
    except Exception as e:
        pytest.skip(f"cannot import OmniVGGT: {e!r}")
    torch.manual_seed(0)
    ag = ZeroAggregator(img_size=56, patch_size=14, embed_dim=32, depth=2, num_heads=2, patch_embed="conv")
    for a in ag.camera_adapters:    # the zero init would hide the pose gate: weights differ from zero after training
        torch.nn.init.normal_(a.weight, std=0.1)
    torch.nn.init.normal_(ag.depth_placeholder, std=0.1)
    return ag.eval()


def _ov_inputs(S=3, H=56, W=56):
    torch.manual_seed(1)
    ext = torch.eye(4)[:3].repeat(1, S, 1, 1)
    ext[0, :, 0, 3] = torch.arange(S).float() * 0.3
    intr = torch.tensor([[50., 0, W / 2], [0, 50., H / 2], [0, 0, 1]]).repeat(1, S, 1, 1)
    return dict(images=torch.rand(1, S, 3, H, W), extrinsics=ext, intrinsics=intr,
                depth=torch.rand(1, S, H, W, 1) + 1, mask=torch.ones(1, S, H, W))


def _ov_tokens(ag, cam, dep):
    with torch.no_grad():
        out, _ = ag.inference(**_ov_inputs(), depth_gt_index=dep, camera_gt_index=cam)
    return out[-1]


def test_omnivggt_real_aggregator_gates():
    ag = tiny_omnivggt()

    class Model:
        aggregator = ag
    m = Model()
    views = {"depth_gt_index": [0, 1, 2], "camera_gt_index": [0, 1, 2]}
    base = _ov_tokens(ag, [0, 1, 2], [0, 1, 2])
    assert not torch.allclose(base, _ov_tokens(ag, [0, 1, 2], [])), "the depth hint must matter"

    model_io_omnivggt.set_hint_gate(m, views, np.ones(3, np.float32), np.ones(3, np.float32))
    assert torch.allclose(_ov_tokens(ag, [0, 1, 2], [0, 1, 2]), base), "gamma = 1 must change nothing"
    model_io_omnivggt.clear_hint_gate(m)

    model_io_omnivggt.set_hint_gate(m, views, np.zeros(3, np.float32), None)
    assert torch.allclose(_ov_tokens(ag, [0, 1, 2], [0, 1, 2]), _ov_tokens_nogate(ag, [0, 1, 2], []), atol=1e-5), \
        "depth gamma = 0 must equal the placeholder of a missing depth"
    model_io_omnivggt.clear_hint_gate(m)

    model_io_omnivggt.set_hint_gate(m, views, None, np.zeros(3, np.float32))
    assert torch.allclose(_ov_tokens(ag, [0, 1, 2], [0, 1, 2]), _ov_tokens_nogate(ag, [], [0, 1, 2]), atol=1e-5), \
        "pose gamma = 0 must equal a missing camera"
    model_io_omnivggt.clear_hint_gate(m)
    assert torch.equal(_ov_tokens(ag, [0, 1, 2], [0, 1, 2]), base), "clearing must remove every hook"

    g = np.ones((3, 4, 4), np.float32)
    g[1, 2, 2] = 0
    model_io_omnivggt.set_hint_gate(m, views, g, None)
    patched = _ov_tokens(ag, [0, 1, 2], [0, 1, 2])
    assert not torch.allclose(patched, base), "a per-patch gate must act"
    model_io_omnivggt.clear_hint_gate(m)


def _ov_tokens_nogate(ag, cam, dep):
    """Runs without touching whichever gate is installed: the hooks only see views listed in the index lists."""
    saved = [dict(ag.depth_patch_embed._forward_hooks)] + [dict(p._forward_hooks) for p in ag.pose_embeddings]
    ag.depth_patch_embed._forward_hooks.clear()
    for p in ag.pose_embeddings:
        p._forward_hooks.clear()
    try:
        return _ov_tokens(ag, cam, dep)
    finally:
        ag.depth_patch_embed._forward_hooks.update(saved[0])
        for p, s in zip(ag.pose_embeddings, saved[1:]):
            p._forward_hooks.update(s)


def test_omnivggt_real_token_capture():
    ag = tiny_omnivggt()

    class Model:
        aggregator = ag
    cap = model_io_omnivggt.TokenCapture(Model(), [1], 3)
    _ov_tokens(ag, [0, 1, 2], [0, 1, 2])
    assert cap.tokens[1].shape == (3, 16, 32), "16 patches per view, camera and register tokens left out"
    cap.remove()


def fake_io(clip_by_id):
    """A backend whose output depends on the hint, so analyses see real differences. Bad hints raise the error."""
    class IO:
        __name__ = "exp1.model_io"
        last = {}

        @staticmethod
        def load_model(ck, dev):
            return object()

        @staticmethod
        def build_views(images, K, poses, inputs, pose_mask, metric, depth, depth_views):
            return dict(images=images, K=K, poses=poses, inputs=list(inputs), depth=depth, depth_views=list(depth_views))

        @staticmethod
        def set_hint_gate(model, views, gd, gp):
            IO.last = dict(gd=gd, gp=gp)

        @staticmethod
        def clear_hint_gate(model):
            IO.last = {}

        class TokenCapture:
            cur = None

            def __init__(self, model, layers, n):
                self.tokens, self.layers, self.n = {}, layers, n
                IO.TokenCapture.cur = self

            def remove(self):
                pass

        @staticmethod
        def run(model, views, device):
            imgs = views["images"]
            N, H, W = imgs.shape[:3]
            clip = next(c for c in clip_by_id.values() if c["images"] is imgs or np.array_equal(c["images"], imgs))
            gt, d = clip["depth"], np.array(clip["depth"], np.float32) * 1.05
            hint = views["depth"]
            gd = IO.last.get("gd")
            if "depth" in views["inputs"] and hint is not None:
                for i in views["depth_views"]:
                    w = 1.0 if gd is None or gd.ndim != 1 else float(gd[i])
                    r = np.where(gt[i] > 0, hint[i] / np.maximum(gt[i], 1e-3), 1)
                    d[i] = gt[i] * (1 + w * 0.5 * np.abs(np.median(r[hint[i] > 0]) - 1 if (hint[i] > 0).any() else 0))
            K = clip["K"]
            yy, xx = np.mgrid[:H, :W]
            pts = np.zeros((N, H, W, 3), np.float32)
            for i in range(N):
                z = d[i]
                cam = np.stack([(xx - K[i, 0, 2]) / K[i, 0, 0] * z, (yy - K[i, 1, 2]) / K[i, 1, 1] * z, z], -1)
                pts[i] = cam @ clip["poses"][i][:3, :3].T + clip["poses"][i][:3, 3]
            cap = IO.TokenCapture.cur
            if cap is not None:
                seed = abs(hash((tuple(views["inputs"]), float(np.nansum(hint)) if hint is not None else 0))) % 2**31
                r = torch.Generator().manual_seed(seed)
                cap.tokens = {k: torch.randn(N, 6, 4, generator=r) for k in cap.layers}
                IO.TokenCapture.cur = None
            return {"pts3d": pts, "depth_z": d.astype(np.float32), "pose": np.array(clip["poses"]), "conf": np.ones((N, H, W), np.float32),
                    "mask": gt > 0, "intrinsics": K, "metric_scaling_factor": 1.0, "runtime_s": 0.0, "peak_mem_gb": 0.0}
    return IO


@pytest.fixture
def clip_files(tmp_path):
    out = {}
    for name, rooms in (("dev__room__c0", True), ("dev__eth__c0", False)):
        c = synth_clip(rooms=rooms, seed=1 if rooms else 2, H=98, W=168)
        c["clip_id"] = name
        c["scene"] = name.split("__")[1]
        np.savez(tmp_path / f"{name}.npz", **{k: v for k, v in c.items() if k != "clip_id"})
        out[name] = c
    return str(tmp_path / "dev__*.npz"), out


def _run(mp, config, clips, by_id, out, backend="mapanything"):
    io = fake_io(by_id)
    mp.setattr(run_gate, "get_backend", lambda name: io)
    mp.setattr(sys, "argv", ["run_gate", "--config", os.path.join(ROOT, "configs", config), "--clips", clips, "--out", out,
                             "--device", "cpu", "--backend", backend])
    run_gate.main()
    fp = os.path.join(out, "failures.jsonl")
    assert not os.path.exists(fp), open(fp).read()[:1500]
    return pd.read_csv(os.path.join(out, "per_run.csv"))


def test_eval_clips_refused(tmp_path):
    with pytest.raises(SystemExit, match="frozen tag"):
        run_gate.check_split(["data/eval__x__c0.npz"], False)
    run_gate.check_split(["data/eval__x__c0.npz"], True)
    run_gate.check_split(["data/dev__x__c0.npz"], False)


def test_inputs_adapted_for_omnivggt():
    assert run_gate.adapt_inputs("omnivggt", ["intrinsics"]) == []
    assert run_gate.adapt_inputs("omnivggt", ["intrinsics", "depth"]) == ["depth"]
    assert run_gate.adapt_inputs("omnivggt", ["intrinsics", "pose"]) == ["intrinsics", "pose"]
    assert run_gate.adapt_inputs("mapanything", ["intrinsics"]) == ["intrinsics"]


@pytest.mark.parametrize("backend", ["mapanything", "omnivggt"])
def test_g1_runs_and_resumes(monkeypatch, tmp_path, clip_files, backend):
    clips, by_id = clip_files
    out = str(tmp_path / f"g1_{backend}")
    df = _run(monkeypatch, "g1_oracle.json", clips, by_id, out, backend)
    assert (df.status == "ok").all() and df.run_key.is_unique
    assert set(df.dataset) == {"rooms", "eth"}
    man = json.load(open(os.path.join(out, "corruption_manifest.json")))
    conds = set(df.condition)
    cfg = json.load(open(os.path.join(ROOT, "configs", "g1_oracle.json")))
    hinted = {c["name"] for c in cfg["conditions"] if "depth" in c or isinstance(c.get("pose"), dict)}
    assert {m["condition"] for m in man} == hinted & conds, "every hint used needs a manifest entry"
    assert os.path.exists(os.path.join(out, "checks.csv"))
    n = len(df)
    assert len(_run(monkeypatch, "g1_oracle.json", clips, by_id, out, backend)) == n, "a rerun must add nothing"


def test_g1_analysis_end_to_end(monkeypatch, tmp_path, clip_files, capsys):
    clips, by_id = clip_files
    d = str(tmp_path / "g1_ma")
    _run(monkeypatch, "g1_oracle.json", clips, by_id, d)
    tab = g_analysis.oracle_summary(g_analysis.load_runs([d]))
    assert {"depth_lidar_view_bad", "depth_lidar_patch_bad", "pose_outlier30"} <= set(tab.scenario)
    assert (tab[tab.role == "always"].gap.dropna().round(6) == 1).all() and (tab[tab.role == "clean"].gap.dropna().round(6) == 0).all()
    ver = g_analysis.oracle_verdicts(tab)
    gate = ver[ver.scenario == "depth_lidar_view_bad"].iloc[0]
    assert gate.best_gate == "oracle_gate_view" and gate.gate_gap < 1, "the fake model obeys the gate, so damage must fall"
    parts = g_analysis.decomposition_table(g_analysis.load_runs([d]))
    assert {"ai", "ai_fix_view", "gt_ai_view", "gt_ai_global"} <= set(parts.condition)
    monkeypatch.setattr(sys, "argv", ["g", "oracle", "--runs", d, "--out", str(tmp_path / "sum")])
    g_analysis.main()
    assert "PILOT" in capsys.readouterr().out
    assert os.path.exists(tmp_path / "sum" / "verdicts_proposed.csv")


def test_g2_runs_and_analysis(monkeypatch, tmp_path, clip_files):
    clips, by_id = clip_files
    d = str(tmp_path / "g2_ma")
    df = _run(monkeypatch, "g2_signal.json", clips, by_id, d)
    assert (df.status == "ok").all()
    rows, feats = g_analysis.load_tokens([d])
    assert len(rows) == len(feats) and {0, 1} <= set(rows.label)
    v = rows[(rows.task == "pose_outlier") & (rows.label == 1)]
    assert len(v) and (v.dis_rot > 1).all(), "a turned camera must disagree with the photos-only poses in the record"
    assert (rows[rows.kind == "view"]["clip"] != "").all()
    tab = g_analysis.signal_summary(rows, feats)
    assert set(tab.backend) == {"mapanything"} and len(tab)


def test_auroc():
    assert g_analysis.auroc([2, 3], [0, 1]) == 1.0 and g_analysis.auroc([0, 1], [2, 3]) == 0.0
    assert g_analysis.auroc([1], [1]) == 0.5 and np.isnan(g_analysis.auroc([], [1]))


def test_ridge_scores_rank_signal():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(20), np.zeros(20)]
    X = rng.normal(size=(40, 200))
    X[:, :20] += 2 * y[:, None]
    yte = np.r_[np.ones(10), np.zeros(10)]
    Xte = rng.normal(size=(20, 200))
    Xte[:, :20] += 2 * yte[:, None]
    s = g_analysis.ridge_scores(X, y, Xte, lam=10.0)
    assert g_analysis.auroc(s[yte == 1], s[yte == 0]) > 0.9
