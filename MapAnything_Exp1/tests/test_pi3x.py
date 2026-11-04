"""Pi3X backend (exp1/model_io_pi3x.py): input building, and the E2-D grid through the runner with a FAKE Pi3X.

The fake consumes the real build_views output (explicit per-view masks, view 0 posed, sizes multiple of 14). It
returns GT geometry, copies pose hints, and turns unhinted cameras by 1 deg. Plumbing only; the real model runs on
the VM. If third_party/Pi3 is present, the keyword names are checked against Pi3X.forward without loading weights.
"""

import ast
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from exp1 import model_io_pi3x, run_exp1  # noqa: E402
from exp1.datasets import load_clip, prepare_eth3d  # noqa: E402
from exp1.geometry import so3_exp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
import test_e0_clip_pipeline as synth  # noqa: E402


def _inputs(n=4, h=28, w=42):
    rng = np.random.default_rng(0)
    images = rng.integers(0, 255, (n, h, w, 3), dtype=np.uint8)
    K = np.tile(np.array([[30.0, 0, w / 2], [0, 30.0, h / 2], [0, 0, 1]]), (n, 1, 1))
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, 0, 3] = np.arange(n)
    depth = rng.uniform(1, 3, (n, h, w)).astype(np.float32)
    return images, K, poses, depth


def test_build_views_masks_and_shapes():
    images, K, poses, depth = _inputs()
    v = model_io_pi3x.build_views(images, K, poses, ["pose", "intrinsics", "depth"],
                                  pose_mask=[True, False, True, True], depth=depth, depth_views=(1, 3))
    assert tuple(v["imgs"].shape) == (1, 4, 3, 28, 42) and float(v["imgs"].max()) <= 1.0
    assert v["mask_add_pose"].tolist() == [[True, False, True, True]]
    assert v["mask_add_ray"].tolist() == [[True] * 4]
    assert v["mask_add_depth"].tolist() == [[False, True, False, True]]
    assert torch.equal(v["poses"][0, 1], torch.eye(4))           # unposed view: identity placeholder, mask off
    assert np.allclose(v["poses"][0, 2].numpy(), poses[2])
    assert float(v["depths"][0, 0].abs().sum()) == 0 and np.allclose(v["depths"][0, 3].numpy(), depth[3])
    none = model_io_pi3x.build_views(images, K, poses, [])
    assert none["poses"] is None and none["intrinsics"] is None and none["depths"] is None
    assert not none["mask_add_pose"].any() and not none["mask_add_ray"].any() and not none["mask_add_depth"].any()


def test_build_views_rejects_what_pi3x_cannot_take():
    images, K, poses, depth = _inputs()
    with pytest.raises(ValueError, match="view 0"):
        model_io_pi3x.build_views(images, K, poses, ["pose"], pose_mask=[False, True, True, True])
    with pytest.raises(ValueError, match="multiple of 14"):
        model_io_pi3x.build_views(images[:, :27], K, poses, [])
    with pytest.raises(ValueError, match="unknown"):
        model_io_pi3x.build_views(images, K, poses, ["rays"])


def test_keywords_match_pi3x_forward():
    src = os.path.join(model_io_pi3x.REPO, "pi3", "models", "pi3x.py")
    if not os.path.exists(src):
        pytest.skip("third_party/Pi3 not present")
    tree = ast.parse(open(src, encoding="utf-8").read())
    fwd = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "forward"
               and any(a.arg == "mask_add_pose" for a in n.args.args))
    names = {a.arg for a in fwd.args.args}
    images, K, poses, depth = _inputs()
    v = model_io_pi3x.build_views(images, K, poses, ["pose", "intrinsics", "depth"], depth=depth, depth_views=(0,))
    assert set(v) | {"with_prior"} <= names


def fake_pi3x(clip):
    N, H, W = clip["depth"].shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    def run(model, views, device):
        assert tuple(views["imgs"].shape) == (1, N, 3, H, W)
        for k in ("mask_add_pose", "mask_add_ray", "mask_add_depth"):
            assert views[k] is not None and tuple(views[k].shape) == (1, N)   # never random (pi3x.py:322-326)
        posed = views["mask_add_pose"][0].numpy()
        assert not posed.any() or posed[0]
        out = {"pts3d": [], "depth_z": [], "pose": [], "conf": [], "mask": [], "intrinsics": []}
        for i in range(N):
            if posed[i] and posed.sum() > 1:
                T = views["poses"][0, i].double().numpy()
            else:
                T = clip["poses"][i].copy()
                T[:3, :3] = T[:3, :3] @ so3_exp(np.radians(1.0) * np.array([0.0, 1.0, 0.0]))
            K = clip["K"][i]
            z = np.where(clip["depth"][i] > 0, clip["depth"][i], np.median(clip["depth"][i]))
            pc = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], -1)
            out["pts3d"].append(pc @ T[:3, :3].T + T[:3, 3])
            out["depth_z"].append(z)
            out["pose"].append(T)
            out["conf"].append(np.full((H, W), 0.9 if posed[i] else 0.5))
            out["mask"].append(np.ones((H, W), bool))
            out["intrinsics"].append(K)
        out = {k: np.stack(x) for k, x in out.items()}
        out["runtime_s"], out["peak_mem_gb"], out["metric_scaling_factor"] = 0.01, 0.0, 1.0
        return out

    return run


@pytest.fixture(scope="module")
def pi3x_run(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("pi3x"))
    synth.write_scene(root, n=10)
    clips = os.path.join(root, "clips")
    prepare_eth3d(root, clips, n_views=10, clips_per_scene=1, stride=1, tw=168, th=98, scenes=["synth"])
    clip_path = os.path.join(clips, "dev__synth__c0.npz")
    out = os.path.join(root, "pi3x")
    mp = pytest.MonkeyPatch()
    mp.setattr(run_exp1, "model_io", run_exp1.model_io)  # main() rebinds it; undo() restores it
    mp.setattr(model_io_pi3x, "load_model", lambda ckpt, device: "fake")
    mp.setattr(model_io_pi3x, "run", fake_pi3x(load_clip(clip_path)))
    mp.setattr(run_exp1, "git_commit", lambda path: "test")
    mp.setattr(sys, "argv", ["run_exp1", "--config", os.path.join(ROOT, "configs", "e2d_full.json"),
                             "--clips", clip_path, "--out", out, "--device", "cpu", "--save_extras",
                             "--backend", "pi3x"])
    run_exp1.main()
    mp.undo()
    return out


def test_e2d_grid_runs_completely_on_pi3x_views(pi3x_run):
    fp = os.path.join(pi3x_run, "failures.jsonl")
    assert not os.path.exists(fp), open(fp).read()
    pr = pd.read_csv(os.path.join(pi3x_run, "per_run.csv"))
    runs = pr[pr.view_subset == "all"]
    assert (runs.status == "ok").all() and len(runs) == 138
    assert json.load(open(os.path.join(pi3x_run, "run_info.json")))["backend"] == "pi3x"
    pv = pd.read_csv(os.path.join(pi3x_run, "per_view.csv"))
    o = pv[(pv.condition == "c1_rot_rot_deg=15") & (pv.was_corrupted == 1) & (pv.control == "none")]
    assert len(o) == 2 and np.allclose(o.absorption_rot, 1.0, atol=1e-3)  # the fake copies pose hints
