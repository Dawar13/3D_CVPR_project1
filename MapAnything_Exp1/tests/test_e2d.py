"""E2-D grid (configs/e2d_full.json) through the runner with --backend omnivggt and a FAKE OmniVGGT.

The fake consumes the real model_io_omnivggt.build_views output, so every condition must build a
hint set OmniVGGT accepts (camera = pose AND intrinsics, view 0 hinted, sizes multiple of 14). It
returns GT geometry, copies camera hints, and turns unhinted cameras by 1 deg. Plumbing only.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from exp1 import model_io_omnivggt, run_exp1  # noqa: E402
from exp1.datasets import load_clip, prepare_eth3d  # noqa: E402
from exp1.geometry import so3_exp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
import test_e0_clip_pipeline as synth  # noqa: E402


def fake_omnivggt(clip):
    N, H, W = clip["depth"].shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    def run(model, views, device):
        assert tuple(views["images"].shape) == (N, 3, H, W)
        cam = views["camera_gt_index"]
        assert not cam or cam[0] == 0
        out = {"pts3d": [], "depth_z": [], "pose": [], "conf": [], "mask": [], "intrinsics": []}
        for i in range(N):
            if i in cam:
                T = np.linalg.inv(np.vstack([views["extrinsics"][0, i].double().numpy(), [0, 0, 0, 1]]))
                K = views["intrinsics"][0, i].double().numpy()
            else:
                T = clip["poses"][i].copy()
                T[:3, :3] = T[:3, :3] @ so3_exp(np.radians(1.0) * np.array([0.0, 1.0, 0.0]))
                K = clip["K"][i]
            z = np.where(clip["depth"][i] > 0, clip["depth"][i], np.median(clip["depth"][i]))
            pc = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], -1)
            out["pts3d"].append(pc @ T[:3, :3].T + T[:3, 3])
            out["depth_z"].append(z)
            out["pose"].append(T)
            out["conf"].append(np.full((H, W), 2.0 if i in cam else 1.0))
            out["mask"].append(np.ones((H, W), bool))
            out["intrinsics"].append(K)
        out = {k: np.stack(x) for k, x in out.items()}
        out["runtime_s"], out["peak_mem_gb"], out["metric_scaling_factor"] = 0.01, 0.0, float("nan")
        return out

    return run


@pytest.fixture(scope="module")
def e2d_run(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("e2d"))
    synth.write_scene(root, n=10)
    clips = os.path.join(root, "clips")
    prepare_eth3d(root, clips, n_views=10, clips_per_scene=1, stride=1, tw=168, th=98, scenes=["synth"])
    clip_path = os.path.join(clips, "dev__synth__c0.npz")
    out = os.path.join(root, "e2d")
    mp = pytest.MonkeyPatch()
    mp.setattr(run_exp1, "model_io", run_exp1.model_io)  # main() rebinds it; undo() restores it
    mp.setattr(model_io_omnivggt, "load_model", lambda ckpt, device: "fake")
    mp.setattr(model_io_omnivggt, "run", fake_omnivggt(load_clip(clip_path)))
    mp.setattr(run_exp1, "git_commit", lambda path: "test")
    mp.setattr(sys, "argv", ["run_exp1", "--config", os.path.join(ROOT, "configs", "e2d_full.json"),
                             "--clips", clip_path, "--out", out, "--device", "cpu", "--save_extras",
                             "--backend", "omnivggt"])
    run_exp1.main()
    mp.undo()
    return out


def test_e2d_grid_runs_completely_on_omnivggt_views(e2d_run):
    fp = os.path.join(e2d_run, "failures.jsonl")
    assert not os.path.exists(fp), open(fp).read()
    pr = pd.read_csv(os.path.join(e2d_run, "per_run.csv"))
    runs = pr[pr.view_subset == "all"]
    assert (runs.status == "ok").all() and len(runs) == 138
    assert json.load(open(os.path.join(e2d_run, "run_info.json")))["backend"] == "omnivggt"
    pv = pd.read_csv(os.path.join(e2d_run, "per_view.csv"))
    o = pv[(pv.condition == "c1_rot_rot_deg=15") & (pv.was_corrupted == 1) & (pv.control == "none")]
    assert len(o) == 2 and np.allclose(o.absorption_rot, 1.0, atol=1e-3)  # the fake copies camera hints


def test_e2d_corruptions_equal_mapanything_grids():
    """Same condition names as E2-C / E1, so the same random draws: definitions must be verbatim."""
    load = lambda f: {c.get("name", c.get("family")): c for c in json.load(open(os.path.join(ROOT, "configs", f)))["conditions"]}
    d, c, e = load("e2d_full.json"), load("e2c_full.json"), load("e1_full.json")
    for k, v in d.items():
        src = c.get(k) if k.startswith("c") and k in c else e.get(k)
        if v.get("family") == "reference":
            continue
        assert src is not None, k
        for key in ("inputs", "sweep", "values", "levels", "corruption", "seeds", "pair_with"):
            assert v.get(key) == src.get(key), (k, key)
