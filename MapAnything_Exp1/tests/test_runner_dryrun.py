"""Dry run of the whole runner on CPU with a FAKE model (no MapAnything, no GPU).

Checks the plumbing only: run_exp1 (conditions, corruption, controls, contamination
subsets, manifest, resume), metrics, aggregate and the E0 smoke checker. The fake
"model" returns GT geometry and trusts every supplied pose completely; views without
a pose get the GT pose turned by 1 deg. Its numbers mean nothing about MapAnything.
"""

import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from exp1 import aggregate, model_io, run_exp1  # noqa: E402
from exp1.datasets import load_clip, prepare_eth3d  # noqa: E402
from exp1.geometry import so3_exp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
import test_e0_clip_pipeline as synth  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "e0_check_smoke", os.path.join(ROOT, "scripts", "e0_check_smoke.py"))
e0_check_smoke = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e0_check_smoke)

ONLY = ["rgb_only", "rgb_K", "clean_pose", "clean_pose_noK", "zero_noise", "outlier_rot_rot_deg=15"]


def fake_run_factory(clip):
    N, H, W = clip["depth"].shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    def fake_run(model, views, device):
        assert len(views) == N
        out = {"pts3d": [], "depth_z": [], "pose": [], "conf": [], "mask": []}
        for i, view in enumerate(views):
            assert tuple(view["img"].shape) == (H, W, 3)
            if "camera_poses" in view:
                T = view["camera_poses"].double().numpy()
            else:
                T = clip["poses"][i].copy()
                T[:3, :3] = T[:3, :3] @ so3_exp(np.radians(1.0) * np.array([0.0, 1.0, 0.0]))
            K = clip["K"][i]
            z = np.where(clip["depth"][i] > 0, clip["depth"][i], np.median(clip["depth"][i]))
            pc = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], -1)
            out["pts3d"].append(pc @ T[:3, :3].T + T[:3, 3])
            out["depth_z"].append(z)
            out["pose"].append(T)
            out["conf"].append(np.ones((H, W)))
            out["mask"].append(np.ones((H, W), bool))
        out = {k: np.stack(x) for k, x in out.items()}
        out["runtime_s"], out["peak_mem_gb"] = 0.01, 0.0
        return out

    return fake_run


@pytest.fixture(scope="module")
def clip_path(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("eth3d"))
    synth.write_scene(root, n=6)
    clips = os.path.join(root, "clips")
    prepare_eth3d(root, clips, n_views=6, clips_per_scene=1, stride=1, tw=170, th=100,
                  scenes=["synth"])
    return os.path.join(clips, "dev__synth__c0.npz")


@pytest.fixture(scope="module")
def run_dir(clip_path):
    out = os.path.join(os.path.dirname(os.path.dirname(clip_path)), "smoke")
    mp = pytest.MonkeyPatch()
    mp.setattr(model_io, "load_model", lambda ckpt, device: "fake")
    mp.setattr(model_io, "run", fake_run_factory(load_clip(clip_path)))
    argv = ["run_exp1", "--config", os.path.join(ROOT, "configs", "pilot.json"),
            "--clips", clip_path, "--out", out, "--device", "cpu", "--only", *ONLY]
    for _ in range(2):  # second pass exercises resume
        mp.setattr(sys, "argv", argv)
        run_exp1.main()
    mp.undo()
    return out


def test_all_rows_ok_no_failures_no_duplicates(run_dir):
    df = pd.read_csv(os.path.join(run_dir, "per_run.csv"))
    assert (df.status == "ok").all()
    assert not os.path.exists(os.path.join(run_dir, "failures.jsonl"))
    assert df.run_key.is_unique
    assert set(df.condition) == set(ONLY)


def test_zero_noise_identical_to_clean(run_dir):
    man = json.load(open(os.path.join(run_dir, "corruption_manifest.json")))
    zn = [e for e in man if e["condition"] == "zero_noise"]
    assert zn and all(e["max_abs_diff_vs_clean_pts3d"] == 0.0 for e in zn if "max_abs_diff_vs_clean_pts3d" in e)


def test_controls_drop_the_right_views(run_dir):
    df = pd.read_csv(os.path.join(run_dir, "per_run.csv"))
    c = df[(df.view_subset == "all") & df.control.isin(["drop_oracle", "filter_rgbcheck"])]
    assert set(c.control) == {"drop_oracle", "filter_rgbcheck"}
    assert len(c) == 6  # 2 controls x 3 seeds
    assert (c.n_pose_dropped >= 1).all() and (c.dropped_are_corrupted == 1).all()


def test_contamination_rows_present(run_dir):
    df = pd.read_csv(os.path.join(run_dir, "per_run.csv"))
    sub = df[df.condition.str.startswith("outlier") & df.view_subset.isin(["corrupted", "uncorrupted"])]
    assert {"none", "ref:rgb_only", "ref:clean_pose"} <= set(sub.control)


def test_smoke_checker_and_aggregate_run(run_dir, capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x", "--runs", run_dir])
    e0_check_smoke.main()
    text = capsys.readouterr().out
    assert "[FAIL]" not in text, text
    monkeypatch.setattr(sys, "argv", ["x", "--run", run_dir, "--metric", "pts_relerr_scale"])
    aggregate.main()
    assert os.path.exists(os.path.join(run_dir, "aggregate_metrics.csv"))


def test_demo_recon_writes_ply_png_metrics(clip_path, tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "e0_demo_recon", os.path.join(ROOT, "scripts", "e0_demo_recon.py"))
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    monkeypatch.setattr(model_io, "load_model", lambda ckpt, device: "fake")
    monkeypatch.setattr(model_io, "run", fake_run_factory(load_clip(clip_path)))
    monkeypatch.setattr(sys, "argv", ["x", "--clip", clip_path, "--out", str(tmp_path),
                                      "--config", os.path.join(ROOT, "configs", "pilot.json"),
                                      "--device", "cpu"])
    demo.main()
    cid = "dev__synth__c0"
    for name in ("gt", "rgb_only", "clean_pose", "outlier_rot_rot_deg_15"):
        with open(tmp_path / f"{cid}__{name}.ply", "rb") as f:
            head = f.read(200).decode("latin-1")
        n = int(head.split("element vertex ")[1].split("\n")[0])
        size = os.path.getsize(tmp_path / f"{cid}__{name}.ply")
        assert n > 0 and size == head.index("end_header\n") + len("end_header\n") + 15 * n
    assert (tmp_path / f"{cid}__overview.png").stat().st_size > 10_000
    m = json.load(open(tmp_path / f"{cid}__metrics.json"))
    # the fake model trusts poses: clean hints reproduce GT, the 15 deg outlier does not
    assert m["clean_pose"]["pts_relerr_metric"] < 1e-6 < m["outlier_rot_rot_deg=15"]["pts_relerr_metric"]


def test_demo_recon_refuses_eval_clips(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "e0_demo_recon", os.path.join(ROOT, "scripts", "e0_demo_recon.py"))
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    monkeypatch.setattr(sys, "argv", ["x", "--clip", "data/clips/eval__x__c0.npz", "--out", "unused"])
    with pytest.raises(SystemExit):
        demo.main()
