"""E2-E (real device hints) and E2-F (remedies): runner plumbing with fake models, and the ARKit/SceneFun3D
clip geometry (exp1/arkit_clips.py) on synthetic data. No network, no model."""

import csv
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from exp1 import arkit_clips as ak  # noqa: E402
from exp1 import model_io, model_io_omnivggt, run_exp1  # noqa: E402
from exp1.datasets import load_clip, prepare_eth3d  # noqa: E402
from exp1.geometry import so3_exp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
import test_e0_clip_pipeline as synth  # noqa: E402
from test_e2d import fake_omnivggt  # noqa: E402
from test_runner_dryrun import fake_run_factory  # noqa: E402


def _run(mp, config, clip_path, out, backend="mapanything"):
    clip = load_clip(clip_path)
    mp.setattr(run_exp1, "model_io", run_exp1.model_io)
    if backend == "omnivggt":
        mp.setattr(model_io_omnivggt, "load_model", lambda ckpt, device: "fake")
        mp.setattr(model_io_omnivggt, "run", fake_omnivggt(clip))
    else:
        mp.setattr(model_io, "load_model", lambda ckpt, device: "fake")
        mp.setattr(model_io, "run", fake_run_factory(clip))
    mp.setattr(run_exp1, "git_commit", lambda path: "test")
    mp.setattr(sys, "argv", ["run_exp1", "--config", os.path.join(ROOT, "configs", config), "--clips", clip_path,
                             "--out", out, "--device", "cpu", "--save_extras", "--backend", backend])
    run_exp1.main()
    fp = os.path.join(out, "failures.jsonl")
    assert not os.path.exists(fp), open(fp).read()
    return pd.read_csv(os.path.join(out, "per_run.csv")), pd.read_csv(os.path.join(out, "per_view.csv"))


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    """A 10-view synthetic clip (sizes multiple of 14) and a copy with device hints added."""
    root = str(tmp_path_factory.mktemp("e2ef"))
    synth.write_scene(root, n=10)
    cdir = os.path.join(root, "clips")
    prepare_eth3d(root, cdir, n_views=10, clips_per_scene=1, stride=1, tw=168, th=98, scenes=["synth"])
    path = os.path.join(cdir, "dev__synth__c0.npz")
    c = dict(np.load(path))
    rng = np.random.default_rng(0)
    dev = c["poses"].copy()
    for i in range(1, 10):  # a small real-looking error on every view but view 0
        dev[i][:3, :3] = dev[i][:3, :3] @ so3_exp(np.radians(0.5) * rng.normal(size=3) / np.sqrt(3))
        dev[i][:3, 3] += 0.01 * rng.normal(size=3)
    c.update(dev_poses=dev, dev_K=c["K"] * np.array([[1.01, 1, 1], [1, 1.01, 1], [1, 1, 1]]),
             dev_depth=(c["depth"] * 0.98).astype(np.float32), mono_depth=(c["depth"] * 1.3).astype(np.float32))
    dpath = os.path.join(root, "e2e", "dev__synthdev__c0.npz")
    os.makedirs(os.path.dirname(dpath))
    np.savez_compressed(dpath, **c)
    return root, path, dpath


def test_e2df_grid_runs_with_all_remedies(clips):
    root, path, _ = clips
    mp = pytest.MonkeyPatch()
    pr, pv = _run(mp, "e2df_mapanything.json", path, os.path.join(root, "e2df"))
    mp.undo()
    runs = pr[(pr.view_subset == "all") & ~pr.control.str.startswith("ref:")]
    assert (runs.status == "ok").all() and len(runs) == 183
    assert set(runs.control) == {"none", "filter_rgbcheck", "drop_lowconf", "drop_oracle"}
    assert "drop_oracle" not in set(runs[runs.family == "anchor_error"].control)
    # the fake model is equally confident in every posed view, so R3 drops nothing
    assert (runs[runs.control == "drop_lowconf"].n_pose_dropped == 0).all()


def test_drop_lowconf_flags_the_low_confidence_view(clips, monkeypatch):
    """R3 drops exactly the view whose mean confidence is below tau x the median, never view 0."""
    root, path, _ = clips
    clip = load_clip(path)
    base = fake_run_factory(clip)

    def low_on_view_4(model, views, device):
        out = base(model, views, device)
        if all("camera_poses" in v for v in views):   # only the corrupted base run; the rerun lacks some poses
            out["conf"][4] *= 0.3
            out["conf"][0] *= 0.1                        # view 0 is never dropped
        return out

    monkeypatch.setattr(run_exp1, "model_io", run_exp1.model_io)
    monkeypatch.setattr(model_io, "load_model", lambda ckpt, device: "fake")
    monkeypatch.setattr(model_io, "run", low_on_view_4)
    out = os.path.join(root, "r3")
    monkeypatch.setattr(sys, "argv", ["run_exp1", "--config", os.path.join(ROOT, "configs", "e2df_mapanything.json"),
                                      "--clips", path, "--out", out, "--device", "cpu", "--only", "rgb_only",
                                      "outlier_rot_rot_deg=15"])
    run_exp1.main()
    pv = pd.read_csv(os.path.join(out, "per_view.csv"))
    r3 = pv[(pv.condition == "outlier_rot_rot_deg=15") & (pv.control == "drop_lowconf") & (pv.seed == 0)]
    assert set(r3[r3.has_pose_hint == 0].view_idx) == {4}


def test_e2e_device_hints_replace_ground_truth(clips):
    root, _, dpath = clips
    mp = pytest.MonkeyPatch()
    pr, pv = _run(mp, "e2e_mapanything.json", dpath, os.path.join(root, "e2e_ma"))
    pr2, pv2 = _run(mp, "e2e_omnivggt.json", dpath, os.path.join(root, "e2e_ov"), backend="omnivggt")
    mp.undo()
    for p, n in ((pr, 17), (pr2, 9)):
        runs = p[(p.view_subset == "all") & ~p.control.str.startswith("ref:")]
        assert (runs.status == "ok").all() and len(runs) == n
    clip = load_clip(dpath)
    man = {e["condition"]: e for e in json.load(open(os.path.join(root, "e2e_ma", "corruption_manifest.json")))}
    assert man["dev_all"]["affected_views"] == list(range(1, 10)) and man["dev_depth"]["affected_views"] == list(range(10))
    # the fake copies pose hints: dev_pose's predicted camera is the device pose, gt_pose's is the truth
    z = np.load(os.path.join(root, "e2e_ma", "extras", "dev__synthdev__c0__dev_pose__0__none__all.npz"))
    assert np.allclose(z["pred_pose_rel"], clip["dev_poses"], atol=1e-6)
    assert np.allclose(z["hint_pose_rel"], clip["dev_poses"], atol=1e-9)
    g = np.load(os.path.join(root, "e2e_ma", "extras", "dev__synthdev__c0__gt_pose__0__none__all.npz"))
    assert np.allclose(g["pred_pose_rel"], clip["poses"], atol=1e-6)
    d = pv[(pv.condition == "dev_depth")]
    assert (d.has_depth_hint == 1).all() and np.allclose(d.hint_depth_scale, 0.98, atol=1e-6)
    assert np.allclose(pv[pv.condition == "dev_K"].hint_focal_rel_err, 0.01, atol=1e-9)
    for cond in ("ai_depth", "dev_pose_ai_depth"):  # the AI depth estimate reaches the model as the depth hint
        a = pv[(pv.condition == cond) & (pv.control == "none")]
        assert (a.has_depth_hint == 1).all() and np.allclose(a.hint_depth_scale, 1.3, atol=1e-6)
    assert man["dev_pose_ai_depth"]["affected_views"] == list(range(1, 10))


# ---------------------------------------------------------------- ARKit / SceneFun3D geometry
def _traj_line(t, c2w):
    from scipy.spatial.transform import Rotation as Rot
    E = np.linalg.inv(c2w)
    return f"{t:.3f} " + " ".join(f"{x:.12f}" for x in (*Rot.from_matrix(E[:3, :3]).as_rotvec(), *E[:3, 3]))


def test_parse_traj_inverts_world_to_camera_and_sorts():
    A = np.eye(4); A[:3, :3] = so3_exp(np.array([0.1, -0.2, 0.3])); A[:3, 3] = [1, 2, 3]
    B = np.eye(4); B[:3, 3] = [0, 0, 1]
    ts, P = ak.parse_traj("\n".join([_traj_line(2.0, B), _traj_line(1.0, A), "bad line"]))
    assert list(ts) == [1.0, 2.0] and np.allclose(P[0], A, atol=1e-9) and np.allclose(P[1], B, atol=1e-9)


def test_device_K_turn_matches_the_measured_scenefun3d_case():
    """Video 42445441: ARKit lowres (cx, cy) = (129.204, 95.6055) maps to (950.97, 722.96) in the upright
    hires frame; SceneFun3D's hires intrinsics there are (951.0, 723.1)."""
    K = np.array([[214.679, 0, 129.204], [0, 214.679, 95.6055], [0, 0, 1]])
    D = ak.device_K(K, 180)
    assert np.allclose(D[0, 0], 214.679 * 7.5) and abs(D[0, 2] - 951.0) < 0.1 and abs(D[1, 2] - 723.0) < 0.1
    assert np.allclose(ak.device_K(K, 0)[:2, 2], [969.03, 717.04], atol=0.01)


def test_lidar_on_grid_covers_the_crop_and_turns():
    nw, nh, left, top = ak.resize_crop_params(ak.HW, ak.HH, ak.TW, ak.TH)
    d = np.arange(ak.LW * ak.LH, dtype=np.float32).reshape(ak.LH, ak.LW)
    g = ak.lidar_on_grid(d, 0, nw, nh, left, top)
    assert g.shape == (ak.TH, ak.TW)
    # 1920x1440 -> 523x392, 2 px cropped left: the first pixel centre sits at LiDAR x = 2.5 * 256 / 523 = 1.22,
    # the last at (519.5) * 256 / 523 = 254.3; rows span 0.24 .. 191.75
    assert (nw, nh, left, top) == (523, 392, 2, 0)
    assert g[0, 0] == d[0, 1] and g[-1, -1] == d[191, 254] and g[0, -1] == d[0, 254] and g[-1, 0] == d[191, 1]
    assert np.array_equal(ak.lidar_on_grid(d, 180, nw, nh, left, top), ak.lidar_on_grid(np.rot90(d, 2), 0, nw, nh, left, top))


def test_device_turn_detects_0_and_180():
    rng = np.random.default_rng(1)
    gt = []
    for i in range(20):
        T = np.eye(4); T[:3, :3] = so3_exp(rng.normal(size=3)); T[:3, 3] = rng.normal(size=3); gt.append(T)
    gt, ts = np.array(gt), np.arange(20) * 0.1
    L2A = np.eye(4); L2A[:3, :3] = so3_exp(np.array([0, 0, 1.0])); L2A[:3, 3] = [5, 0, 0]
    for turn in (0, 180):
        dev = np.array([L2A @ T @ (ak.R180 if turn else np.eye(4)) for T in gt])
        got, measured, resid = ak.device_turn(ts, gt, ts, dev, L2A)
        assert got == turn and abs(measured - turn) < 1e-6 and resid < 1e-6


def test_keyframes_orientation_and_selection_rule(tmp_path):
    P = np.array([np.eye(4)] * 50)
    for i in range(50):
        P[i, 0, 3] = 0.11 * i                             # 11 cm per frame -> a keyframe every 3 frames
    assert ak.keyframes(P) == list(range(0, 50, 3))
    assert ak.orientation(np.eye(4) @ np.diag([1.0, -1, -1, 1])) in range(4)
    up = np.eye(4); up[:3, :3] = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])  # camera y = world -z: upright
    assert ak.orientation(up) == 0
    long_path = np.array([up.copy() for _ in range(400)])
    long_path[:, 0, 3] = 0.1 * np.arange(400)             # 133 keyframes
    short = long_path[:50]
    rows = [("visit_id", "video_id"), ("1", "11"), ("1", "12"), ("2", "21"), ("3", "31"), ("4", "41")]
    p = tmp_path / "test.csv"
    with open(p, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    traj = {"11": short, "12": long_path, "21": short, "31": long_path, "41": long_path}
    sel = ak.select_videos(str(p), n_dev=1, n_eval=1, traj_of=lambda v, vid: (np.arange(len(traj[vid])), traj[vid]))
    assert len(sel) == 2 and sel[0][0] == "dev" and sel[1][0] == "eval"
    assert all(vid in ("12", "31", "41") for _, _, vid in sel)     # visit 2 (too short) never used; visit 1 -> longer video


# ---------------------------------------------------------------- automatic dev gates (scripts/dev_gates.py)
def _gates():
    import importlib.util
    spec = importlib.util.spec_from_file_location("dev_gates", os.path.join(ROOT, "scripts", "dev_gates.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_dev_gates_pass_complete_runs_and_catch_problems(clips, tmp_path):
    root, _, dpath = clips
    g = _gates()
    mp = pytest.MonkeyPatch()
    out = str(tmp_path / "e2e")
    _run(mp, "e2e_mapanything.json", dpath, out)
    mp.undo()
    cfg = os.path.join(ROOT, "configs", "e2e_mapanything.json")
    assert g.complete(out, cfg) == []
    r = g.runs(out)
    assert g.hints_help(r, "gt_pose", unit="scene", need=1) == []      # the fake copies the true poses
    assert g.hints_help(r, "gt_pose", unit="scene", need=2) != []      # only one scene here
    # a missing run and a failures file are both caught
    pr = pd.read_csv(os.path.join(out, "per_run.csv"))
    pr[pr.condition != "dev_K"].to_csv(os.path.join(out, "per_run.csv"), index=False)
    open(os.path.join(out, "failures.jsonl"), "w").write("{}\n")
    why = g.complete(out, cfg)
    assert any("expected" in w for w in why) and any("failures" in w for w in why)
