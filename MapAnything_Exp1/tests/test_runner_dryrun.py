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
        out = {"pts3d": [], "depth_z": [], "pose": [], "conf": [], "mask": [], "intrinsics": []}
        for i, view in enumerate(views):
            assert tuple(view["img"].shape) == (H, W, 3)
            if "depth_z" in view:
                assert "intrinsics" in view and tuple(view["depth_z"].shape) == (H, W)
            if "camera_poses" in view:
                T = view["camera_poses"].double().numpy()
            else:
                T = clip["poses"][i].copy()
                T[:3, :3] = T[:3, :3] @ so3_exp(np.radians(1.0) * np.array([0.0, 1.0, 0.0]))
            K = view["intrinsics"].double().numpy() if "intrinsics" in view else clip["K"][i]
            z = np.where(clip["depth"][i] > 0, clip["depth"][i], np.median(clip["depth"][i]))
            pc = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], -1)
            out["pts3d"].append(pc @ T[:3, :3].T + T[:3, 3])
            out["depth_z"].append(z)
            out["pose"].append(T)
            out["conf"].append(np.full((H, W), 2.0 if "camera_poses" in view else 1.0))
            out["mask"].append(np.ones((H, W), bool))
            out["intrinsics"].append(K)
        out = {k: np.stack(x) for k, x in out.items()}
        out["runtime_s"], out["peak_mem_gb"], out["metric_scaling_factor"] = 0.01, 0.0, 1.0
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


@pytest.fixture(scope="module")
def e1_run(tmp_path_factory):
    """The FULL E1 grid on an 8-view synthetic clip (depth hints on views 3 and 7), fake model."""
    root = str(tmp_path_factory.mktemp("e1"))
    synth.write_scene(root, n=8)
    clips = os.path.join(root, "clips")
    prepare_eth3d(root, clips, n_views=8, clips_per_scene=1, stride=1, tw=170, th=100, scenes=["synth"])
    clip_path = os.path.join(clips, "dev__synth__c0.npz")
    out = os.path.join(root, "e1")
    mp = pytest.MonkeyPatch()
    mp.setattr(model_io, "load_model", lambda ckpt, device: "fake")
    mp.setattr(model_io, "run", fake_run_factory(load_clip(clip_path)))
    mp.setattr(sys, "argv", ["run_exp1", "--config", os.path.join(ROOT, "configs", "e1_full.json"),
                             "--clips", clip_path, "--out", out, "--device", "cpu"])
    run_exp1.main()
    mp.undo()
    return out


def test_e1_grid_runs_completely(e1_run):
    pr = pd.read_csv(os.path.join(e1_run, "per_run.csv"))
    pv = pd.read_csv(os.path.join(e1_run, "per_view.csv"))
    assert not os.path.exists(os.path.join(e1_run, "failures.jsonl"))
    runs = pr[pr.view_subset == "all"]
    assert (runs.status == "ok").all() and len(runs) == 235
    assert len(pv) == 8 * len(runs)
    man = json.load(open(os.path.join(e1_run, "corruption_manifest.json")))
    eq = {e["condition"]: e["max_abs_diff_vs_clean_pts3d"] for e in man if "max_abs_diff_vs_clean_pts3d" in e}
    assert eq == {"zero_noise": 0.0, "scale_all_x1.0": 0.0}
    # depth hints only on views 3 and 7; the metric flag is recorded
    d = pv[pv.condition == "clean_pose_depth2"]
    assert set(d[d.has_depth_hint == 1].view_idx) == {3, 7}
    assert set(pr[pr.condition == "metric_flag_off"].metric_flag) == {0}
    # the fake model copies pose hints, so absorption of a single 15 deg outlier is 1
    o = pv[(pv.condition == "outlier_rot_rot_deg=15") & (pv.was_corrupted == 1) & (pv.control == "none")]
    assert np.allclose(o.absorption_rot, 1.0, atol=1e-3)
    # evaluation is gauge-invariant: moving every camera together changes nothing
    g = pr[(pr.condition == "gauge_rigid") & (pr.view_subset == "all")].pts_relerr_scale
    c = pr[(pr.condition == "clean_pose") & (pr.view_subset == "all")].pts_relerr_scale.iloc[0]
    assert np.allclose(g, c, atol=1e-6)


def test_e1_analysis_runs_on_the_full_grid(e1_run, monkeypatch):
    from exp1 import e1_analysis
    monkeypatch.setattr(sys, "argv", ["x", "--run", e1_run, "--checks"])
    e1_analysis.main()
    agg = os.path.join(e1_run, "agg")
    for f in ("dose_response.csv", "contamination.csv", "confidence_auroc.csv", "absorption.csv", "fix.csv",
              "scale.csv", "signals.json", "signals.md", "dev_checks.md", "fig_dose_response.png",
              "fig_scene_heatmap.png", "fig_contamination.png", "fig_absorption.png", "fig_scale.png"):
        assert os.path.exists(os.path.join(agg, f)), f
    sig = json.load(open(os.path.join(agg, "signals.json")))
    assert sig["decision"] in ("Strong GO", "Weak GO", "STOP")
    # the fake model trusts every hint: blind trust (S3) must be detected
    assert sig["S3"]["value"] is True


def test_save_extras_poses_and_point_displacement(clip_path, tmp_path, monkeypatch):
    """--save_extras writes one npz per run. The fake model copies pose hints, so the corrupted
    view's predicted pose equals its hint and only that view's points move vs clean_pose."""
    out = str(tmp_path / "extras_run")
    monkeypatch.setattr(model_io, "load_model", lambda ckpt, device: "fake")
    monkeypatch.setattr(model_io, "run", fake_run_factory(load_clip(clip_path)))
    monkeypatch.setattr(sys, "argv", [
        "run_exp1", "--config", os.path.join(ROOT, "configs", "pilot.json"), "--clips", clip_path,
        "--out", out, "--device", "cpu", "--save_extras",
        "--only", "rgb_only", "clean_pose", "outlier_rot_rot_deg=15"])
    run_exp1.main()
    files = sorted(os.listdir(os.path.join(out, "extras")))
    assert len(files) == 2 + 3 * 3  # rgb_only, clean_pose, 3 seeds x (none + 2 controls)
    pv = pd.read_csv(os.path.join(out, "per_view.csv"))
    bad = pv[(pv.condition == "outlier_rot_rot_deg=15") & (pv.seed == 0) & (pv.control == "none")
             & (pv.was_corrupted == 1)].view_idx.tolist()
    z = np.load(os.path.join(out, "extras", "dev__synth__c0__outlier_rot_rot_deg=15__0__none__all.npz"))
    assert np.allclose(z["pred_pose_rel"], z["hint_pose_rel"], atol=1e-9)
    assert np.allclose(z["clean_pose_rel"], load_clip(clip_path)["poses"], atol=1e-9)
    good = [i for i in range(len(z["pts_disp_median_m"])) if i not in bad]
    assert len(bad) == 1 and z["pts_disp_median_m"][bad[0]] > 0
    assert np.allclose(z["pts_disp_median_m"][good], 0, atol=1e-9)
    assert np.isnan(z["pts_disp_along_shift_m"]).all()  # rotation-only: no camera shift
    r = np.load(os.path.join(out, "extras", "dev__synth__c0__rgb_only__0__none__all.npz"))
    assert "hint_pose_rel" not in r.files and "clean_pose_rel" not in r.files
    assert "start=" in open(os.path.join(out, "gpu_log.txt")).read()


@pytest.fixture(scope="module")
def e2c_run(tmp_path_factory):
    """The FULL E2-C grid on an 8-view synthetic clip, fake model, with --save_extras."""
    root = str(tmp_path_factory.mktemp("e2c"))
    synth.write_scene(root, n=8)
    clips = os.path.join(root, "clips")
    prepare_eth3d(root, clips, n_views=8, clips_per_scene=1, stride=1, tw=170, th=100, scenes=["synth"])
    clip_path = os.path.join(clips, "dev__synth__c0.npz")
    out = os.path.join(root, "e2c")
    stats_used = []
    mp = pytest.MonkeyPatch()
    mp.setattr(model_io, "load_model", lambda ckpt, device: "fake")
    mp.setattr(model_io, "run", fake_run_factory(load_clip(clip_path)))
    mp.setattr(model_io, "set_pose_scale_statistic", stats_used.append)
    mp.setattr(sys, "argv", ["run_exp1", "--config", os.path.join(ROOT, "configs", "e2c_full.json"),
                             "--clips", clip_path, "--out", out, "--device", "cpu", "--save_extras"])
    run_exp1.main()
    mp.undo()
    return out, stats_used


def test_e2c_grid_runs_completely(e2c_run):
    out, stats_used = e2c_run
    assert not os.path.exists(os.path.join(out, "failures.jsonl")), open(os.path.join(out, "failures.jsonl")).read()
    pr = pd.read_csv(os.path.join(out, "per_run.csv"))
    runs = pr[pr.view_subset == "all"]
    assert (runs.status == "ok").all() and len(runs) == 132
    assert len(os.listdir(os.path.join(out, "extras"))) == 132
    # the median probe is switched on and off around every median run, and never left on
    n_median = int(runs.condition.str.startswith(("c4m_", "clean_pose_median")).sum())
    assert stats_used == ["median", "mean"] * n_median


def test_e2c_pairing_partial_poses_and_matched_jitter(e2c_run):
    out, _ = e2c_run
    man = {(e["condition"], e["base_seed"]): e for e in json.load(open(os.path.join(out, "corruption_manifest.json")))}
    for s in (0, 1):  # median runs reuse the mean runs' corruption
        a, b = man[("c1_trans_trans_frac=1", s)], man[("c4m_trans_trans_frac=1", s)]
        assert a["affected_views"] == b["affected_views"] and a["per_view"] == b["per_view"]
    pv = pd.read_csv(os.path.join(out, "per_view.csv"))
    for frac, k in ((0.3, 2), (0.6, 5)):
        for name in (f"c7_clean_frac={frac}", f"c7_bad_frac={frac}"):
            d = pv[(pv.condition == name) & (pv.control == "none")]
            assert (d.groupby("seed").has_pose_hint.sum() == k).all()
        bad = pv[(pv.condition == f"c7_bad_frac={frac}") & (pv.control == "none") & (pv.was_corrupted == 1)]
        assert (bad.has_pose_hint == 1).all() and (bad.view_idx > 0).all()
    j = pv[(pv.condition == "c6_drift_jitter_level=2") & (pv.control == "none")].set_index(["seed", "view_idx"])
    d = pv[(pv.condition == "c6_drift_level=2") & (pv.control == "none")].set_index(["seed", "view_idx"])
    assert np.allclose(j.hint_rot_err, d.loc[j.index].hint_rot_err, atol=1e-6)
    assert np.allclose(j.hint_trans_err_m, d.loc[j.index].hint_trans_err_m, atol=1e-9)
    # pixel size of the hint error is recorded on corrupted views of every hint channel (undefined
    # once the turned camera sees its points from behind, so only up to 60 deg)
    for fam in ("c1_rot", "c1_trans", "c1_focal", "c1_depth", "c4_radial", "c4_orbit"):
        c = pv[(pv.family == fam) & (pv.was_corrupted == 1) & (pv.control == "none") & (pv.severity != 1.0)
               & ((pv.family != "c1_rot") | (pv.severity <= 60))]
        assert c.hint_reproj_px.notna().all() and (c.hint_reproj_px > 0).all(), fam
    assert pv.view_shape_err_sim3.notna().mean() > 0.9


def test_e2c_analysis_runs_and_fake_model_copies_fully(e2c_run, monkeypatch, capsys):
    from exp1 import e2c_analysis
    out, _ = e2c_run
    clips = os.path.join(os.path.dirname(out), "clips", "*.npz")
    agg = os.path.join(out, "agg")
    monkeypatch.setattr(sys, "argv", ["x", "--run", out, "--clips", clips, "--out", agg])
    e2c_analysis.main()
    pt = pd.read_csv(os.path.join(agg, "primary_tests.csv"))
    assert list(pt.test) == [p[0] for p in e2c_analysis.PRIMARY]
    bv = pd.read_csv(os.path.join(agg, "bad_views.csv"))
    # the fake model outputs every pose hint exactly, so absorption is 1 for every pose channel
    assert np.allclose(bv.absorption, 1.0, atol=1e-3)  # float32 hint rounding only
    for f in ("c1_c3_trust_curves.csv", "c1_adoption_shared_channels.csv", "c4_channels_harm.csv",
              "c4_median_probe.csv", "c5_evidence.csv", "c6_correlated_vs_independent.csv",
              "c7_partial_hints.csv", "c8_photos_alone.csv"):
        assert len(pd.read_csv(os.path.join(agg, f))) > 0, f
