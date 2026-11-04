"""E3 control (exp1/e3.py, run_exp1.run_e3) through the runner with a FAKE model (Pi3X input format). Plumbing and
decision logic only: the fake answers images-only passes with the GT poses plus a small order-dependent wobble (so
the witness ensemble has a spread), copies pose and depth hints, and otherwise returns GT depth."""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from exp1 import model_io_pi3x, run_exp1  # noqa: E402
from exp1.datasets import load_clip, prepare_eth3d  # noqa: E402
from exp1.geometry import so3_exp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_e0_clip_pipeline as synth  # noqa: E402

CONFIG = {
    "checkpoint": "fake", "seeds": [0], "depth_views": [3, 7], "filter_rot_tau_deg": 10.0, "filter_dir_tau_deg": 15.0,
    "lowconf_tau": 0.4572,
    "conditions": [
        {"name": "rgb_only", "family": "reference", "inputs": []},
        {"name": "clean_pose", "family": "reference", "inputs": ["intrinsics", "pose"], "controls": ["witness", "witness_drop"]},
        {"family": "outlier_rot", "inputs": ["intrinsics", "pose"], "sweep": "rot_deg", "values": [15], "seeds": 1,
         "corruption": {"mode": "subset", "n": 1, "trans_frac": 0.0}, "controls": ["witness"]},
        {"family": "shared_heading_bias", "inputs": ["intrinsics", "pose"], "sweep": "rot_deg", "values": [10], "seeds": 1,
         "corruption": {"type": "shared_rot"}, "controls": ["witness", "witness_drop"]},
        {"family": "anchor_error", "inputs": ["intrinsics", "pose"], "levels": [{"rot_deg": 15.0, "trans_frac": 0.0}],
         "seeds": 1, "corruption": {"views": [0]}, "controls": ["witness"]},
        {"family": "depth_scale", "inputs": ["intrinsics", "pose", "depth"], "sweep": "factor", "values": [1.5], "seeds": 1,
         "corruption": {"target": "depth", "type": "scale"}, "controls": ["witness", "witness_drop"]},
        {"name": "clean_pose_depth2", "family": "reference", "inputs": ["intrinsics", "pose", "depth"], "controls": ["witness"]},
    ]}


def fake_model(clip):
    N, H, W = clip["depth"].shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    ref = clip["images"].reshape(N, -1).astype(np.float64)

    def run(model, views, device):
        imgs = (views["imgs"][0].permute(0, 2, 3, 1).numpy() * 255).round().reshape(N, -1)
        order = np.array([int(np.argmin(np.abs(ref - im).sum(1))) for im in imgs])   # which clip view is where
        posed = views["mask_add_pose"][0].numpy() & (views["mask_add_pose"][0].numpy().sum() > 1)
        dmask = views["mask_add_depth"][0].numpy()
        shift = int(order[0])
        P0 = np.linalg.inv(clip["poses"][order[0]])
        out = {"pts3d": [], "depth_z": [], "pose": [], "conf": [], "mask": [], "intrinsics": []}
        for pos, i in enumerate(order):
            if posed[pos]:
                T = views["poses"][0, pos].double().numpy()
            else:
                T = P0 @ clip["poses"][i]
                if not posed.any():                       # images only: small order-dependent wobble
                    T = T.copy()
                    T[:3, :3] = T[:3, :3] @ so3_exp(np.radians(0.1 * (shift + 1)) * np.array([0.0, 1.0, 0.0]))
            K = clip["K"][i]
            z = views["depths"][0, pos].numpy().astype(np.float64) if dmask[pos] else clip["depth"][i]
            z = np.where(z > 0, z, np.median(clip["depth"][i]))
            pc = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], -1)
            out["pts3d"].append(pc @ T[:3, :3].T + T[:3, 3])
            out["depth_z"].append(z)
            out["pose"].append(T)
            out["conf"].append(np.full((H, W), 0.8))
            out["mask"].append(np.ones((H, W), bool))
            out["intrinsics"].append(K)
        out = {k: np.stack(x) for k, x in out.items()}
        out["runtime_s"], out["peak_mem_gb"], out["metric_scaling_factor"] = 0.01, 0.0, 1.0
        return out

    return run


@pytest.fixture(scope="module")
def e3_run(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("e3"))
    synth.write_scene(root, n=10)
    clips = os.path.join(root, "clips")
    prepare_eth3d(root, clips, n_views=10, clips_per_scene=1, stride=1, tw=168, th=98, scenes=["synth"])
    clip_path = os.path.join(clips, "dev__synth__c0.npz")
    cfg_path = os.path.join(root, "e3.json")
    json.dump(CONFIG, open(cfg_path, "w"))
    out = os.path.join(root, "out")
    mp = pytest.MonkeyPatch()
    mp.setattr(run_exp1, "model_io", run_exp1.model_io)
    mp.setattr(model_io_pi3x, "load_model", lambda ckpt, device: "fake")
    mp.setattr(model_io_pi3x, "run", fake_model(load_clip(clip_path)))
    mp.setattr(run_exp1, "git_commit", lambda path: "test")
    mp.setattr(sys, "argv", ["run_exp1", "--config", cfg_path, "--clips", clip_path, "--out", out, "--device", "cpu",
                             "--backend", "pi3x"])
    run_exp1.main()
    mp.undo()
    return out


def _decisions(out):
    return {json.loads(x)["run_key"].split("|", 1)[1]: json.loads(x) for x in open(os.path.join(out, "e3_decisions.jsonl"))}


def test_e3_runs_without_failures(e3_run):
    fp = os.path.join(e3_run, "failures.jsonl")
    assert not os.path.exists(fp), open(fp).read()
    pr = pd.read_csv(os.path.join(e3_run, "per_run.csv"))
    assert (pr.status == "ok").all()
    assert {"witness", "witness_drop"} <= set(pr.control)


def test_e3_leaves_clean_hints_alone(e3_run):
    d = _decisions(e3_run)
    for k in ("clean_pose|0|witness", "clean_pose|0|witness_drop", "clean_pose_depth2|0|witness"):
        assert not (d[k]["fire_shared"] or d[k]["fire_mount"] or d[k]["fire_drift"]) and d[k]["drop_pose_views"] == [], k
    assert not d["clean_pose_depth2|0|witness"]["fire_depth_scale"]
    assert abs(d["clean_pose_depth2|0|witness"]["depth_k"] - 1) < 1e-6
    pr = pd.read_csv(os.path.join(e3_run, "per_run.csv")).set_index(["condition", "control", "view_subset"])
    assert pr.loc[("clean_pose", "witness", "all"), "rre_deg"] == pytest.approx(pr.loc[("clean_pose", "none", "all"), "rre_deg"])


def test_e3_drops_one_bad_camera(e3_run):
    d = _decisions(e3_run)["c1_rot_rot_deg=15|0|witness"] if "c1_rot_rot_deg=15|0|witness" in _decisions(e3_run) \
        else _decisions(e3_run)["outlier_rot_rot_deg=15|0|witness"]
    man = json.load(open(os.path.join(e3_run, "corruption_manifest.json")))
    bad = next(m["affected_views"] for m in man if m["condition"].startswith("outlier_rot"))
    assert d["drop_pose_views"] == bad and not d["fire_shared"]


def test_e3_repairs_shared_heading_and_anchor(e3_run):
    d = _decisions(e3_run)
    sh = next(v for k, v in d.items() if k.startswith("shared_heading_bias") and k.endswith("|witness"))
    assert (sh["fire_shared"] or sh["fire_mount"]) and sh["drop_pose_views"] == []
    an = next(v for k, v in d.items() if k.startswith("anchor_error") and k.endswith("|witness"))
    assert an["fire_shared"] and abs(an["shared_deg"] - 15) < 1.0
    drop = next(v for k, v in d.items() if k.startswith("shared_heading_bias") and k.endswith("|witness_drop"))
    assert drop["pose_hints_dropped"] and "pose" not in drop["inputs_final"]
    pr = pd.read_csv(os.path.join(e3_run, "per_run.csv"))
    pr = pr[(pr.view_subset == "all") & pr.condition.str.startswith("shared_heading_bias")].set_index("control")
    assert pr.loc["witness", "rre_deg"] < 0.3 * pr.loc["none", "rre_deg"]     # repaired close to clean


def test_e3_rescales_wrong_depth(e3_run):
    d = _decisions(e3_run)
    dz = next(v for k, v in d.items() if k.startswith("depth_scale") and k.endswith("|witness"))
    assert dz["fire_depth_scale"] and abs(dz["depth_k"] - 1.5) < 1e-3 and dz["depth_scale_from"] == "pose"
    dd = next(v for k, v in d.items() if k.startswith("depth_scale") and k.endswith("|witness_drop"))
    assert "depth" not in dd["inputs_final"]


def _device_clip(root):
    """The synthetic clip plus device-like hints (as tests/test_e2ef.py) and an AI depth 1.3x too far."""
    synth.write_scene(root, n=10)
    clips = os.path.join(root, "clips")
    prepare_eth3d(root, clips, n_views=10, clips_per_scene=1, stride=1, tw=168, th=98, scenes=["synth"])
    path = os.path.join(clips, "dev__synth__c0.npz")
    c = dict(np.load(path))
    rng = np.random.default_rng(0)
    dev = c["poses"].copy()
    for i in range(1, 10):
        dev[i][:3, :3] = dev[i][:3, :3] @ so3_exp(np.radians(0.5) * rng.normal(size=3) / np.sqrt(3))
        dev[i][:3, 3] += 0.01 * rng.normal(size=3)
    c.update(dev_poses=dev, dev_K=c["K"] * np.array([[1.01, 1, 1], [1, 1.01, 1], [1, 1, 1]]),
             dev_depth=(c["depth"] * 0.98).astype(np.float32), mono_depth=(c["depth"] * 1.3).astype(np.float32))
    np.savez_compressed(path, **c)
    return path


@pytest.mark.parametrize("backend,kind", [("pi3x", "synth"), ("omnivggt", "synth"), ("mapanything", "synth"),
                                          ("pi3x", "real"), ("omnivggt", "real"), ("mapanything", "real")])
def test_e3cal_config_runs_completely(tmp_path, backend, kind):
    """The E3 calibration grids (configs/e2x_e3cal_<kind>_<model>.json) build and run every condition and control."""
    from exp1 import model_io, model_io_omnivggt
    import test_e2d
    import test_runner_dryrun
    clip_path = _device_clip(str(tmp_path))
    io = {"pi3x": model_io_pi3x, "omnivggt": model_io_omnivggt, "mapanything": model_io}[backend]
    fake = {"pi3x": fake_model, "omnivggt": test_e2d.fake_omnivggt,
            "mapanything": test_runner_dryrun.fake_run_factory}[backend]
    out = os.path.join(str(tmp_path), "out")
    mp = pytest.MonkeyPatch()
    mp.setattr(run_exp1, "model_io", run_exp1.model_io)
    mp.setattr(io, "load_model", lambda ckpt, device: "fake")
    mp.setattr(io, "run", fake(load_clip(clip_path)))
    mp.setattr(run_exp1, "git_commit", lambda path: "test")
    cfg = os.path.join(HERE, "..", "configs", f"e2x_e3cal_{kind}_{backend}.json")
    mp.setattr(sys, "argv", ["run_exp1", "--config", cfg, "--clips", clip_path, "--out", out, "--device", "cpu",
                             "--backend", backend])
    try:
        run_exp1.main()
    finally:
        mp.undo()
    fp = os.path.join(out, "failures.jsonl")
    assert not os.path.exists(fp), open(fp).read()[:2000]
    pr = pd.read_csv(os.path.join(out, "per_run.csv"))
    r = pr[(pr.view_subset == "all") & ~pr.control.astype(str).str.startswith("ref:")]
    c = json.load(open(cfg))
    per_clip = sum(len(run_exp1.seeds_for(x, c)) * (1 + len(x.get("controls", []))) for x in run_exp1.expand_conditions(c))
    assert len(r) == per_clip and (r.status == "ok").all()
    for line in open(os.path.join(out, "e3_decisions.jsonl")):
        d = json.loads(line)
        if backend == "omnivggt":   # dropping all poses also drops intrinsics (OmniVGGT cannot take them alone)
            assert not ("intrinsics" in d["inputs_final"] and "pose" not in d["inputs_final"]), d["run_key"]


def test_calibration_recomputes_the_logged_decisions(e3_run):
    """exp1/e3_calibrate recomputes every logged decision exactly at the logged settings (default z, taus)."""
    from exp1 import e3, e3_calibrate as cal
    rows = cal.load_decisions([e3_run])
    assert rows
    for r in rows:
        if "shared_deg" in r:
            g, nd, _ = cal.pose_actions(r, e3.DEFAULTS["e3_z"])
            assert g == (r["fire_shared"] or r["fire_mount"] or r["fire_drift"]) and nd == len(r["drop_pose_views"])
        if r.get("depth_scale_from") in ("pose", "images"):
            s, dd, _ = cal.depth_actions(r, e3.DEFAULTS["e3_tau_depth_shared"], e3.DEFAULTS["e3_tau_depth_view"])
            assert s == r["fire_depth_scale"] and dd == len(r["drop_depth_views"])


def test_calibration_rule_picks_the_smallest_safe_value():
    from exp1 import e3_calibrate as cal
    base = {"control": "witness", "condition": "gt_pose", "sigma_group_deg": {"shared": 0.0, "mount": 0.0, "drift": 0.0},
            "mount_deg": 0.0, "drift_total_deg": 0.0, "sigma_view_deg": [0.0] * 10, "rot_res_deg": [0.0] * 10}
    # 100 clean clips; the shared fit is 1.2 deg on 10 of them: z = 2 (threshold 1.0 deg) acts on 10% of clips,
    # z = 2.5 (1.25 deg) on none
    rows = [dict(base, clip_id=f"cal__x{i}__c0", shared_deg=1.2 if i < 10 else 0.1) for i in range(100)]
    table, chosen = cal.calibrate(rows)
    assert chosen["e3_z"] == 2.5
    rows = [dict(base, clip_id=f"cal__x{i}__c0", shared_deg=9.0) for i in range(100)]   # never safe
    assert cal.calibrate(rows)[1]["e3_z"] == "NOT CALIBRATED"
