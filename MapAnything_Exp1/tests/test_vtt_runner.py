"""E4-A controls in the runner (exp1/run_exp1.py, controls vtt / vtt_gate) on CPU with the fake model of
test_runner_dryrun: rows and vtt_decisions.jsonl are written for every control, no run fails, the gate is applied and
cleared, and the decisions carry the posteriors."""

import json
import os
import sys

import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("cv2")

from exp1 import model_io, run_exp1  # noqa: E402
from exp1.datasets import load_clip, prepare_eth3d  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_e0_clip_pipeline as synth  # noqa: E402
from test_runner_dryrun import fake_run_factory  # noqa: E402

CFG = {"checkpoint": "fake", "seeds": [0], "depth_views": [0, 2], "vtt_kappa_rot": 100.0, "vtt_kappa_dir": 50.0,
       "conditions": [
           {"name": "rgb_only", "family": "reference", "inputs": []},
           {"name": "clean_pose", "family": "reference", "inputs": ["intrinsics", "pose"],
            "controls": ["vtt", "vtt_gate"]},
           {"name": "one15", "family": "outlier_rot", "inputs": ["intrinsics", "pose"],
            "corruption": {"mode": "subset", "n": 1, "trans_frac": 0.0, "rot_deg": 15}, "controls": ["vtt", "vtt_gate"]},
           {"name": "depth15", "family": "depth_scale", "inputs": ["intrinsics", "pose", "depth"],
            "corruption": {"target": "depth", "type": "scale", "factor": 1.5}, "controls": ["vtt"]}]}


def test_vtt_controls_in_runner(tmp_path):
    root = str(tmp_path)
    synth.write_scene(root, n=6)
    cdir = os.path.join(root, "clips")
    prepare_eth3d(root, cdir, n_views=6, clips_per_scene=1, stride=1, tw=170, th=100, scenes=["synth"])
    path = os.path.join(cdir, "dev__synth__c0.npz")
    cfgp = os.path.join(root, "vtt.json")
    json.dump(CFG, open(cfgp, "w"))
    current, gates = {}, []

    def load(p):
        clip = load_clip(p)
        current["run"] = fake_run_factory(clip)
        return clip

    mp = pytest.MonkeyPatch()
    mp.setattr(run_exp1, "load_clip", load)
    mp.setattr(model_io, "load_model", lambda ckpt, device: "fake")
    mp.setattr(model_io, "run", lambda model, views, device: current["run"](model, views, device))
    mp.setattr(model_io, "set_pose_scale_statistic", lambda stat: None)
    mp.setattr(model_io, "set_hint_gate", lambda model, views, gd, gp: gates.append(("set", None if gp is None else list(gp))))
    mp.setattr(model_io, "clear_hint_gate", lambda model: gates.append(("clear",)))
    out = os.path.join(root, "out")
    mp.setattr(sys, "argv", ["run_exp1", "--config", cfgp, "--clips", path, "--out", out, "--device", "cpu"])
    run_exp1.main()
    mp.undo()
    assert not os.path.exists(os.path.join(out, "failures.jsonl"))
    R = pd.read_csv(os.path.join(out, "per_run.csv"))
    R = R[R.view_subset == "all"]
    got = set(zip(R.condition, R.control))
    for c, k in [("clean_pose", "vtt"), ("clean_pose", "vtt_gate"), ("one15", "vtt"), ("one15", "vtt_gate"),
                 ("depth15", "vtt")]:
        assert (c, k) in got, (c, k)
    D = [json.loads(x) for x in open(os.path.join(out, "vtt_decisions.jsonl"))]
    assert len(D) == 5 and all("w_hint" in d for d in D)
    assert any("depth" in d for d in D)
    assert [g[0] for g in gates] == ["set", "clear"] * 2 and len(gates[0][1]) == 6
