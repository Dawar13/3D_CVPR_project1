"""exp1/run_xm.py plumbing on CPU with a FAKE backend (no model, no GPU): every experiment row is written once,
resume repeats nothing, eval clips are refused, and the state statistics are consistent. The fake model copies pose
hints (test_runner_dryrun.fake_run_factory); its numbers mean nothing about any real model."""

import glob
import json
import os
import subprocess
import sys
import types

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from exp1 import model_io, run_xm  # noqa: E402
from exp1.datasets import load_clip, prepare_eth3d  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_e0_clip_pipeline as synth  # noqa: E402
from test_runner_dryrun import fake_run_factory  # noqa: E402

LAYERS = [0, 2, 4]
CFG = {"checkpoint": "fake", "backend": "mapanything", "inputs": ["intrinsics", "pose"], "depth_views": [0, 2],
       "layers": LAYERS, "dial": [0, 0.5, 1],
       "cases": [{"name": "one_cam_15", "corruption": {"mode": "subset", "n": 1, "trans_frac": 0.0, "rot_deg": 15}},
                 {"name": "depth_x1.5", "inputs": ["intrinsics", "pose", "depth"],
                  "corruption": {"target": "depth", "type": "scale", "factor": 1.5}}]}


def fake_io(clip, calls):
    io = types.SimpleNamespace()
    io.build_views = model_io.build_views
    run = fake_run_factory(clip)
    io.load_model = lambda ckpt, device: "fake"
    io.run = lambda model, views, device: run(model, views, device)
    io.set_hint_gate = lambda model, views, gd, gp: calls.append(("gate", gd is not None, gp is not None))
    io.clear_hint_gate = lambda model: None

    counter = [0]

    class Cap:
        def __init__(self, model, layers, n, full=False):
            counter[0] += 1                          # a different fake state for every captured run
            rng = np.random.default_rng(counter[0])
            self.tokens = {k: rng.normal(size=(n, 5, 3)).astype(np.float32) for k in layers}

        def remove(self):
            pass

    class Patch:
        def __init__(self, model, layer, views, tokens):
            calls.append(("patch", layer, tuple(views)))

        def remove(self):
            pass

    io.TokenCapture, io.TokenPatch = Cap, Patch
    return io


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("xm"))
    synth.write_scene(root, n=6)
    cdir = os.path.join(root, "clips")
    prepare_eth3d(root, cdir, n_views=6, clips_per_scene=1, stride=1, tw=170, th=100, scenes=["synth"])
    path = os.path.join(cdir, "dev__synth__c0.npz")
    cfgp = os.path.join(root, "xm.json")
    json.dump(CFG, open(cfgp, "w"))
    return root, path, cfgp


def _main(mp, io, cfgp, clips, out, extra=()):
    mp.setattr(run_xm, "get_backend", lambda name: io)
    mp.setattr(sys, "argv", ["run_xm", "--config", cfgp, "--clips", clips, "--out", out, "--device", "cpu", *extra])
    run_xm.main()


def test_rows_resume_and_stats(setup):
    root, path, cfgp = setup
    calls = []
    io = fake_io(load_clip(path), calls)
    mp = pytest.MonkeyPatch()
    out = os.path.join(root, "out")
    _main(mp, io, cfgp, path, out)
    assert not os.path.exists(os.path.join(out, "failures.jsonl"))
    R = pd.read_csv(os.path.join(out, "per_run.csv"))
    per_case = 4 + len(CFG["dial"]) + 3 * len(LAYERS)   # 4 references, M1, M3 (clean, drop), M5
    assert len(R) == 2 * per_case and R.run_key.is_unique
    one = R[R.case == "one_cam_15"]
    bad = one.bad_views.iloc[0]
    assert len(str(bad).split()) == 1 and str(bad) != "0"
    clean = one[(one.experiment == "ref") & (one.source == "clean")].iloc[0]
    worse = one[(one.experiment == "ref") & (one.source == "bad")].iloc[0]
    assert worse.bad_rot_err_gt_deg > 14 and clean.bad_rot_err_gt_deg < 0.05   # the fake model copies the hint
    assert worse.bad_rot_err_hint_deg < 0.05   # float32 round-off
    dep = R[R.case == "depth_x1.5"]
    assert set(dep.bad_views.astype(str)) == {"0 2"}
    assert ("gate", True, False) in calls and ("gate", False, True) in calls
    assert sum(c[0] == "patch" for c in calls) == 2 * 3 * len(LAYERS)
    S = pd.read_csv(os.path.join(out, "state_stats.csv"))
    assert len(S) == 2 * len(LAYERS) * 6
    first = S[S.layer == LAYERS[0]]
    assert np.allclose(first.cos_with_first_layer, 1.0, atol=1e-6)
    gate = subprocess.run([sys.executable, os.path.join(HERE, "..", "scripts", "e2x", "a2_gate.py"), "xm", "--run", out],
                          capture_output=True, text=True)
    assert gate.returncode == 0 and "PASS" in gate.stdout, gate.stdout + gate.stderr
    n = len(calls)
    _main(mp, io, cfgp, path, out)                           # resume: nothing is run again
    mp.undo()
    assert len(calls) == n and len(pd.read_csv(os.path.join(out, "per_run.csv"))) == len(R)


def test_eval_clips_need_allow_eval(setup, tmp_path):
    root, path, cfgp = setup
    ev = tmp_path / "eval__synth__c0.npz"
    ev.write_bytes(open(path, "rb").read())
    mp = pytest.MonkeyPatch()
    with pytest.raises(SystemExit):
        _main(mp, fake_io(load_clip(path), []), cfgp, str(ev), str(tmp_path / "o"))
    mp.undo()
