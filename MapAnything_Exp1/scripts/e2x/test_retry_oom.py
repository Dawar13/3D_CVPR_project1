"""scripts/e2x/retry_oom.py: runs that failed with CUDA out-of-memory, reset and run again, must give exactly the
files of a run where nothing failed (CPU, fake model as in tests/test_shard.py).

  python -m pytest scripts/e2x/test_retry_oom.py -q

Two shards over three synthetic clips. In the first attempt clip c1's first inference (rgb_only, a cached reference:
later runs of c1 then fail with other errors) and one later inference of clip c0 raise CUDA out-of-memory. reset +
rerun + dedupe + merge must reproduce the one-process run byte for byte, and a non-memory failure must stay as it is.
"""

import glob
import importlib.util
import json
import os
import sys

import pytest

torch = pytest.importorskip("torch")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from exp1 import model_io, run_exp1, shard  # noqa: E402
from exp1.datasets import load_clip, prepare_eth3d  # noqa: E402
import test_e0_clip_pipeline as synth  # noqa: E402
from test_runner_dryrun import fake_run_factory  # noqa: E402

_spec = importlib.util.spec_from_file_location("retry_oom", os.path.join(HERE, "retry_oom.py"))
retry_oom = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(retry_oom)

ONLY = ["rgb_only", "clean_pose", "c1_rot_rot_deg=15", "c6_drift_jitter_level=2", "c1_focal_factor=1.1"]
FILES = ("per_run.csv", "per_view.csv", "corruption_manifest.json")


def _run(mp, clips_glob, out, fail=None, error=None):
    """fail: {clip_id: set of inference numbers (0-based, per clip) that raise}."""
    current = {}

    def load(path):
        clip = load_clip(path)
        current["run"], current["cid"], current["k"] = fake_run_factory(clip), clip["clip_id"], 0
        return clip

    def run(model, views, device):
        k = current["k"]
        current["k"] += 1
        if fail and k in fail.get(current["cid"], ()):
            if error:
                raise error("bad input")
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return current["run"](model, views, device)

    mp.setattr(run_exp1, "load_clip", load)
    mp.setattr(model_io, "load_model", lambda ckpt, device: "fake")
    mp.setattr(model_io, "run", run)
    mp.setattr(model_io, "set_pose_scale_statistic", lambda stat: None)
    mp.setattr(sys, "argv", ["run_exp1", "--config", os.path.join(ROOT, "configs", "e2c_full.json"),
                             "--clips", clips_glob, "--out", out, "--device", "cpu", "--save_extras",
                             "--only", *ONLY])
    run_exp1.main()


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("oom"))
    synth.write_scene(root, n=12)
    cdir = os.path.join(root, "clips")
    prepare_eth3d(root, cdir, n_views=4, clips_per_scene=3, stride=1, tw=170, th=100, scenes=["synth"],
                  non_overlapping=True)
    return root, sorted(glob.glob(os.path.join(cdir, "*.npz")))


def test_reset_rerun_equals_clean_run(clips):
    root, paths = clips
    mp = pytest.MonkeyPatch()
    single = os.path.join(root, "single")
    _run(mp, os.path.join(os.path.dirname(paths[0]), "*.npz"), single)
    out = os.path.join(root, "sharded")
    jobs = shard.prepare(out, paths, 2)
    fail = {"dev__synth__c1": {0}, "dev__synth__c0": {4}}
    for cglob, sout in jobs:
        _run(mp, cglob, sout, fail)
    shard.merge(out, paths)
    fails = open(os.path.join(out, "failures.jsonl")).read().splitlines()
    assert any("OutOfMemoryError" in x for x in fails) and any("dev__synth__c1" in x for x in fails)
    n = retry_oom.reset(out)
    assert n >= 2
    assert retry_oom.reset(out) == 0   # nothing left to reset
    for cglob, sout in jobs:           # the same shards again: resume
        _run(mp, cglob, sout)
    assert retry_oom.dedupe(out) >= 1
    shard.merge(out, paths)
    mp.undo()
    assert not os.path.exists(os.path.join(out, "failures.jsonl"))
    for f in FILES:
        assert open(os.path.join(single, f), "rb").read() == open(os.path.join(out, f), "rb").read(), f
    log = [json.loads(x) for x in open(os.path.join(out, "oom_retries.jsonl")).read().splitlines()]
    assert sum(len(x["runs_reset"]) for x in log) == n
    assert all(os.path.isdir(os.path.join(out, x["backup"])) for x in log)


def test_other_errors_are_not_reset(clips):
    root, paths = clips
    mp = pytest.MonkeyPatch()
    out = os.path.join(root, "other_error")
    _run(mp, paths[0], out, {"dev__synth__c0": {3}}, ValueError)
    mp.undo()
    before = {f: open(os.path.join(out, f), "rb").read() for f in ("per_run.csv", "failures.jsonl")}
    assert retry_oom.reset(out) == 0
    assert {f: open(os.path.join(out, f), "rb").read() for f in before} == before
    assert not os.path.exists(os.path.join(out, "oom_retries.jsonl"))
