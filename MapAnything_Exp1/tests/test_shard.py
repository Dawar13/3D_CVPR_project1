"""exp1.shard: N parallel runner processes + merge must give exactly the files of one process.

Fake model (as in test_runner_dryrun), three synthetic clips, a mix of E2-C conditions (controls,
matched jitter, partial poses, median probe, lens corruption) with --save_extras. Two shards, so
the merge has to interleave clips c0, c2 (shard 0) and c1 (shard 1) back into sorted order.
"""

import glob
import json
import os
import sys

import numpy as np
import pytest

pytest.importorskip("torch")

from exp1 import model_io, run_exp1, shard  # noqa: E402
from exp1.datasets import load_clip, prepare_eth3d  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
import test_e0_clip_pipeline as synth  # noqa: E402
from test_runner_dryrun import fake_run_factory  # noqa: E402

ONLY = ["rgb_only", "clean_pose", "c1_rot_rot_deg=15", "c6_indep_rot_rot_deg=5", "c6_drift_jitter_level=2",
        "c7_bad_frac=0.6", "c4m_trans_trans_frac=1", "c1_focal_factor=1.1"]
FILES = ("per_run.csv", "per_view.csv", "corruption_manifest.json")


def _run(mp, clips_glob, out):
    current = {}

    def load(path):
        clip = load_clip(path)
        current["run"] = fake_run_factory(clip)
        return clip

    mp.setattr(run_exp1, "load_clip", load)
    mp.setattr(model_io, "load_model", lambda ckpt, device: "fake")
    mp.setattr(model_io, "run", lambda model, views, device: current["run"](model, views, device))
    mp.setattr(model_io, "set_pose_scale_statistic", lambda stat: None)
    mp.setattr(sys, "argv", ["run_exp1", "--config", os.path.join(ROOT, "configs", "e2c_full.json"),
                             "--clips", clips_glob, "--out", out, "--device", "cpu", "--save_extras",
                             "--only", *ONLY])
    run_exp1.main()


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("shard"))
    synth.write_scene(root, n=12)
    cdir = os.path.join(root, "clips")
    prepare_eth3d(root, cdir, n_views=4, clips_per_scene=3, stride=1, tw=170, th=100, scenes=["synth"],
                  non_overlapping=True)
    pattern = os.path.join(cdir, "*.npz")
    clips = sorted(glob.glob(pattern))
    assert len(clips) == 3
    mp = pytest.MonkeyPatch()
    single = os.path.join(root, "single")
    _run(mp, pattern, single)
    sharded = os.path.join(root, "sharded")
    jobs = shard.prepare(sharded, clips, 2)
    for cglob, sout in jobs:
        _run(mp, cglob, sout)
    mp.undo()
    n = shard.merge(sharded, clips)
    return single, sharded, jobs, n


def test_split_round_robin_and_links(runs):
    _, sharded, jobs, _ = runs
    names = [sorted(os.path.basename(p) for p in glob.glob(g)) for g, _ in jobs]
    assert names == [["dev__synth__c0.npz", "dev__synth__c2.npz"], ["dev__synth__c1.npz"]]
    assert shard.split(["b", "a", "c"], 5) == [["a"], ["b"], ["c"]]


def test_merged_files_identical_to_one_process(runs):
    single, sharded, _, n = runs
    assert not os.path.exists(os.path.join(single, "failures.jsonl"))
    assert not os.path.exists(os.path.join(sharded, "failures.jsonl"))
    for f in FILES:
        assert open(os.path.join(single, f), "rb").read() == open(os.path.join(sharded, f), "rb").read(), f
    assert n["per_run.csv"] > 0 and n["per_view.csv"] > n["per_run.csv"]


def test_run_info_and_extras_match(runs):
    single, sharded, _, _ = runs
    a = json.load(open(os.path.join(single, "run_info.json")))
    b = json.load(open(os.path.join(sharded, "run_info.json")))
    assert b.pop("shards") == 2 and a == b
    ea, eb = sorted(os.listdir(os.path.join(single, "extras"))), sorted(os.listdir(os.path.join(sharded, "extras")))
    assert ea == eb and len(ea) > 0
    for f in ea:
        za, zb = np.load(os.path.join(single, "extras", f)), np.load(os.path.join(sharded, "extras", f))
        assert za.files == zb.files
        for k in za.files:
            assert np.array_equal(za[k], zb[k], equal_nan=True), (f, k)
    assert open(os.path.join(sharded, "gpu_log.txt")).read().count("shard=") == 2


STUB = '''import argparse, csv, glob, json, os
ap = argparse.ArgumentParser()
ap.add_argument("--config"); ap.add_argument("--clips"); ap.add_argument("--out"); ap.add_argument("--flag", action="store_true")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
clips = sorted(glob.glob(a.clips))
json.dump({"config": a.config, "clips": clips, "flag": a.flag, "cwd": os.getcwd()}, open(os.path.join(a.out, "run_info.json"), "w"))
with open(os.path.join(a.out, "per_run.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["run_key", "clip_id", "view_subset"])
    for c in clips:
        cid = os.path.splitext(os.path.basename(c))[0]
        w.writerow([cid + "|x|0|none|all", cid, "all"])
'''


def test_run_launches_processes_in_workdir_and_merges(tmp_path):
    """shard.run with a stand-in runner: arguments, working folder, clip links and the merge."""
    work = tmp_path / "tree"
    (work / "exp1").mkdir(parents=True)
    (work / "exp1" / "__init__.py").write_text("")
    (work / "exp1" / "run_exp1.py").write_text(STUB)
    (work / "data").mkdir()
    for k in range(5):
        (work / "data" / f"eval__s{k}__c0.npz").write_bytes(b"x")
    n = shard.run(3, str(work), ["--config", "c.json", "--clips", "data/eval__*.npz", "--out", "res/x", "--flag"], stagger=0)
    assert n["per_run.csv"] == 5
    out = work / "res" / "x"
    info = json.load(open(out / "run_info.json"))
    assert info["clips"] == [f"data/eval__s{k}__c0.npz" for k in range(5)] and info["shards"] == 3
    assert info["flag"] and info["config"] == "c.json" and os.path.samefile(info["cwd"], work)
    ids = [line.split(",")[1] for line in open(out / "per_run.csv").read().splitlines()[1:]]
    assert ids == [f"eval__s{k}__c0" for k in range(5)]


def test_merge_again_is_stable(runs):
    _, sharded, _, _ = runs
    before = {f: open(os.path.join(sharded, f), "rb").read() for f in FILES}
    clips = json.load(open(os.path.join(sharded, "run_info.json")))["clips"]
    shard.merge(sharded, clips)
    assert {f: open(os.path.join(sharded, f), "rb").read() for f in FILES} == before
    assert len(os.listdir(os.path.join(sharded, "extras"))) > 0  # extras already moved stay merged
