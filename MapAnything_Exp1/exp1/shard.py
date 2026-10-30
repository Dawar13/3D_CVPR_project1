"""Run one run_exp1 job as N parallel processes on the same GPU, then merge into one run folder.

  python -m exp1.shard run --n 4 --workdir ~/3D-CVPR_Exp/MapAnything_Exp1 -- \
      --config configs/e2c_full.json --clips 'data/clips_e2/eval__*.npz' --out results/e2c_eval --save_extras

The runner treats every clip on its own (its caches are per clip, its random draws depend on
(seed, clip, condition) only), so clips are split round-robin over N shards. Each shard is an
ordinary, resumable run_exp1 invocation on <out>/_shards/<k>/ with a folder of links to its clips
(same file names, so the same clip ids). merge() then writes <out>/ exactly as one process would:
rows, manifest entries and failures in sorted clip order, all extras, and run_info.json with the
full clip list. The shard folders stay the source of truth: rerunning `run` resumes them and
merges again. --workdir is the code tree that runs (e.g. a frozen tag); this module only launches
and merges, it never imports the runner.
"""

import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import time


def split(clips, n):
    clips = sorted(clips)
    return [clips[k::n] for k in range(n) if clips[k::n]]


def _link(src, dst):
    if os.path.lexists(dst):
        return
    try:
        os.symlink(os.path.abspath(src), dst)
    except OSError:  # no symlink rights (Windows tests)
        shutil.copy2(src, dst)


def prepare(out, clips, n):
    """Make <out>/_shards/<k>/clips/ with links to that shard's clips; return [(clip_glob, shard_out)]."""
    jobs = []
    for k, part in enumerate(split(clips, n)):
        cdir = os.path.join(out, "_shards", str(k), "clips")
        os.makedirs(cdir, exist_ok=True)
        for c in part:
            _link(c, os.path.join(cdir, os.path.basename(c)))
        jobs.append((os.path.join(cdir, "*.npz"), os.path.join(out, "_shards", str(k), "run")))
    return jobs


def _clip_of_key(key):
    return key.split("|")[0]


def _merge_csv(srcs, dst, order):
    header, rows = None, {}
    for s in srcs:
        if not os.path.exists(s):
            continue
        with open(s, newline="") as f:
            r = csv.reader(f)
            h = next(r)
            assert header is None or h == header, f"header differs in {s}"
            header = h
            ci = h.index("clip_id")
            for row in r:
                rows.setdefault(row[ci], []).append(row)
    if header is None:
        return 0
    with open(dst, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for c in order:
            w.writerows(rows.get(c, []))
    return sum(len(v) for v in rows.values())


def merge(out, clips):
    """Merge <out>/_shards/*/run into <out>. `clips` = the full clip list (paths as given to the runner)."""
    runs = sorted(glob.glob(os.path.join(out, "_shards", "*", "run")), key=lambda p: int(p.split(os.sep)[-2]))
    order = [os.path.splitext(os.path.basename(c))[0] for c in sorted(clips)]
    n = {}
    for name in ("per_run.csv", "per_view.csv"):
        n[name] = _merge_csv([os.path.join(r, name) for r in runs], os.path.join(out, name), order)
    man = {}
    for r in runs:
        p = os.path.join(r, "corruption_manifest.json")
        for e in (json.load(open(p)) if os.path.exists(p) else []):
            man.setdefault(e["clip_id"], []).append(e)
    json.dump([e for c in order for e in man.get(c, [])], open(os.path.join(out, "corruption_manifest.json"), "w"),
              indent=1)
    fails = {}
    for r in runs:
        p = os.path.join(r, "failures.jsonl")
        for line in (open(p).read().splitlines() if os.path.exists(p) else []):
            fails.setdefault(_clip_of_key(json.loads(line)["key"]), []).append(line)
    if fails:
        with open(os.path.join(out, "failures.jsonl"), "w") as f:
            f.writelines(line + "\n" for c in order for line in fails.get(c, []))
    elif os.path.exists(os.path.join(out, "failures.jsonl")):
        os.remove(os.path.join(out, "failures.jsonl"))
    infos = [json.load(open(os.path.join(r, "run_info.json"))) for r in runs]
    for i in infos[1:]:
        assert {k: v for k, v in i.items() if k != "clips"} == {k: v for k, v in infos[0].items() if k != "clips"}, \
            "shards ran with different settings"
    info = dict(infos[0], clips=sorted(clips), shards=len(runs))
    json.dump(info, open(os.path.join(out, "run_info.json"), "w"), indent=1)
    ex = os.path.join(out, "extras")
    for r in runs:
        for p in glob.glob(os.path.join(r, "extras", "*.npz")):
            os.makedirs(ex, exist_ok=True)
            os.replace(p, os.path.join(ex, os.path.basename(p)))
    with open(os.path.join(out, "gpu_log.txt"), "w") as f:
        for k, r in enumerate(runs):
            p = os.path.join(r, "gpu_log.txt")
            if os.path.exists(p):
                f.writelines(f"shard={k} {line}" for line in open(p))
    return n


def _arg(argv, flag):
    i = argv.index(flag)
    return argv[i + 1], argv[:i] + argv[i + 2:]


def run(n, workdir, runner_argv, stagger=20):
    """Launch the shards, wait, merge. Returns the merged row counts; raises if any shard failed."""
    pattern, rest = _arg(runner_argv, "--clips")
    out, rest = _arg(rest, "--out")
    clips = sorted(glob.glob(os.path.join(workdir, pattern)))
    if not clips:
        raise SystemExit(f"no clips match {pattern} in {workdir}")
    out_abs = os.path.join(workdir, out)
    jobs = prepare(out_abs, clips, n)
    env = dict(os.environ, OMP_NUM_THREADS=str(max(1, (os.cpu_count() or n) // len(jobs))))
    procs = []
    for k, (cglob, sout) in enumerate(jobs):
        log = open(os.path.join(out_abs, "_shards", str(k), "log.txt"), "a")
        cmd = [sys.executable, "-m", "exp1.run_exp1", *rest, "--clips", cglob, "--out", sout]
        procs.append((subprocess.Popen(cmd, cwd=workdir, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        time.sleep(stagger if k < len(jobs) - 1 else 0)  # stagger model loading
    codes = [p.wait() for p, _ in procs]
    for _, log in procs:
        log.close()
    if any(codes):
        raise SystemExit(f"shard exit codes {codes}; see {out_abs}/_shards/*/log.txt")
    rel = [os.path.relpath(c, workdir).replace(os.sep, "/") for c in clips]
    return merge(out_abs, rel)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--n", type=int, required=True)
    r.add_argument("--workdir", default=".")
    m = sub.add_parser("merge")
    m.add_argument("--out", required=True)
    m.add_argument("--clips", required=True, help="the glob given to the runner, relative to --workdir")
    m.add_argument("--workdir", default=".")
    argv = sys.argv[1:]
    runner_argv = []
    if "--" in argv:
        i = argv.index("--")
        argv, runner_argv = argv[:i], argv[i + 1:]
    a = ap.parse_args(argv)
    if a.cmd == "run":
        print(run(a.n, a.workdir, runner_argv))
    else:
        clips = sorted(glob.glob(os.path.join(a.workdir, a.clips)))
        rel = [os.path.relpath(c, a.workdir).replace(os.sep, "/") for c in clips]
        print(merge(os.path.join(a.workdir, a.out), rel))


if __name__ == "__main__":
    main()
