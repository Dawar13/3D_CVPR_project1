"""Shared-GPU helper for scripts/session_queue_e2x.sh (job scripts only; the runner exp1/run_exp1.py is unchanged).

run_exp1 records a run that raised an error as a FAILED row and skips that run on resume. On a shared GPU a run can
fail only because another experiment held the memory at that moment (CUDA out-of-memory). The runner is
deterministic (E0: bit-identical repeats; tests/test_shard.py: shards = one process), so running such a run again with
the same code, settings and clip gives exactly what it would have given the first time.

  python scripts/e2x/retry_oom.py reset <out>
      For every run folder of the job (<out>/_shards/*/run, or <out> itself when it has no shards): if a clip has at
      least one failure whose error is a CUDA out-of-memory error, that clip is reset from its first failed run on:
      every run of the clip from that run onwards (in the order the runner wrote them) is removed from per_run.csv,
      per_view.csv and e3_decisions.jsonl, and the clip's failures from failures.jsonl. All later runs are included
      because a failed cached reference makes later runs of the clip fail with other errors, and because the runner
      then writes them again in its own order, so the files end up byte-identical to a run without failures
      (scripts/e2x/test_retry_oom.py). The removed lines are kept in <run>/_retried/<UTC time>/, and one line per
      reset is appended to <out>/oom_retries.jsonl (fetched with the results). Prints the number of runs reset.
  python scripts/e2x/retry_oom.py dedupe <out>
      A rerun appends its corruption-manifest entry again (the same entry, it is deterministic). Removes exact
      duplicates from every corruption_manifest.json of the job, keeping the first. Prints the number removed.
Failures in clips without an out-of-memory error are left as they are: they are reported, not retried.
"""

import csv
import glob
import json
import os
import re
import sys
import time

OOM = re.compile(r"out of memory|OutOfMemoryError|ALLOC_FAILED|NOT_ENOUGH_WORKSPACE", re.I)


def run_dirs(out):
    d = sorted(glob.glob(os.path.join(out, "_shards", "*", "run")))
    return d if d else ([out] if os.path.exists(os.path.join(out, "per_run.csv")) else [])


def _base(run_key):
    """clip|condition|seed: the runner's unit of work (rows add |control|subset, per-view rows |v<i>)."""
    return "|".join(run_key.split("|")[:3])


def _belongs(run_key, keys):
    return _base(run_key) in keys


def _filter_csv(path, keys, backup):
    if not os.path.exists(path):
        return 0
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return 0
    head, body = rows[0], rows[1:]
    ki = head.index("run_key")
    drop = [r for r in body if _belongs(r[ki], keys)]
    if not drop:
        return 0
    with open(os.path.join(backup, os.path.basename(path)), "w", newline="") as f:
        csv.writer(f).writerows([head] + drop)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows([head] + [r for r in body if not _belongs(r[ki], keys)])
    return len(drop)


def _filter_jsonl(path, keys, field, backup):
    if not os.path.exists(path):
        return 0
    lines = open(path).read().splitlines()
    drop = [x for x in lines if _belongs(json.loads(x)[field], keys)]
    if not drop:
        return 0
    with open(os.path.join(backup, os.path.basename(path)), "w") as f:
        f.writelines(x + "\n" for x in drop)
    keep = [x for x in lines if not _belongs(json.loads(x)[field], keys)]
    if keep:
        with open(path, "w") as f:
            f.writelines(x + "\n" for x in keep)
    else:
        os.remove(path)
    return len(drop)


def reset(out):
    total = 0
    for r in run_dirs(out):
        fp = os.path.join(r, "failures.jsonl")
        if not os.path.exists(fp):
            continue
        fails = [json.loads(x) for x in open(fp).read().splitlines()]
        oom_clips = {f["key"].split("|")[0] for f in fails if OOM.search(f.get("error", ""))}
        failed = {f["key"] for f in fails if f["key"].split("|")[0] in oom_clips}
        if not failed:
            continue
        with open(os.path.join(r, "per_run.csv"), newline="") as f:
            order = [_base(x["run_key"]) for x in csv.DictReader(f)]
        keys = set(failed)
        for c in oom_clips:   # every run of the clip from its first failed run on
            mine = [b for b in order if b.split("|")[0] == c]
            first = next((i for i, b in enumerate(mine) if b in failed), len(mine))
            keys |= set(mine[first:])
        keys = sorted(keys)
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        backup = os.path.join(r, "_retried", stamp)
        os.makedirs(backup, exist_ok=True)
        n = {"per_run": _filter_csv(os.path.join(r, "per_run.csv"), keys, backup),
             "per_view": _filter_csv(os.path.join(r, "per_view.csv"), keys, backup),
             "e3_decisions": _filter_jsonl(os.path.join(r, "e3_decisions.jsonl"), keys, "run_key", backup),
             "failures": _filter_jsonl(fp, keys, "key", backup)}
        errs = sorted({f["error"][:200] for f in fails if f["key"] in failed})
        with open(os.path.join(out, "oom_retries.jsonl"), "a") as f:
            f.write(json.dumps({"time_utc": stamp, "run_dir": os.path.relpath(r, out), "runs_reset": keys, "runs_failed": sorted(failed),
                                "lines_removed": n, "errors": errs, "backup": os.path.relpath(backup, out)}) + "\n")
        total += len(keys)
    return total


def dedupe(out):
    removed = 0
    for r in run_dirs(out):
        p = os.path.join(r, "corruption_manifest.json")
        if not os.path.exists(p):
            continue
        man = json.load(open(p))
        seen, keep = set(), []
        for e in man:
            s = json.dumps(e, sort_keys=True)
            if s not in seen:
                seen.add(s)
                keep.append(e)
        if len(keep) < len(man):
            removed += len(man) - len(keep)
            json.dump(keep, open(p, "w"), indent=1)   # as run_exp1 writes it
    return removed


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("reset", "dedupe"):
        raise SystemExit("usage: retry_oom.py reset|dedupe <out>")
    print({"reset": reset, "dedupe": dedupe}[sys.argv[1]](sys.argv[2]))
