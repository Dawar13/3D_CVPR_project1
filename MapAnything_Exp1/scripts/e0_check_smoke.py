"""E0 Step 7: read smoke-run outputs and print the E0 check table.

  python scripts/e0_check_smoke.py --runs results/smoke results/smoke_ctrl

Reads per_run.csv, failures.jsonl, corruption_manifest.json and run_info.json of each
run. Prints facts plus a mechanical verdict per check; "clearly better" remains a
human judgement, so the numbers are always shown next to the verdict.
"""

import argparse
import json
import os

import pandas as pd

REFS = ["rgb_only", "rgb_K", "clean_pose", "clean_pose_noK", "zero_noise"]
METRICS = ["pts_relerr_scale", "rre_deg", "depth_absrel_scale", "pts_relerr_metric",
           "depth_absrel_metric", "rte_deg", "mean_conf"]


def load(run):
    df = pd.read_csv(os.path.join(run, "per_run.csv"))
    fp = os.path.join(run, "failures.jsonl")
    fails = [json.loads(l) for l in open(fp)] if os.path.exists(fp) else []
    mp = os.path.join(run, "corruption_manifest.json")
    man = json.load(open(mp)) if os.path.exists(mp) else []
    info = json.load(open(os.path.join(run, "run_info.json")))
    return df, fails, man, info


def verdict(ok):
    return "PASS" if ok else "FAIL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    a = ap.parse_args()
    dfs, all_fails, man = [], [], []
    for run in a.runs:
        df, fails, m, info = load(run)
        print(f"== {run}: {len(df)} rows, {len(fails)} failures; commit "
              f"{info['mapanything_commit'][:10]} ckpt {info['checkpoint']} torch {info['torch']} "
              f"cuda {info['cuda']} gpu {info['gpu']}")
        dfs.append(df.assign(run=run))
        all_fails += fails
        man += m
    df = pd.concat(dfs, ignore_index=True)

    n_bad = int((df.status != "ok").sum())
    print(f"\n[{verdict(n_bad == 0 and not all_fails)}] all rows ok ({n_bad} not ok), "
          f"failures.jsonl lines: {len(all_fails)}")
    for f in all_fails[:3]:
        print(f"    {f['key']}: {f['error']}")
    resized = any("resized" in f["error"] for f in all_fails)
    print(f"[{verdict(not resized)}] no resize error (metrics.evaluate raises if pred size != GT)")

    zn = [e for e in man if e["condition"] == "zero_noise"]
    if zn and "max_abs_diff_vs_clean_pts3d" in zn[-1]:
        d, L = zn[-1]["max_abs_diff_vs_clean_pts3d"], zn[-1]["scene_length_L_m"]
        print(f"[{verdict(d < 1e-3 * L)}] zero_noise vs clean_pose: max|d pts3d| = {d:.3g} m "
              f"= {d / L:.3g} L (threshold 1e-3 L, L = {L:.3g} m)")
    else:
        print("[FAIL] zero_noise vs clean_pose diff not found in manifest")

    base = df[(df.status == "ok") & (df.control == "none") & (df.view_subset == "all")]
    tab = base[base.condition.isin(REFS)].groupby(["condition"])[METRICS].mean()
    tab = tab.reindex([c for c in REFS if c in tab.index])
    print("\nreference conditions (control=none, all views; lower is better except mean_conf):")
    print(tab.to_string(float_format=lambda x: f"{x:.4f}"))
    for cond, ref in (("clean_pose", "rgb_only"), ("rgb_K", "rgb_only")):
        if cond in tab.index and ref in tab.index:
            rel = {m: (tab.loc[cond, m] - tab.loc[ref, m]) / tab.loc[ref, m]
                   for m in ("pts_relerr_scale", "rre_deg")}
            better = all(r < 0 for r in rel.values())
            print(f"[{verdict(better) if cond == 'clean_pose' else 'INFO'}] {cond} vs {ref}: "
                  + ", ".join(f"{m} {r:+.1%}" for m, r in rel.items()))

    ctrl = df[df.control.isin(["drop_oracle", "filter_rgbcheck"]) & (df.view_subset == "all")]
    print(f"\n[{verdict(set(ctrl.control) == {'drop_oracle', 'filter_rgbcheck'} and (ctrl.status == 'ok').all())}]"
          " controls ran:")
    if len(ctrl):
        print(ctrl[["condition", "seed", "control", "status", "n_pose_dropped",
                    "dropped_are_corrupted", "pts_relerr_scale", "rre_deg"]].to_string(index=False))

    print(f"\nruntime per 10-view inference: median {base.runtime_s.median():.3f} s "
          f"(min {base.runtime_s.min():.3f}, max {base.runtime_s.max():.3f}); "
          f"peak memory max {base.peak_mem_gb.max():.2f} GB")


if __name__ == "__main__":
    main()
