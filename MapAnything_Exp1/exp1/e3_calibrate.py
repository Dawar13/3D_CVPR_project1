"""E3 calibration (experiments/E2X.md section 5): choose the E3 cut-offs from the decisions logged on 'cal' and dev
clips, by the rule fixed before any calibration run. No model is run: every decision is recomputed from the logged
fit values and witness uncertainties (run_exp1.run_e3 -> e3_decisions.jsonl).

Rule, per model:
  - e3_z: the smallest value in Z_GRID whose false-action rate on GROUND-TRUTH hints is <= 5% of clips for any group
    action (shared / mount / drift repair) AND <= 2% of views (i > 0) for single-view drops.
  - e3_tau_depth_shared / e3_tau_depth_view: the smallest values in their grids with <= 5% of clips rescaled and
    <= 2% of depth views dropped, on ground-truth depth.
  - If no value of a grid meets the rule, that part of E3 cannot be calibrated for this model: it is reported as
    NOT CALIBRATED (an E3 stop signal for that part), never loosened.
Floors stay at exp1.e3.DEFAULTS. Detection rates on corrupted hints at the chosen values are reported, not used.

  python -m exp1.e3_calibrate --runs results/e2x_e3cal_*_ma_* --out results/e2x_e3cal_agg/mapanything
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from .e3 import DEFAULTS

Z_GRID = [2.0, 2.5, 3.0, 4.0, 5.0]
TAU_SHARED_GRID = [0.05, 0.10, 0.15, 0.20]
TAU_VIEW_GRID = [0.10, 0.15, 0.25]
GT_CONDITIONS = {"clean_pose", "clean_pose_depth2", "gt_pose", "gt_all"}
GT_DEPTH_CONDITIONS = {"clean_pose_depth2", "gt_all"}
MAX_GROUP, MAX_VIEW = 0.05, 0.02


def load_decisions(run_dirs):
    rows = []
    for d in run_dirs:
        p = os.path.join(d, "e3_decisions.jsonl")
        if not os.path.exists(p):
            continue
        for line in open(p):
            r = json.loads(line)
            parts = r["run_key"].split("|")
            r.update(run_dir=d, clip_id=parts[0], condition=parts[1], seed=parts[2], control=parts[3])
            rows.append(r)
    return rows


def pose_actions(r, z, floors=DEFAULTS):
    """(group action?, number of views dropped, number of views considered) at multiplier z."""
    sig = r["sigma_group_deg"]
    th = {"shared": z * np.hypot(sig["shared"], floors["e3_floor_shared_deg"]),
          "mount": z * np.hypot(sig["mount"], floors["e3_floor_shared_deg"]),
          "drift": z * np.hypot(sig["drift"], floors["e3_floor_drift_deg"])}
    group = (r["shared_deg"] > th["shared"]) or (r["mount_deg"] > th["mount"]) or (r["drift_total_deg"] > th["drift"])
    res = np.asarray(r["rot_res_deg"])[1:]
    thv = z * np.hypot(np.asarray(r["sigma_view_deg"])[1:], floors["e3_floor_view_deg"])
    return bool(group), int((res > thv).sum()), len(res)


def depth_actions(r, tau_shared, tau_view):
    """(rescale?, depth views dropped, depth views considered) for one decision with a depth check."""
    k = r["depth_k"]
    ratios = np.array([np.nan if x is None else x for x in r["depth_ratios"]], float)
    ok = np.isfinite(ratios)
    drop = np.abs(np.log(ratios[ok] / k)) > tau_view
    return bool(abs(np.log(k)) > tau_shared), int(drop.sum()), int(ok.sum())


def calibrate(rows):
    """Returns (table of false-action rates per setting, chosen settings dict)."""
    gt = [r for r in rows if r["control"] == "witness" and r["condition"] in GT_CONDITIONS and "shared_deg" in r]
    gtd = [r for r in rows if r["control"] == "witness" and r["condition"] in GT_DEPTH_CONDITIONS
           and r.get("depth_scale_from") in ("pose", "images")]
    table, chosen = [], {}
    for z in Z_GRID:
        acts = [pose_actions(r, z) for r in gt]
        g = np.mean([a[0] for a in acts]) if acts else np.nan
        v = sum(a[1] for a in acts) / max(sum(a[2] for a in acts), 1) if acts else np.nan
        table.append({"part": "pose", "setting": f"e3_z={z}", "n": len(acts), "group_action_rate": g,
                      "view_drop_rate": v, "meets_rule": bool(acts) and g <= MAX_GROUP and v <= MAX_VIEW})
    ok = [t for t in table if t["part"] == "pose" and t["meets_rule"]]
    chosen["e3_z"] = float(ok[0]["setting"].split("=")[1]) if ok else "NOT CALIBRATED"
    for ts in TAU_SHARED_GRID:
        acts = [depth_actions(r, ts, TAU_VIEW_GRID[-1]) for r in gtd]
        g = np.mean([a[0] for a in acts]) if acts else np.nan
        table.append({"part": "depth_shared", "setting": f"e3_tau_depth_shared={ts}", "n": len(acts),
                      "group_action_rate": g, "view_drop_rate": np.nan, "meets_rule": bool(acts) and g <= MAX_GROUP})
    for tv in TAU_VIEW_GRID:
        acts = [depth_actions(r, TAU_SHARED_GRID[-1], tv) for r in gtd]
        v = sum(a[1] for a in acts) / max(sum(a[2] for a in acts), 1) if acts else np.nan
        table.append({"part": "depth_view", "setting": f"e3_tau_depth_view={tv}", "n": len(acts),
                      "group_action_rate": np.nan, "view_drop_rate": v, "meets_rule": bool(acts) and v <= MAX_VIEW})
    for part, key in (("depth_shared", "e3_tau_depth_shared"), ("depth_view", "e3_tau_depth_view")):
        ok = [t for t in table if t["part"] == part and t["meets_rule"]]
        chosen[key] = float(ok[0]["setting"].split("=")[1]) if ok else "NOT CALIBRATED"
    chosen["n_gt_pose_decisions"], chosen["n_gt_depth_decisions"] = len(gt), len(gtd)
    return pd.DataFrame(table), chosen


def detection(rows, chosen):
    """Descriptive only: how often E3 acts on each corrupted condition at the chosen settings."""
    out = []
    z = chosen.get("e3_z")
    for r in rows:
        if r["control"] != "witness" or r["condition"] in GT_CONDITIONS or not isinstance(z, float):
            continue
        row = {"condition": r["condition"], "clip_id": r["clip_id"]}
        if "shared_deg" in r:
            g, nd, _ = pose_actions(r, z)
            row.update(group_action=g, views_dropped=nd)
        if r.get("depth_scale_from") in ("pose", "images") and isinstance(chosen.get("e3_tau_depth_shared"), float):
            ds, dd, _ = depth_actions(r, chosen["e3_tau_depth_shared"], chosen.get("e3_tau_depth_view", 0.25)
                                      if isinstance(chosen.get("e3_tau_depth_view"), float) else 0.25)
            row.update(depth_rescaled=ds, depth_views_dropped=dd)
        out.append(row)
    df = pd.DataFrame(out)
    return df.groupby("condition").mean(numeric_only=True).reset_index() if len(df) else df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run folders (globs allowed) with e3_decisions.jsonl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dirs = sorted({d for pat in a.runs for d in glob.glob(pat)})
    if any(os.path.basename(os.path.normpath(d)).endswith("_eval") for d in dirs):
        raise SystemExit("calibration uses cal and dev runs only")
    rows = load_decisions(dirs)
    if not rows:
        raise SystemExit(f"no e3_decisions.jsonl in {dirs}")
    bad = sorted({r["clip_id"] for r in rows if r["clip_id"].startswith("eval__")})
    if bad:
        raise SystemExit(f"eval clips in the calibration data: {bad[:3]}")
    os.makedirs(a.out, exist_ok=True)
    table, chosen = calibrate(rows)
    table.to_csv(os.path.join(a.out, "false_action_rates.csv"), index=False)
    json.dump(chosen, open(os.path.join(a.out, "chosen.json"), "w"), indent=1)
    detection(rows, chosen).to_csv(os.path.join(a.out, "detection_rates.csv"), index=False)
    print(table.to_string(index=False))
    print("chosen:", json.dumps(chosen))


if __name__ == "__main__":
    main()
