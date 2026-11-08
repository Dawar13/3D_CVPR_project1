"""POST HOC analyses of the E3 evaluation (2026-11-08, after the preregistered verdict; nothing here changes it).
Run from MapAnything_Exp1/:  python scripts/e2x/posthoc_e3.py

  results/e3_eval_agg[_pi3x]/posthoc_fixed_remedies.csv
      each remedy applied to EVERY condition, as a deployable system must (the preregistered BEST test compares E3
      with the best baseline chosen per condition, i.e. with knowledge of the error type). ETH3D eval, rotation:
      mean and worst over the bad-pose conditions (scene means), and the error on correct hints (clean_pose).
  results/e3_eval_agg/posthoc_support_vs_false_actions.csv
      calibration data only (SceneFun3D cal + room dev; no eval data): MapAnything E3 at z = 5 on CORRECT hints, the
      share of clips where it acts, by photo match support (pairs with a verified two-view pose, geometry referee
      rev 1, results/e2x_geometry_referee_deg_rev1). Motivates the evidence gate of the next method (E4).
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from exp1 import e2def_analysis as E  # noqa: E402
from exp1.e1_analysis import load  # noqa: E402
from exp1.e3_calibrate import GT_CONDITIONS, load_decisions, pose_actions  # noqa: E402

REMEDIES = ["none", "witness", "witness_drop", "filter_rgbcheck", "drop_lowconf", "rgb_only"]


def fixed_remedies(run):
    ok = E.ok_runs(load(run)[0])
    sm = lambda c, k: ok[(ok.condition == c) & (ok.control == k)].groupby("scene")["rre_deg"].mean()
    rgb = ok[(ok.condition == "rgb_only") & (ok.control == "none")].groupby("scene")["rre_deg"].mean()
    bad = sorted(c for c in ok.condition.unique()
                 if c not in ("rgb_only", "clean_pose", "clean_pose_depth2") and not c.startswith("depth"))
    rows = []
    for r in REMEDIES:
        vals = [(rgb if r == "rgb_only" else sm(c, r)).mean() for c in bad]
        rows.append({"label": "POST HOC", "remedy": r, "n_bad_conditions": len(bad), "mean_bad_rot_deg": np.mean(vals),
                     "worst_bad_rot_deg": max(vals),
                     "correct_hints_rot_deg": (rgb if r == "rgb_only" else sm("clean_pose", r)).mean()})
    return pd.DataFrame(rows)


def support_vs_false_actions():
    runs = ["results/e2x_e3cal_synth_ma_cal", "results/e2x_e3cal_real_ma_cal", "results/e2x_e3cal_real_ma_roomdev"]
    gt = [r for r in load_decisions(runs) if r["control"] == "witness" and r["condition"] in GT_CONDITIONS
          and "shared_deg" in r]
    act = pd.DataFrame([{"clip_id": r["clip_id"], "act": pose_actions(r, 5.0)[0] or pose_actions(r, 5.0)[1] > 0}
                        for r in gt]).groupby("clip_id").act.max().reset_index()
    R = pd.concat([pd.read_csv(f"results/e2x_geometry_referee_deg_rev1/{k}/runs.csv") for k in ("synth", "real")])
    sup = R[R.condition.isin(["clean_pose", "gt_pose"])].groupby("clip_id").n_pairs_posed.max().reset_index()
    d = act.merge(sup, on="clip_id", how="left")
    d["support"] = pd.cut(d.n_pairs_posed.fillna(0), [-1, 2, 9, 1000], labels=["0-2 pairs", "3-9 pairs", ">=10 pairs"])
    t = d.groupby("support", observed=False).agg(clips=("act", "size"), acts_on_correct_hints=("act", "sum")).reset_index()
    t["rate"] = t.acts_on_correct_hints / t.clips
    t.insert(0, "label", "POST HOC, calibration data only, MapAnything E3 z = 5")
    return t


if __name__ == "__main__":
    for run, out in (("results/e3_eval_synth_ma", "results/e3_eval_agg"), ("results/e3_eval_synth_px", "results/e3_eval_agg_pi3x")):
        if os.path.exists(os.path.join(run, "per_run.csv")):
            df = fixed_remedies(run)
            df.to_csv(os.path.join(out, "posthoc_fixed_remedies.csv"), index=False)
            print(out); print(df.round(2).to_string(index=False))
    s = support_vs_false_actions()
    s.to_csv("results/e3_eval_agg/posthoc_support_vs_false_actions.csv", index=False)
    print(s.round(3).to_string(index=False))
