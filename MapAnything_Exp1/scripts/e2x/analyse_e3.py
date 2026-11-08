"""E3 evaluation (experiments/E3.md, frozen at tag e3-prereg before any E3 eval run). Run on the laptop:

  python scripts/e2x/analyse_e3.py --synth results/e3_eval_synth_ma --real results/e3_eval_real_ma_room \
      results/e3_eval_real_ma_e2e --out results/e3_eval_agg

E3 = the `witness` control (repair) at the frozen thresholds. Scene means over clips, paired over scenes.
Primary tests (one Holm family over all of them, alpha 0.05, practical thresholds of E2: shape 1 pp, rotation 0.25 deg,
real size 5 pp):
  SAFE   on correct hints, E3 vs using the hints as given: Delta = E[c, E3] - E[c, none]; harmful = Holm-clear and
         Delta >= threshold. Conditions: clean_pose, clean_pose_depth2 (ETH3D eval); gt_pose, gt_all, dev_pose,
         dev_all (real clips). E3 is SAFE if no test is harmful.
  HELP   on bad hints, E3 vs using them as given: helpful = Holm-clear and Delta <= -threshold. Conditions (fixed here):
         outlier_rot 15/30/90, outlier_frac 0.2/0.4, shared_heading_bias 5/10, c6_shared_rot 5, drift 2/3 (rotation);
         depth_scale 0.67/1.5, depth_one_scale 1.5 (shape); ai_depth, dev_pose_ai_depth on real clips (shape).
  BEST   E3 vs the best existing remedy for that condition, chosen on these same data in the baselines' favour (the
         lowest mean error among R0 images only, R2 filter_rgbcheck, R3 drop_lowconf): better = Holm-clear and
         Delta <= -threshold.
Decision (fixed before the run): the E3 method claim holds if E3 is SAFE, and HELP and BEST both hold on at least one
agreeing-error condition (shared_heading_bias, c6_shared_rot, drift) or one real AI-depth condition.
Descriptive: benefit kept / damage removed per remedy (safety-vs-benefit), E3 drop variant (witness_drop), ORACLE.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from exp1 import e2def_analysis as E  # noqa: E402
from exp1.e1_analysis import boot_ci, load  # noqa: E402
from exp1.e2_audit import ALPHA, THRESH, holm, signflip_p, ttest_p  # noqa: E402

SAFE_SYNTH = ["clean_pose", "clean_pose_depth2"]
SAFE_REAL = ["gt_pose", "gt_all", "dev_pose", "dev_all"]
HELP_SYNTH = {**{c: "rotation" for c in ["outlier_rot_rot_deg=15", "outlier_rot_rot_deg=30", "outlier_rot_rot_deg=90",
                                          "outlier_frac_frac=0.2", "outlier_frac_frac=0.4",
                                          "shared_heading_bias_rot_deg=5", "shared_heading_bias_rot_deg=10",
                                          "c6_shared_rot_rot_deg=5", "drift_level=2", "drift_level=3"]},
              **{c: "shape" for c in ["depth_scale_factor=0.67", "depth_scale_factor=1.5", "depth_one_scale_factor=1.5"]}}
HELP_REAL = {"ai_depth": "shape", "dev_pose_ai_depth": "shape"}
AGREEING = ("shared_heading_bias", "c6_shared_rot", "drift", "ai_depth", "dev_pose_ai_depth")
BASELINES = ["rgb_only", "filter_rgbcheck", "drop_lowconf"]


def scene_mean(ok, cond, ctrl, col):
    if ctrl == "rgb_only":
        return ok[(ok.condition == "rgb_only") & (ok.control == "none")].groupby("scene")[col].mean()
    return ok[(ok.condition == cond) & (ok.control == ctrl)].groupby("scene")[col].mean()


def row(test, data, cond, metric, d, vs):
    lo, hi = boot_ci(d.values) if len(d) > 1 else (np.nan, np.nan)
    return {"test": test, "data": data, "condition": cond, "metric": metric, "vs": vs, "delta": d.mean(), "ci_lo": lo,
            "ci_hi": hi, "n_scenes": len(d), "threshold": THRESH[E.METRIC_COL[metric]],
            "p_t": ttest_p(d.values) if len(d) > 1 else np.nan, "p_signflip": signflip_p(d.values) if len(d) > 1 else np.nan}


def tests_for(ok, data, safe, help_):
    rows = []
    for c in safe:
        if not (ok.condition == c).any():
            continue
        for m in ["shape", "rotation"] + (["real_size"] if data != "synth" else []):
            col = E.METRIC_COL[m]
            d = (scene_mean(ok, c, "witness", col) - scene_mean(ok, c, "none", col)).dropna()
            rows.append(row("SAFE", data, c, m, d, "hints as given"))
    for c, m in help_.items():
        if not (ok.condition == c).any():
            continue
        col = E.METRIC_COL[m]
        e3 = scene_mean(ok, c, "witness", col)
        rows.append(row("HELP", data, c, m, (e3 - scene_mean(ok, c, "none", col)).dropna(), "hints as given"))
        cands = {b: scene_mean(ok, c, b, col) for b in BASELINES}
        cands = {b: v for b, v in cands.items() if len(v)}
        best = min(cands, key=lambda b: cands[b].mean())
        rows.append(row("BEST", data, c, m, (e3 - cands[best]).dropna(), best))
    return rows


def descriptive(ok, data, conds):
    """benefit kept on correct hints and damage removed on bad hints, per remedy (shape and rotation)."""
    rows = []
    for col in ("rre_deg", "pts_relerr_scale"):
        rgb = scene_mean(ok, None, "rgb_only", col)
        good = "clean_pose" if data == "synth" else "gt_pose"
        clean = scene_mean(ok, good, "none", col)
        for ctrl in ["none", "witness", "witness_drop", "filter_rgbcheck", "drop_lowconf", "drop_oracle", "rgb_only"]:
            kept = ((rgb - scene_mean(ok, good, ctrl, col)) / (rgb - clean)).mean() if ctrl != "drop_oracle" else np.nan
            rem = []
            for c in conds:
                dmg = scene_mean(ok, c, "none", col)
                fix = scene_mean(ok, c, ctrl, col)
                if len(fix) and (dmg - clean).mean() > 0:
                    rem.append(((dmg - fix) / (dmg - clean).mean()).mean())
            rows.append({"label": "DESCRIPTIVE", "data": data, "metric": col, "remedy": ctrl, "benefit_kept": kept,
                         "damage_removed": float(np.nanmean(rem)) if rem else np.nan, "n_bad_conditions": len(rem)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", nargs="*", default=[])
    ap.add_argument("--real", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows, desc = [], []
    for run in a.synth:
        if os.path.exists(os.path.join(run, "per_run.csv")):
            ok = E.ok_runs(load(run)[0])
            rows += tests_for(ok, "synth", SAFE_SYNTH, HELP_SYNTH)
            desc.append(descriptive(ok, "synth", list(HELP_SYNTH)))
    for run in a.real:
        if os.path.exists(os.path.join(run, "per_run.csv")):
            ok = E.ok_runs(load(run)[0])
            name = "real_" + os.path.basename(run.rstrip("/")).split("_")[-1]
            rows += tests_for(ok, name, SAFE_REAL, HELP_REAL)
            desc.append(descriptive(ok, name, list(HELP_REAL)))
    T = pd.DataFrame(rows)
    T["p_holm"] = holm(T.p_t.fillna(1.0))
    T["clear"] = T.p_holm < ALPHA
    T["harmful"] = T.clear & (T.delta >= T.threshold)
    T["better"] = T.clear & (T.delta <= -T.threshold)
    T.to_csv(os.path.join(a.out, "e3_tests.csv"), index=False)
    if desc:
        pd.concat(desc, ignore_index=True).to_csv(os.path.join(a.out, "e3_safety_benefit.csv"), index=False)
    safe = not T[T.test == "SAFE"].harmful.any()
    key = lambda df: set(df[df.better & df.condition.str.startswith(AGREEING)].condition)
    helped, beat = key(T[T.test == "HELP"]), key(T[T.test == "BEST"])
    claim = safe and len(helped & beat) > 0
    verdict = {"safe": bool(safe), "help_on_agreeing_or_ai_depth": sorted(helped),
               "best_on_agreeing_or_ai_depth": sorted(beat), "method_claim_holds": bool(claim),
               "n_tests": int(len(T))}
    pd.Series(verdict).to_json(os.path.join(a.out, "e3_verdict.json"), indent=1)
    print(verdict)


if __name__ == "__main__":
    main()
