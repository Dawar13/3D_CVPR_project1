"""E4-A evaluation (experiments/E4.md, frozen at tag e4-prereg before any E4 eval run). Run on the laptop, per model:

  python scripts/e4/analyse_e4.py --model mapanything \
      --synth results/e4_eval_synth_ma --real results/e4_eval_real_ma_room results/e4_eval_real_ma_e2e \
      --e3 results/e3_eval_synth_ma results/e3_eval_real_ma_room results/e3_eval_real_ma_e2e --out results/e4_eval_agg_ma

VtT = the `vtt` control. The E4 runs hold none, filter_rgbcheck (R2), drop_lowconf (R3), vtt (and vtt_gate for
MapAnything); images only (R0) is the rgb_only condition. E3 (witness), E3-drop (witness_drop) and ORACLE
(drop_oracle) rows come from the E3 eval runs of the same clips, conditions and seeds (--e3, paired in order with
--synth then --real); the rows both runs share (none, R2, R3, rgb_only) are compared first (parity.csv).
Scene means over clips, paired over scenes; 95% percentile-bootstrap CIs; t-test p; practical thresholds of E2
(shape 1 pp, rotation 0.25 deg, real size 5 pp).
  SAFE   non-inferiority on correct hints (clean_pose, clean_pose_depth2; gt_pose, gt_all, dev_pose, dev_all):
         Delta = E[vtt] - E[none]; passes if the CI's upper end < threshold (all SAFE tests must pass; an
         intersection-union test, no multiplicity correction needed). E3's "harmful" flag is reported too.
  HELP   on bad hints (E3's list): Delta = E[vtt] - E[none]; helpful = Holm-clear and Delta <= -threshold.
  FIXED  VtT vs each fixed deployable remedy (R0, R2, R3, E3, E3-drop), each applied to every condition of one data
         set: per scene, the mean over that data set's SAFE + HELP conditions of one metric (rotation: conditions
         with pose hints; shape: all); better = Holm-clear and Delta <= -threshold; non-inferior = CI upper < threshold.
  BEST   (continuity with E3) VtT vs the best of R0 / R2 / R3 per condition, chosen on these data.
One Holm family per model over HELP, FIXED and BEST.
Decision (fixed before the run): the VtT method claim holds for a model if (1) SAFE, (2) HELP on at least one
agreeing-error condition (shared_heading_bias, c6_shared_rot, drift) or real AI-depth condition, and (3) on ETH3D
(synth) VtT is better than every fixed deployable remedy on pooled rotation and non-inferior to each on pooled shape.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from exp1 import e2def_analysis as E  # noqa: E402
from exp1.e1_analysis import boot_ci, load  # noqa: E402
from exp1.e2_audit import ALPHA, THRESH, holm, ttest_p  # noqa: E402

SAFE_SYNTH = ["clean_pose", "clean_pose_depth2"]
SAFE_REAL = ["gt_pose", "gt_all", "dev_pose", "dev_all"]
HELP_SYNTH = {**{c: "rotation" for c in ["outlier_rot_rot_deg=15", "outlier_rot_rot_deg=30", "outlier_rot_rot_deg=90",
                                          "outlier_frac_frac=0.2", "outlier_frac_frac=0.4",
                                          "shared_heading_bias_rot_deg=5", "shared_heading_bias_rot_deg=10",
                                          "c6_shared_rot_rot_deg=5", "drift_level=2", "drift_level=3"]},
              **{c: "shape" for c in ["depth_scale_factor=0.67", "depth_scale_factor=1.5", "depth_one_scale_factor=1.5"]}}
HELP_REAL = {"ai_depth": "shape", "dev_pose_ai_depth": "shape"}
AGREEING = ("shared_heading_bias", "c6_shared_rot", "drift", "ai_depth", "dev_pose_ai_depth")
BEST_BASELINES = ["rgb_only", "filter_rgbcheck", "drop_lowconf"]
FIXED_REMEDIES = ["rgb_only", "filter_rgbcheck", "drop_lowconf", "witness", "witness_drop"]
FROM_E3 = ("witness", "witness_drop", "drop_oracle")
SHARED = ("none", "filter_rgbcheck", "drop_lowconf")
PARITY_COLS = ["pts_relerr_scale", "rre_deg", "pts_relerr_metric", "depth_absrel_scale"]


def scene_mean(ok, cond, ctrl, col):
    if ctrl == "rgb_only":
        return ok[(ok.condition == "rgb_only") & (ok.control == "none")].groupby("scene")[col].mean()
    return ok[(ok.condition == cond) & (ok.control == ctrl)].groupby("scene")[col].mean()


def row(test, data, cond, metric, d, vs):
    lo, hi = boot_ci(d.values) if len(d) > 1 else (np.nan, np.nan)
    return {"test": test, "data": data, "condition": cond, "metric": metric, "vs": vs, "delta": d.mean(), "ci_lo": lo,
            "ci_hi": hi, "n_scenes": len(d), "threshold": THRESH[E.METRIC_COL[metric]],
            "p_t": ttest_p(d.values) if len(d) > 1 else np.nan}


def merged(e4_run, e3_run):
    """E4 rows plus the E3-only controls of the same runs; parity of the shared rows."""
    a = E.ok_runs(load(e4_run)[0])
    b = E.ok_runs(load(e3_run)[0]) if e3_run and os.path.exists(os.path.join(e3_run, "per_run.csv")) else None
    par = []
    if b is not None:
        key = ["clip_id", "condition", "seed", "control"]
        m = a[a.control.isin(SHARED)].merge(b[b.control.isin(SHARED)], on=key, suffixes=("_e4", "_e3"))
        for c in PARITY_COLS:
            d = (m[f"{c}_e4"] - m[f"{c}_e3"]).abs()
            par.append({"e4_run": e4_run, "e3_run": e3_run, "metric": c, "n_rows": len(m),
                        "n_exact": int((d <= 1e-9).sum()),   # equal up to CSV round-off
                        "max_abs_diff": float(d.max()) if len(d) else np.nan})
        a = pd.concat([a, b[b.control.isin(FROM_E3)]], ignore_index=True)
    return a, par


def tests_for(ok, data, safe, help_):
    rows = []
    for c in safe:
        if not (ok.condition == c).any():
            continue
        for m in ["shape", "rotation"] + (["real_size"] if data != "synth" else []):
            col = E.METRIC_COL[m]
            rows.append(row("SAFE", data, c, m, (scene_mean(ok, c, "vtt", col) - scene_mean(ok, c, "none", col)).dropna(),
                            "hints as given"))
    for c, m in help_.items():
        if not (ok.condition == c).any():
            continue
        col = E.METRIC_COL[m]
        v = scene_mean(ok, c, "vtt", col)
        rows.append(row("HELP", data, c, m, (v - scene_mean(ok, c, "none", col)).dropna(), "hints as given"))
        cands = {b: scene_mean(ok, c, b, col) for b in BEST_BASELINES}
        cands = {b: x for b, x in cands.items() if len(x)}
        best = min(cands, key=lambda b: cands[b].mean())
        rows.append(row("BEST", data, c, m, (v - cands[best]).dropna(), best))
    pools = {"rotation": list(safe) + [c for c, m in help_.items() if m == "rotation"] +
                         (["dev_pose_ai_depth"] if "dev_pose_ai_depth" in help_ else []),
             "shape": list(safe) + list(help_)}
    for m, conds in pools.items():
        col = E.METRIC_COL[m]
        conds = [c for c in conds if (ok.condition == c).any()]
        if not conds:
            continue
        pooled = lambda ctrl: pd.concat([scene_mean(ok, c, ctrl, col) for c in conds], axis=1).mean(axis=1)
        v = pooled("vtt")
        for r in FIXED_REMEDIES:
            if r != "rgb_only" and not (ok.control == r).any():
                continue
            other = scene_mean(ok, None, "rgb_only", col) if r == "rgb_only" else pooled(r)
            rows.append(dict(row("FIXED", data, "+".join(conds), m, (v - other).dropna(), r), n_conditions=len(conds)))
    return rows


def descriptive(ok, data, conds, good):
    rows = []
    for c in conds:
        for col in ("rre_deg", "pts_relerr_scale", "pts_relerr_metric"):
            r = {"data": data, "condition": c, "metric": col}
            for ctrl in ["none", "vtt", "vtt_gate", "filter_rgbcheck", "drop_lowconf", "witness", "witness_drop",
                         "drop_oracle", "rgb_only"]:
                x = scene_mean(ok, c, ctrl, col)
                r[ctrl] = x.mean() if len(x) else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--synth", nargs="*", default=[])
    ap.add_argument("--real", nargs="*", default=[])
    ap.add_argument("--e3", nargs="*", default=[], help="E3 eval runs, in the order of --synth then --real")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = [(r, "synth") for r in a.synth] + [(r, "real") for r in a.real]
    e3 = list(a.e3) + [None] * (len(runs) - len(a.e3))
    rows, desc, par = [], [], []
    for (run, kind), e3run in zip(runs, e3):
        if not os.path.exists(os.path.join(run, "per_run.csv")):
            continue
        ok, p = merged(run, e3run)
        par += p
        if kind == "synth":
            rows += tests_for(ok, "synth", SAFE_SYNTH, HELP_SYNTH)
            desc.append(descriptive(ok, "synth", SAFE_SYNTH + list(HELP_SYNTH), "clean_pose"))
        else:
            name = "real_" + os.path.basename(run.rstrip("/")).split("_")[-1]
            rows += tests_for(ok, name, SAFE_REAL, HELP_REAL)
            desc.append(descriptive(ok, name, SAFE_REAL + list(HELP_REAL), "gt_pose"))
    T = pd.DataFrame(rows)
    fam = T.test != "SAFE"
    T["p_holm"] = np.nan
    T.loc[fam, "p_holm"] = holm(T.loc[fam, "p_t"].fillna(1.0))
    T["clear"] = T.p_holm < ALPHA
    T["noninferior"] = T.ci_hi < T.threshold
    T["harmful"] = (T.test == "SAFE") & (T.delta >= T.threshold) & (T.ci_lo > 0)
    T["better"] = T.clear & (T.delta <= -T.threshold)
    T.to_csv(os.path.join(a.out, "e4_tests.csv"), index=False)
    if desc:
        pd.concat(desc, ignore_index=True).to_csv(os.path.join(a.out, "e4_per_condition.csv"), index=False)
    pd.DataFrame(par).to_csv(os.path.join(a.out, "parity.csv"), index=False)
    S = T[T.test == "SAFE"]
    safe = bool(len(S)) and bool(S.noninferior.all())
    key = lambda df: set(df[df.better & df.condition.str.startswith(AGREEING)].condition)
    helped = key(T[T.test == "HELP"])
    F = T[(T.test == "FIXED") & (T.data == "synth")]
    rot_better = F[F.metric == "rotation"].set_index("vs").better
    shape_ni = F[F.metric == "shape"].set_index("vs").noninferior
    fixed_ok = bool(len(rot_better)) and bool(rot_better.all()) and bool(len(shape_ni)) and bool(shape_ni.all())
    verdict = {"model": a.model, "safe_noninferior": safe, "safe_failures": S[~S.noninferior][["data", "condition", "metric"]].to_dict("records"),
               "help_on_agreeing_or_ai_depth": sorted(helped),
               "fixed_rotation_better_than": sorted(rot_better[rot_better].index), "fixed_rotation_not_better_than": sorted(rot_better[~rot_better].index),
               "fixed_shape_noninferior_to": sorted(shape_ni[shape_ni].index), "fixed_shape_not_noninferior_to": sorted(shape_ni[~shape_ni].index),
               "method_claim_holds": bool(safe and helped and fixed_ok), "n_tests": int(len(T)),
               "parity_all_exact": bool(all(p["n_exact"] == p["n_rows"] for p in par)) if par else None}
    json.dump(verdict, open(os.path.join(a.out, "e4_verdict.json"), "w"), indent=1)
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    main()
