"""Session A2 readouts (experiments/E2X.md, X-M "A2 grid and readouts", frozen at e2x-a2-prereg). All DESCRIPTIVE:
scene means over clips, then the mean over scenes with a bootstrap 95% CI; "n_scenes_pattern" counts scenes showing
the pattern (a finding needs >= 7 of 10 eval scenes and a CI excluding the opposite).

  python scripts/e2x/analyse_xm.py --xm results/xm_ma_eval results/xm_px_eval --m8 results/m8_ma_eval results/m8_px_eval \
      --out results/e2x_xm_agg

Error per case: bad-view rotation error vs GT (deg) for pose cases; shape error (pts_relerr_scale) for the depth case.
  m1_dial.csv     per case and gamma: error; straight-line fit R^2 per scene; switch index (E(1/6)-E(0))/(E(1)-E(0)).
                  Linear = R^2 >= 0.95; switch-like = index > 0.5.
  m3_patch.csv    per case, source (clean / drop) and layer: share of the damage removed (E_bad-E_patch)/(E_bad-E_clean);
                  point of no return = first layer whose share falls below 0.5.
  m5_spread.csv   per case and layer: the same share when only the OTHER views are patched, and the good-view error.
  m2_m4_state.csv per case and layer: relative size of (bad - clean) state of the bad views, its cosine with the first
                  layer; copy path = cosine >= 0.8 up to the last layer.
  m8_scale.csv    per pose-hint scale x depth-hint scale: output scale vs ground truth (metric_scale_ratio), and the same
                  for images only and correct hints.
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from exp1.e1_analysis import boot_ci  # noqa: E402


def err_col(case):
    return "pts_relerr_scale" if case.startswith("depth") else "bad_rot_err_gt_deg"


def summarise(df, by, val, label):
    """scene means -> mean over scenes with CI."""
    s = df.groupby(by + ["scene"])[val].mean().reset_index()
    rows = []
    for k, g in s.groupby(by):
        lo, hi = boot_ci(g[val].values) if len(g) > 1 else (np.nan, np.nan)
        rows.append({**dict(zip(by, k if isinstance(k, tuple) else (k,))), label: g[val].mean(), "ci_lo": lo,
                     "ci_hi": hi, "n_scenes": len(g)})
    return pd.DataFrame(rows), s


def model_of(run):
    return "pi3x" if "_px_" in run else "omnivggt" if "_ov_" in run else "mapanything"


def xm(run):
    R = pd.read_csv(os.path.join(run, "per_run.csv"))
    R = R[R.status == "ok"]
    m = model_of(run)
    out = {}
    refs = R[R.experiment == "ref"]
    base = []
    for case, g in refs.groupby("case"):
        col = err_col(case)
        p = g.pivot_table(index=["clip_id", "scene"], columns="source", values=col, aggfunc="first").reset_index()
        p["case"] = case
        base.append(p)
    B = pd.concat(base, ignore_index=True)

    # M1
    d = R[R.experiment == "m1_dial"].copy()
    if len(d):
        d["gamma"] = d.gamma.astype(float)
        d["err"] = [r[err_col(r.case)] for _, r in d.iterrows()]
        rows = []
        for (case, scene), g in d.groupby(["case", "scene"]):
            e = g.groupby("gamma").err.mean()
            x, y = e.index.values, e.values
            fit = np.polyfit(x, y, 1)
            ss = ((y - y.mean()) ** 2).sum()
            r2 = 1 - ((y - np.polyval(fit, x)) ** 2).sum() / ss if ss > 0 else np.nan
            span = e.loc[1.0] - e.loc[0.0]
            g16 = e.iloc[np.argmin(np.abs(x - 1 / 6))]
            rows.append({"case": case, "scene": scene, "r2": r2, "switch_index": (g16 - e.loc[0.0]) / span if span else np.nan,
                         "err_gamma0": e.loc[0.0], "err_gamma1": e.loc[1.0]})
        S = pd.DataFrame(rows)
        t = []
        for case, g in S.groupby("case"):
            lo, hi = boot_ci(g.r2.dropna().values)
            slo, shi = boot_ci(g.switch_index.dropna().values)
            t.append({"model": m, "case": case, "r2_mean": g.r2.mean(), "r2_ci_lo": lo, "r2_ci_hi": hi,
                      "n_scenes_linear": int((g.r2 >= 0.95).sum()), "switch_index_mean": g.switch_index.mean(),
                      "switch_ci_lo": slo, "switch_ci_hi": shi, "n_scenes_switch": int((g.switch_index > 0.5).sum()),
                      "n_scenes": len(g), "err_gamma0": g.err_gamma0.mean(), "err_gamma1": g.err_gamma1.mean()})
        curve = d.groupby(["case", "gamma", "scene"]).err.mean().groupby(["case", "gamma"]).mean().reset_index()
        curve.insert(0, "model", m)
        out["m1_dial"] = pd.DataFrame(t)
        out["m1_curve"] = curve

    # M3 and M5
    for exp, name in (("m3_patch_bad_views", "m3_patch"), ("m5_patch_other_views", "m5_spread")):
        p = R[R.experiment == exp].copy()
        if not len(p):
            continue
        p["err"] = [r[err_col(r.case)] for _, r in p.iterrows()]
        p = p.merge(B[["clip_id", "case", "bad", "clean"]], on=["clip_id", "case"])
        dmg = p.bad - p.clean
        p["share_removed"] = np.where(np.abs(dmg) > 1e-9, (p.bad - p.err) / dmg, np.nan)
        keys = ["case", "source", "layer"] if exp.startswith("m3") else ["case", "layer"]
        T, _ = summarise(p, keys, "share_removed", "share_removed")
        if exp.startswith("m5"):
            G, _ = summarise(p, keys, "good_rot_err_gt_deg", "good_rot_err_gt_deg")
            T = T.merge(G[keys + ["good_rot_err_gt_deg"]], on=keys)
        T.insert(0, "model", m)
        if exp.startswith("m3"):
            pnr = []
            for (case, src), g in T.sort_values("layer").groupby(["case", "source"]):
                below = g[g.share_removed < 0.5]
                pnr.append({"model": m, "case": case, "source": src,
                            "point_of_no_return_layer": below.layer.iloc[0] if len(below) else "none"})
            out["m3_point_of_no_return"] = pd.DataFrame(pnr)
        out[name] = T

    # M2 / M4
    st = os.path.join(run, "state_stats.csv")
    if os.path.exists(st):
        S = pd.read_csv(st)
        sc = R[["clip_id", "scene"]].drop_duplicates()
        S = S[S.is_bad == 1].merge(sc, on="clip_id")
        A, _ = summarise(S, ["case", "layer"], "rel_diff_bad_vs_clean", "rel_diff_bad_vs_clean")
        Cn, _ = summarise(S, ["case", "layer"], "cos_with_first_layer", "cos_with_first_layer")
        A = A.merge(Cn[["case", "layer", "cos_with_first_layer"]], on=["case", "layer"])
        A.insert(0, "model", m)
        out["m2_m4_state"] = A
    return out


def m8(run):
    R = pd.read_csv(os.path.join(run, "per_run.csv"))
    R = R[(R.status == "ok") & (R.control == "none") & (R.view_subset == "all")]
    rows = []
    for cond, g in R.groupby("condition"):
        s = g.groupby("scene").metric_scale_ratio.mean()
        lo, hi = boot_ci(s.values) if len(s) > 1 else (np.nan, np.nan)
        p = d = np.nan
        if cond.startswith("m8_"):
            p = float(cond.split("pose")[1].split("_")[0])
            d = float(cond.split("depth")[1])
        rows.append({"model": model_of(run), "condition": cond, "pose_hint_scale": p, "depth_hint_scale": d,
                     "output_scale_vs_gt": s.mean(), "ci_lo": lo, "ci_hi": hi, "n_scenes": len(s),
                     "shape_err": g.groupby("scene").pts_relerr_scale.mean().mean()})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xm", nargs="*", default=[])
    ap.add_argument("--m8", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    acc = {}
    for run in a.xm:
        if os.path.exists(os.path.join(run, "per_run.csv")):
            for k, v in xm(run).items():
                acc.setdefault(k, []).append(v)
    for run in a.m8:
        if os.path.exists(os.path.join(run, "per_run.csv")):
            acc.setdefault("m8_scale", []).append(m8(run))
    for k, v in acc.items():
        df = pd.concat(v, ignore_index=True)
        df.insert(0, "label", "DESCRIPTIVE")
        df.to_csv(os.path.join(a.out, f"{k}.csv"), index=False)
        print(f"{k}: {len(df)} rows")


if __name__ == "__main__":
    main()
