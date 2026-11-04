"""Extended-E2 analysis (experiments/E2X.md). Run on the laptop after scripts/vm_fetch_e2x.sh, from MapAnything_Exp1/:

  python scripts/e2x/analyse_e2x.py --out results/e2x_agg

Preregistered parts reuse the frozen E2 code unchanged (exp1/e2def_analysis):
  X-A  Pi3X on ETH3D: d_harm and d_behaviour, then the side-by-side classes against MapAnything (E2-D companion run
       results/e2df_eval, E2-C primary tests) and OmniVGGT (results/e2d_eval).
  X-B  real hints: e_tests per model on the room eval runs (Holm per model) and for Pi3X on the E2-E clips; e_costs,
       e_real_errors (room clips), f_real (R2/R3 on real pose hints).
Descriptive additions (E2X.md section 5, "angles"), labelled descriptive in every output:
  break_even   per model and C1 family: Delta vs the floor at each error size (scene means, bootstrap CI) and the
               smallest size whose CI lies above 0, i.e. where hints stop helping.
  safety       per remedy on the E3 calibration runs: share of the good-hint benefit kept and share of the bad-hint
               damage removed (dev / cal data: a preview, not a result).
Missing runs are skipped, so the same script runs on dev runs.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from exp1 import e2def_analysis as E  # noqa: E402
from exp1.e1_analysis import boot_ci, load  # noqa: E402

E.E_CONDS.setdefault("pi3x", list(E.E_CONDS["mapanything"]))   # Pi3X takes every hint set MapAnything takes
C1 = {"c1_rot": ("rot_deg", "rgb_only", "rre_deg"), "c1_trans": ("trans_frac", "rgb_only", "rre_deg"),
      "c1_scale": ("factor", "rgb_only", "pts_relerr_scale"), "c1_focal": ("factor", "rgb_only", "pts_relerr_scale"),
      "c1_depth": ("factor", "clean_pose", "pts_relerr_scale")}


def exists(run):
    return run and os.path.exists(os.path.join(run, "per_run.csv"))


def x_a(px, ma, ov, ma_c_primary, clips):
    out = {}
    dh_px = E.d_harm(px, "pi3x")
    dh_px = E.refresh_holm(dh_px[~dh_px.metric.isin(["real_size", "scale_ratio"])])   # Pi3X normalises hint scale
    out["xa_harm_pi3x"] = dh_px
    db_px = E.d_behaviour(px, clips, "pi3x")
    out["xa_behaviour_pi3x"] = db_px
    relabel = lambda df: df.replace({"class": {"OmniVGGT only": "Pi3X only"}}).replace(
        {"class": {"MapAnything only (no OmniVGGT readout)": "MapAnything only (no Pi3X readout)"}})
    if exists(ma):
        sbs, fam = E.side_by_side_harm(E.d_harm(ma, "mapanything"), dh_px)
        out["xa_harm_ma_vs_pi3x"], out["xa_family_ma_vs_pi3x"] = relabel(sbs), relabel(fam)
    if exists(ov):
        sbs, fam = E.side_by_side_harm(E.d_harm(ov, "omnivggt"), dh_px)
        sbs = sbs.replace({"class": {"MapAnything only": "OmniVGGT only", "OmniVGGT only": "Pi3X only"}})
        out["xa_harm_ov_vs_pi3x"] = sbs
    if ma_c_primary and os.path.exists(ma_c_primary):
        pt = pd.read_csv(ma_c_primary)
        pt = pt[pt.test.isin(E.DB_TESTS)].copy()
        out["xa_behaviour_ma_vs_pi3x"] = relabel(E.side_by_side_behaviour(pt, db_px))
    return out


def x_b(room, e2e_px, room_clips):
    out = {}
    for model, run in room.items():
        if exists(run):
            out[f"xb_room_tests_{model}"] = E.e_tests({model: run})            # Holm per model
            out[f"xb_room_costs_{model}"] = E.e_costs({model: run})
            out[f"xb_room_remedies_{model}"] = E.f_real(run)
    if exists(e2e_px):
        out["xb_e2e_tests_pi3x"] = E.e_tests({"pi3x": e2e_px})
    if glob.glob(room_clips):
        out["xb_room_real_errors"] = E.e_real_errors(room_clips)
    return out


def break_even(run, model):
    """DESCRIPTIVE. Delta vs the floor (pose families: images only; depth: clean poses) per C1 setting."""
    pr, _, _ = load(run)
    ok = E.ok_runs(pr)
    ok = ok[ok.control == "none"]
    rows = []
    for fam, (key, floor, col) in C1.items():
        conds = sorted(c for c in ok.condition.unique() if c.startswith(fam + "_"))
        for c in conds:
            d = E.scene_delta(ok, c, floor, col)
            if not len(d):
                continue
            lo, hi = boot_ci(d.values)
            rows.append({"model": model, "family": fam, "condition": c, "size": float(c.split("=")[-1]),
                         "metric": col, "delta_vs_floor": d.mean(), "ci_lo": lo, "ci_hi": hi, "n_scenes": len(d)})
    df = pd.DataFrame(rows)
    if len(df):
        df["hint_worse_than_floor"] = df.ci_lo > 0
        df.insert(0, "label", "DESCRIPTIVE")
    return df


def safety(run, model):
    """DESCRIPTIVE preview on calibration runs: benefit kept on good hints, damage removed on bad hints, per remedy."""
    pr, _, _ = load(run)
    ok = E.ok_runs(pr)
    rows = []
    for col in ("rre_deg", "pts_relerr_scale"):
        sm = lambda cond, ctrl="none": ok[(ok.condition == cond) & (ok.control == ctrl)].groupby("scene")[col].mean()
        rgb, clean = sm("rgb_only"), sm("clean_pose")
        for ctrl in sorted(set(ok.control) - {"drop_oracle"}):
            kept = (rgb - sm("clean_pose", ctrl)) / (rgb - clean) if ctrl != "none" else (rgb - clean) / (rgb - clean)
            bad = [c for c in ok.condition.unique() if c not in ("rgb_only", "clean_pose", "clean_pose_depth2")
                   and not c.startswith("depth")]
            rem = []
            for c in bad:
                dmg, fix = sm(c), sm(c, ctrl)
                if len(fix) and (dmg - clean).mean() > 0:
                    rem.append(((dmg - fix) / (dmg - clean).mean()).mean())
            rows.append({"label": "DESCRIPTIVE (cal/dev)", "model": model, "metric": col, "remedy": ctrl,
                         "benefit_kept": float(np.nanmean(kept)) if len(kept) else np.nan,
                         "damage_removed": float(np.nanmean(rem)) if rem else np.nan, "n_bad_conditions": len(rem)})
        rows.append({"label": "DESCRIPTIVE (cal/dev)", "model": model, "metric": col, "remedy": "R0 images only",
                     "benefit_kept": 0.0, "damage_removed": float(np.nanmean(
                         [((sm(c) - rgb) / (sm(c) - clean).mean()).mean() for c in ok.condition.unique()
                          if c not in ("rgb_only", "clean_pose", "clean_pose_depth2") and not c.startswith("depth")
                          and (sm(c) - clean).mean() > 0] or [np.nan]))})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="eval", choices=["eval", "dev"], help="dev: run everything on the dev runs")
    a = ap.parse_args()
    R = "results/"
    s = a.split
    os.makedirs(a.out, exist_ok=True)
    out = {}
    if exists(R + f"e2x_pi3x_eth3d_{s}"):
        out.update(x_a(R + f"e2x_pi3x_eth3d_{s}", R + f"e2df_{s}", R + f"e2d_{s}",
                       R + "e2c_eval/agg/primary_tests.csv" if s == "eval" else None, f"data/clips_e2/{s}__*.npz"))
    room = {m: R + f"e2x_room_{t}_{s}" for m, t in (("mapanything", "ma"), ("omnivggt", "ov"), ("pi3x", "px"))}
    out.update(x_b(room, R + f"e2x_e2e_px_{s}", f"data/clips_e2x_room/{s}__*.npz"))
    be = [break_even(run, m) for m, run in (("mapanything", R + f"e2c_{s}"), ("omnivggt", R + f"e2d_{s}"),
                                             ("pi3x", R + f"e2x_pi3x_eth3d_{s}")) if exists(run)]
    if be:
        out["angle_break_even"] = pd.concat(be, ignore_index=True)
    sf = [safety(run, m) for m, run in (("mapanything", R + "e2x_e3cal_synth_ma_cal"), ("omnivggt", R + "e2x_e3cal_synth_ov_cal"),
                                         ("pi3x", R + "e2x_e3cal_synth_px_cal")) if exists(run)]
    if sf:
        out["angle_safety_preview"] = pd.concat(sf, ignore_index=True)
    for name, df in out.items():
        df.to_csv(os.path.join(a.out, f"{name}.csv"), index=False)
        print(f"{name}: {len(df)} rows")
    json.dump(sorted(out), open(os.path.join(a.out, "index.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
