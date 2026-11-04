"""Exploratory (DEV ONLY): witness fit on E2-C dev extras. Not a result."""
import glob, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\djtde\Downloads\3D-CVPR_Exp\MapAnything_Exp1")
from exp1 import witness  # noqa: E402

R = r"C:\Users\djtde\Downloads\3D-CVPR_Exp\MapAnything_Exp1\results"
rows = []
for res in ["e2c_dev", "e2df_dev", "e2e_ma_dev"]:
    ex = os.path.join(R, res, "extras")
    pv = pd.read_csv(os.path.join(R, res, "per_view.csv"))
    for f in sorted(glob.glob(os.path.join(ex, "*__0__none__all.npz"))):
        clip, cond = os.path.basename(f).split("__")[0:3:2] if False else (None, None)
        parts = os.path.basename(f)[:-4].split("__")
        clip_id, cond = "__".join(parts[:3]), parts[3]
        rgb = os.path.join(ex, f"{clip_id}__rgb_only__0__none__all.npz")
        if cond == "rgb_only" or not os.path.exists(rgb):
            continue
        d, p = np.load(f), np.load(rgb)
        if "hint_pose_rel" not in d.files:
            continue
        H = d["hint_pose_rel"]
        if not np.isfinite(H).all() or np.allclose(H, np.eye(4)):
            continue
        out = witness.fit(H, p["pred_pose_rel"])
        cor = pv[(pv.clip_id == clip_id) & (pv.condition == cond) & (pv.control == "none") & (pv.seed == 0)].sort_values("view_idx")
        wc = cor.was_corrupted.values.astype(bool) if len(cor) == len(H) else np.zeros(len(H), bool)
        hr = cor.hint_rot_err.values if len(cor) == len(H) else np.full(len(H), np.nan)
        rows.append(dict(src=res, clip=clip_id, cond=cond, shared=out["shared_deg"], drift=out["drift_total_deg"], scale=out["scale"],
                         res_bad=np.nanmax(np.where(wc, out["rot_res_deg"], np.nan)) if wc.any() else np.nan,
                         res_good=np.nanmax(np.where(~wc, out["rot_res_deg"], np.nan)[1:]),
                         hint_err_max=np.nanmax(hr[1:]) if np.isfinite(hr[1:]).any() else np.nan))
df = pd.DataFrame(rows)
df.to_csv("e3_detect_dev.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
g = df.groupby(["src", "cond"]).agg(n=("clip", "size"), shared=("shared", "median"), drift=("drift", "median"), scale=("scale", "median"),
                                    res_bad=("res_bad", "median"), res_good=("res_good", "median"), hint_err=("hint_err_max", "median")).round(2)
print(g.to_string())
