"""Exploratory: ADVIO 10-view clip-level hint errors (device vs GT), several clip spans. Not a result."""
import sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp

root = sys.argv[1]


def load(path):
    a = np.loadtxt(path, delimiter=",")
    return a[:, 0], a[:, 1:4], Rot.from_quat(a[:, 4:8])  # x,y,z,w (probe 1)


def at(t, p, R, tq):
    tq = np.clip(tq, t[0], t[-1])
    return np.stack([np.interp(tq, t, p[:, k]) for k in range(3)], 1), Slerp(t, R)(tq)


tg, pg, Rg = load(f"{root}/ground-truth/pose.csv")
tf = np.loadtxt(f"{root}/iphone/frames.csv", delimiter=",")[:, 0]
for dev in ["iphone/arkit.csv", "pixel/arcore.csv"]:
    td, pd_, Rd = load(f"{root}/{dev}")
    lo, hi = max(td[0], tg[0], tf[0]) + 0.5, min(td[-1], tg[-1], tf[-1]) - 0.5
    for camconv in ["R=world2cam", "R=cam2world"]:
        for span in [3, 10, 30]:
            rows = []
            for t0 in np.arange(lo, hi - span, span / 2):
                ts = t0 + np.linspace(0, span, 10)
                Pg, Qg = at(tg, pg, Rg, ts)
                Pd, Qd = at(td, pd_, Rd, ts)
                if camconv == "R=world2cam":  # cam->world rotation = Q^-1
                    Cg, Cd = Qg.inv(), Qd.inv()
                else:
                    Cg, Cd = Qg, Qd
                # relative to view 0 (camera frame of view 0)
                rg = Cg[0].inv() * Cg
                rd = Cd[0].inv() * Cd
                rot_err = np.degrees((rg.inv() * rd).magnitude())[1:]
                tgr = Cg[0].inv().apply(Pg - Pg[0])[1:]
                tdr = Cd[0].inv().apply(Pd - Pd[0])[1:]
                Lg, Ld = np.linalg.norm(tgr, axis=1), np.linalg.norm(tdr, axis=1)
                ok = Lg > 0.3
                cosang = np.sum(tgr * tdr, 1) / np.maximum(Lg * Ld, 1e-9)
                dir_err = np.degrees(np.arccos(np.clip(cosang, -1, 1)))[ok]
                scale = np.median(Ld[ok] / Lg[ok]) if ok.any() else np.nan
                rows.append((rot_err.max(), np.median(rot_err), np.median(dir_err) if ok.any() else np.nan,
                             dir_err.max() if ok.any() else np.nan, scale, Lg.max()))
            r = np.array(rows)
            print(f"{dev:18s} {camconv:12s} span {span:2d}s n={len(r):3d} | rot max-view median {np.median(r[:,0]):5.2f} p90 {np.nanpercentile(r[:,0],90):5.2f} max {r[:,0].max():6.2f} deg"
                  f" | dir median {np.nanmedian(r[:,2]):5.1f} max-view p90 {np.nanpercentile(r[:,3],90):5.1f} deg"
                  f" | scale med {np.nanmedian(r[:,4]):.3f} [p10 {np.nanpercentile(r[:,4],10):.3f}, p90 {np.nanpercentile(r[:,4],90):.3f}]"
                  f" | GT extent med {np.median(r[:,5]):.1f} m")
