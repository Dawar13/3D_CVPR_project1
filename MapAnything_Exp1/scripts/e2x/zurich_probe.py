"""Exploratory: Zurich Urban MAV (AGZ) device hints (GPS + autopilot attitude) vs Pix4D GT. Logs only. Not a result."""
import sys, itertools
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as Rot, Slerp

root = sys.argv[1]
gt = pd.read_csv(f"{root}/Log Files/GroundTruthAGL.csv", skipinitialspace=True).dropna(axis=1, how="all")
gps = pd.read_csv(f"{root}/Log Files/OnboardGPS.csv", skipinitialspace=True).dropna(axis=1, how="all")
pose = pd.read_csv(f"{root}/Log Files/OnboardPose.csv", skipinitialspace=True).dropna(axis=1, how="all")
gt.columns = [c.strip() for c in gt.columns]; gps.columns = [c.strip() for c in gps.columns]; pose.columns = [c.strip() for c in pose.columns]
print(len(gt), "GT frames; imgid range", gt.imgid.min(), gt.imgid.max())
ts = gps.set_index("imgid")["Timpstemp"]
gt = gt[gt.imgid.isin(ts.index)].reset_index(drop=True)
gt["t"] = ts.loc[gt.imgid].values * 1e-6
P = gt[["x_gt", "y_gt", "z_gt"]].values
G = gt[["x_gps", "y_gps", "z_gps"]].values
d = G - P
print("GPS - GT position (m): median |err| %.2f, p90 %.2f, max %.2f; mean offset %s" % (
    np.median(np.linalg.norm(d, axis=1)), np.percentile(np.linalg.norm(d, axis=1), 90), np.linalg.norm(d, axis=1).max(), np.round(d.mean(0), 2)))
step = np.linalg.norm(np.diff(P, axis=0), axis=1)
print("GT motion per GT frame (1/30 imgs): median %.2f m, p90 %.2f m" % (np.median(step), np.percentile(step, 90)))

# GT rotations: try photogrammetric omega-phi-kappa conventions
opk = np.radians(gt[["omega_gt", "phi_gt", "kappa_gt"]].values)
pt = pose["Timpstemp"].values * 1e-6
qa = Rot.from_quat(pose[["Attitude_x", "Attitude_y", "Attitude_z", "Attitude_w"]].values)
ok = (gt.t > pt[0]) & (gt.t < pt[-1])
Ra = Slerp(pt, qa)(gt.t[ok].values)


def kabsch(A, B):
    U, S, Vt = np.linalg.svd(A.T @ B)
    dd = np.sign(np.linalg.det(Vt.T @ U.T))
    return Rot.from_matrix(Vt.T @ np.diag([1, 1, dd]) @ U.T)


cands = {}
for seq in ["XYZ", "xyz", "ZYX", "zyx"]:
    for inv in [False, True]:
        R = Rot.from_euler(seq, opk[ok.values])
        cands[f"{seq}{'^-1' if inv else ''}"] = R.inv() if inv else R
best = None
for (gname, Rg), (aname, Rd) in itertools.product(cands.items(), [("att", Ra), ("att^-1", Ra.inv())]):
    k = 3  # 3 GT frames = ~3 s
    rg = (Rg[:-k].inv() * Rg[k:]).as_rotvec()
    rd = (Rd[:-k].inv() * Rd[k:]).as_rotvec()
    m = np.linalg.norm(rg, axis=1) > np.radians(3)
    if m.sum() < 20:
        continue
    for _ in range(3):  # robust refit
        C = kabsch(rd[m], rg[m])
        e = np.degrees((Rot.from_rotvec(rg).inv() * C * Rot.from_rotvec(rd) * C.inv()).magnitude())
        m = m & (e < np.percentile(e[m], 80))
    if best is None or np.median(e) < best[0]:
        best = (np.median(e), gname, aname, C, e, rg)
s, gname, aname, C, e, rg = best
print(f"best convention GT={gname} attitude={aname}; mount C (xyz deg) {np.round(C.as_euler('xyz', degrees=True), 1)}")
print(f"  3 s relative-rotation diff autopilot vs GT: median {np.median(e):.2f}, p75 {np.percentile(e, 75):.2f}, p90 {np.percentile(e, 90):.2f}, max {e.max():.1f} deg"
      f" (GT 3 s rotation median {np.degrees(np.median(np.linalg.norm(rg, axis=1))):.1f} deg)")

# rigidity check: residual vs GT rotation size, for 3 s and 10 s windows (same convention + mount)
Rg = cands[gname]; Rd = Ra.inv() if aname == "att^-1" else Ra
for k in [3, 10]:
    rg = (Rg[:-k].inv() * Rg[k:]); rd = (Rd[:-k].inv() * Rd[k:])
    e = np.degrees((rg.inv() * C * rd * C.inv()).magnitude()); mg = np.degrees(rg.magnitude()); md = np.degrees(rd.magnitude())
    for lo, hi in [(0, 2), (2, 5), (5, 10), (10, 30), (30, 180)]:
        m = (mg >= lo) & (mg < hi)
        if m.sum():
            print(f"  k={k:2d}s GT rot {lo:3d}-{hi:3d} deg: n={m.sum():4d}  residual median {np.median(e[m]):5.2f}  p90 {np.percentile(e[m], 90):5.2f}  | |rot| autopilot median {np.median(md[m]):5.2f} vs GT {np.median(mg[m]):5.2f}")
