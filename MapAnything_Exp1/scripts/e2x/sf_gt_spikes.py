"""Exploratory: who jumps? per-source spike test (frame vs midpoint of its neighbours), GT vs ARKit. Not a result."""
import csv, os, sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp
sys.path.insert(0, r"C:\Users\djtde\Downloads\3D-CVPR_Exp\MapAnything_Exp1")
from exp1.arkit_clips import parse_traj, device_turn, nearest, R180, POSE_TOL  # noqa: E402
cache = sys.argv[1]
tot = {"gt": [0, 0, 0], "dev": [0, 0, 0]}; nfr = 0; vids_with = {"gt": set(), "dev": set()}; nv = 0
per = []
for r in csv.DictReader(open(os.path.join(cache, "test_set.csv"))):
    vid = r["video_id"]
    try:
        gts, gP = parse_traj(open(os.path.join(cache, f"{vid}_hires.traj")).read())
        dts, dP = parse_traj(open(os.path.join(cache, f"{vid}_lowres_wide.traj")).read())
        T = np.load(os.path.join(cache, f"{vid}_transform.npy"))
    except FileNotFoundError:
        continue
    try:
        turn, meas, resid = device_turn(gts, gP, dts, dP, T)
    except IndexError:  # no device pose within POSE_TOL of any GT frame
        continue
    if resid > 5:
        continue
    nv += 1
    src = {"gt": (gts, gP), "dev": (dts, dP)}
    for k, (ts, P) in src.items():
        R = Rot.from_matrix(P[:, :3, :3]); x = P[:, :3, 3]
        # spike: frame i vs interpolation of i-1 and i+1 at its timestamp (rotation deg, position m)
        mid = Slerp([0, 1], Rot.concatenate([R[0], R[1]]))  # placeholder to keep API simple
        sr, sp = [], []
        for i in range(1, len(R) - 1):
            a = (ts[i] - ts[i - 1]) / max(ts[i + 1] - ts[i - 1], 1e-6)
            Ri = Slerp([0, 1], Rot.concatenate([R[i - 1], R[i + 1]]))([a])[0]
            sr.append(np.degrees((Ri.inv() * R[i]).magnitude()))
            sp.append(np.linalg.norm(x[i] - ((1 - a) * x[i - 1] + a * x[i + 1])))
        sr, sp = np.array(sr), np.array(sp)
        if k == "gt":
            nfr += len(sr)
        tot[k][0] += int((sr > 5).sum()); tot[k][1] += int((sr > 30).sum()); tot[k][2] += int((sp > 0.3).sum())
        if (sr > 5).any():
            vids_with[k].add(vid)
        per.append((vid, k, len(sr), int((sr > 5).sum()), float(sr.max()), float(sp.max())))
print(f"{nv} videos, {nfr} GT frames")
for k in tot:
    print(f"{k:4s}: frames that jump away from their neighbours by >5 deg: {tot[k][0]}, >30 deg: {tot[k][1]}; position spikes >0.3 m: {tot[k][2]}; videos with a >5 deg spike: {len(vids_with[k])}")
