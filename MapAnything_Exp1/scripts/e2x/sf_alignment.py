"""Exploratory: per-frame device-vs-GT world alignment over time (drift vs jumps vs 180-deg flips). Not a result."""
import csv, os, sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, r"C:\Users\djtde\Downloads\3D-CVPR_Exp\MapAnything_Exp1")
from exp1.arkit_clips import parse_traj, device_turn, nearest, orientation, R180, POSE_TOL  # noqa: E402

cache = sys.argv[1]
rows = list(csv.DictReader(open(os.path.join(cache, "test_set.csv"))))
out = []
for r in rows:
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
    W, tt, up = [], [], []
    for i, t in enumerate(gts):
        j, dt = nearest(dts, t)
        if dt <= POSE_TOL:
            Pd = dP[j] @ R180 if turn else dP[j]
            W.append((T @ gP[i]) @ np.linalg.inv(Pd))  # GT->ARKit-world alignment implied by this frame
            tt.append(t); up.append(orientation(gP[i]) == 0)
    W = np.array(W); up = np.array(up)
    R = Rot.from_matrix(W[:, :3, :3])
    Rm = R[up].mean()
    ang = np.degrees((Rm.inv() * R).magnitude())
    pos = W[:, :3, 3] - np.median(W[up, :3, 3], 0)
    step = np.degrees((R[:-1].inv() * R[1:]).magnitude())
    flips = np.mean(ang[up] > 90)
    out.append((vid, r["visit_id"], len(W), up.mean(), np.median(ang[up]), np.percentile(ang[up], 95), ang[up].max(), flips,
                step.max(), np.linalg.norm(pos[up], axis=1).max(), tt[-1] - tt[0]))
out.sort(key=lambda x: -x[6])
print("video     visit   n   up%  |alignment deviation over the video, deg: median p95 max| flip>90 | max frame-to-frame jump deg | max pos dev m | duration s")
for o in out[:25]:
    print(f"{o[0]} {o[1]} {o[2]:4d} {o[3]:4.0%} | {o[4]:6.2f} {o[5]:6.2f} {o[6]:6.1f} | {o[7]:5.1%} | {o[8]:6.2f} | {o[9]:5.2f} | {o[10]:5.0f}")
a = np.array([o[4:10] for o in out], float)
print(f"\n{len(out)} videos. median-over-videos of p95 deviation {np.median(a[:,1]):.2f} deg; videos with p95 > 2 deg: {np.mean(a[:,1] > 2):.0%}, > 5 deg: {np.mean(a[:,1] > 5):.0%};"
      f" videos with any >90 deg frame: {np.mean(a[:,3] > 0):.0%}; videos with a frame-to-frame jump > 2 deg: {np.mean(a[:,4] > 2):.0%}")
