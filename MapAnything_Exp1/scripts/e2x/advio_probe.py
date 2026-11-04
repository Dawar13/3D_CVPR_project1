"""Exploratory: ADVIO device tracks vs ground truth (conventions + error sizes). Not a result."""
import sys, itertools
import numpy as np
from scipy.spatial.transform import Rotation as Rot

root = sys.argv[1]


def load(path, wfirst):
    a = np.loadtxt(path, delimiter=",")
    t, p, q = a[:, 0], a[:, 1:4], a[:, 4:8]
    q = q[:, [1, 2, 3, 0]] if wfirst else q  # scipy wants x,y,z,w
    return t, p, Rot.from_quat(q)


def interp_rot(t, R, tq):
    from scipy.spatial.transform import Slerp
    ok = (tq >= t[0]) & (tq <= t[-1])
    out = Slerp(t, R)(np.clip(tq, t[0], t[-1]))
    return out, ok


def rel(R, i, j, body):
    return (R[i].inv() * R[j]) if body else (R[j] * R[i].inv())


def kabsch(A, B):  # rotation C with B ~ C A
    H = A.T @ B
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Rot.from_matrix(Vt.T @ np.diag([1, 1, d]) @ U.T)


tg, pg, _ = load(f"{root}/ground-truth/pose.csv", True)
for dev in ["iphone/arkit.csv", "pixel/arcore.csv", "tango/raw.csv"]:
    best = None
    for wf_d, wf_g, body_d, body_g in itertools.product([True, False], repeat=4):
        td, pd_, Rd = load(f"{root}/{dev}", wf_d)
        _, _, Rg = load(f"{root}/ground-truth/pose.csv", wf_g)
        ts = np.arange(max(td[0], tg[0]) + 1, min(td[-1], tg[-1]) - 2, 0.5)
        Rgi, _ = interp_rot(tg, Rg, ts)
        Rdi, _ = interp_rot(td, Rd, ts)
        i, j = np.arange(len(ts) - 2), np.arange(2, len(ts))  # 1 s apart
        rg = rel(Rgi, i, j, body_g).as_rotvec()
        rd = rel(Rdi, i, j, body_d).as_rotvec()
        m = np.linalg.norm(rg, axis=1) > np.deg2rad(5)
        if m.sum() < 10:
            continue
        C = kabsch(rd[m], rg[m])
        err = np.degrees(np.linalg.norm((Rot.from_rotvec(rg).inv() * C * Rot.from_rotvec(rd) * C.inv()).as_rotvec(), axis=1))
        s = np.median(err)
        if best is None or s < best[0]:
            best = (s, (wf_d, wf_g, body_d, body_g), C, err, ts, np.degrees(np.linalg.norm(rg, axis=1)))
    s, conv, C, err, ts, mag = best
    print(f"{dev}: conv(wfirst_dev, wfirst_gt, body_dev, body_gt)={conv}  C(deg)={np.round(C.as_euler('xyz', degrees=True), 1)}")
    print(f"   1 s relative-rotation diff vs GT: median {np.median(err):.2f}  p90 {np.percentile(err, 90):.2f}  max {err.max():.1f} deg"
          f"  (GT motion median {np.median(mag):.1f} deg/s)  n={len(err)}  span {ts[0]:.0f}-{ts[-1]:.0f}s")
