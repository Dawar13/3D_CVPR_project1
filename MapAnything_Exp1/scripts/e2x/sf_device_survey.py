"""Exploratory: ARKit (device) vs SceneFun3D GT pose errors for 10-view clips of growing span, all test videos.
Trajectories only (no images). Not a result yet."""
import csv, io, os, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np

sys.path.insert(0, r"C:\Users\djtde\Downloads\3D-CVPR_Exp\MapAnything_Exp1")
from exp1 import remote_zip  # noqa: E402
from exp1.arkit_clips import SF_URL, AK_URL, parse_traj, device_turn, nearest, keyframes, orientation, R180, POSE_TOL  # noqa: E402

cache, out = sys.argv[1], sys.argv[2]
os.makedirs(cache, exist_ok=True)
TEST = "https://raw.githubusercontent.com/SceneFun3D/scenefun3d/main/benchmark_file_lists/test_set.csv"


def cached(name, url):
    p = os.path.join(cache, name)
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        data = remote_zip.fetch(url)   # fetch BEFORE opening: a failed fetch must not leave an empty file
        with open(p + ".part", "wb") as f:
            f.write(data)
        os.replace(p + ".part", p)
    return p


rows = list(csv.DictReader(open(cached("test_set.csv", TEST))))
vids = [(r["visit_id"], r["video_id"]) for r in rows]
print(len(vids), "test videos")


def load(vv):
    v, vid = vv
    try:
        g = parse_traj(open(cached(f"{vid}_hires.traj", SF_URL.format(visit=v, video=vid) + "hires_poses.traj")).read())
        d = parse_traj(open(cached(f"{vid}_lowres_wide.traj", AK_URL.format(video=vid) + "lowres_wide.traj")).read())
        T = np.load(cached(f"{vid}_transform.npy", SF_URL.format(visit=v, video=vid) + f"{vid}_refined_transform.npy"))
        return v, vid, g, d, T, None
    except Exception as e:
        return v, vid, None, None, None, repr(e)


with ThreadPoolExecutor(8) as ex:
    data = list(ex.map(load, vids))


def rel_err(Pg, Pd):
    """10 c2w GT and device poses -> per-view rot err (deg), trans-dir err (deg), length ratio; relative to view 0."""
    rg = np.linalg.inv(Pg[0]) @ Pg
    rd = np.linalg.inv(Pd[0]) @ Pd
    R = np.einsum("nji,njk->nik", rg[:, :3, :3], rd[:, :3, :3])
    rot = np.degrees(np.arccos(np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)))[1:]
    tg, td = rg[1:, :3, 3], rd[1:, :3, 3]
    Lg, Ld = np.linalg.norm(tg, axis=1), np.linalg.norm(td, axis=1)
    cos = np.sum(tg * td, 1) / np.maximum(Lg * Ld, 1e-9)
    return rot, np.degrees(np.arccos(np.clip(cos, -1, 1))), Ld / np.maximum(Lg, 1e-9), Lg


res = []
for v, vid, g, d, T, err in data:
    if err:
        print("skip", vid, err)
        continue
    (gts, gP), (dts, dP) = g, d
    try:
        turn, meas, resid = device_turn(gts, gP, dts, dP, T)
    except IndexError:  # no device pose within POSE_TOL of any GT frame
        print("skip", vid, "no matched frames"); continue
    if resid > 5:  # portrait captures etc. (same rule as the clip builder)
        continue
    idx, dev = [], []
    for i, t in enumerate(gts):
        j, dt = nearest(dts, t)
        if dt <= POSE_TOL and orientation(gP[i]) == 0:
            idx.append(i)
            dev.append(dP[j] @ R180 if turn else dP[j])
    # drop GT frames that jump away from their neighbours (COLMAP misregistrations; sf_spikes.py)
    from scipy.spatial.transform import Rotation as Rot, Slerp
    R = Rot.from_matrix(gP[:, :3, :3]); x = gP[:, :3, 3]
    bad = set()
    for i in range(1, len(gP) - 1):
        a = (gts[i] - gts[i - 1]) / max(gts[i + 1] - gts[i - 1], 1e-6)
        Ri = Slerp([0, 1], Rot.concatenate([R[i - 1], R[i + 1]]))([a])[0]
        if np.degrees((Ri.inv() * R[i]).magnitude()) > 2 or np.linalg.norm(x[i] - ((1 - a) * x[i - 1] + a * x[i + 1])) > 0.1:
            bad.update([i])
    keep = [k for k, i in enumerate(idx) if i not in bad]
    idx, dev = [idx[k] for k in keep], [dev[k] for k in keep]
    if len(idx) < 40:
        continue
    gq, dq = gP[idx], np.array(dev)
    kf = keyframes(gq)
    n = len(kf)
    path = np.sum(np.linalg.norm(np.diff(gq[:, :3, 3], axis=0), axis=1))
    for span_name, frac in [("consec", None), ("25%", 0.25), ("50%", 0.5), ("100%", 1.0)]:
        if frac is None:
            starts, wins = range(0, max(n - 10, 1), 10), [lambda s: kf[s:s + 10]]
        else:
            w = max(int(round(frac * n)), 10)
            starts = range(0, max(n - w, 0) + 1, max(w // 2, 1))
            wins = [lambda s, w=w: [kf[k] for k in np.linspace(s, s + w - 1, 10).round().astype(int)]]
        for s in starts:
            sel = wins[0](s)
            if len(sel) < 10:
                continue
            rot, dire, ratio, Lg = rel_err(gq[sel], dq[sel])
            ok = Lg > 0.2
            res.append(dict(visit=v, video=vid, span=span_name, start=s, path_m=path, extent_m=Lg.max(),
                            rot_max=rot.max(), n_gt_dropped=len(bad), rot_med=np.median(rot), dir_med=np.median(dire[ok]) if ok.any() else np.nan,
                            scale=np.median(ratio[ok]) if ok.any() else np.nan, n_kf=n))

import pandas as pd  # noqa: E402
df = pd.DataFrame(res)
df.to_csv(out, index=False)
print(df.video.nunique(), "videos used")
for sp, g in df.groupby("span", sort=False):
    vmax = g.groupby("video").rot_max.max()
    print(f"{sp:7s} clips {len(g):5d} | extent median {g.extent_m.median():5.2f} m | worst-view rot err: median {g.rot_max.median():.2f}, p90 {g.rot_max.quantile(.9):.2f}, "
          f"p99 {g.rot_max.quantile(.99):.2f}, max {g.rot_max.max():.1f} deg; clips >2 deg {np.mean(g.rot_max > 2):.1%}, >5 deg {np.mean(g.rot_max > 5):.1%}"
          f" | move-dir median {g.dir_med.median():.1f} deg | length ratio p10-p90 {g.scale.quantile(.1):.3f}-{g.scale.quantile(.9):.3f}"
          f" | videos with any clip >5 deg: {np.mean(vmax > 5):.0%}")
