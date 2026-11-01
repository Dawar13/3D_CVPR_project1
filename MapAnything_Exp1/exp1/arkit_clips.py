"""E2-E clips with REAL device hints: SceneFun3D test split (= ARKitScenes Validation), experiments/E2E_data.md.

  python -m exp1.arkit_clips --test_csv <SceneFun3D benchmark_file_lists/test_set.csv> --out data/clips_e2e \
      --traj_cache data/e2e_traj

Each clip .npz has the usual keys (images, K, poses, depth: GROUND TRUTH, poses relative to view 0) plus the
device hints for the same frames: dev_K (ARKit intrinsics), dev_poses (ARKit VIO, relative to view 0), dev_depth
(iPad LiDAR, high-confidence pixels only, 0 elsewhere).

Ground truth: SceneFun3D hires_poses.traj (SuperGlue + COLMAP in the laser-scan frame), hires_depth (rendered
from the Faro scan), hires_wide_intrinsics. Device: ARKitScenes lowres_wide.traj, lowres_wide_intrinsics,
lowres_depth + confidence. Only the frames a clip uses are downloaded (exp1/remote_zip.py).

Fixed rules (set before any model run; E2-E preregistration):
  - Videos: the longest video of each visit (SceneFun3D's own "one video" rule); visits in a seeded random order
    (SEED); a video is used if it yields CLIPS_PER_VIDEO clips and its device camera frame matches the GT frame
    after a 0 or 180 deg turn (median residual <= 5 deg; this excludes portrait captures, stored 1440x1920);
    the first N_DEV used visits are dev, the next N_EVAL eval. Every visit looked at is logged (selection_log.csv).
  - Frames: only frames stored upright (SceneFun3D decide_pose == 0) with a device pose within 5 ms, a LiDAR frame
    within 20 ms and all files present; keyframes along the ground-truth path whenever the camera moved KF_TRANS m
    or turned KF_ROT deg since the last keyframe; clips = N_VIEWS consecutive keyframes, CLIPS_PER_VIDEO
    non-overlapping clips spread evenly with the first at keyframe 0 (datasets.clip_starts).
  - Device frame: the ARKit camera frame of a video may be turned 180 deg about the optical axis relative to the
    SceneFun3D frames (stored upright). The turn is measured from all frames of the video (GT pose mapped into the
    ARKit frame with SceneFun3D's transform vs the ARKit pose), snapped to 0 or 180 deg, and applied to the device
    pose, intrinsics and LiDAR map together.
  - Size: TW x TH = 518 x 392 (MapAnything's 4:3 resolution, image.py:43), cropped and resized exactly like the
    ETH3D clips (datasets.resize_crop_params / adjust_K / resize_depth_nearest).
"""

import argparse
import csv
import io
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as Rot

from . import remote_zip
from .datasets import adjust_K, clip_starts, resize_crop_params, resize_depth_nearest
from .geometry import relative_to_first, validate_pose

SF_URL = "https://cvg-data.inf.ethz.ch/scenefun3d/v1/test/{visit}/{video}/"
AK_URL = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Validation/{video}/"
SEED, N_DEV, N_EVAL = 0, 3, 15
N_VIEWS, CLIPS_PER_VIDEO = 10, 3
KF_TRANS, KF_ROT = 0.30, 15.0
TW, TH = 518, 392
HW, HH, LW, LH = 1920, 1440, 256, 192        # hires_wide and lowres (LiDAR) image sizes
POSE_TOL, DEPTH_TOL = 0.005, 0.020           # s
R180 = np.diag([-1.0, -1.0, 1.0, 1.0])       # camera turned 180 deg about its optical axis
Z_ORIEN = np.array([[0, -1, 0], [-1, 0, 0], [0, 1, 0], [1, 0, 0]], float)  # SceneFun3D decide_pose


# ------------------------------------------------------------------ parsing
def parse_traj(text):
    """.traj rows: timestamp, angle-axis (rad), translation (m), world->camera. Returns sorted ts, cam2world."""
    ts, P = [], []
    for line in text.splitlines():
        t = line.split()
        if len(t) != 7:
            continue
        E = np.eye(4)
        E[:3, :3] = Rot.from_rotvec(np.array(t[1:4], float)).as_matrix()
        E[:3, 3] = np.array(t[4:7], float)
        ts.append(float(t[0]))
        P.append(np.linalg.inv(E))
    o = np.argsort(ts)
    return np.array(ts)[o], np.array(P)[o]


def parse_pincam(text):
    w, h, fx, fy, cx, cy = (float(x) for x in text.split())
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]), (int(w), int(h))


def orientation(c2w):
    """SceneFun3D decide_pose: 0 upright, 1 left, 2 upside-down, 3 right."""
    return int(np.argmax(Z_ORIEN @ c2w[2, :3]))


def stamp(name):
    """'.../42445441_50257.080.jpg' -> 50257.08"""
    return float(os.path.splitext(os.path.basename(name))[0].split("_")[1])


def nearest(ts_sorted, t):
    j = int(np.clip(np.searchsorted(ts_sorted, t), 1, len(ts_sorted) - 1))
    j = j - 1 if abs(ts_sorted[j - 1] - t) <= abs(ts_sorted[j] - t) else j
    return j, abs(ts_sorted[j] - t)


# ------------------------------------------------------------------ selection
def keyframes(P, trans=KF_TRANS, rot=KF_ROT):
    k = [0]
    for i in range(1, len(P)):
        A, B = P[k[-1]], P[i]
        ang = np.degrees(np.arccos(np.clip((np.trace(A[:3, :3].T @ B[:3, :3]) - 1) / 2, -1, 1)))
        if np.linalg.norm(B[:3, 3] - A[:3, 3]) >= trans or ang >= rot:
            k.append(i)
    return k


def device_turn(gt_ts, gt_P, dev_ts, dev_P, laser_to_arkit):
    """Measured camera-frame offset device vs GT over all matched frames: (snapped 0/180, median angle deg,
    median residual deg after the snap)."""
    C = []
    for t, P in zip(gt_ts, gt_P):
        j, d = nearest(dev_ts, t)
        if d <= POSE_TOL:
            C.append(np.linalg.inv(laser_to_arkit @ P) @ dev_P[j])
    R = Rot.from_matrix(np.array(C)[:, :3, :3])
    ang = np.degrees(np.linalg.norm(R.as_rotvec(), axis=1))
    turn = 180 if np.median(ang) > 90 else 0
    snap = Rot.from_matrix(R180[:3, :3]) if turn else Rot.identity()
    resid = np.degrees(np.linalg.norm((snap.inv() * R).as_rotvec(), axis=1))
    return turn, float(np.median(ang)), float(np.median(resid))


def usable_frames(gt_ts, gt_P, dev_ts, depth_ts, have):
    """Indices of GT frames that are upright and have a device pose, a LiDAR frame and all files."""
    ok = []
    for i, (t, P) in enumerate(zip(gt_ts, gt_P)):
        if orientation(P) != 0 or f"{t:.3f}" not in have:
            continue
        if nearest(dev_ts, t)[1] <= POSE_TOL and nearest(depth_ts, t)[1] <= DEPTH_TOL:
            ok.append(i)
    return ok


def select_videos(test_csv, n_dev=N_DEV, n_eval=N_EVAL, seed=SEED, traj_of=None, aligned=None, log=None):
    """[(split, visit, video)] by the fixed rule. traj_of(visit, video) -> (ts, c2w) of the GT trajectory.
    aligned(visit, video) -> (ok, reason): the device camera frame matches the GT frame after a 0/180 deg turn
    (excludes portrait captures, whose SceneFun3D frames are stored 1440x1920 and turned 90 deg). log gets
    (visit, video, decision) for every visit looked at."""
    rows = list(csv.DictReader(open(test_csv)))
    by_visit = {}
    for r in rows:
        by_visit.setdefault(r["visit_id"], []).append(r["video_id"])
    visits = sorted(by_visit)
    order = [visits[i] for i in np.random.default_rng(seed).permutation(len(visits))]
    out = []
    for v in order:
        lengths = [(len(traj_of(v, vid)[0]), -int(vid), vid) for vid in sorted(by_visit[v])]
        video = max(lengths)[2]                            # longest; ties -> smaller id
        ts, P = traj_of(v, video)
        up = [i for i in range(len(P)) if orientation(P[i]) == 0]
        if len(keyframes(P[up])) < N_VIEWS * CLIPS_PER_VIDEO:
            log and log((v, video, "skipped: too few upright keyframes"))
            continue
        if aligned is not None:
            ok, reason = aligned(v, video)
            if not ok:
                log and log((v, video, f"skipped: {reason}"))
                continue
        out.append(("dev" if len(out) < n_dev else "eval", v, video))
        log and log((v, video, out[-1][0]))
        if len(out) == n_dev + n_eval:
            break
    return out


# ------------------------------------------------------------------ building
def _read(z, name):
    return remote_zip.read_member(z, name)  # one range request per file


def _png(z, name):
    return np.asarray(Image.open(io.BytesIO(_read(z, name))))


def lidar_on_grid(d, turn, nw, nh, left, top):
    """Nearest LiDAR sample (LW x LH) at every pixel of the TW x TH crop of the HW x HH image (continuous
    coordinates: the LiDAR covers the same field of view as hires_wide)."""
    if turn:
        d = np.rot90(d, 2)
    u = (np.arange(left, left + TW) + 0.5) * HW / nw * LW / HW
    v = (np.arange(top, top + TH) + 0.5) * HH / nh * LH / HH
    iu = np.clip(np.floor(u).astype(int), 0, LW - 1)
    iv = np.clip(np.floor(v).astype(int), 0, LH - 1)
    return d[np.ix_(iv, iu)]


def device_K(K_low, turn):
    """ARKit lowres intrinsics -> the hires frame (scale, then the 180 deg turn in continuous coordinates)."""
    K = K_low.copy()
    K[:2] *= HW / LW
    if turn:
        K[0, 2], K[1, 2] = HW - K[0, 2], HH - K[1, 2]
    return K


class Video:
    """Remote handles and trajectories of one video; reads only what a clip needs."""

    def __init__(self, visit, video):
        sf, ak = SF_URL.format(visit=visit, video=video), AK_URL.format(video=video)
        self.visit, self.video = visit, video
        self.gt_ts, self.gt_P = parse_traj(remote_zip.fetch(sf + "hires_poses.traj").decode())
        self.dev_ts, self.dev_P = parse_traj(remote_zip.fetch(ak + "lowres_wide.traj").decode())
        self.T = np.load(io.BytesIO(remote_zip.fetch(sf + f"{video}_refined_transform.npy")))
        self.z = {a: remote_zip.open_zip(sf + a + ".zip", 1 << 18) for a in ("hires_wide", "hires_depth", "hires_wide_intrinsics")}
        self.z.update({a: remote_zip.open_zip(ak + a + ".zip", 1 << 18)
                       for a in ("lowres_depth", "confidence", "lowres_wide_intrinsics")})
        self.names = {a: {f"{stamp(n):.3f}": n for n in z.namelist() if not n.endswith("/")} for a, z in self.z.items()}
        self.depth_ts = np.array(sorted(stamp(n) for n in self.names["lowres_depth"].values()))
        self.kin_ts = np.array(sorted(stamp(n) for n in self.names["lowres_wide_intrinsics"].values()))
        self.turn, self.turn_deg, self.turn_resid = device_turn(self.gt_ts, self.gt_P, self.dev_ts, self.dev_P, self.T)
        have = set(self.names["hires_wide"]) & set(self.names["hires_depth"]) & set(self.names["hires_wide_intrinsics"])
        self.frames = usable_frames(self.gt_ts, self.gt_P, self.dev_ts, self.depth_ts, have)

    def bytes_fetched(self):
        return sum(z.fp.bytes_fetched for z in self.z.values())

    def frame(self, i, nw, nh, left, top):
        t = self.gt_ts[i]
        key = f"{t:.3f}"
        im = Image.open(io.BytesIO(_read(self.z["hires_wide"], self.names["hires_wide"][key]))).convert("RGB")
        if im.size != (HW, HH):
            raise ValueError(f"{self.video} {key}: image {im.size}")
        im = np.asarray(im.resize((nw, nh), Image.BICUBIC).crop((left, top, left + TW, top + TH)), np.uint8)
        K_gt, _ = parse_pincam(_read(self.z["hires_wide_intrinsics"], self.names["hires_wide_intrinsics"][key]).decode())
        gd = _png(self.z["hires_depth"], self.names["hires_depth"][key]).astype(np.float32) / 1000.0
        jd, _ = nearest(self.depth_ts, t)
        dkey = f"{self.depth_ts[jd]:.3f}"
        ld = _png(self.z["lowres_depth"], self.names["lowres_depth"][dkey]).astype(np.float32) / 1000.0
        conf = _png(self.z["confidence"], self.names["confidence"][dkey])
        ld[conf < 2] = 0.0                                     # high-confidence LiDAR only
        jk, _ = nearest(self.kin_ts, t)
        K_low, _ = parse_pincam(_read(self.z["lowres_wide_intrinsics"],
                                      self.names["lowres_wide_intrinsics"][f"{self.kin_ts[jk]:.3f}"]).decode())
        jp, _ = nearest(self.dev_ts, t)
        P_dev = self.dev_P[jp] @ R180 if self.turn else self.dev_P[jp]
        return dict(image=im, K=adjust_K(K_gt, HW, HH, nw, nh, left, top), pose=self.gt_P[i],
                    depth=resize_depth_nearest(gd, HW, HH, nw, nh, left, top, TW, TH),
                    dev_K=adjust_K(device_K(K_low, self.turn), HW, HH, nw, nh, left, top), dev_pose=P_dev,
                    dev_depth=lidar_on_grid(ld, self.turn, nw, nh, left, top), stamp=key)


def build_video(split, visit, video, out_dir, log=print):
    v = Video(visit, video)
    if v.turn_resid > 5.0:
        raise ValueError(f"{video}: device frame does not match GT frame after a {v.turn} deg turn "
                         f"(median residual {v.turn_resid:.1f} deg)")
    kf = [v.frames[i] for i in keyframes(v.gt_P[v.frames])]
    if len(kf) < N_VIEWS * CLIPS_PER_VIDEO:
        raise ValueError(f"{video}: {len(kf)} usable keyframes")
    nw, nh, left, top = resize_crop_params(HW, HH, TW, TH)
    paths = []
    for c, s in enumerate(clip_starts(len(kf), N_VIEWS, CLIPS_PER_VIDEO, non_overlapping=True)):
        fr = [v.frame(i, nw, nh, left, top) for i in kf[s:s + N_VIEWS]]
        st = lambda k: np.stack([f[k] for f in fr])
        poses, dev_poses = relative_to_first(st("pose")), relative_to_first(st("dev_pose"))
        for i in range(N_VIEWS):
            validate_pose(poses[i], f"{video} GT {i}")
            validate_pose(dev_poses[i], f"{video} device {i}")
        scene = f"sf{visit}"
        clip = dict(images=st("image"), K=st("K"), poses=poses, depth=st("depth").astype(np.float32),
                    dev_K=st("dev_K"), dev_poses=dev_poses, dev_depth=st("dev_depth").astype(np.float32),
                    names=np.array([f["stamp"] for f in fr]), scene=np.array(scene), video_id=np.array(video),
                    device_turn_deg=np.array(v.turn), device_turn_measured_deg=np.array(v.turn_deg),
                    device_turn_residual_deg=np.array(v.turn_resid))
        path = os.path.join(out_dir, f"{split}__{scene}__c{c}.npz")
        np.savez_compressed(path, **clip)
        paths.append(path)
        log(f"[ok] {path}  frames {fr[0]['stamp']}..{fr[-1]['stamp']}  GT depth valid {(clip['depth'] > 0).mean():.0%}  "
            f"LiDAR valid {(clip['dev_depth'] > 0).mean():.0%}  turn {v.turn} (measured {v.turn_deg:.1f}, "
            f"residual {v.turn_resid:.2f} deg)")
    log(f"     {video}: {v.bytes_fetched() / 1e6:.0f} MB fetched")
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--traj_cache", required=True, help="folder for the GT trajectories used for selection")
    ap.add_argument("--only", nargs="*", help="video ids (debugging)")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.traj_cache, exist_ok=True)

    def traj_of(visit, video):
        p = os.path.join(a.traj_cache, f"{video}_hires.traj")
        if not os.path.exists(p):
            open(p, "wb").write(remote_zip.fetch(SF_URL.format(visit=visit, video=video) + "hires_poses.traj"))
        return parse_traj(open(p).read())

    def cached(name, url):
        p = os.path.join(a.traj_cache, name)
        if not os.path.exists(p):
            open(p, "wb").write(remote_zip.fetch(url))
        return p

    def aligned(visit, video):
        sf, akurl = SF_URL.format(visit=visit, video=video), AK_URL.format(video=video)
        dev_ts, dev_P = parse_traj(open(cached(f"{video}_lowres_wide.traj", akurl + "lowres_wide.traj")).read())
        T = np.load(cached(f"{video}_transform.npy", sf + f"{video}_refined_transform.npy"))
        gt_ts, gt_P = traj_of(visit, video)
        turn, measured, resid = device_turn(gt_ts, gt_P, dev_ts, dev_P, T)
        return resid <= 5.0, f"device frame off by {measured:.1f} deg (residual {resid:.1f} after a {turn} deg turn)"

    decisions = []
    sel = select_videos(a.test_csv, traj_of=traj_of, aligned=aligned, log=decisions.append)
    with open(os.path.join(a.out, "selection.csv"), "w", newline="") as f:
        csv.writer(f).writerows([("split", "visit_id", "video_id"), *sel])
    with open(os.path.join(a.out, "selection_log.csv"), "w", newline="") as f:
        csv.writer(f).writerows([("visit_id", "video_id", "decision"), *decisions])
    todo = [s for s in sel if (not a.only or s[2] in a.only) and not all(
        os.path.exists(os.path.join(a.out, f"{s[0]}__sf{s[1]}__c{c}.npz")) for c in range(CLIPS_PER_VIDEO))]
    failed = []
    with ProcessPoolExecutor(a.workers) as ex:  # network latency dominates: videos in parallel
        for s, (paths, err) in zip(todo, ex.map(_build, [(*s, a.out) for s in todo])):
            print(f"done {s[2]}: {len(paths)} clips" + (f"  FAILED: {err}" if err else ""), flush=True)
            if err:
                failed.append(s[2])
    if failed:
        raise SystemExit(f"failed videos: {failed}")


def _build(args):
    try:
        return build_video(*args, log=lambda m: print(m, flush=True)), None
    except Exception as e:  # one bad video must not stop the others
        return [], repr(e)


if __name__ == "__main__":
    main()
