"""Photo check of a clip's camera poses: do the poses explain the feature matches between the clip's own images?

For every pair of views (i, j), SIFT matches (ratio test 0.75) are first cleaned with a RANSAC fundamental matrix
fitted to the matches alone (RANSAC_PX; no pose is used), then scored by their Sampson distance (px) to the
epipolar geometry implied by K and the relative pose under test. A pair with fewer than MIN_MATCHES clean matches is
not scored. A view's score is the median, over its scored partners, of the pair's median Sampson distance; a
misregistered camera disagrees with all its partners, a good one with none.

Used to find ground-truth poses that do not match the photos (SceneFun3D has some; extended E2) and to measure how
well device poses fit the photos. Thresholds are set from ETH3D ground truth (laser-scan registered), see
experiments/E2X.md.

  python -m exp1.photo_check --clips "data/clips/*.npz" --out results/e2x_photo_check/eth3d.csv [--key poses]
"""

import argparse
import glob
import os

import cv2
import numpy as np
import pandas as pd

MIN_MATCHES = 30
RANSAC_PX = 1.5


def _features(images):
    sift = cv2.SIFT_create(2000)
    out = []
    for im in images:
        k, d = sift.detectAndCompute(cv2.cvtColor(np.ascontiguousarray(im), cv2.COLOR_RGB2GRAY), None)
        out.append((np.float32([p.pt for p in k]) if k else np.zeros((0, 2), np.float32), d))
    return out


def fundamental(Ki, Kj, Pi, Pj):
    """F with x_j^T F x_i = 0 for cam2world poses Pi, Pj (OpenCV axes)."""
    T = np.linalg.inv(Pj) @ Pi                     # camera i -> camera j
    R, t = T[:3, :3], T[:3, 3]
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    return np.linalg.inv(Kj).T @ tx @ R @ np.linalg.inv(Ki)


def sampson(F, xi, xj):
    hi = np.c_[xi, np.ones(len(xi))]
    hj = np.c_[xj, np.ones(len(xj))]
    Fx, Ftx = hi @ F.T, hj @ F
    num = np.sum(hj * Fx, 1) ** 2
    return np.sqrt(num / (Fx[:, 0] ** 2 + Fx[:, 1] ** 2 + Ftx[:, 0] ** 2 + Ftx[:, 1] ** 2 + 1e-12))


def match_pairs(images):
    """{(i, j): (xi, xj)} for pairs with at least MIN_MATCHES ratio-test matches."""
    feats = _features(images)
    bf = cv2.BFMatcher()
    pairs = {}
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            (pi, di), (pj, dj) = feats[i], feats[j]
            if di is None or dj is None or len(pi) < 2 or len(pj) < 2:
                continue
            m = [a for a, b in bf.knnMatch(di, dj, k=2) if a.distance < 0.75 * b.distance]
            if len(m) < MIN_MATCHES:
                continue
            xi, xj = pi[[a.queryIdx for a in m]], pj[[a.trainIdx for a in m]]
            # keep matches consistent with SOME epipolar geometry (RANSAC on the matches alone, no pose used)
            F, inl = cv2.findFundamentalMat(xi, xj, cv2.FM_RANSAC, RANSAC_PX, 0.999, 5000)
            if F is None or inl is None or int(inl.sum()) < MIN_MATCHES:
                continue
            inl = inl.ravel().astype(bool)
            pairs[(i, j)] = (xi[inl], xj[inl])
    return pairs


def score(pairs, K, poses, n):
    """Per-pair median Sampson distance (px) and per-view score (median over scored partners; NaN if none)."""
    pair_err = {}
    for (i, j), (xi, xj) in pairs.items():
        if np.linalg.norm(poses[i][:3, 3] - poses[j][:3, 3]) < 1e-6:
            continue                                   # no baseline: epipolar geometry undefined
        pair_err[(i, j)] = float(np.median(sampson(fundamental(K[i], K[j], poses[i], poses[j]), xi, xj)))
    view = np.full(n, np.nan)
    for v in range(n):
        e = [x for (i, j), x in pair_err.items() if v in (i, j)]
        if e:
            view[v] = float(np.median(e))
    return pair_err, view


def check_clip(clip, keys=("poses",)):
    pairs = match_pairs(clip["images"])
    n = len(clip["images"])
    rows = []
    for key in keys:
        K = clip["K"] if key == "poses" else clip.get("dev_K", clip["K"])
        pair_err, view = score(pairs, K, clip[key], n)
        for v in range(n):
            partners = sum(v in p for p in pair_err)
            rows.append(dict(key=key, view=v, sampson_px=view[v], n_partners=partners))
    return rows, len(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", nargs="+", default=["poses"], help="pose arrays to score (poses = GT, dev_poses = device)")
    a = ap.parse_args()
    rows = []
    for path in sorted(glob.glob(a.clips)):
        clip = dict(np.load(path, allow_pickle=False))
        keys = [k for k in a.key if k in clip]
        r, npairs = check_clip(clip, keys)
        cid = os.path.splitext(os.path.basename(path))[0]
        for x in r:
            x.update(clip_id=cid, n_pairs=npairs)
        rows += r
        worst = max((x["sampson_px"] for x in r if x["key"] == "poses" and np.isfinite(x["sampson_px"])), default=np.nan)
        print(f"{cid}: {npairs} matched pairs, worst GT view {worst:.2f} px", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    pd.DataFrame(rows)[["clip_id", "key", "view", "sampson_px", "n_partners", "n_pairs"]].to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
