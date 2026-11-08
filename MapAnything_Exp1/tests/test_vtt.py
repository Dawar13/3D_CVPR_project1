"""exp1/vtt.py (E4-A Verify-then-Trust) on synthetic scenes with known geometry: correct hints stay, a wrong camera is
corrected and flagged, a camera the photos cannot see keeps its hint, a heading shared by all hints is corrected via
the translation directions, a wrong depth scale is corrected and weak depth evidence changes nothing."""

import os

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from exp1 import vtt  # noqa: E402

W, H_IMG = 518, 392
K1 = np.array([[500.0, 0, 259.0], [0, 500.0, 196.0], [0, 0, 1]])


def look_at(c, target):
    z = target - c
    z /= np.linalg.norm(z)
    x = np.cross(z, np.array([0.0, 1.0, 0.0]))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], 1)


def scene(n_cam=8, n_pts=400, seed=0, px_noise=0.5):
    rng = np.random.default_rng(seed)
    X = rng.uniform([-2, -1.5, 6], [2, 1.5, 9], size=(n_pts, 3))
    target = np.array([0.0, 0.0, 7.5])
    P = []
    for k in range(n_cam):
        a = np.radians(-25 + 50 * k / (n_cam - 1))
        c = np.array([4 * np.sin(a), 0.3 * np.sin(3 * a), 7.5 - 6 * np.cos(a)])
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = look_at(c, target), c
        P.append(T)
    P = np.array(P)
    K = np.stack([K1] * n_cam)
    uv, vis, depth = [], [], np.zeros((n_cam, H_IMG, W), np.float32)
    for T in P:
        Xc = (X - T[:3, 3]) @ T[:3, :3]
        p = Xc @ K1.T
        u = p[:, :2] / p[:, 2:3] + rng.normal(0, px_noise, size=(n_pts, 2))
        ok = (Xc[:, 2] > 0.1) & (u[:, 0] > 0) & (u[:, 0] < W) & (u[:, 1] > 0) & (u[:, 1] < H_IMG)
        uv.append(u)
        vis.append(ok)
    for v in range(n_cam):   # GT depth hint: the true z of every visible point at its pixel
        Xc = (X - P[v][:3, 3]) @ P[v][:3, :3]
        for (x, y), z, ok in zip(np.round(uv[v]).astype(int), Xc[:, 2], vis[v]):
            if ok and 0 <= x < W and 0 <= y < H_IMG:
                depth[v, y, x] = z
    pairs = {}
    for i in range(n_cam):
        for j in range(i + 1, n_cam):
            m = vis[i] & vis[j]
            if m.sum() >= 30:
                pairs[(i, j)] = (uv[i][m].astype(np.float32), uv[j][m].astype(np.float32))
    return P, K, pairs, depth


@pytest.fixture(scope="module")
def calibrated():
    P, K, pairs, depth = scene()
    edges = vtt.pair_edges(pairs, K)
    (kr, sr, kd, sd), p, _ = vtt.fit_kappa_em(vtt.edge_residuals(edges, P[:, :3, :3], P[:, :3, 3]))
    cfg = dict(vtt_kappa_rot=kr, vtt_kappa_dir=kd, vtt_sys_rot_deg=np.degrees(sr), vtt_sys_dir_deg=np.degrees(sd))
    return P, K, pairs, depth, edges, cfg


def rot_about(axis, deg):
    return vtt.exp_so3(np.radians(deg) * np.asarray(axis, float) / np.linalg.norm(axis))


def test_correct_hints_are_kept(calibrated):
    P, K, pairs, depth, edges, cfg = calibrated
    R, w, pi, info = vtt.fuse_rotations(P[:, :3, :3], P[:, :3, 3], edges, cfg)
    assert np.array_equal(R, P[:, :3, :3])          # labelled right: returned exactly
    assert w.min() > 0.9


def test_one_confident_wrong_edge_does_not_overrule_correct_hints(calibrated):
    """Whole-room clips: often a single matched pair, sometimes wrong by tens of degrees with a tight covariance.
    One edge against the hints is decided by prior reliabilities (Laplace evidence), not by its confidence."""
    P, K, pairs, depth, edges, cfg = calibrated
    ed = dict(next(e for e in edges if (e["i"], e["j"]) == (2, 3)))
    ed["R"] = ed["R"] @ rot_about([0.3, 1, 0.1], 25)     # wrong matches: a wrong relative rotation AND direction
    ed["t"] = rot_about(np.cross(ed["t"], [0.0, 1.0, 0.3]), 30) @ ed["t"]   # axis perpendicular to t
    ed["B"] = vtt.tangent_basis(ed["t"])
    R, w, pi, info = vtt.fuse_rotations(P[:, :3, :3], P[:, :3, 3], [ed], cfg)
    assert max(vtt.angle_deg(P[i, :3, :3], R[i]) for i in range(len(P))) < 0.01, info
    assert info["edge_inlier_prob"][0] < 0.05


def test_one_wrong_camera_is_corrected_and_flagged(calibrated):
    P, K, pairs, depth, edges, cfg = calibrated
    Hh = P[:, :3, :3].copy()
    Hh[3] = Hh[3] @ rot_about([0.3, 1, 0.2], 15)
    R, w, pi, info = vtt.fuse_rotations(Hh, P[:, :3, 3], edges, cfg)
    assert vtt.angle_deg(P[3, :3, :3], R[3]) < 1.0
    assert info["replaced"][3] and sum(info["replaced"]) == 1


def test_camera_without_evidence_keeps_its_hint(calibrated):
    P, K, pairs, depth, edges, cfg = calibrated
    sub = [ed for ed in edges if 5 not in (ed["i"], ed["j"])]
    Hh = P[:, :3, :3].copy()
    Hh[5] = Hh[5] @ rot_about([1, 0, 0], 15)
    R, w, pi, info = vtt.fuse_rotations(Hh, P[:, :3, 3], sub, cfg)
    assert vtt.angle_deg(Hh[5], R[5]) < 1e-3          # no photo evidence: the posterior is the prior


def test_shared_heading_is_corrected_by_directions(calibrated):
    P, K, pairs, depth, edges, cfg = calibrated
    S = rot_about([0, 1, 0], 10)                        # world vertical in this scene
    Hh = P[:, :3, :3].copy()
    for i in range(1, len(P)):
        Hh[i] = S @ Hh[i]
    R, w, pi, info = vtt.fuse_rotations(Hh, P[:, :3, 3], edges, cfg)
    before = np.mean([vtt.angle_deg(P[i, :3, :3], Hh[i]) for i in range(1, len(P))])
    after = np.mean([vtt.angle_deg(P[i, :3, :3], R[i]) for i in range(1, len(P))])
    assert before > 9 and after < 2.0


def test_eval_style_shared_rotation_is_corrected(calibrated):
    """corrupt_geometry 'shared_rot' (E1/E3 shared heading and mount families): R' = R Exp(theta u) on every view but
    view 0; the relative rotations then side with the 7 wrong cameras, only the directions show the error."""
    P, K, pairs, depth, edges, cfg = calibrated
    D = rot_about([0.2, 1, -0.3], 5)
    Hh = P[:, :3, :3].copy()
    for i in range(1, len(P)):
        Hh[i] = Hh[i] @ D
    R, w, pi, info = vtt.fuse_rotations(Hh, P[:, :3, 3], edges, cfg)
    after = max(vtt.angle_deg(P[i, :3, :3], R[i]) for i in range(len(P)))
    assert after < 1.0, (after, info["start"], info["neg_log_evidence_by_start"], info["replaced"])


def test_noisy_centres_do_not_move_correct_rotations(calibrated):
    P, K, pairs, depth, edges, cfg = calibrated
    rng = np.random.default_rng(1)
    C = P[:, :3, 3] + rng.normal(0, 0.05, size=(len(P), 3))   # 5 cm centre noise, baselines ~0.5-4 m
    R, w, pi, info = vtt.fuse_rotations(P[:, :3, :3], C, edges, cfg)
    assert max(vtt.angle_deg(P[i, :3, :3], R[i]) for i in range(len(P))) < 0.3
    assert info["sigma_c"] > 0.01 and w.min() > 0.9


def test_depth_scale_corrected_and_weak_evidence_kept(calibrated):
    P, K, pairs, depth, edges, cfg = calibrated
    bad = depth.copy()
    bad[[2, 5]] *= 1.5
    new_poses, new_depth, info = vtt.sanitize(pairs, K, P, bad, (2, 5), ["intrinsics", "pose", "depth"], cfg, edges)
    for v in (2, 5):
        k = np.median(new_depth[v][depth[v] > 0] / depth[v][depth[v] > 0])
        assert abs(k - 1.0) < 0.03
    # one pair per depth view with 4 matches (< 10 samples): no evidence, no change
    few = {k: (v[0][:4], v[1][:4]) for k, v in pairs.items() if k in ((1, 2), (4, 5))}
    _, kept, info2 = vtt.sanitize(few, K, P, bad, (2, 5), ["intrinsics", "pose", "depth"], cfg, edges)
    assert np.array_equal(kept, bad) and all(d["s"] == 0.0 for d in info2["depth"].values())


def test_depth_pooling(calibrated):
    """A scale error shared by the depth hints reaches a view without matches; one wrong view among correct ones does
    not drag the others."""
    P, K, pairs, depth, edges, cfg = calibrated
    inputs = ["intrinsics", "pose", "depth"]
    bad = depth.copy()
    bad[[2, 5, 7]] *= 1.5
    no7 = {k: v for k, v in pairs.items() if 7 not in k}
    _, nd, info = vtt.sanitize(no7, K, P, bad, (2, 5, 7), inputs, cfg, edges)
    assert info["depth"][7]["m"] is None
    for v in (2, 5, 7):
        k = np.median(nd[v][depth[v] > 0] / depth[v][depth[v] > 0])
        assert abs(k - 1.0) < 0.03, (v, k)
    one = depth.copy()
    one[5] *= 1.5
    _, nd, info = vtt.sanitize(pairs, K, P, one, (2, 5, 7), inputs, cfg, edges)
    for v, want in ((2, 1.0), (5, 1.0), (7, 1.0)):
        k = np.median(nd[v][depth[v] > 0] / depth[v][depth[v] > 0])
        assert abs(k - want) < 0.03, (v, k, info["depth"])


def test_one_to_one_matches():
    xi = np.array([[10, 10], [10, 10], [50, 50], [60, 60], [70, 70]], float)
    xj = np.array([[20, 20], [20, 20], [90, 90], [90, 90], [30, 30]], float)
    a, b = vtt.one_to_one(xi, xj)
    # the exact repeat counts once; the two keypoints matched to the same (90, 90) are ambiguous and dropped
    assert a.tolist() == [[10, 10], [70, 70]] and b.tolist() == [[20, 20], [30, 30]]


def test_inputs_without_pose_are_untouched(calibrated):
    P, K, pairs, depth, edges, cfg = calibrated
    poses, d, info = vtt.sanitize(pairs, K, P, depth, (2,), ["intrinsics", "depth"], cfg, edges)
    assert np.array_equal(poses, P) and np.array_equal(d, depth) and "w_hint" not in info


def test_resolution_is_a_no_op_where_the_matches_decide(calibrated):
    """Synthetic scene, well conditioned: correct hints leave every pair as is; a 5 deg wrong hint does not pull a
    pair to it (the restart from the hint returns to the photo mode or costs clearly more)."""
    P, K, pairs, depth, edges, cfg = calibrated
    out, n = vtt.resolve_with_hints(edges, P[:, :3, :3], P[:, :3, 3], cfg)
    assert n == 0
    Hh = P[:, :3, :3].copy()
    Hh[3] = Hh[3] @ rot_about([0.3, 1, 0.2], 5)
    out, n = vtt.resolve_with_hints(edges, Hh, P[:, :3, 3], cfg)
    assert n == 0


REAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "clips_e2x_cal",
                    "cal__sf483632__c2.npz")


@pytest.mark.skipif(not os.path.exists(REAL), reason="SceneFun3D cal clip not on this machine")
def test_ambiguous_video_pairs_do_not_overrule_correct_hints():
    """cal__sf483632__c2 (SceneFun3D cal): several pairs with hundreds of matches have a second relative pose, 12-25
    deg away, that fits the matches as well as the ground truth does (both at sub-pixel Sampson error)."""
    import json
    from exp1 import photo_check
    from exp1.datasets import load_clip
    cal = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "e4_vtt_cal", "chosen.json")
    cfg = {k: v for k, v in json.load(open(cal)).items() if k.startswith("vtt_")}
    clip = load_clip(REAL)
    pairs = photo_check.match_pairs(clip["images"])
    new, _, info = vtt.sanitize(pairs, clip["K"], clip["poses"], None, (), ["intrinsics", "pose"], cfg)
    assert info["n_edges_resolved_to_hint"] > 0
    assert max(vtt.angle_deg(clip["poses"][i, :3, :3], new[i, :3, :3]) for i in range(len(new))) < 0.01
