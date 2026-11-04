"""Extended E2 helpers: room-mode windows and GT frame filters (exp1/arkit_clips.py), the photo check
(exp1/photo_check.py) and the E3 witness fit (exp1/witness.py). Synthetic data only."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from exp1 import arkit_clips as ak
from exp1 import photo_check, witness


def _traj(n=60, seed=0):
    rng = np.random.default_rng(seed)
    ts = np.arange(n) * 0.1
    P = np.tile(np.eye(4), (n, 1, 1))
    rv = np.cumsum(rng.normal(0, 0.01, (n, 3)), 0)
    P[:, :3, :3] = Rot.from_rotvec(rv).as_matrix()
    P[:, :3, 3] = np.c_[np.linspace(0, 3, n), np.sin(np.linspace(0, 2, n)), np.zeros(n)]
    return ts, P


def test_room_windows_spread_and_halves():
    w = ak.room_windows(40)
    assert [len(x) for x in w] == [10, 10, 10]
    assert w[0][0] == 0 and w[0][-1] == 39 and w[1][-1] == 19 and w[2][0] == 20 and w[2][-1] == 39
    with pytest.raises(ValueError):
        ak.room_windows(19)


def test_gt_spikes_finds_single_frame_jumps_only():
    ts, P = _traj()
    assert ak.gt_spikes(ts, P) == set()
    Q = P.copy()
    Q[20, :3, :3] = Q[20, :3, :3] @ Rot.from_euler("z", 180, degrees=True).as_matrix()
    Q[40, :3, 3] += [0.5, 0, 0]
    assert ak.gt_spikes(ts, Q) == {19, 20, 21, 39, 40, 41}  # a jump also spoils its neighbours' interpolation


def test_gt_gross_flags_frames_far_from_device_after_alignment():
    ts, P = _traj()
    T = np.eye(4)
    T[:3, :3] = Rot.from_euler("y", 40, degrees=True).as_matrix()      # laser -> ARKit world
    dev = np.array([T @ p for p in P])                                  # device = GT in the ARKit world
    assert ak.gt_gross(ts, P, ts, dev, np.eye(4), 0) == set()           # one video-wide alignment absorbs T
    G = P.copy()
    for i in (5, 6, 7):                                                 # a flipped GT segment
        G[i, :3, :3] = G[i, :3, :3] @ Rot.from_euler("z", 179, degrees=True).as_matrix()
    assert ak.gt_gross(ts, G, ts, dev, np.eye(4), 0) == {5, 6, 7}
    # a device turned 180 deg about the optical axis is undone by `turn`
    assert ak.gt_gross(ts, P, ts, np.array([d @ ak.R180 for d in dev]), np.eye(4), 180) == set()


def test_photo_check_scores_true_poses_low_and_a_wrong_pose_high():
    rng = np.random.default_rng(0)
    X = rng.uniform([-2, -1.5, 4], [2, 1.5, 7], (400, 3))
    K = np.array([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]])
    poses = []
    for k in range(4):
        P = np.eye(4)
        P[:3, :3] = Rot.from_euler("y", 4 * k, degrees=True).as_matrix()
        P[:3, 3] = [0.3 * k, 0.05 * k, 0]
        poses.append(P)
    proj = []
    for P in poses:
        Xc = (np.linalg.inv(P) @ np.c_[X, np.ones(len(X))].T).T[:, :3]
        x = (K @ Xc.T).T
        proj.append(x[:, :2] / x[:, 2:])
    pairs = {(i, j): (proj[i].astype(np.float32), proj[j].astype(np.float32)) for i in range(4) for j in range(i + 1, 4)}
    _, good = photo_check.score(pairs, [K] * 4, poses, 4)
    assert np.all(good < 1e-3)
    bad = [p.copy() for p in poses]
    bad[2][:3, :3] = bad[2][:3, :3] @ Rot.from_euler("x", 5, degrees=True).as_matrix()
    _, v = photo_check.score(pairs, [K] * 4, bad, 4)
    assert v[2] > 5 and np.argmax(v) == 2


def _poses(n=10, seed=1):
    rng = np.random.default_rng(seed)
    P = np.tile(np.eye(4), (n, 1, 1))
    for i in range(1, n):
        P[i, :3, :3] = Rot.from_rotvec(rng.normal(0, 0.2, 3)).as_matrix()
        P[i, :3, 3] = rng.normal(0, 1, 3)
    return P


def _left(P, L):
    Q = P.copy()
    Q[1:, :3, :3] = L.as_matrix() @ P[1:, :3, :3]
    Q[1:, :3, 3] = L.apply(P[1:, :3, 3])
    return Q


def test_witness_separates_shared_drift_and_single_errors():
    P = _poses()
    clean = witness.fit(P, P)
    assert clean["shared_deg"] < 1e-6 and clean["drift_total_deg"] < 1e-6 and abs(clean["scale"] - 1) < 1e-9
    sh = witness.fit(_left(P, Rot.from_euler("z", 8, degrees=True)), P)
    assert abs(sh["shared_deg"] - 8) < 0.05 and sh["rot_res_deg"].max() < 0.05
    H = P.copy()
    H[4, :3, :3] = H[4, :3, :3] @ Rot.from_euler("x", 15, degrees=True).as_matrix()
    one = witness.fit(H, P)
    assert one["shared_deg"] < 0.5 and abs(one["rot_res_deg"][4] - 15) < 0.5
    assert np.delete(one["rot_res_deg"], 4).max() < 0.5
    D = P.copy()
    for i in range(1, 10):
        D[i, :3, :3] = Rot.from_euler("z", 1.0 * i, degrees=True).as_matrix() @ P[i, :3, :3]
        D[i, :3, 3] = Rot.from_euler("z", 1.0 * i, degrees=True).apply(P[i, :3, 3])
    dr = witness.fit(D, P)
    assert abs(dr["drift_deg_per_view"] - 1.0) < 0.05 and dr["rot_res_deg"].max() < 0.1
    S = P.copy()
    S[:, :3, 3] *= 2
    assert abs(witness.fit(S, P)["scale"] - 2) < 1e-6
