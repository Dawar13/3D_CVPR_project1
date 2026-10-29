"""SE(3)/SO(3) helpers used by the corruption utility and the metrics.

Convention throughout this experiment: camera poses are 4x4 **OpenCV cam2world**
matrices (x right, y down, z forward), which is what MapAnything's
`camera_poses` input expects according to its README. Verify this against the
installed repo before running (see README.md, "Things to verify").
"""

import numpy as np

ROT_ORTHO_TOL = 1e-5
ROT_DET_TOL = 1e-5


def so3_exp(omega):
    """Rodrigues: axis-angle vector (rad) -> 3x3 rotation matrix."""
    omega = np.asarray(omega, dtype=np.float64)
    theta = np.linalg.norm(omega)
    if theta < 1e-12:
        return np.eye(3)
    k = omega / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def so3_log(R):
    """Inverse of so3_exp: 3x3 rotation -> axis-angle vector (rad), angle in [0, pi]."""
    theta = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if theta < 1e-8:
        return 0.5 * w
    if np.pi - theta < 1e-3:  # sin(theta) ~ 0: read the axis from sym(R) - cos I = (1 - cos) n n^T
        A = (0.5 * (R + R.T) - np.cos(theta) * np.eye(3)) / (1.0 - np.cos(theta))
        k = int(np.argmax(np.diag(A)))
        n = A[:, k] / np.sqrt(A[k, k])
        if n @ w < 0:
            n = -n
        return theta * n / np.linalg.norm(n)
    return theta / (2.0 * np.sin(theta)) * w


def rotation_angle_deg(R):
    """Geodesic angle of a rotation matrix, in degrees, numerically clamped."""
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def rotation_error_deg(R_a, R_b):
    return rotation_angle_deg(R_a.T @ R_b)


def rotation_error_deg_precise(R_a, R_b):
    """Same angle as rotation_error_deg, via atan2(|vee(R - R^T)| / 2, (tr R - 1) / 2), which stays
    accurate near 0 deg (arccos of float32-rounded matrices has a ~0.03 deg floor). Used by the
    E2 metrics; the E1 metrics keep rotation_error_deg."""
    R = R_a.T @ R_b
    s = 0.5 * np.linalg.norm([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return float(np.degrees(np.arctan2(s, (np.trace(R) - 1.0) / 2.0)))


def random_unit_vector(rng):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def validate_pose(T, name="pose"):
    """Raise ValueError unless T is a finite 4x4 rigid transform."""
    T = np.asarray(T)
    if T.shape != (4, 4):
        raise ValueError(f"{name}: expected shape (4,4), got {T.shape}")
    if not np.all(np.isfinite(T)):
        raise ValueError(f"{name}: non-finite values")
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError(f"{name}: bottom row is {T[3]}, expected [0,0,0,1]")
    R = T[:3, :3]
    ortho = np.abs(R.T @ R - np.eye(3)).max()
    if ortho > ROT_ORTHO_TOL:
        raise ValueError(f"{name}: R^T R deviates from I by {ortho:.2e}")
    det = np.linalg.det(R)
    if abs(det - 1.0) > ROT_DET_TOL:
        raise ValueError(f"{name}: det(R) = {det:.6f}, expected +1")
    return True


def invert_pose(T):
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def relative_to_first(poses):
    """Re-express cam2world poses so that view 0 is the identity."""
    T0_inv = invert_pose(poses[0])
    return np.stack([T0_inv @ T for T in poses])


def camera_centres(poses):
    return np.asarray(poses)[:, :3, 3]


def trajectory_scale(poses):
    """Scene-length reference L used to define translation noise.

    L = median distance of the selected camera centres from their centroid.
    It is computed on the CLEAN poses of the clip and stored in the manifest.
    """
    c = camera_centres(poses)
    return float(np.median(np.linalg.norm(c - c.mean(0), axis=1)))


def quat_wxyz_to_R(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
