"""Prior-rejection controls (Stage G). No learned components."""

import numpy as np

from .geometry import rotation_error_deg


def rgbcheck_mask(input_poses, rgb_pred_poses, rot_tau, dir_tau):
    """True = keep the pose. Both pose sets are cam2world relative to their view 0."""
    T0 = np.linalg.inv(rgb_pred_poses[0])
    pred = np.stack([T0 @ T for T in rgb_pred_poses])
    keep = np.ones(len(input_poses), bool)
    for i in range(1, len(input_poses)):
        rot_err = rotation_error_deg(input_poses[i][:3, :3], pred[i][:3, :3])
        a, b = input_poses[i][:3, 3], pred[i][:3, 3]
        cosang = a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
        dir_err = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        keep[i] = rot_err <= rot_tau and dir_err <= dir_tau
    return keep
