"""E4-A "Verify-then-Trust" (experiments/E4.md): test-time hint sanitising by robust MAP fusion of the hints with the
photos' own two-view geometry. Model-agnostic (it only rewrites the hints), CPU only, no training.

Model (per clip, cameras i = 0..N-1, hinted cam2world poses P_i = [H_i | c_i], OpenCV axes):
  unknowns    camera rotations R_i (the hinted centres c_i are kept; E1: position errors are mostly absorbed).
  hint prior  each hint rotation is right, H_i = R_i Exp(n), n ~ N(0, sigma_h^2 I), with probability pi, or wrong
              (uniform over the rotation ball) with probability 1 - pi; sigma_h = vtt_sigma_h_deg and pi = a / (a + b)
              are declared priors (fitting them per clip needs a trustworthy reference, which the photos are not
              where their graph has no cycles). w_i = P(hint i is right | data) is reported.
  photo edges for every image pair with RANSAC-verified matches (exp1/photo_check.match_pairs): the essential matrix
              in the hinted intrinsics, refined on all inliers by robust Gauss-Newton on the Sampson error (the
              "gold standard" two-view estimate), gives a relative rotation Rhat_ij (camera i -> j), the direction
              that_ij of camera i's centre seen from camera j, and their first-order covariances Sr_ij (3x3) and
              St_ij (2x2, tangent plane of that_ij) from the same Jacobian. The covariances carry the geometry
              (parallax, field of view, number and spread of the matches); a pair that cannot fix its direction
              (pure rotation) gets an unbounded St. Each edge is right (Gaussian residuals with covariances
              kappa_rot^2 Sr, kappa_dir^2 St) or wrong (uniform rotation and direction); the prior that it is right
              grows with its inlier count, pe(n) = sigmoid(pe_a + pe_b log n). kappa_* and pe_* are fitted by EM on
              dev / cal clips only (exp1/vtt_calibrate.py; kappa on ACCURATE ground truth).
  centres     the hinted centres are noisy too (GPS / odometry): the direction d_ij = unit(c_i - c_j) then has angular
              noise sqrt(2) sigma_c / |c_i - c_j| per axis, added to the direction covariance. sigma_c is estimated
              once per clip (empirical Bayes) at the photo rotations placed by the directions (start 3 below): centre
              noise is independent per camera, so it is what remains once the best global rotation is removed.
  energy      E = - sum_edges log(pe N(r; kappa_rot^2 Sr) N(a; C_dir) + (1 - pe) U_rot U_dir)
                  - sum_i log(pi N(Log(H_i^T R_i); sigma_h^2 I) + (1 - pi) U_rot) - log Beta(pi; a, b),
              C_dir = kappa_dir^2 St + 2 sigma_c^2 / |c_i - c_j|^2 I: the negative log joint, both mixtures summed
              over their labels. EM at the fixed hyperparameters: responsibilities of hints (w) and edges (v), then a
              Gauss-Newton step on SO(3) (right perturbations).
  starts      E is not convex: EM from three starts, each converging to one explanation (which hints and edges are
              wrong); the explanation kept has the highest Laplace evidence, E + 1/2 log det(Hessian), not the
              lowest E: integrating over the rotations removes the edge a single confident measurement has in a
              peak-density comparison, so one edge against one hint is decided by their prior reliabilities, while
              edges that confirm each other outweigh a hint. All explanations share the hyperparameters (sigma_c is
              estimated once per clip, before and outside the EM). The starts: (1) the hints;
              (2) the photos' own rotations (rotation averaging per connected component) placed in the world by
              the rotation hints (robust L1 mean); (3) the same photo rotations placed in the world by the
              translation directions to the hinted centres (robust weighted Wahba fit). (3) is what makes an error
              shared by all rotation hints visible: relative rotations cannot see it, directions can.
  decision    per hint, a leave-one-out posterior-predictive check inside the chosen explanation: camera i's posterior
              from everything except hint i (mode R_-i, covariance Sigma_-i); with r = Log(H_i^T R_-i), the hint is
              inconsistent if r^T (sigma_h^2 I + Sigma_-i)^-1 r > chi2_3(1 - alpha), alpha = vtt_alpha (0.01). An
              inconsistent hint is replaced only if the evidence against it survives the removal of any single photo
              pair (unless that pair is a priori more reliable than a hint, pe(n) > pi: for one pair against one hint
              the posterior odds are the prior odds), by its posterior mode, re-estimated with the other hints fixed;
              a consistent hint is returned exactly (nudging it gains of the order of its own small noise but carries
              the full risk of unmodelled photo bias). A camera without photo edges has a wide prediction and keeps
              its hint. (The mixture labels w_i are not used for the decision: against a uniform outlier density a
              hint 5 sigma from a precise posterior still counts as "right".)
Depth (only with metric pose hints, which fix the units): matches triangulated with the sanitised rotations and
hinted centres give log(depth hint / triangulated depth) per match in each depth view; median m_v with standard error
se_v (depth_evidence). The same label logic (depth_decide): the scale shared by the depth hints, and each view alone,
is right or wrong (prior as for hints); evidence noise se_v^2 + sys^2 (a calibrated model-error floor); the most
probable labels by exact enumeration; only depth hints labelled wrong are rescaled (posterior mean).
"""

import numpy as np

try:
    import cv2
except ImportError:   # tests that supply pairs and edges directly do not need OpenCV
    cv2 = None

DEFAULTS = dict(vtt_sigma_h_deg=1.0, vtt_sigma_h_min_deg=0.05, vtt_pi_a=9.0, vtt_pi_b=1.0, vtt_kappa_rot=1.0,
                vtt_kappa_dir=1.0, vtt_pe_a=1.5, vtt_pe_b=0.0, vtt_tau_depth=0.35, vtt_iters=30, vtt_min_inliers=15,
                vtt_min_parallax_deg=1.0, vtt_sigma_min_px=0.1, vtt_sys_rot_deg=0.0, vtt_sys_dir_deg=0.0,
                vtt_depth_sys=0.0, vtt_alpha=0.01, vtt_pose_scale_sd=0.05)
CHI2_3_Q = {0.05: 7.8147, 0.01: 11.3449, 0.001: 16.2662}   # chi-square (3 dof) upper quantiles
CAUCHY_C = 2.3849   # Cauchy loss of the two-view refinement (robust to wrong matches within a pair)
U_BALL = 3.0 / (4.0 * np.pi ** 4)   # uniform density over the rotation-vector ball of radius pi
SC_GRID = np.r_[0.0, np.geomspace(1e-4, 1.0, 49)]   # sigma_c candidates, in units of the median baseline
DIR_MAX_SD = 0.5    # rad: a direction whose own standard deviation exceeds this carries no usable information
NULL_VAR = 1e12     # variance factor for directions the two-view Jacobian does not constrain


def cfg_get(cfg, k):
    return (cfg or {}).get(k, DEFAULTS[k])


# ------------------------------------------------------------------------------------------------- SO(3) helpers
def hat(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def exp_so3(v):
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3) + hat(v)
    k = hat(v / th)
    return np.eye(3) + np.sin(th) * k + (1 - np.cos(th)) * k @ k


def log_so3(R):
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    th = np.arccos(c)
    if th < 1e-9:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2
    if np.pi - th < 1e-6:   # near pi: axis from the symmetric part
        A = (R + np.eye(3)) / 2
        ax = A[:, np.argmax(np.diag(A))]
        return th * ax / np.linalg.norm(ax)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return th / (2 * np.sin(th)) * w


def angle_deg(Ra, Rb):
    return float(np.degrees(np.linalg.norm(log_so3(Ra.T @ Rb))))


def tangent_basis(t):
    """3x2 orthonormal basis of the plane orthogonal to the unit vector t."""
    a = np.array([1.0, 0.0, 0.0]) if abs(t[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    b1 = np.cross(t, a)
    b1 /= np.linalg.norm(b1)
    return np.stack([b1, np.cross(t, b1)], 1)


def sphere_log(t, u, B):
    """Tangent vector (in basis B at t) pointing from unit t to unit u, with length = the angle between them."""
    th = np.arccos(np.clip(t @ u, -1.0, 1.0))
    v = u - (t @ u) * t
    nv = np.linalg.norm(v)
    if nv < 1e-15:
        return np.array([th, 0.0])
    return th * (B.T @ v) / nv


# ---------------------------------------------------------------------------------------------- photo geometry
def _cauchy(x):
    return 0.5 * CAUCHY_C ** 2 * np.log1p((x / CAUCHY_C) ** 2)


def _cauchy_w(x):
    """IRLS weight of the Cauchy loss at the standardised residual x."""
    return 1.0 / (1.0 + (x / CAUCHY_C) ** 2)


def _sampson(R, t, a, b):
    """Signed Sampson residuals (normalised coordinates) of homogeneous points a (camera i), b (camera j), E = [t]x R."""
    E = hat(t) @ R
    Ea = a @ E.T
    Eb = b @ E
    return np.sum(b * Ea, 1) / np.sqrt(Ea[:, 0] ** 2 + Ea[:, 1] ** 2 + Eb[:, 0] ** 2 + Eb[:, 1] ** 2 + 1e-18)


def _retract(R, t, p):
    t2 = t + tangent_basis(t) @ p[3:]
    return R @ exp_so3(p[:3]), t2 / np.linalg.norm(t2)


def refine_two_view(R, t, a, b, iters=20, sig_min=0.0):
    """Robust Gauss-Newton on the signed Sampson residuals of all inliers (Hartley & Zisserman, 2nd ed., 11.4.3),
    5 dof: R <- R Exp(p[:3]), t <- unit(t + B(t) p[3:]); Cauchy IRLS at the robust residual scale.
    Returns R, t, sigma (robust residual scale) and Cov (5x5) = sigma^2 (J^T W J)^-1 at the solution, in that
    parametrisation (directions the data do not constrain get variance NULL_VAR * sigma^2). sigma is floored at
    sig_min, the keypoint localisation precision: identical or near-identical images match with zero residual,
    which says nothing about the noise."""
    h = 1e-7

    def lin(R, t):
        r0 = _sampson(R, t, a, b)
        J = np.empty((len(r0), 5))
        for k in range(5):
            p = np.zeros(5)
            p[k] = h
            J[:, k] = (_sampson(*_retract(R, t, p), a, b) - r0) / h
        sig = max(1.4826 * float(np.median(np.abs(r0))), sig_min, 1e-12)
        return r0, J, sig, _cauchy_w(np.abs(r0) / sig)

    for _ in range(iters):
        r0, J, sig, wt = lin(R, t)
        A = (J * wt[:, None]).T @ J
        dp = -np.linalg.solve(A + 1e-12 * np.trace(A) * np.eye(5), (J * wt[:, None]).T @ r0)
        R, t = _retract(R, t, dp)
        if np.linalg.norm(dp) < 1e-10:
            break
    r0, J, sig, wt = lin(R, t)
    A = (J * wt[:, None]).T @ J
    ev, V = np.linalg.eigh(A)
    inv = np.where(ev > 1e-12 * max(ev.max(), 1e-300), 1.0 / np.maximum(ev, 1e-300), NULL_VAR)
    return R, t, sig, sig ** 2 * (V * inv) @ V.T


def one_to_one(xi, xj, grid_px=0.5):
    """Correspondences as evidence: a match repeated exactly (SIFT repeats a keypoint at one location with several
    orientations) counts once, and a keypoint matched to more than one distinct partner is ambiguous and dropped
    (the ratio test does not enforce one-to-one matching). Locations are compared on a grid_px grid."""
    xi, xj = np.asarray(xi, np.float64), np.asarray(xj, np.float64)
    if len(xi) == 0:
        return xi, xj
    ki, kj = np.round(xi / grid_px).astype(np.int64), np.round(xj / grid_px).astype(np.int64)
    _, first = np.unique(np.c_[ki, kj], axis=0, return_index=True)
    first = np.sort(first)
    ki, kj, xi, xj = ki[first], kj[first], xi[first], xj[first]
    _, inv_i, cnt_i = np.unique(ki, axis=0, return_inverse=True, return_counts=True)
    _, inv_j, cnt_j = np.unique(kj, axis=0, return_inverse=True, return_counts=True)
    keep = (cnt_i[inv_i.ravel()] == 1) & (cnt_j[inv_j.ravel()] == 1)
    return xi[keep], xj[keep]


def pair_edges(pairs, K, cfg=None):
    """Two-view geometry per matched pair. pairs: {(i, j): (xi, xj)} pixel matches (RANSAC-clean), reduced to
    one-to-one correspondences first (one_to_one). Returns a list of dicts: i, j, R (i -> j), t (unit, camera i's
    centre in camera j's frame), Sr (3x3 covariance of R, right perturbation), St (2x2 covariance of t in the tangent
    basis B), B, n (inliers), e (robust Sampson residual scale, normalised coordinates), parallax (rad, median ray
    angle; logged only)."""
    if cv2 is None:
        raise ImportError("OpenCV is needed to estimate pair geometry")
    edges = []
    for (i, j), (xi, xj) in pairs.items():
        xi, xj = one_to_one(xi, xj)
        ni = cv2.undistortPoints(np.asarray(xi, np.float64).reshape(-1, 1, 2), K[i], None).reshape(-1, 2)
        nj = cv2.undistortPoints(np.asarray(xj, np.float64).reshape(-1, 1, 2), K[j], None).reshape(-1, 2)
        if len(ni) < 5:
            continue
        f = 0.5 * (K[i][0, 0] + K[j][0, 0])
        E, inl = cv2.findEssentialMat(ni, nj, np.eye(3), cv2.RANSAC, 0.999, 1.5 / f)
        if E is None or E.shape != (3, 3) or inl is None:
            continue
        n, R, t, mask = cv2.recoverPose(E, ni, nj, np.eye(3), mask=inl.copy())
        m = mask.ravel().astype(bool)
        if n < cfg_get(cfg, "vtt_min_inliers"):
            continue
        t = t.ravel() / max(np.linalg.norm(t), 1e-12)
        a, b = np.c_[ni[m], np.ones(m.sum())], np.c_[nj[m], np.ones(m.sum())]
        R, t, sig, Cov = refine_two_view(R, t, a, b, sig_min=cfg_get(cfg, "vtt_sigma_min_px") / f)
        ra = a / np.linalg.norm(a, axis=1, keepdims=True)
        rb = (b @ R) / np.linalg.norm(b, axis=1, keepdims=True)
        par = float(np.median(np.arccos(np.clip(np.sum(ra * rb, 1), -1, 1))))
        edges.append(dict(i=int(i), j=int(j), R=R, t=t, B=tangent_basis(t), Sr=Cov[:3, :3], St=Cov[3:, 3:],
                          n=int(n), e=sig, parallax=par, a=a, b=b, sig_min=cfg_get(cfg, "vtt_sigma_min_px") / f))
    return edges


def robust_cost(R, t, a, b, sig):
    """Sum of the Cauchy loss of the Sampson residuals at scale sig (the refinement's own loss)."""
    x = _sampson(R, t, a, b) / sig
    return float(np.sum(0.5 * CAUCHY_C ** 2 * np.log1p((x / CAUCHY_C) ** 2)))


def resolve_with_hints(edges, H, centres, cfg=None):
    """Two-view ambiguity (short-baseline video: rotation-translation / bas-relief; near-planar views): the matches
    of a pair can be explained by more than one relative pose, and the refinement may settle in the wrong one; its
    covariance is local and cannot show this. For each pair, the refinement is restarted from the HINTED relative
    pose (R_j^T R_i, and the direction to the hinted centre where the centres differ); if it converges to another
    mode whose robust Sampson cost is statistically equivalent, 2 (cost_hint - cost_photo) / kappa_rot^2 <
    chi2_3(1 - alpha) (the same alpha and the same kappa that scale the pair's covariance), the matches cannot tell
    the two apart and the pair enters the fusion at the hint-consistent mode (the prior breaks the tie). Where the
    matches contradict the hint, the restart returns to the photo mode or costs clearly more, and the pair is
    unchanged. Returns (edges, n_resolved)."""
    q = CHI2_3_Q[cfg_get(cfg, "vtt_alpha")]
    k2 = cfg_get(cfg, "vtt_kappa_rot") ** 2
    out, nres = [], 0
    for ed in edges:
        i, j = ed["i"], ed["j"]
        if "a" not in ed:
            out.append(ed)
            continue
        Rh = H[j].T @ H[i]
        d = centres[i] - centres[j] if centres is not None else np.zeros(3)
        th = H[j].T @ d / np.linalg.norm(d) if np.linalg.norm(d) > 1e-9 else ed["t"]
        if angle_deg(Rh, ed["R"]) < 1e-3:
            out.append(ed)
            continue
        R2, t2, sig2, Cov2 = refine_two_view(Rh, th, ed["a"], ed["b"], sig_min=ed["sig_min"])
        lrt = 2.0 * (robust_cost(R2, t2, ed["a"], ed["b"], ed["e"]) - robust_cost(ed["R"], ed["t"], ed["a"], ed["b"], ed["e"]))
        if lrt / k2 < q and angle_deg(R2, ed["R"]) > 0.5 * angle_deg(Rh, ed["R"]) and angle_deg(R2, ed["R"]) > 1e-3:
            out.append(dict(ed, R=R2, t=t2, B=tangent_basis(t2), Sr=Cov2[:3, :3], St=Cov2[3:, 3:], e=sig2,
                            resolved_from_deg=angle_deg(R2, ed["R"]), lrt=float(lrt)))
            nres += 1
        else:
            out.append(ed)
    return out, nres


# ------------------------------------------------------------------------------------------------- fusion (MAP)
LOG_U_BALL = float(np.log(U_BALL))
LOG_U_SPHERE = float(-np.log(4 * np.pi))   # uniform density of a direction on the unit sphere


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -700, 700)))


def edge_prior(ed, cfg):
    """Prior probability that a two-view edge is right, sigmoid(pe_a + pe_b log n) (exp1/vtt_calibrate.py)."""
    p = _sigmoid(cfg_get(cfg, "vtt_pe_a") + cfg_get(cfg, "vtt_pe_b") * np.log(max(ed["n"], 1)))
    return float(np.clip(p, 1e-6, 1 - 1e-6))


def _dir_ok(ed, centres):
    """A direction term exists if the hinted centres differ and the photos fix the direction (bounded St)."""
    if centres is None or np.linalg.norm(centres[ed["i"]] - centres[ed["j"]]) <= 1e-9:
        return False
    return np.sqrt(np.linalg.eigvalsh(ed["St"]).max()) < DIR_MAX_SD


def rot_cov(ed, cfg):
    """Covariance of an edge's relative rotation: kappa_rot^2 Sr (first order, scaled) + sys_rot^2 I, the model-error
    floor (rolling shutter, residual distortion, ...: errors that do not shrink with more matches)."""
    return cfg_get(cfg, "vtt_kappa_rot") ** 2 * ed["Sr"] + np.radians(cfg_get(cfg, "vtt_sys_rot_deg")) ** 2 * np.eye(3)


def dir_cov_photo(ed, cfg):
    """Covariance of an edge's translation direction from the photos alone (same form as rot_cov)."""
    return cfg_get(cfg, "vtt_kappa_dir") ** 2 * ed["St"] + np.radians(cfg_get(cfg, "vtt_sys_dir_deg")) ** 2 * np.eye(2)


def gauss_logpdf(x, C):
    _, logdet = np.linalg.slogdet(2 * np.pi * C)
    return float(-0.5 * (x @ np.linalg.solve(C, x) + logdet))


def _rot_part(R, ed, cfg):
    """M = R_j^T R_i, residual r = Log(Rhat^T M), its covariance kappa_rot^2 Sr, inlier log density."""
    M = R[ed["j"]].T @ R[ed["i"]]
    r = log_so3(ed["R"].T @ M)
    C = rot_cov(ed, cfg)
    return M, r, C, gauss_logpdf(r, C)


def _dir_cov(ed, centres, cfg, sc):
    b = np.linalg.norm(centres[ed["i"]] - centres[ed["j"]])
    return dir_cov_photo(ed, cfg) + (2.0 * sc ** 2 / b ** 2) * np.eye(2)


def _dir_part(R, ed, centres, cfg, sc):
    """u = R_j^T unit(c_i - c_j), residual a (tangent plane at that), covariance C, inlier log density."""
    d = centres[ed["i"]] - centres[ed["j"]]
    u = R[ed["j"]].T @ (d / np.linalg.norm(d))
    a = sphere_log(ed["t"], u, ed["B"])
    C = _dir_cov(ed, centres, cfg, sc)
    return u, a, C, gauss_logpdf(a, C)


def _edge_mix(R, ed, centres, cfg, sc, dirs):
    """One edge as a two-component mixture: right (rotation and, if usable, direction Gaussian with their
    covariances) with prior edge_prior, or wrong (uniform rotation and direction). Returns the parts and the two log
    likelihoods (inlier, outlier)."""
    pe = edge_prior(ed, cfg)
    M, r, Cr, lr = _rot_part(R, ed, cfg)
    li, lo = np.log(pe) + lr, np.log(1 - pe) + LOG_U_BALL
    D = None
    if dirs and _dir_ok(ed, centres):
        u, a, Cd, ld = _dir_part(R, ed, centres, cfg, sc)
        li, lo = li + ld, lo + LOG_U_SPHERE
        D = (u, a, Cd)
    return M, r, Cr, D, li, lo


def _hint_logs(rh, pi, sh):
    """Per hint, log of (inlier) pi N(rh; 0, sh^2 I) and (outlier) (1 - pi) U."""
    li = np.log(pi) - 1.5 * np.log(2 * np.pi * sh ** 2) - 0.5 * np.sum(rh ** 2, 1) / sh ** 2
    return li, np.full(len(rh), np.log(1 - pi) + LOG_U_BALL)


def _log_beta(pi, cfg):
    a, b = cfg_get(cfg, "vtt_pi_a"), cfg_get(cfg, "vtt_pi_b")
    return (a - 1) * np.log(max(pi, 1e-300)) + (b - 1) * np.log(max(1 - pi, 1e-300))


def _fit_sigma_c(R, centres, edges, cfg):
    """Empirical Bayes: the sigma_c (on SC_GRID x the median baseline) maximising the mixture likelihood of the
    direction terms at R. Called at the photo rotations placed by the directions (fuse_rotations): what is left of
    the direction residuals once the best global rotation is removed is the centre-hint noise (independent per
    camera), not a shared rotation error of the hints."""
    D = [ed for ed in edges if _dir_ok(ed, centres)]
    if not D:
        return 0.0
    b = np.array([np.linalg.norm(centres[ed["i"]] - centres[ed["j"]]) for ed in D])
    pe = np.array([edge_prior(ed, cfg) for ed in D])
    grid = SC_GRID * np.median(b)
    nll = []
    for sc in grid:
        ld = np.array([_dir_part(R, ed, centres, cfg, sc)[3] for ed in D])
        nll.append(-np.sum(np.logaddexp(np.log(pe) + ld, np.log(1 - pe) + LOG_U_SPHERE)))
    return float(grid[int(np.argmin(nll))])


def energy(R, H, centres, edges, cfg, pi, sh, sc):
    """E = -log p(edges, hints, R | pi, sigma_h, sigma_c) - log Beta(pi): the negative log joint, both mixtures
    summed over their labels."""
    E = 0.0
    for ed in edges:
        li, lo = _edge_mix(R, ed, centres, cfg, sc, True)[4:]
        E -= np.logaddexp(li, lo)
    rh = np.array([log_so3(H[i].T @ R[i]) for i in range(len(H))])
    E -= np.sum(np.logaddexp(*_hint_logs(rh, pi, sh))) + _log_beta(pi, cfg)
    return float(E)


def _system(R, H, centres, edges, cfg, use_hints, anchor, w, sh, sc, dirs, v_fixed=None):
    """Gauss-Newton normal equations (Hm, g) at R with the current responsibilities (hint w; edge v computed here,
    or v_fixed). Returns Hm, g, v."""
    N = len(R)
    Hm = np.zeros((3 * N, 3 * N))
    g = np.zeros(3 * N)
    if use_hints:   # the hint prior (a small floor keeps a camera without edges at its hint)
        for i in range(N):
            wi = max(w[i], 1e-4) / sh ** 2
            Hm[3 * i:3 * i + 3, 3 * i:3 * i + 3] += wi * np.eye(3)
            g[3 * i:3 * i + 3] += wi * log_so3(H[i].T @ R[i])
    else:           # gauge: the anchor stays where it is; cameras without edges stay too
        for i in range(N):
            wi = 1e6 if i == anchor else 1e-6
            Hm[3 * i:3 * i + 3, 3 * i:3 * i + 3] += wi * np.eye(3)
    v = np.zeros(len(edges))
    for k, ed in enumerate(edges):
        i, j = ed["i"], ed["j"]
        M, r, Cr, D, li, lo = _edge_mix(R, ed, centres, cfg, sc, dirs)
        v[k] = _sigmoid(li - lo) if v_fixed is None else v_fixed[k]
        L = v[k] * np.linalg.inv(Cr)
        Ji, Jj = np.eye(3), -M.T
        for (p, Jp) in ((i, Ji), (j, Jj)):
            g[3 * p:3 * p + 3] += Jp.T @ L @ r
            for (q, Jq) in ((i, Ji), (j, Jj)):
                Hm[3 * p:3 * p + 3, 3 * q:3 * q + 3] += Jp.T @ L @ Jq
        if D is not None:
            u, a, Cd = D
            Ld = v[k] * np.linalg.inv(Cd)
            Jd = ed["B"].T @ hat(u)   # d(a)/d(delta_j): u <- u + [u]x delta_j
            g[3 * j:3 * j + 3] += Jd.T @ Ld @ a
            Hm[3 * j:3 * j + 3, 3 * j:3 * j + 3] += Jd.T @ Ld @ Jd
    return Hm + 1e-9 * np.eye(3 * N), g, v


def _solve(R0, H, centres, edges, cfg, use_hints=True, anchor=None, sc=0.0, sh=None, pi=None, clamp=None,
           w_fixed=None, v_fixed=None):
    """EM / Gauss-Newton from R0 at FIXED hyperparameters (sigma_c sc; sigma_h sh in rad and pi, by default their
    prior values vtt_sigma_h_deg and a / (a + b)): E-step (responsibilities of hints w and edges v), then one Gauss-
    Newton step on the rotations. use_hints=False: edges only, the gauge fixed at camera `anchor` (photo-only rotation
    averaging). clamp (bool array): those cameras are held at their hints and the others' hints count as wrong (the
    final solve of fuse_rotations; no hint E-step). w_fixed / v_fixed: hold the hint / edge responsibilities (the
    leave-one-out solves, inside one explanation). Returns R, w, v, Sigma (N, 3, 3; Laplace posterior covariances),
    Hm (the final Gauss-Newton Hessian)."""
    N = len(H)
    a, b = cfg_get(cfg, "vtt_pi_a"), cfg_get(cfg, "vtt_pi_b")
    sh = np.radians(cfg_get(cfg, "vtt_sigma_h_deg")) if sh is None else sh
    pi = a / (a + b) if pi is None else pi
    R = np.array([np.asarray(r, float).copy() for r in R0])
    w = np.ones(N) if clamp is None else np.where(clamp, 1e8, 0.0)
    if w_fixed is not None:
        w = np.asarray(w_fixed, float)
    fixed_w = clamp is not None or w_fixed is not None
    if clamp is not None:
        R[clamp] = np.asarray(H, float)[clamp]
    dirs = use_hints and centres is not None

    def e_step(R):
        rh = np.array([log_so3(H[i].T @ R[i]) for i in range(N)])
        return _sigmoid(np.subtract(*_hint_logs(rh, pi, sh)))

    for _ in range(int(cfg_get(cfg, "vtt_iters"))):
        if use_hints and not fixed_w:
            w = e_step(R)
        Hm, g, v = _system(R, H, centres, edges, cfg, use_hints, anchor, w, sh, sc, dirs, v_fixed)
        delta = -np.linalg.solve(Hm, g)
        for i in range(N):
            d_i = delta[3 * i:3 * i + 3]
            nd = np.linalg.norm(d_i)
            R[i] = R[i] @ exp_so3(d_i if nd <= 0.5 else d_i * 0.5 / nd)   # trust region: at most 0.5 rad a step
        if np.max(np.abs(delta)) < 1e-8:
            break
    if use_hints and not fixed_w:
        w = e_step(R)
    Hm, g, v = _system(R, H, centres, edges, cfg, use_hints, anchor, w, sh, sc, dirs, v_fixed)
    Cf = np.linalg.inv(Hm)
    return R, w, v, np.array([Cf[3 * i:3 * i + 3, 3 * i:3 * i + 3] for i in range(N)]), Hm


def _components(N, edges):
    parent = list(range(N))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for ed in edges:
        parent[find(ed["i"])] = find(ed["j"])
    comps = {}
    for i in range(N):
        comps.setdefault(find(i), []).append(i)
    return list(comps.values())


def _l1_mean(X, iters=50):
    """Robust (L1, Weiszfeld) mean of rotations X (k, 3, 3)."""
    G = X[0].copy()
    for _ in range(iters):
        v = np.array([log_so3(G.T @ x) for x in X])
        nrm = np.maximum(np.linalg.norm(v, axis=1), 1e-9)
        step = (v / nrm[:, None]).sum(0) / (1.0 / nrm).sum()
        G = G @ exp_so3(step)
        if np.linalg.norm(step) < 1e-10:
            break
    return G


def _wahba(V, D, wts):
    """The rotation G minimising sum_k wts_k |D_k - G V_k|^2 (weighted Kabsch)."""
    B = (D * wts[:, None]).T @ V
    U, _, Vt = np.linalg.svd(B)
    return U @ np.diag([1.0, 1.0, 1.0 if np.linalg.det(U @ Vt) >= 0 else -1.0]) @ Vt


def photo_frames(H, edges, cfg=None):
    """Photo-only rotations per connected component of the edge graph (>= 2 cameras): a spanning tree along the most
    precise edges (smallest tr Sr) from the camera with the most edge precision, then the photo-only EM solve.
    Returns [(component, Rc, Sc)]: Rc (N, 3, 3) valid on the component, in its own gauge; Sc its covariances."""
    N = len(H)
    adj = {i: [] for i in range(N)}
    for ed in edges:
        p = 1.0 / max(np.trace(rot_cov(ed, cfg)), 1e-30)
        adj[ed["i"]].append((ed["j"], ed["R"].T, p))    # R_j = R_i Rhat_ij^T
        adj[ed["j"]].append((ed["i"], ed["R"], p))      # R_i = R_j Rhat_ij
    out = []
    for comp in _components(N, edges):
        if len(comp) < 2:
            continue
        root = max(comp, key=lambda c: sum(p for _, _, p in adj[c]))
        P = {root: np.eye(3)}
        while True:
            cand = [(p, a_, b_, Rel) for a_ in P for (b_, Rel, p) in adj[a_] if b_ not in P]
            if not cand:
                break
            p, a_, b_, Rel = max(cand, key=lambda t: t[0])
            P[b_] = P[a_] @ Rel
        Rc = np.array([P.get(i, np.eye(3)) for i in range(N)])
        sub = [ed for ed in edges if ed["i"] in P and ed["j"] in P]
        Rc, _, _, Sc, _ = _solve(Rc, H, None, sub, cfg, use_hints=False, anchor=root)
        out.append((comp, Rc, Sc))
    return out


def align_by_directions(comp, Rc, H, centres, edges, cfg=None, iters=20):
    """World placement G (R = G Rc) of a photo component from the translation directions to the hinted centres:
    each usable edge says unit(c_i - c_j) = G (Rc_j that_ij), weight 1 / (kappa_dir^2 tr(St) / 2); the rotation
    hints enter with a weak weight (sigma 1 rad) only to fix what the directions leave free (e.g. cameras on a line).
    Robust weighted Wahba fit (IRLS, Cauchy); only a start for the EM. None if no usable direction."""
    D = [ed for ed in edges if ed["i"] in comp and ed["j"] in comp and _dir_ok(ed, centres)]
    if not D:
        return None
    V = np.array([Rc[ed["j"]] @ ed["t"] for ed in D])
    T = np.array([(centres[ed["i"]] - centres[ed["j"]]) / np.linalg.norm(centres[ed["i"]] - centres[ed["j"]])
                  for ed in D])
    s = np.array([np.sqrt(np.trace(dir_cov_photo(ed, cfg)) / 2) for ed in D])
    Vh = np.concatenate([Rc[i].T for i in comp])     # rows: Rc_i e_k
    Th = np.concatenate([H[i].T for i in comp])      # rows: H_i e_k
    G = _l1_mean(np.array([H[i] @ Rc[i].T for i in comp]))
    for _ in range(iters):
        rd = np.linalg.norm(T - V @ G.T, axis=1)
        rh = np.linalg.norm(Th - Vh @ G.T, axis=1)
        wts = np.r_[_cauchy_w(rd / s) / s ** 2, _cauchy_w(rh)]
        G_new = _wahba(np.r_[V, Vh], np.r_[T, Th], wts)
        done = np.linalg.norm(log_so3(G.T @ G_new)) < 1e-10
        G = G_new
        if done:
            break
    return G


def fuse_rotations(H, centres, edges, cfg=None):
    """Posterior rotations. The EM is run from three starts (the hints; the photo rotations placed by the rotation
    hints; the photo rotations placed by the translation directions); each converges to one explanation of the data
    (which hints and which edges are wrong). The explanation kept is the one with the highest Laplace evidence,
    -log p(data) ~ E + 1/2 log det(Hessian): integrating over the rotations removes the advantage a single confident
    measurement has in a peak-density (MAP) comparison, so one edge against one hint is decided by their prior
    reliabilities, while edges that confirm each other outweigh a hint.
    Returns R (N, 3, 3) the posterior mode within that explanation, w (N,) hint inlier probabilities, pi, info."""
    H = np.asarray(H, float)
    N = len(H)
    deg = np.zeros(N, int)
    for ed in edges:
        deg[ed["i"]] += 1
        deg[ed["j"]] += 1
    starts = [("hints", H)]
    frames = photo_frames(H, edges, cfg) if edges else []
    RB, has_b = None, False
    if frames:
        RA, RB = H.copy(), H.copy()
        for comp, Rc, _ in frames:
            GA = _l1_mean(np.array([H[i] @ Rc[i].T for i in comp]))
            GB = align_by_directions(comp, Rc, H, centres, edges, cfg) if centres is not None else None
            has_b |= GB is not None
            for i in comp:
                RA[i] = GA @ Rc[i]
                RB[i] = (GA if GB is None else GB) @ Rc[i]
        starts.append(("photos_by_hints", RA))
        if has_b:
            starts.append(("photos_by_directions", RB))
    sc = _fit_sigma_c(RB, centres, edges, cfg) if has_b else 0.0
    a, b = cfg_get(cfg, "vtt_pi_a"), cfg_get(cfg, "vtt_pi_b")
    sh, pi = np.radians(cfg_get(cfg, "vtt_sigma_h_deg")), a / (a + b)   # the declared priors, not fitted per clip
    best, evid = None, {}
    for name, R0 in starts:   # every explanation at the same hyperparameters
        R, w, v, Sigma, Hm = _solve(R0, H, centres, edges, cfg, sc=sc, sh=sh, pi=pi)
        L = energy(R, H, centres, edges, cfg, pi, sh, sc) + 0.5 * np.linalg.slogdet(Hm)[1]
        evid[name] = float(L)
        if best is None or L < best[0] - 1e-9:
            best = (L, R, w, v, Sigma, name)
    L, Rmap, w, v, Sigma, name = best
    # leave-one-out posterior-predictive check, inside the chosen explanation: camera i's posterior from everything
    # but its own hint; the hint is inconsistent if d2 = r^T (sigma_h^2 I + Sigma_-i)^-1 r > chi2_3(1 - alpha)
    d2 = np.zeros(N)
    for i in range(N):
        wl = w.copy()
        wl[i] = 0.0
        Rl, _, _, Sl, _ = _solve(Rmap, H, centres, edges, cfg, sc=sc, sh=sh, pi=pi, w_fixed=wl, v_fixed=v)
        r = log_so3(H[i].T @ Rl[i])
        d2[i] = float(r @ np.linalg.solve(sh ** 2 * np.eye(3) + Sl[i], r))
    q = CHI2_3_Q[cfg_get(cfg, "vtt_alpha")]
    wrong = d2 > q
    # breakdown requirement: the evidence against a hint must survive the removal of any single photo pair, unless
    # that pair is a priori more reliable than a hint (one pair against one hint: the posterior odds are the prior odds)
    single = np.zeros(N, bool)
    for i in np.where(wrong)[0]:
        for k, ed in enumerate(edges):
            if i not in (ed["i"], ed["j"]) or edge_prior(ed, cfg) > pi:
                continue
            wl = w.copy()
            wl[i] = 0.0
            sub, vs = edges[:k] + edges[k + 1:], np.delete(v, k)
            Rl, _, _, Sl, _ = _solve(Rmap, H, centres, sub, cfg, sc=sc, sh=sh, pi=pi, w_fixed=wl, v_fixed=vs)
            r = log_so3(H[i].T @ Rl[i])
            if float(r @ np.linalg.solve(sh ** 2 * np.eye(3) + Sl[i], r)) <= q:
                wrong[i], single[i] = False, True
                break
    R = H.copy()
    if wrong.any():   # replace only the inconsistent hints, re-estimated with the others held at their hints
        Rf, _, v, Sigma, _ = _solve(Rmap, H, centres, edges, cfg, sc=sc, sh=sh, pi=pi, clamp=~wrong)
        R[wrong] = Rf[wrong]
    res = [float(np.degrees(np.linalg.norm(log_so3(H[i].T @ Rmap[i])))) for i in range(N)]
    post = [float(np.degrees(np.sqrt(max(np.trace(S), 0.0) / 3))) for S in Sigma]
    return R, w, pi, dict(replaced=wrong.tolist(), loo_d2=[float(x) for x in d2], kept_single_pair=single.tolist(),
                          map_residual_deg=res,
                          posterior_sd_deg=post,
                          photo_degree=deg.tolist(), start=name,
                          neg_log_evidence=float(L), neg_log_evidence_by_start=evid,
                          sigma_h_deg=float(np.degrees(sh)), sigma_c=float(sc),
                          edge_inlier_prob=[float(x) for x in v], edge_pairs=[[e["i"], e["j"]] for e in edges],
                          n_dir_edges=int(sum(_dir_ok(ed, centres) for ed in edges)))


# ------------------------------------------------------------------------------------------------- calibration
CHI3_MEDIAN = 1.5382      # median of a chi distribution with 3 dof (rotation-vector noise)
RAYLEIGH_MEDIAN = 1.1774  # median of a chi distribution with 2 dof (direction noise on the sphere)


def edge_residuals(edges, R_true, centres_true):
    """For calibration against ground truth: per edge, the true residuals of the photo estimate (rotation r, direction
    a or None where the photos do not fix it) with the edge's own covariances and inlier count."""
    out = []
    for ed in edges:
        i, j = ed["i"], ed["j"]
        r = log_so3(ed["R"].T @ (R_true[j].T @ R_true[i]))
        a = None
        if _dir_ok(ed, centres_true):
            d = centres_true[i] - centres_true[j]
            a = sphere_log(ed["t"], R_true[j].T @ (d / np.linalg.norm(d)), ed["B"])
        out.append(dict(r=r, Sr=ed["Sr"], a=a, St=ed["St"], n=ed["n"]))
    return out


def edge_standardised_errors(edges, R_true, centres_true, cfg=None):
    """Whitened true errors at kappa = 1 (chi_3 / chi_2 distributed if the covariances were exact); direction NaN
    where the photos do not fix it. Diagnostics."""
    rot, dirs = [], []
    for x in edge_residuals(edges, R_true, centres_true):
        rot.append(float(np.sqrt(max(x["r"] @ np.linalg.solve(x["Sr"], x["r"]), 0.0))))
        dirs.append(np.nan if x["a"] is None else float(np.sqrt(max(x["a"] @ np.linalg.solve(x["St"], x["a"]), 0.0))))
    return np.asarray(rot), np.asarray(dirs)


def calibrate_kappa(rot_std, dir_std):
    """Median match (diagnostic and EM start): the kappa that makes the median whitened error equal to the median of
    the noise model (chi_3 for rotations, Rayleigh for directions). NaNs are ignored."""
    rot_std, dir_std = np.asarray(rot_std, float), np.asarray(dir_std, float)
    rot_std, dir_std = rot_std[np.isfinite(rot_std)], dir_std[np.isfinite(dir_std)]
    kr = float(np.median(rot_std) / CHI3_MEDIAN) if len(rot_std) else 1.0
    kd = float(np.median(dir_std) / RAYLEIGH_MEDIAN) if len(dir_std) else 1.0
    return kr, kd


def _mix_logs(x, nz, s_gt, log_pe, log_qe):
    """Inlier / outlier log likelihoods of one calibration residual set x (edge_residuals) at noise parameters
    nz = (kappa_rot, sys_rot, kappa_dir, sys_dir) (sys in rad) and extra ground-truth noise s_gt (rad)."""
    kr, sr, kd, sd = nz
    li = log_pe + gauss_logpdf(x["r"], kr ** 2 * x["Sr"] + (sr ** 2 + s_gt ** 2) * np.eye(3))
    lo = log_qe + LOG_U_BALL
    if x["a"] is not None:
        li += gauss_logpdf(x["a"], kd ** 2 * x["St"] + (sd ** 2 + s_gt ** 2) * np.eye(2))
        lo += LOG_U_SPHERE
    return li, lo


def _fit_scale_floor(vals, w, S, dim, start):
    """Weighted maximum likelihood of (kappa, floor) for residuals vals[k] ~ N(0, kappa^2 S[k] + floor^2 I_dim)."""
    from scipy.optimize import minimize

    def nll(th):
        k, f = np.exp(th)
        return -sum(wi * gauss_logpdf(v, k ** 2 * Sk + f ** 2 * np.eye(dim)) for v, wi, Sk in zip(vals, w, S))
    r = minimize(nll, np.log(np.maximum(start, 1e-6)), method="Nelder-Mead",
                 options=dict(xatol=1e-6, fatol=1e-8, maxiter=2000))
    return tuple(np.exp(r.x))


def fit_kappa_em(res, iters=100):
    """Stage 1 (accurate ground truth only): maximum likelihood noise model of the right edges, covariance kappa^2 S +
    floor^2 I for rotations and for directions, and the share of right edges p, under the edge mixture (right:
    Gaussian; wrong: uniform), by EM. Returns (kappa_rot, sys_rot, kappa_dir, sys_dir) (sys in rad), p, v."""
    rs = [np.sqrt(max(x["r"] @ np.linalg.solve(x["Sr"], x["r"]), 0.0)) for x in res]
    ds = [np.nan if x["a"] is None else np.sqrt(max(x["a"] @ np.linalg.solve(x["St"], x["a"]), 0.0)) for x in res]
    kr, kd = calibrate_kappa(rs, ds)
    nz, p = (kr, np.radians(0.05), kd, np.radians(0.05)), 0.8
    D = [x for x in res if x["a"] is not None]
    for _ in range(iters):
        v = np.array([_sigmoid(np.subtract(*_mix_logs(x, nz, 0.0, np.log(p), np.log(1 - p)))) for x in res])
        kr_, sr_ = _fit_scale_floor([x["r"] for x in res], v, [x["Sr"] for x in res], 3, nz[:2])
        vd = np.array([vi for vi, x in zip(v, res) if x["a"] is not None])
        kd_, sd_ = _fit_scale_floor([x["a"] for x in D], vd, [x["St"] for x in D], 2, nz[2:]) if D else nz[2:]
        new, p_new = (kr_, sr_, kd_, sd_), float(np.clip(v.mean(), 1e-3, 1 - 1e-3))
        done = np.allclose(new, nz, rtol=1e-5, atol=1e-9) and abs(p_new - p) < 1e-7
        nz, p = tuple(float(t) for t in new), p_new
        if done:
            break
    return nz, p, v


def fit_edge_prior_em(res, src_noisy, nz, iters=100, s_grid_deg=np.r_[0.0, np.geomspace(0.05, 5.0, 30)]):
    """Stage 2 (all dev / cal clips, noise model nz fixed): the prior that an edge is right as a function of its
    inlier count, sigmoid(a + b log n), by EM with a weighted logistic M-step; clips whose ground truth is itself
    noisy (src_noisy[k] True, e.g. SceneFun3D) get an extra ground-truth noise s_gt fitted on a grid, so their
    ground-truth error is not counted as wrong edges. Returns a, b, s_gt (deg), responsibilities."""
    x = np.log(np.array([max(r["n"], 1) for r in res], float))
    noisy = np.asarray(src_noisy, bool)
    a, b, s_gt = 1.5, 0.0, 0.0
    for _ in range(iters):
        pe = np.clip(_sigmoid(a + b * x), 1e-6, 1 - 1e-6)
        L = [_mix_logs(r, nz, s_gt if n_ else 0.0, np.log(q), np.log(1 - q)) for r, n_, q in zip(res, noisy, pe)]
        v = np.array([_sigmoid(li - lo) for li, lo in L])
        a_old, b_old, s_old = a, b, s_gt
        for _ in range(25):   # weighted logistic regression with soft targets v (Newton)
            q = _sigmoid(a + b * x)
            gr = np.array([np.sum(v - q), np.sum((v - q) * x)])
            W = q * (1 - q)
            Hs = np.array([[W.sum(), (W * x).sum()], [(W * x).sum(), (W * x * x).sum()]]) + 1e-9 * np.eye(2)
            st = np.linalg.solve(Hs, gr)
            a, b = a + st[0], b + st[1]
            if np.abs(st).max() < 1e-10:
                break
        if noisy.any():   # ground-truth noise of the noisy sources: maximum marginal likelihood on the grid
            pe = np.clip(_sigmoid(a + b * x), 1e-6, 1 - 1e-6)
            best = None
            for sg in np.radians(s_grid_deg):
                ll = sum(np.logaddexp(*_mix_logs(r, nz, sg, np.log(q), np.log(1 - q)))
                         for r, n_, q in zip(res, noisy, pe) if n_)
                if best is None or ll > best[0]:
                    best = (ll, sg)
            s_gt = float(best[1])
        if abs(a - a_old) < 1e-7 and abs(b - b_old) < 1e-7 and s_gt == s_old:
            break
    return float(a), float(b), float(np.degrees(s_gt)), v


# ------------------------------------------------------------------------------------------------------- depth
def depth_evidence(pairs, K, poses, depth, depth_views, cfg=None):
    """Per depth view v (in the order of depth_views): m_v, the median of log(depth hint / triangulated depth) over
    the matches seen in v whose triangulation angle exceeds vtt_min_parallax_deg (NaN with fewer than 10), and its
    standard error se_v (inf without evidence). Triangulation uses `poses` (cam2world), i.e. their units."""
    minpar = np.radians(cfg_get(cfg, "vtt_min_parallax_deg"))
    W2C = [np.linalg.inv(P) for P in poses]
    acc = {v: [] for v in depth_views}
    for (i, j), (xi, xj) in pairs.items():
        if i not in acc and j not in acc:
            continue
        Pi, Pj = K[i] @ W2C[i][:3], K[j] @ W2C[j][:3]
        Xh = cv2.triangulatePoints(Pi, Pj, np.asarray(xi, np.float64).T, np.asarray(xj, np.float64).T)
        X = (Xh[:3] / Xh[3:]).T
        ci, cj = poses[i][:3, 3], poses[j][:3, 3]
        va, vb = X - ci, X - cj
        cos = np.sum(va * vb, 1) / np.maximum(np.linalg.norm(va, axis=1) * np.linalg.norm(vb, axis=1), 1e-12)
        good = np.arccos(np.clip(cos, -1, 1)) > minpar
        for v, x in ((i, xi), (j, xj)):
            if v not in acc:
                continue
            z = (W2C[v][:3, :3] @ X.T + W2C[v][:3, 3:4])[2]
            u = np.clip(np.round(np.asarray(x)[:, 0]).astype(int), 0, depth.shape[2] - 1)
            r = np.clip(np.round(np.asarray(x)[:, 1]).astype(int), 0, depth.shape[1] - 1)
            dh = depth[v][r, u]
            ok = good & (z > 1e-6) & (dh > 0)
            acc[v].extend(np.log(dh[ok] / z[ok]).tolist())
    V = list(depth_views)
    m, se = np.full(len(V), np.nan), np.full(len(V), np.inf)
    for k, v in enumerate(V):
        l = np.asarray(acc[v])
        if len(l) >= 10:
            m[k] = float(np.median(l))
            se[k] = float(1.2533 * 1.4826 * np.median(np.abs(l - m[k])) / np.sqrt(len(l))) + 1e-6
    return m, se


def depth_decide(m, se, cfg=None):
    """Which depth hints are wrong, and by how much (log scale), from the evidence m, se (depth_evidence).
    Model: s_v = z0 mu + z_v delta_v with labels z0 (the scale shared by the depth hints is wrong) and z_v (view v
    alone is wrong), each wrong with prior probability 1 - pi0 (pi0 = a / (a + b) of the hint prior); mu, delta_v ~
    N(0, tau^2); m_v ~ N(s_v, se_v^2 + sys^2) plus a log-scale error shared by all views, N(0, ps^2), ps =
    vtt_pose_scale_sd: the depth is measured in the units of the pose hints, which carry their own scale error
    (odometry scale drift); sys = vtt_depth_sys (the evidence's model-error floor, calibrated on dev / cal clips with
    correct depth and ground-truth poses). The label configuration with the highest posterior is found by exact
    enumeration over the views with evidence (greedy beyond 14 views); the log scale removed from each view is its
    posterior mean under that configuration (0 for a view labelled right, unless the shared scale is wrong; a view
    without evidence gets the shared part). Returns s (per view), z0, z (per view; None without evidence)."""
    import itertools
    tau, sys_ = cfg_get(cfg, "vtt_tau_depth"), cfg_get(cfg, "vtt_depth_sys")
    ps = cfg_get(cfg, "vtt_pose_scale_sd")
    a, b = cfg_get(cfg, "vtt_pi_a"), cfg_get(cfg, "vtt_pi_b")
    lp_ok, lp_bad = np.log(a / (a + b)), np.log(b / (a + b))
    V = len(m)
    O = np.where(np.isfinite(m))[0]
    if len(O) == 0:
        return np.zeros(V), False, [None] * V
    mo, D = m[O], np.diag(se[O] ** 2 + sys_ ** 2) + ps ** 2 * np.ones((len(O), len(O)))   # + the pose hints' scale error

    def score(z0, zo):
        Ss = z0 * tau ** 2 * np.ones((len(O), len(O))) + np.diag(np.asarray(zo, float) * tau ** 2)
        lp = (lp_bad if z0 else lp_ok) + sum(lp_bad if z else lp_ok for z in zo)
        return lp + gauss_logpdf(mo, Ss + D)
    if len(O) <= 14:
        best = max(((score(z0, zo), z0, zo) for z0 in (0, 1) for zo in itertools.product((0, 1), repeat=len(O))),
                   key=lambda t: t[0])
    else:
        best = None
        for z0 in (0, 1):
            zo = [0] * len(O)
            while True:   # greedy single-label flips
                cur = score(z0, zo)
                cand = max(((score(z0, zo[:k] + [1 - zo[k]] + zo[k + 1:]), k) for k in range(len(O))), key=lambda t: t[0])
                if cand[0] <= cur + 1e-12:
                    break
                zo[cand[1]] = 1 - zo[cand[1]]
            if best is None or cur > best[0]:
                best = (cur, z0, tuple(zo))
    _, z0, zo = best
    z = np.zeros(V)
    z[O] = zo
    Sall = z0 * tau ** 2 * np.ones((V, V)) + np.diag(z * tau ** 2)   # prior covariance of s under the labels
    s = Sall[:, O] @ np.linalg.solve(Sall[np.ix_(O, O)] + D, mo)
    zl = [None] * V
    for k, o in enumerate(O):
        zl[o] = bool(zo[k])
    return s, bool(z0), zl


# ------------------------------------------------------------------------------------------------- the sanitiser
def sanitize(pairs, K, poses, depth, depth_views, inputs, cfg=None, edges=None):
    """Returns (new_poses, new_depth, info). Poses are rewritten only if pose hints are given; depth only if both
    pose and depth hints are given (the pose hints fix the units). info holds every posterior for the log."""
    poses = np.asarray(poses, float)
    new_poses, new_depth = poses.copy(), None if depth is None else depth.copy()
    info = {"variant": "vtt"}
    if "pose" in inputs:
        edges = pair_edges(pairs, K, cfg) if edges is None else edges
        edges, n_res = resolve_with_hints(edges, poses[:, :3, :3], poses[:, :3, 3], cfg)
        R, w, pi, extra = fuse_rotations(poses[:, :3, :3], poses[:, :3, 3], edges, cfg)
        extra["n_edges_resolved_to_hint"] = n_res
        new_poses[:, :3, :3] = R
        info.update(w_hint=w.tolist(), pi=float(pi), n_edges=len(edges), **extra,
                    correction_deg=[angle_deg(poses[i, :3, :3], R[i]) for i in range(len(R))])
    if "pose" in inputs and "depth" in inputs and depth is not None and len(depth_views):
        V = list(depth_views)
        m, se = depth_evidence(pairs, K, new_poses, depth, V, cfg)
        sv, z0, zv = depth_decide(m, se, cfg)
        for k, v in enumerate(V):
            new_depth[v] = depth[v] * np.exp(-sv[k])
        info["depth"] = {int(v): {"m": None if not np.isfinite(m[k]) else float(m[k]),
                                  "se": float(se[k]) if np.isfinite(se[k]) else None, "s": float(sv[k]),
                                  "view_wrong": zv[k]} for k, v in enumerate(V)}
        info["depth_shared_wrong"] = z0
    return new_poses, new_depth, info
