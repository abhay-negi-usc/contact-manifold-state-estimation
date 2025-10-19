import numpy as np 
import torch 
import math 
RAD2DEG = 180.0 / math.pi

# TODO: check if this is consistent with rotation convention used in cicp 
def torch_matrix_to_pose_xyzabc(matrix):
    """Differentiable version of 4x4 matrix to pose (x, y, z, rz, ry, rx) using ZYX Euler angles"""
    translation = matrix[:, :3, 3]

    r11, r12, r13 = matrix[:, 0, 0], matrix[:, 0, 1], matrix[:, 0, 2]
    r21, r22, r23 = matrix[:, 1, 0], matrix[:, 1, 1], matrix[:, 1, 2]
    r31, r32, r33 = matrix[:, 2, 0], matrix[:, 2, 1], matrix[:, 2, 2]

    ry = torch.asin(-r31)
    rz = torch.atan2(r21, r11)
    rx = torch.atan2(r32, r33)

    euler_deg = torch.stack([rz, ry, rx], dim=-1) * RAD2DEG
    pose = torch.cat([translation, euler_deg], dim=-1)
    return pose

# TODO: check if this is consistent with rotation convention used in cicp 
def torch_pose_xyzabc_to_matrix(pose):
    """Differentiable version of pose (x, y, z, rz, ry, rx) to 4x4 matrix using ZYX Euler angles"""
    batch_size = pose.shape[0]
    translation = pose[:, :3]
    angles_rad = pose[:, 3:6] * math.pi / 180.0
    rz, ry, rx = angles_rad[:, 0], angles_rad[:, 1], angles_rad[:, 2]

    # Create rotation matrices for each sample in the batch
    cos_rz, sin_rz = torch.cos(rz), torch.sin(rz)
    cos_ry, sin_ry = torch.cos(ry), torch.sin(ry)
    cos_rx, sin_rx = torch.cos(rx), torch.sin(rx)

    # Rz rotation matrix
    Rz = torch.zeros(batch_size, 3, 3, device=pose.device, dtype=pose.dtype)
    Rz[:, 0, 0] = cos_rz
    Rz[:, 0, 1] = -sin_rz
    Rz[:, 1, 0] = sin_rz
    Rz[:, 1, 1] = cos_rz
    Rz[:, 2, 2] = 1

    # Ry rotation matrix
    Ry = torch.zeros(batch_size, 3, 3, device=pose.device, dtype=pose.dtype)
    Ry[:, 0, 0] = cos_ry
    Ry[:, 0, 2] = sin_ry
    Ry[:, 1, 1] = 1
    Ry[:, 2, 0] = -sin_ry
    Ry[:, 2, 2] = cos_ry

    # Rx rotation matrix
    Rx = torch.zeros(batch_size, 3, 3, device=pose.device, dtype=pose.dtype)
    Rx[:, 0, 0] = 1
    Rx[:, 1, 1] = cos_rx
    Rx[:, 1, 2] = -sin_rx
    Rx[:, 2, 1] = sin_rx
    Rx[:, 2, 2] = cos_rx

    # Combine rotations: R = Rz * Ry * Rx
    rotation_matrix = torch.bmm(torch.bmm(Rz, Ry), Rx)
    
    # Create 4x4 transformation matrix
    matrix = torch.eye(4, device=pose.device, dtype=pose.dtype).unsqueeze(0).repeat(batch_size, 1, 1)
    matrix[:, :3, :3] = rotation_matrix
    matrix[:, :3, 3] = translation
    return matrix


# ---------- so(3) <-> SO(3) helpers ----------

def _hat(v):
    x, y, z = v
    return np.array([[0, -z,  y],
                     [z,  0, -x],
                     [-y, x,  0]], dtype=float)

def _so3_exp(w):
    """
    Exponential map: R = exp([w]^)
    Uses stable series for small angles.
    """
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        return np.eye(3)
    K = _hat(w / theta)
    s = np.sin(theta)
    c = np.cos(theta)
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)

def _so3_log(R):
    """
    Log map: w = log(R) (axis * angle).
    Handles edge cases near 0 and near pi robustly.
    """
    # Clamp trace to valid range to avoid nan from numeric noise
    tr = np.trace(R)
    cos_theta = (tr - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if theta < 1e-12:
        return np.zeros(3)

    # Regular branch (well away from pi): use skew part
    if theta < np.pi - 1e-6:
        w = (theta / (2.0 * np.sin(theta))) * np.array([
            R[2,1] - R[1,2],
            R[0,2] - R[2,0],
            R[1,0] - R[0,1]
        ])
        return w

    # Near pi: sin(theta) ~ 0 -> use diagonal to get axis reliably
    # Based on Hartley-Zisserman / Markley-style extraction
    A = (R + np.eye(3)) * 0.5  # = I + R over 2; diag holds cos^2(θ/2)+axis^2*sin^2(θ/2)
    # Pick the largest diagonal element to avoid division by small numbers
    i = np.argmax(np.diag(A))
    v = np.zeros(3)
    if i == 0:
        v[0] = np.sqrt(max(A[0,0], 0.0))
        denom = v[0] if v[0] > 1e-12 else 1.0
        v[1] = A[0,1] / denom
        v[2] = A[0,2] / denom
    elif i == 1:
        v[1] = np.sqrt(max(A[1,1], 0.0))
        denom = v[1] if v[1] > 1e-12 else 1.0
        v[0] = A[0,1] / denom
        v[2] = A[1,2] / denom
    else:
        v[2] = np.sqrt(max(A[2,2], 0.0))
        denom = v[2] if v[2] > 1e-12 else 1.0
        v[0] = A[0,2] / denom
        v[1] = A[1,2] / denom

    # Normalize axis; sign is ambiguous at π, either is fine but be consistent
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-12:
        # Fallback to skew part if diagonal route failed
        axis = np.array([
            R[2,1] - R[1,2],
            R[0,2] - R[2,0],
            R[1,0] - R[0,1]
        ])
        axis_norm = np.linalg.norm(axis)
        axis = axis / axis_norm if axis_norm > 1e-12 else np.array([1.0, 0.0, 0.0])
    else:
        axis = v / norm_v

    return theta * axis


# ---------- Intrinsic (Karcher) mean on SO(3) ----------

def mean_rotation_from_rotvecs(rotvecs, max_iters=100, tol=1e-12, return_rotvec=True):
    """
    Compute the intrinsic (Karcher/Fréchet) mean rotation from N×3 rotation vectors.
    Args:
        rotvecs: (N, 3) array of axis–angle vectors (log map, in radians).
        max_iters: maximum iterations for Karcher mean.
        tol: stop when update tangent vector ||Δ|| < tol (radians).
        return_rotvec: if True, also return the mean as a rotation vector.
    Returns:
        R_mean: (3,3) rotation matrix of the mean.
        w_mean (optional): (3,) rotation vector (log map) of the mean.
    """
    rotvecs = np.asarray(rotvecs, dtype=float)
    if rotvecs.ndim != 2 or rotvecs.shape[1] != 3:
        raise ValueError("rotvecs must be an (N, 3) array.")

    N = rotvecs.shape[0]
    if N == 0:
        raise ValueError("Empty input.")

    # Convert all to rotation matrices
    Rs = np.stack([_so3_exp(w) for w in rotvecs], axis=0)

    # Initialize: chordal/SVD init via “Procrustes” is okay; simple pick works fine for clustered data
    R_bar = Rs[0].copy()

    for _ in range(max_iters):
        # Accumulate tangent vectors at current mean
        Delta = np.zeros(3)
        for R in Rs:
            Delta += _so3_log(R_bar.T @ R)
        Delta /= float(N)

        # Convergence check
        if np.linalg.norm(Delta) < tol:
            break

        # Retract to the manifold
        R_bar = R_bar @ _so3_exp(Delta)

    if return_rotvec:
        return R_bar, _so3_log(R_bar)
    return R_bar