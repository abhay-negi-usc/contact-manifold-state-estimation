import torch 
import math 
RAD2DEG = 180.0 / math.pi

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