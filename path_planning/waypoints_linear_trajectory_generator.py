#!/usr/bin/env python3
"""
waypoints_linear_trajectory_generator.py

Create a dense pose trajectory from sparse waypoints using:
- Linear interpolation in position
- SLERP in rotation (quaternions)

Sampling is done so that step-to-step changes do not exceed:
- pos_res (meters)
- rot_res_deg (degrees)

Waypoint CSV format:
x,y,z,qx,qy,qz,qw   (quaternion in x,y,z,w order)

Output CSV columns:
t,x,y,z,qx,qy,qz,qw,seg_idx,alpha
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


# ============================================================
# CONFIG
# ============================================================

CONFIG = dict(
    # -------- IO --------
    input_waypoints_csv="./path_planning/data/cylinder_assembly_waypoints.csv",     # input waypoint CSV
    output_traj_csv="./path_planning/data/cylinder_assembly_trajectory.csv",  # output trajectory CSV
    write_example_waypoints=False,            # if True, writes example waypoints and exits

    # -------- Resolution --------
    pos_res=0.0001,        # meters
    rot_res_deg=0.1,      # degrees

    # -------- Validation --------
    require_unit_quat=True,
)


# ============================================================
# Quaternion utilities
# ============================================================

def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.where(n > 0.0, n, 1.0)
    return q / n


def quat_dot(q1: np.ndarray, q2: np.ndarray) -> float:
    return float(np.dot(q1, q2))


def quat_slerp(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    q0 = quat_normalize(q0.reshape(1, 4))[0]
    q1 = quat_normalize(q1.reshape(1, 4))[0]

    dot = quat_dot(q0, q1)

    # shortest-path
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = max(min(dot, 1.0), -1.0)

    # near-linear
    if dot > 0.9995:
        q = (1.0 - u) * q0 + u * q1
        return quat_normalize(q.reshape(1, 4))[0]

    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)

    theta = theta_0 * u
    sin_theta = math.sin(theta)

    s0 = math.sin(theta_0 - theta) / sin_theta_0
    s1 = sin_theta / sin_theta_0

    q = s0 * q0 + s1 * q1
    return quat_normalize(q.reshape(1, 4))[0]


def quat_relative_angle_rad(q0: np.ndarray, q1: np.ndarray) -> float:
    q0 = quat_normalize(q0.reshape(1, 4))[0]
    q1 = quat_normalize(q1.reshape(1, 4))[0]
    d = abs(quat_dot(q0, q1))
    d = max(min(d, 1.0), -1.0)
    return 2.0 * math.acos(d)


# ============================================================
# Trajectory generation
# ============================================================

@dataclass
class Waypoints:
    pos: np.ndarray   # (N,3)
    quat: np.ndarray  # (N,4)


def compute_segment_steps(p0, p1, q0, q1, pos_res, rot_res_rad) -> int:
    dp = float(np.linalg.norm(p1 - p0))
    dtheta = quat_relative_angle_rad(q0, q1)

    pos_steps = math.ceil(dp / pos_res) if pos_res > 0 else 1
    rot_steps = math.ceil(dtheta / rot_res_rad) if rot_res_rad > 0 else 1

    return max(1, int(pos_steps), int(rot_steps))


def densify_waypoints(
    wps: Waypoints,
    pos_res: float,
    rot_res_deg: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    rot_res_rad = math.radians(rot_res_deg)

    P = wps.pos
    Q = quat_normalize(wps.quat)

    out_pos = [P[0]]
    out_quat = [Q[0]]
    out_seg = [0]
    out_alpha = [0.0]

    for seg_idx in range(len(P) - 1):
        k = compute_segment_steps(
            P[seg_idx], P[seg_idx + 1],
            Q[seg_idx], Q[seg_idx + 1],
            pos_res, rot_res_rad,
        )

        for j in range(1, k + 1):
            u = j / k
            p = (1.0 - u) * P[seg_idx] + u * P[seg_idx + 1]
            q = quat_slerp(Q[seg_idx], Q[seg_idx + 1], u)

            out_pos.append(p)
            out_quat.append(q)
            out_seg.append(seg_idx)
            out_alpha.append(u)

    out_pos = np.asarray(out_pos)
    out_quat = np.asarray(out_quat)
    meta = np.stack([out_seg, out_alpha], axis=1)

    t = np.linspace(0.0, 1.0, out_pos.shape[0])

    return t, out_pos, out_quat, meta


# ============================================================
# IO
# ============================================================

REQUIRED_COLS = ["x", "y", "z", "qx", "qy", "qz", "qw"]


def load_waypoints_csv(path: Path) -> Waypoints:
    data = np.genfromtxt(path, delimiter=",", names=True)
    missing = [c for c in REQUIRED_COLS if c not in data.dtype.names]
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")

    pos = np.vstack([data["x"], data["y"], data["z"]]).T
    quat = np.vstack([data["qx"], data["qy"], data["qz"], data["qw"]]).T
    return Waypoints(pos=pos, quat=quat)


def save_trajectory_csv(path: Path, t, pos, quat, meta):
    out = np.column_stack([t, pos, quat, meta[:, 0], meta[:, 1]])
    header = "t,x,y,z,qx,qy,qz,qw,seg_idx,alpha"
    np.savetxt(path, out, delimiter=",", header=header, comments="", fmt="%.10g")


def write_example_waypoints(path: Path):
    s = math.sin(math.radians(45))
    c = math.cos(math.radians(45))
    example = np.array([
        [0, 0, 0, 0, 0, 0, 1],
        [0.1, 0, 0, 0, 0, s, c],
    ])
    np.savetxt(
        path, example, delimiter=",",
        header="x,y,z,qx,qy,qz,qw", comments="", fmt="%.10g"
    )


# ============================================================
# Main entry
# ============================================================

def main(cfg: dict):

    out_csv = Path(cfg["output_traj_csv"])

    if cfg["write_example_waypoints"]:
        write_example_waypoints(out_csv)
        print(f"[INFO] Wrote example waypoints to {out_csv}")
        return

    wps = load_waypoints_csv(Path(cfg["input_waypoints_csv"]))

    t, pos, quat, meta = densify_waypoints(
        wps,
        pos_res=cfg["pos_res"],
        rot_res_deg=cfg["rot_res_deg"],
    )

    save_trajectory_csv(out_csv, t, pos, quat, meta)

    print(f"[INFO] Waypoints: {wps.pos.shape[0]}")
    print(f"[INFO] Trajectory samples: {pos.shape[0]}")
    print(f"[INFO] Output written to: {out_csv}")


if __name__ == "__main__":
    main(CONFIG)
