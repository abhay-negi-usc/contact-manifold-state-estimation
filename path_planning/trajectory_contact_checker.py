#!/usr/bin/env python3
"""
trajectory_contact_checker.py

Given:
- trajectory CSV of peg poses w.r.t. hole: x,y,z,qx,qy,qz,qw  (quat in x,y,z,w order)
- watertight peg OBJ and hole OBJ

Compute per-pose binary contact (and diagnostics) using Kaolin:
- contact_by_distance: min(surface-to-surface distance) <= contact_threshold
- contact_by_penetration: any sampled surface points are inside the other mesh (check_sign)

Prints trajectory-wide contact stats.
CSV output is optional.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch

from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

CONFIG: Dict = dict(
    # --- IO ---
    traj_csv="./path_planning/data/BNC_disassembly_trajectory.csv",
    peg_obj="./data_generation/assets/meshes/BNC/BNC_peg.obj",
    hole_obj="./data_generation/assets/meshes/BNC/BNC_hole.obj",

    save_csv=True,                     # <-- OPTIONAL OUTPUT
    out_csv="./data_generation/assets/meshes/BNC/BNC_disassembly_trajectory_contact_check.csv",

    # --- Device ---
    device="cuda" if torch.cuda.is_available() else "cpu",
    dtype=torch.float32,

    # --- Sampling ---
    n_surface_samples=20000,            # points sampled per mesh per pose
    seed=0,
    use_tqdm = True,  # set False if running in environments where tqdm is undesirable

    # --- Contact decision ---
    # With clearance ~0.2–2mm, using 0.2mm as "contact or essentially touching".
    # Increase to 5e-4 or 1e-3 if you want a "near-contact" band.
    contact_threshold=2e-4,             # meters
    use_penetration_test=True,          # watertight meshes -> good
    penetration_any_is_contact=True,

    # --- Performance ---
    chunk_size=50000,
    verbose_every=50,                   # print per-pose progress every N poses (0 to disable)
    
    # Kaolin check_sign acceleration structure resolution
    check_sign_hash_resolution=512,
    # Optional uniform downsampling of query points for check_sign
    # None disables. You can set either stride or max_points or both.
    check_sign_downsample_max_points=None,   # e.g. 5000
    check_sign_downsample_stride=None,       # e.g. 5
)

# -------------------------
# Kaolin imports (guarded)
# -------------------------

def _import_kaolin():
    import kaolin
    from kaolin.io.obj import import_mesh
    from kaolin.ops.mesh import index_vertices_by_faces, check_sign
    from kaolin.metrics.trianglemesh import point_to_mesh_distance
    return kaolin, import_mesh, index_vertices_by_faces, check_sign, point_to_mesh_distance


# ============================================================
# Quaternion utilities
# ============================================================

def quat_normalize_xyzw(q: torch.Tensor) -> torch.Tensor:
    return q / (q.norm(dim=-1, keepdim=True).clamp_min(1e-12))


def quat_dot(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    return (q1 * q2).sum(dim=-1)


def quat_to_rotmat_xyzw(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion (x,y,z,w) to rotation matrix.
    q: (4,)
    """
    q = quat_normalize_xyzw(q)
    x, y, z, w = q.unbind(-1)

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = torch.stack([
        1 - 2 * (yy + zz),     2 * (xy - wz),         2 * (xz + wy),
        2 * (xy + wz),         1 - 2 * (xx + zz),     2 * (yz - wx),
        2 * (xz - wy),         2 * (yz + wx),         1 - 2 * (xx + yy),
    ], dim=-1).reshape(3, 3)
    return R


def transform_points(R: torch.Tensor, t: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    return (pts @ R.T) + t


# ============================================================
# IO
# ============================================================

REQUIRED_COLS = ["x", "y", "z", "qx", "qy", "qz", "qw"]


def read_traj_csv(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.dtype.names is None:
        raise ValueError(f"Could not read header from {path}")
    for c in REQUIRED_COLS:
        if c not in data.dtype.names:
            raise ValueError(f"Missing column '{c}' in {path}. Found: {list(data.dtype.names)}")
    arr = np.vstack([data["x"], data["y"], data["z"], data["qx"], data["qy"], data["qz"], data["qw"]]).T
    return arr.astype(np.float64)


# ============================================================
# Surface sampling
# ============================================================

def sample_points_on_mesh_surface(
    verts: torch.Tensor,
    faces: torch.Tensor,
    n: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Area-weighted triangle surface sampling.
    verts: (V,3)
    faces: (F,3) long
    returns: (n,3)
    """
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    tri_areas = 0.5 * torch.linalg.norm(torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1)  # (F,)
    tri_probs = tri_areas / tri_areas.sum().clamp_min(1e-12)

    tri_idx = torch.multinomial(tri_probs, n, replacement=True, generator=generator)  # (n,)
    a0 = v0[tri_idx]
    a1 = v1[tri_idx]
    a2 = v2[tri_idx]

    u = torch.rand((n, 1), device=verts.device, dtype=verts.dtype, generator=generator)
    v = torch.rand((n, 1), device=verts.device, dtype=verts.dtype, generator=generator)
    su = torch.sqrt(u)
    b0 = 1.0 - su
    b1 = su * (1.0 - v)
    b2 = su * v

    return b0 * a0 + b1 * a1 + b2 * a2


# ============================================================
# Distance / penetration checks
# ============================================================

@torch.no_grad()
def min_point_to_mesh_distance(
    points: torch.Tensor,          # (N,3)
    face_vertices: torch.Tensor,   # (1,F,3,3)
    point_to_mesh_distance_fn,
    chunk_size: int,
) -> float:
    """
    Returns min unsigned distance from points to mesh (meters).
    """
    N = points.shape[0]
    min_d2 = None
    for s in range(0, N, chunk_size):
        p = points[s:s + chunk_size].unsqueeze(0)          # (1,nc,3)
        d2, _, _ = point_to_mesh_distance_fn(p, face_vertices)  # (1,nc)
        d2_min = d2.min()
        min_d2 = d2_min if min_d2 is None else torch.minimum(min_d2, d2_min)
    return float(torch.sqrt(min_d2).detach().cpu())


@torch.no_grad()
def any_points_inside_mesh(
    points: torch.Tensor,
    verts: torch.Tensor,
    faces: torch.Tensor,
    *,
    check_sign_fn,
    hash_resolution: int = 512,
    downsample_max_points: int | None = None,
    downsample_stride: int | None = None,
) -> torch.Tensor:
    """
    Returns whether ANY point is inside the watertight mesh.

    points: (P,3) or (B,P,3)
    verts:  (V,3) or (B,V,3)
    faces:  (F,3) int64/long (vertex indices)
    """
    # Downsample query points (optional)
    points = uniform_downsample_points(points, max_points=downsample_max_points, stride=downsample_stride)

    # Normalize shapes to batched
    squeeze_points = False
    if points.dim() == 2:
        points = points.unsqueeze(0)  # (1,P,3)
        squeeze_points = True
    if verts.dim() == 2:
        verts = verts.unsqueeze(0)    # (1,V,3)

    if faces.dtype != torch.long:
        faces = faces.long()

    # Kaolin expects: check_sign(verts, faces, points, hash_resolution=...)
    inside = check_sign_fn(verts, faces, points, hash_resolution=hash_resolution)  # (B,P) bool
    any_inside = inside.any(dim=1)  # (B,) bool

    return any_inside[0] if squeeze_points else any_inside

# ============================================================
# Stats helpers
# ============================================================

def uniform_downsample_points(points: torch.Tensor,
                              max_points: int | None = None,
                              stride: int | None = None) -> torch.Tensor:
    """
    Uniformly downsample point sets along the point dimension.

    Supports:
      - (P, 3)
      - (B, P, 3)

    If both max_points and stride are None, returns points unchanged.
    If both are provided, stride is applied first, then max_points.
    """
    if points is None:
        return points

    if points.dim() == 2:
        pts = points.unsqueeze(0)  # (1,P,3)
        squeeze_back = True
    elif points.dim() == 3:
        pts = points
        squeeze_back = False
    else:
        raise ValueError(f"points must be (P,3) or (B,P,3), got {tuple(points.shape)}")

    B, P, _ = pts.shape
    idx = torch.arange(P, device=pts.device)

    if stride is not None and stride > 1:
        idx = idx[::stride]

    if max_points is not None and max_points > 0 and idx.numel() > max_points:
        # pick max_points uniformly across idx
        lin = torch.linspace(0, idx.numel() - 1, steps=max_points, device=pts.device)
        sel = lin.round().long()
        idx = idx[sel]

    out = pts[:, idx, :]
    return out.squeeze(0) if squeeze_back else out


def longest_true_run(mask: np.ndarray) -> Tuple[int, Optional[int], Optional[int]]:
    """
    Returns (max_len, start_idx, end_idx_inclusive) for longest contiguous run of True in mask.
    If no True, returns (0, None, None).
    """
    max_len = 0
    max_s = None
    max_e = None

    cur_len = 0
    cur_s = None
    for i, v in enumerate(mask):
        if v:
            if cur_len == 0:
                cur_s = i
            cur_len += 1
            if cur_len > max_len:
                max_len = cur_len
                max_s = cur_s
                max_e = i
        else:
            cur_len = 0
            cur_s = None

    return max_len, max_s, max_e


def print_contact_stats(
    contact: np.ndarray,
    contact_by_distance: np.ndarray,
    contact_by_pen: np.ndarray,
    min_d: np.ndarray,
    thresh: float,
):
    M = contact.shape[0]
    n_contact = int(contact.sum())
    n_dist = int(contact_by_distance.sum())
    n_pen = int(contact_by_pen.sum())

    print("\n================ CONTACT STATS ================")
    print(f"Total poses: {M}")
    print(f"Contact threshold: {thresh:.6g} m ({thresh*1e3:.3f} mm)")
    print(f"Contact (final OR): {n_contact} / {M}  ({100.0*n_contact/M:.3f}%)")
    print(f"  - by distance:     {n_dist} / {M}  ({100.0*n_dist/M:.3f}%)")
    print(f"  - by penetration:  {n_pen} / {M}  ({100.0*n_pen/M:.3f}%)")

    if M > 0:
        md = min_d
        print("\nMin-distance distribution (meters / mm):")
        for name, val in [
            ("min", float(np.min(md))),
            ("median", float(np.median(md))),
            ("mean", float(np.mean(md))),
            ("p95", float(np.percentile(md, 95))),
            ("max", float(np.max(md))),
        ]:
            print(f"  {name:>6}: {val:.6g} m  ({val*1e3:.4f} mm)")

    if n_contact > 0:
        first = int(np.argmax(contact))
        last = int(np.where(contact)[0][-1])
        run_len, run_s, run_e = longest_true_run(contact.astype(bool))
        print("\nContact segments:")
        print(f"  first contact idx: {first}")
        print(f"  last  contact idx: {last}")
        print(f"  longest contact run: {run_len} poses"
              + (f" (idx {run_s}..{run_e})" if run_s is not None else ""))

        # Also longest run for penetration specifically (useful for “true collision”)
        if n_pen > 0:
            pen_run_len, pen_s, pen_e = longest_true_run(contact_by_pen.astype(bool))
            print(f"  longest penetration run: {pen_run_len} poses"
                  + (f" (idx {pen_s}..{pen_e})" if pen_s is not None else ""))
    else:
        print("\nNo contact detected (per chosen criteria).")

    # Optional quick sanity: how many are "near contact" within 2x threshold
    near = int((min_d <= (2.0 * thresh)).sum())
    print(f"\nNear-contact (min_d <= 2x thresh): {near} / {M} ({100.0*near/M:.3f}%)")
    print("================================================\n")


# ============================================================
# Main
# ============================================================

def main(cfg: Dict):

    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))

    _, import_mesh, index_vertices_by_faces, check_sign, point_to_mesh_distance = _import_kaolin()

    device = torch.device(cfg["device"])
    dtype = cfg["dtype"]

    traj = read_traj_csv(Path(cfg["traj_csv"]))  # (M,7)
    M = traj.shape[0]

    # Load meshes
    peg = import_mesh(cfg["peg_obj"])
    hole = import_mesh(cfg["hole_obj"])

    peg_verts = peg.vertices.to(device=device, dtype=dtype)
    peg_faces = peg.faces.to(device=device, dtype=torch.long)

    hole_verts = hole.vertices.to(device=device, dtype=dtype)
    hole_faces = hole.faces.to(device=device, dtype=torch.long)

    # Hole face vertices (static)
    hole_face_vertices = index_vertices_by_faces(hole_verts.unsqueeze(0), hole_faces)  # (1,F,3,3)

    gen = torch.Generator(device=device)
    gen.manual_seed(int(cfg["seed"]))

    # Sample hole surface points once (static)
    hole_surf_pts = sample_points_on_mesh_surface(
        hole_verts, hole_faces, int(cfg["n_surface_samples"]), generator=gen
    )

    contact_thresh = float(cfg["contact_threshold"])
    use_pen = bool(cfg["use_penetration_test"])
    pen_contact = bool(cfg["penetration_any_is_contact"])
    chunk = int(cfg["chunk_size"])

    # Storage for stats
    contact_flags = np.zeros((M,), dtype=np.int32)
    contact_by_dist_flags = np.zeros((M,), dtype=np.int32)
    contact_by_pen_flags = np.zeros((M,), dtype=np.int32)
    min_dist_all = np.zeros((M,), dtype=np.float64)

    out_rows = []  # only used if save_csv True

    iterator = range(M)
    if CONFIG["use_tqdm"]:
        iterator = tqdm(
            iterator,
            total=M,
            desc="Checking peg–hole contact",
            unit="pose",
            dynamic_ncols=True,
        )

    for i in iterator:
        x, y, z, qx, qy, qz, qw = traj[i]
        t = torch.tensor([x, y, z], device=device, dtype=dtype)
        q = torch.tensor([qx, qy, qz, qw], device=device, dtype=dtype)

        R = quat_to_rotmat_xyzw(q)

        # Transform peg vertices into hole frame
        peg_verts_tf = transform_points(R, t, peg_verts)
        peg_face_vertices = index_vertices_by_faces(peg_verts_tf.unsqueeze(0), peg_faces)

        # Sample peg surface points for this pose
        peg_surf_pts = sample_points_on_mesh_surface(
            peg_verts_tf, peg_faces, int(cfg["n_surface_samples"]), generator=gen
        )

        # unsigned distances both directions
        d_p2h = min_point_to_mesh_distance(
            peg_surf_pts, hole_face_vertices, point_to_mesh_distance, chunk_size=chunk
        )
        d_h2p = min_point_to_mesh_distance(
            hole_surf_pts, peg_face_vertices, point_to_mesh_distance, chunk_size=chunk
        )
        d_min = min(d_p2h, d_h2p)
        min_dist_all[i] = d_min

        # penetration (watertight meshes)
        pen = False
        if use_pen:
            
            pen_peg_in_hole = any_points_inside_mesh(
                points=peg_surf_pts,
                verts=hole_verts,
                faces=hole_faces,
                check_sign_fn=check_sign,
                hash_resolution=CONFIG.get("check_sign_hash_resolution", 512),
                downsample_max_points=CONFIG.get("check_sign_downsample_max_points", None),
                downsample_stride=CONFIG.get("check_sign_downsample_stride", None),
            )

            pen_hole_in_peg = any_points_inside_mesh(
                points=hole_surf_pts,
                verts=peg_verts_tf,     # IMPORTANT: use transformed peg verts, not the original peg_verts
                faces=peg_faces,
                check_sign_fn=check_sign,
                hash_resolution=CONFIG.get("check_sign_hash_resolution", 512),
                downsample_max_points=CONFIG.get("check_sign_downsample_max_points", None),
                downsample_stride=CONFIG.get("check_sign_downsample_stride", None),
            )


            pen = bool(pen_peg_in_hole.item()) or bool(pen_hole_in_peg.item())
        else:
            pen_peg_in_hole = False
            pen_hole_in_peg = False

        c_dist = (d_min <= contact_thresh)
        c_pen = pen
        c = c_dist or (c_pen and pen_contact)

        contact_flags[i] = int(c)
        contact_by_dist_flags[i] = int(c_dist)
        contact_by_pen_flags[i] = int(c_pen)

        if cfg["save_csv"]:
            out_rows.append([
                i, x, y, z, qx, qy, qz, qw,
                int(c),
                d_p2h,
                d_h2p,
                int(pen_peg_in_hole.item()),
                int(pen_hole_in_peg.item()),
            ])

        ve = int(cfg["verbose_every"])
        if ve and (i % ve == 0):
            if CONFIG["use_tqdm"]:
                tqdm.write(
                    f"[{i:05d}/{M}] contact={int(c)} "
                    f"min_d={d_min:.6g} m ({d_min*1e3:.4f} mm) "
                    f"(p2h={d_p2h*1e3:.4f} mm, h2p={d_h2p*1e3:.4f} mm) "
                    f"pen={int(c_pen)}"
                )
            else:
                print(
                    f"[{i:05d}/{M}] contact={int(c)} "
                    f"min_d={d_min:.6g} m ({d_min*1e3:.4f} mm) "
                    f"(p2h={d_p2h*1e3:.4f} mm, h2p={d_h2p*1e3:.4f} mm) "
                    f"pen={int(c_pen)}"
                )

    # Print stats
    print_contact_stats(
        contact=contact_flags.astype(bool),
        contact_by_distance=contact_by_dist_flags.astype(bool),
        contact_by_pen=contact_by_pen_flags.astype(bool),
        min_d=min_dist_all,
        thresh=contact_thresh,
    )

    # Optional CSV
    if cfg["save_csv"]:
        out = np.asarray(out_rows, dtype=np.float64)
        out_path = Path(cfg["out_csv"])
        header = (
            "idx,x,y,z,qx,qy,qz,qw,"
            "contact,min_dist_peg_to_hole,min_dist_hole_to_peg,"
            "penetration_peg_in_hole,penetration_hole_in_peg"
        )
        np.savetxt(out_path, out, delimiter=",", header=header, comments="", fmt="%.10g")
        print(f"[DONE] Wrote CSV results to: {out_path}")
    else:
        print("[DONE] CSV output disabled (save_csv=False).")


if __name__ == "__main__":
    main(CONFIG)
