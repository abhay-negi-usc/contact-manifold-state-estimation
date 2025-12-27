#!/usr/bin/env python3
"""
Adaptive mesh–mesh signed distance / separation / penetration using NVIDIA Kaolin (GPU).
CONFIG-FILE version (no argparse).

Usage:
  1) Copy this file.
  2) Edit the CONFIG dict at the top (or point to a separate config .py file if you want).
  3) Run: python kaolin_adaptive_sdf_config.py

If you want an external config file without argparse:
  - Set CONFIG_PATH below to "my_config.py" containing a dict named CONFIG,
    then run this script. (No CLI parsing needed.)
"""

from __future__ import annotations
import math
import os
import time
import csv
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Any
from tqdm import tqdm  # added

import torch

# Kaolin
import kaolin as kal
from kaolin.io.obj import import_mesh as import_obj_mesh
from kaolin.io.mesh import import_mesh as import_any_mesh
from kaolin.ops.mesh import index_vertices_by_faces, sample_points, check_sign
from kaolin.metrics.trianglemesh import point_to_mesh_distance

CONFIG_PATH: Optional[str] = "./data_generation/kaolin/contact_sampling_config.py"  

# -----------------------------
# Mesh + math helpers
# -----------------------------


def batch_face_vertices(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Return per-face vertices: (B,F,3,3) from verts (B,V,3) and faces (F,3) or (B,F,3)."""
    if verts.ndim != 3:
        raise ValueError(f"verts must be (B,V,3); got {tuple(verts.shape)}")
    if faces.ndim == 2:
        faces_b = faces.unsqueeze(0).expand(verts.shape[0], -1, -1)
    elif faces.ndim == 3:
        faces_b = faces
    else:
        raise ValueError(f"faces must be (F,3) or (B,F,3); got {tuple(faces.shape)}")

    if faces_b.dtype != torch.long:
        faces_b = faces_b.long()

    # Advanced indexing is simplest and avoids kaolin's (F,3) constraint.
    b = torch.arange(verts.shape[0], device=verts.device)[:, None, None]
    face_vertices = verts[b, faces_b]  # (B,F,3,3)
    return face_vertices

def signed_point_to_mesh_distance(query_points: torch.Tensor,
                                  mesh_verts_b: torch.Tensor,
                                  mesh_faces: torch.Tensor,
                                  eps: float = 1e-8) -> torch.Tensor:
    """
    query_points: (B, N, 3)
    mesh_verts_b: (B, V, 3)
    mesh_faces:   (F, 3) OR (B, F, 3)
    returns: (B, N) signed distance (negative inside)
    """
    # Normalize faces to (F,3)
    if mesh_faces.ndim == 3:
        mesh_faces = mesh_faces[0]
    if not (mesh_faces.ndim == 2 and mesh_faces.shape[1] == 3):
        raise ValueError(f"faces must be (F,3); got {tuple(mesh_faces.shape)}")

    # Enforce dtype + device
    mesh_faces = mesh_faces.to(device=mesh_verts_b.device, dtype=torch.int64).contiguous()

    # Unsigned distance
    face_vertices = index_vertices_by_faces(mesh_verts_b, mesh_faces)  # (B,F,3,3)
    d2, _, _ = point_to_mesh_distance(query_points, face_vertices)     # (B,N)
    d = torch.sqrt(torch.clamp(d2, min=eps))

    # Correct Kaolin signature: check_sign(verts, faces, points)
    inside = check_sign(mesh_verts_b, mesh_faces, query_points)        # (B,N) bool

    return torch.where(inside, -d, d)


def load_mesh(path: str, triangulate: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".obj":
        mesh = import_obj_mesh(path, triangulate=triangulate)
    else:
        mesh = import_any_mesh(path, triangulate=triangulate)
    verts = mesh.vertices.float()
    faces = mesh.faces.long()
    return verts, faces


def euler_xyz_to_R(angles_deg: torch.Tensor) -> torch.Tensor:
    angles = angles_deg * (math.pi / 180.0)
    ax, ay, az = angles[:, 0], angles[:, 1], angles[:, 2]

    cx, sx = torch.cos(ax), torch.sin(ax)
    cy, sy = torch.cos(ay), torch.sin(ay)
    cz, sz = torch.cos(az), torch.sin(az)

    Rx = torch.stack([
        torch.stack([torch.ones_like(cx), torch.zeros_like(cx), torch.zeros_like(cx)], dim=-1),
        torch.stack([torch.zeros_like(cx), cx, -sx], dim=-1),
        torch.stack([torch.zeros_like(cx), sx, cx], dim=-1),
    ], dim=-2)

    Ry = torch.stack([
        torch.stack([cy, torch.zeros_like(cy), sy], dim=-1),
        torch.stack([torch.zeros_like(cy), torch.ones_like(cy), torch.zeros_like(cy)], dim=-1),
        torch.stack([-sy, torch.zeros_like(cy), cy], dim=-1),
    ], dim=-2)

    Rz = torch.stack([
        torch.stack([cz, -sz, torch.zeros_like(cz)], dim=-1),
        torch.stack([sz, cz, torch.zeros_like(cz)], dim=-1),
        torch.stack([torch.zeros_like(cz), torch.zeros_like(cz), torch.ones_like(cz)], dim=-1),
    ], dim=-2)

    return Rz @ Ry @ Rx


def transform_points(R: torch.Tensor, t: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    if pts.dim() == 2:
        pts = pts.unsqueeze(0).expand(R.shape[0], -1, -1)
    return (pts @ R.transpose(1, 2)) + t.unsqueeze(1)

@torch.no_grad()
def evaluate_batch(
    static_verts: torch.Tensor, static_faces: torch.Tensor,
    moving_verts0: torch.Tensor, moving_faces: torch.Tensor,
    static_samples: torch.Tensor,    # (1,Ns,3)
    moving_samples0: torch.Tensor,   # (1,Nm,3)
    t: torch.Tensor,                 # (B,3)
    r_deg: torch.Tensor              # (B,3)
) -> Dict[str, torch.Tensor]:
    B = t.shape[0]
    R = euler_xyz_to_R(r_deg)

    moved_samples = transform_points(R, t, moving_samples0)        # (B,Nm,3)
    static_verts_b = static_verts.unsqueeze(0).expand(B, -1, -1)
    sd_m2s = signed_point_to_mesh_distance(moved_samples, static_verts_b, static_faces)  # (B,Nm)

    config_sdf = torch.min(sd_m2s, dim=1).values
    min_u_m2s = torch.min(torch.abs(sd_m2s), dim=1).values
    pen_m2s = torch.max(torch.clamp(-sd_m2s, min=0.0), dim=1).values

    moving_verts0_b = moving_verts0.unsqueeze(0).expand(B, -1, -1)
    moved_verts = transform_points(R, t, moving_verts0_b)          # (B,Vm,3)

    static_samples_b = static_samples.expand(B, -1, -1)
    sd_s2m = signed_point_to_mesh_distance(static_samples_b, moved_verts, moving_faces)  # (B,Ns)

    min_u_s2m = torch.min(torch.abs(sd_s2m), dim=1).values
    pen_s2m = torch.max(torch.clamp(-sd_s2m, min=0.0), dim=1).values

    min_sep_raw = torch.minimum(min_u_m2s, min_u_s2m)
    max_pen = torch.maximum(pen_m2s, pen_s2m)
    min_separation = torch.where(max_pen > 0.0, torch.zeros_like(min_sep_raw), min_sep_raw)

    config_sdf_sym = torch.minimum(config_sdf, torch.min(sd_s2m, dim=1).values)

    return {
        "config_sdf": config_sdf_sym,
        "min_separation": min_separation,
        "max_penetration": max_pen
    }

import torch
import torch.nn.functional as F
import math
import random


# -------------------------------------------------------------------------
# Robust triangle surface sampler (GPU, area-weighted, barycentric)
# -------------------------------------------------------------------------

def sample_points_on_triangles(verts, faces, num_samples):
    """
    Uniformly sample points on a triangle mesh.

    Inputs
    ------
    verts : Tensor
        (V,3) or (1,V,3)
    faces : Tensor
        (F,3) or (1,F,3) or (3,F) or (1,3,F)

    Returns
    -------
    pts : Tensor
        (1, num_samples, 3)
    face_idx : Tensor
        (1, num_samples)
    """

    device = verts.device
    dtype = verts.dtype

    # --------------------------
    # Normalize verts -> (1,V,3)
    # --------------------------
    if verts.ndim == 2:
        verts = verts.unsqueeze(0)
    elif verts.ndim == 3 and verts.shape[-1] != 3:
        raise ValueError(f"Unexpected verts shape {verts.shape}")

    verts = verts.contiguous()
    B, V, _ = verts.shape

    # --------------------------
    # Normalize faces -> (1,3,F)
    # --------------------------
    if faces.ndim == 2:  # (F,3) or (3,F)
        if faces.shape[1] == 3:
            faces = faces.transpose(0, 1).unsqueeze(0)
        elif faces.shape[0] == 3:
            faces = faces.unsqueeze(0)
        else:
            raise ValueError(f"Invalid faces shape {faces.shape}")

    elif faces.ndim == 3:
        if faces.shape[1] == 3:
            pass
        elif faces.shape[-1] == 3:
            faces = faces.transpose(1, 2)
        else:
            raise ValueError(f"Invalid faces shape {faces.shape}")
    else:
        raise ValueError(f"Invalid faces ndim {faces.ndim}")

    faces = faces.long().contiguous()
    _, _, F = faces.shape

    # --------------------------
    # Validate indices
    # --------------------------
    if faces.min() < 0 or faces.max() >= V:
        raise ValueError(
            f"Face indices out of bounds: min={faces.min().item()}, "
            f"max={faces.max().item()}, V={V}"
        )

    # --------------------------
    # Gather triangle vertices
    # tri: (B,F,3,3)
    # --------------------------
    faces_bf3 = faces.transpose(1, 2)  # (B,F,3)
    tri = verts.gather(
        1,
        faces_bf3.unsqueeze(-1).expand(B, F, 3, 3).reshape(B, F * 3, 3)
    ).reshape(B, F, 3, 3)

    v0 = tri[:, :, 0]
    v1 = tri[:, :, 1]
    v2 = tri[:, :, 2]

    # --------------------------
    # Triangle areas
    # --------------------------
    areas = 0.5 * torch.linalg.norm(
        torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1
    )  # (B,F)

    probs = areas / (areas.sum(dim=1, keepdim=True) + 1e-12)

    # --------------------------
    # Sample faces
    # --------------------------
    face_idx = torch.multinomial(probs, num_samples, replacement=True)  # (B,N)

    # --------------------------
    # Gather triangle vertices
    # --------------------------
    v0s = v0.gather(1, face_idx.unsqueeze(-1).expand(B, num_samples, 3))
    v1s = v1.gather(1, face_idx.unsqueeze(-1).expand(B, num_samples, 3))
    v2s = v2.gather(1, face_idx.unsqueeze(-1).expand(B, num_samples, 3))

    # --------------------------
    # Barycentric sampling
    # --------------------------
    u = torch.rand((B, num_samples, 1), device=device, dtype=dtype)
    v = torch.rand((B, num_samples, 1), device=device, dtype=dtype)

    su = torch.sqrt(u)
    w0 = 1 - su
    w1 = su * (1 - v)
    w2 = su * v

    pts = w0 * v0s + w1 * v1s + w2 * v2s

    return pts, face_idx


# -------------------------------------------------------------------------
# Replacement for your original make_surface_samples()
# -------------------------------------------------------------------------

def make_surface_samples(verts, faces, num, device, seed):
    """
    Robust replacement for Kaolin sample_points.
    """

    torch.manual_seed(seed)
    random.seed(seed)

    verts = verts.to(device)
    faces = faces.to(device)

    pts, _ = sample_points_on_triangles(verts, faces, num)

    return pts

def _faces_to_bf3(faces: torch.Tensor, B: int, device=None) -> torch.Tensor:
    """
    Normalize faces into (B, F, 3) LongTensor.
    Accepts: (F,3), (3,F), (1,F,3), (B,F,3), (1,3,F), (B,3,F)
    """
    if device is None:
        device = faces.device
    faces = faces.to(device)

    if faces.ndim == 2:
        # (F,3) or (3,F)
        if faces.shape[1] == 3:
            bf3 = faces.unsqueeze(0)  # (1,F,3)
        elif faces.shape[0] == 3:
            bf3 = faces.t().unsqueeze(0)  # (1,F,3)
        else:
            raise ValueError(f"Invalid faces shape {tuple(faces.shape)}")

    elif faces.ndim == 3:
        # (B,F,3) or (B,3,F) or (1,F,3) or (1,3,F)
        if faces.shape[-1] == 3:
            bf3 = faces  # (B,F,3) or (1,F,3)
        elif faces.shape[1] == 3:
            bf3 = faces.transpose(1, 2)  # (B,F,3)
        else:
            raise ValueError(f"Invalid faces shape {tuple(faces.shape)}")
    else:
        raise ValueError(f"Invalid faces ndim {faces.ndim}")

    # Broadcast batch if needed
    if bf3.shape[0] == 1 and B > 1:
        bf3 = bf3.expand(B, -1, -1)

    if bf3.shape[0] != B:
        raise ValueError(f"faces batch {bf3.shape[0]} != verts batch {B}")

    return bf3.long().contiguous()


def batch_face_vertices(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    Batched equivalent of Kaolin index_vertices_by_faces.

    verts: (B,V,3)
    faces: many possible shapes -> normalized to (B,F,3)
    returns: (B,F,3,3)
    """
    assert verts.ndim == 3 and verts.shape[-1] == 3, f"verts must be (B,V,3), got {verts.shape}"
    verts = verts.contiguous()
    B, V, _ = verts.shape

    faces_bf3 = _faces_to_bf3(faces, B, device=verts.device)  # (B,F,3)

    # Validate indices
    fmin = int(faces_bf3.min().item())
    fmax = int(faces_bf3.max().item())
    if fmin < 0 or fmax >= V:
        raise ValueError(f"Face indices out of bounds: min={fmin} max={fmax} V={V}")

    # Gather triangle vertices
    B, F, _ = faces_bf3.shape
    tri = verts.gather(
        1,
        faces_bf3.unsqueeze(-1).expand(B, F, 3, 3).reshape(B, F * 3, 3)
    ).reshape(B, F, 3, 3)
    return tri


# -----------------------------
# Adaptive sampling
# -----------------------------

@dataclass
class Cell6D:
    tmin: torch.Tensor
    tmax: torch.Tensor
    rmin: torch.Tensor
    rmax: torch.Tensor
    score: float


def sample_pose_in_cell(cell: Cell6D, B: int, device: torch.device, rng: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
    u_t = torch.rand((B, 3), generator=rng, device=device)
    u_r = torch.rand((B, 3), generator=rng, device=device)
    t = cell.tmin.to(device) + u_t * (cell.tmax.to(device) - cell.tmin.to(device))
    r = cell.rmin.to(device) + u_r * (cell.rmax.to(device) - cell.rmin.to(device))
    return t, r


def subdivide_cell(cell: Cell6D) -> Tuple[Cell6D, Cell6D]:
    ext = torch.cat([(cell.tmax - cell.tmin).abs(), (cell.rmax - cell.rmin).abs()], dim=0)
    k = int(torch.argmax(ext).item())
    if k < 3:
        mid = 0.5 * (cell.tmin[k] + cell.tmax[k])
        c1 = Cell6D(cell.tmin.clone(), cell.tmax.clone(), cell.rmin.clone(), cell.rmax.clone(), cell.score)
        c2 = Cell6D(cell.tmin.clone(), cell.tmax.clone(), cell.rmin.clone(), cell.rmax.clone(), cell.score)
        c1.tmax[k] = mid
        c2.tmin[k] = mid
    else:
        kk = k - 3
        mid = 0.5 * (cell.rmin[kk] + cell.rmax[kk])
        c1 = Cell6D(cell.tmin.clone(), cell.tmax.clone(), cell.rmin.clone(), cell.rmax.clone(), cell.score)
        c2 = Cell6D(cell.tmin.clone(), cell.tmax.clone(), cell.rmin.clone(), cell.rmax.clone(), cell.score)
        c1.rmax[kk] = mid
        c2.rmin[kk] = mid
    return c1, c2


# -----------------------------
# CSV helpers
# -----------------------------

CSV_HEADER = [
    "tx", "ty", "tz",
    "rx_deg", "ry_deg", "rz_deg",
    "config_sdf",
    "min_separation",
    "max_penetration",
    "contact_band",
    "contact_valid",
    "fidelity"
]


def ensure_csv(path: str):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)


def append_rows_csv(path: str, rows: List[List]):
    if not path or not rows:
        return
    with open(path, "a", newline="") as f:
        csv.writer(f).writerows(rows)


def tensors_to_csv_rows(
    t_cpu: torch.Tensor, r_cpu: torch.Tensor,
    csdf_cpu: torch.Tensor, minsep_cpu: torch.Tensor, maxpen_cpu: torch.Tensor,
    near_thresh: float, epsilon_pen: float,
    fidelity: str
) -> List[List]:
    contact_band = (csdf_cpu.abs() <= near_thresh)
    contact_valid = contact_band & (maxpen_cpu <= epsilon_pen)
    rows: List[List] = []
    for i in range(t_cpu.shape[0]):
        rows.append([
            float(t_cpu[i, 0]), float(t_cpu[i, 1]), float(t_cpu[i, 2]),
            float(r_cpu[i, 0]), float(r_cpu[i, 1]), float(r_cpu[i, 2]),
            float(csdf_cpu[i]),
            float(minsep_cpu[i]),
            float(maxpen_cpu[i]),
            int(contact_band[i].item()),
            int(contact_valid[i].item()),
            fidelity
        ])
    return rows

def to_numpy(x):
    import torch, numpy as np
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

# -----------------------------
# Config loading
# -----------------------------

def load_config() -> Dict[str, Any]:
    if CONFIG_PATH is None:
        return CONFIG
    # Load python file without argparse; must define CONFIG = {...}
    cfg_ns: Dict[str, Any] = {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        code = f.read()
    exec(compile(code, CONFIG_PATH, "exec"), cfg_ns, cfg_ns)
    if "CONFIG" not in cfg_ns:
        raise RuntimeError(f"{CONFIG_PATH} must define CONFIG = {{...}}")
    return cfg_ns["CONFIG"]


def to_tensor3(x: List[float]) -> torch.Tensor:
    assert len(x) == 3
    return torch.tensor(x, dtype=torch.float32)

# -----------------------------
# Main
# -----------------------------

def main():
    cfg = load_config()

    paths = cfg["paths"]
    bounds = cfg["bounds"]
    sampling = cfg["sampling"]
    thr = cfg["thresholds"]
    surf = cfg["surface_samples"]
    io_cfg = cfg["io"]
    ref = cfg["refinement"]

    static_mesh_path = paths["static_mesh"]
    moving_mesh_path = paths["moving_mesh"]
    csv_out = paths.get("csv_out", "")
    npz_out = paths.get("npz_out", "")

    device_name = cfg.get("device", "cuda")
    device = torch.device(device_name if (device_name == "cpu" or torch.cuda.is_available()) else "cpu")

    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)

    ensure_csv(csv_out)

    budget = int(sampling["budget_rows"])
    gpu_batch = int(sampling.get("gpu_batch", 4096))
    probe_per_cell = sampling.get("probe_per_cell", None)
    if probe_per_cell is None:
        probe_per_cell = min(1024, gpu_batch)
    else:
        probe_per_cell = int(min(probe_per_cell, gpu_batch))

    far_thresh = float(thr["far_thresh"])
    near_thresh = float(thr["near_thresh"])
    epsilon_pen = float(thr["epsilon_pen"])

    csv_flush_every = int(io_cfg.get("csv_flush_every", 20000))
    print_every = int(io_cfg.get("print_every", 5000))
    write_npz = bool(io_cfg.get("write_npz", True))

    enable_fine = bool(ref.get("enable_fine_upgrade", True))
    very_near_factor = float(ref.get("very_near_factor", 0.5))
    count_fine = bool(ref.get("count_fine_toward_budget", True))

    # Load meshes
    static_verts, static_faces = load_mesh(static_mesh_path, triangulate=True)
    moving_verts0, moving_faces = load_mesh(moving_mesh_path, triangulate=True)
    static_verts = static_verts.to(device)
    static_faces = static_faces.to(device)
    moving_verts0 = moving_verts0.to(device)
    moving_faces = moving_faces.to(device)

    # Surface samples
    moving_samples0_coarse = make_surface_samples(moving_verts0, moving_faces, int(surf["moving_coarse"]), device, seed + 11)
    moving_samples0_fine   = make_surface_samples(moving_verts0, moving_faces, int(surf["moving_fine"]), device, seed + 12)
    static_samples_fine    = make_surface_samples(static_verts,  static_faces,  int(surf["static_fine"]), device, seed + 21)
    static_probe_cap = int(surf.get("static_probe_cap", 512))
    static_samples_probe = static_samples_fine[:, :min(static_probe_cap, static_samples_fine.shape[1]), :]

    # Root cell
    root = Cell6D(
        tmin=to_tensor3(bounds["tmin"]),
        tmax=to_tensor3(bounds["tmax"]),
        rmin=to_tensor3(bounds["rmin"]),
        rmax=to_tensor3(bounds["rmax"]),
        score=0.0
    )
    queue: List[Cell6D] = [root]

    rng = torch.Generator(device=device)
    rng.manual_seed(seed + 999)

    rows_buffer: List[List] = []
    written = 0

    if write_npz:
        out_t, out_r, out_csdf, out_minsep, out_maxpen = [], [], [], [], []
        out_fid: List[str] = []

    t0 = time.time()
    last_print = 0
    last_flush = 0

    pbar = tqdm(total=budget, desc="Sampling", unit="row") 

    def do_print(force: bool = False):
        nonlocal last_print
        if force or (written - last_print >= print_every) or (written >= budget):
            elapsed = max(1e-6, time.time() - t0)
            rate = written / elapsed
            pct = 100.0 * written / max(1, budget)
            print(f"[{written}/{budget}] ({pct:5.1f}%)  {rate:,.1f} rows/sec  queue={len(queue)}  device={device}")
            last_print = written

    def do_flush(force: bool = False):
        nonlocal last_flush, rows_buffer
        if force or (written - last_flush >= csv_flush_every) or (written >= budget):
            append_rows_csv(csv_out, rows_buffer)
            rows_buffer = []
            last_flush = written
            if csv_out:
                print(f"  -> flushed CSV at {written} rows: {csv_out}")

    while queue and written < budget:
        queue.sort(key=lambda c: c.score)
        cell = queue.pop(0)

        Bp = min(probe_per_cell, budget - written)
        t_probe, r_probe = sample_pose_in_cell(cell, Bp, device, rng)

        metrics_probe = evaluate_batch(
            static_verts, static_faces,
            moving_verts0, moving_faces,
            static_samples_probe,
            moving_samples0_coarse,
            t_probe, r_probe
        )

        # CPU for IO
        t_cpu = t_probe.detach().cpu()
        r_cpu = r_probe.detach().cpu()
        csdf_cpu = metrics_probe["config_sdf"].detach().cpu()
        minsep_cpu = metrics_probe["min_separation"].detach().cpu()
        maxpen_cpu = metrics_probe["max_penetration"].detach().cpu()

        rows_buffer.extend(tensors_to_csv_rows(
            t_cpu, r_cpu, csdf_cpu, minsep_cpu, maxpen_cpu,
            near_thresh=near_thresh, epsilon_pen=epsilon_pen,
            fidelity="coarse"
        ))

        if write_npz:
            out_t.append(t_cpu); out_r.append(r_cpu)
            out_csdf.append(csdf_cpu); out_minsep.append(minsep_cpu); out_maxpen.append(maxpen_cpu)
            out_fid.extend(["coarse"] * t_cpu.shape[0])

        written += Bp
        pbar.update(Bp) 
        do_print()
        do_flush()

        if written >= budget:
            break

        abs_csdf = csdf_cpu.abs()
        far_mask = (abs_csdf > far_thresh) & (minsep_cpu > far_thresh) & (maxpen_cpu == 0.0)
        near_mask = (abs_csdf < near_thresh) | (maxpen_cpu > 0.0)

        cell_score = float(torch.min(abs_csdf).item())

        if torch.all(far_mask).item():
            continue

        if torch.any(near_mask).item():
            c1, c2 = subdivide_cell(cell)
            c1.score = cell_score
            c2.score = cell_score
            queue.append(c1)
            queue.append(c2)

            if enable_fine and written < budget:
                very_near = (abs_csdf < (very_near_factor * near_thresh)) | (maxpen_cpu > 0.0)
                idx = torch.nonzero(very_near).squeeze(1)

                if idx.numel() > 0:
                    # limit fine rows if counting toward budget
                    if count_fine:
                        nf = min(int(idx.numel()), budget - written)
                        idx = idx[:nf]
                    else:
                        # still cap to avoid absurd bursts
                        idx = idx[:min(int(idx.numel()), gpu_batch)]

                    tf = t_probe[idx.to(device)]
                    rf = r_probe[idx.to(device)]

                    metrics_fine = evaluate_batch(
                        static_verts, static_faces,
                        moving_verts0, moving_faces,
                        static_samples_fine,
                        moving_samples0_fine,
                        tf, rf
                    )

                    tf_cpu = tf.detach().cpu()
                    rf_cpu = rf.detach().cpu()
                    csdf_f = metrics_fine["config_sdf"].detach().cpu()
                    minsep_f = metrics_fine["min_separation"].detach().cpu()
                    maxpen_f = metrics_fine["max_penetration"].detach().cpu()

                    rows_buffer.extend(tensors_to_csv_rows(
                        tf_cpu, rf_cpu, csdf_f, minsep_f, maxpen_f,
                        near_thresh=near_thresh, epsilon_pen=epsilon_pen,
                        fidelity="fine"
                    ))

                    if write_npz:
                        out_t.append(tf_cpu); out_r.append(rf_cpu)
                        out_csdf.append(csdf_f); out_minsep.append(minsep_f); out_maxpen.append(maxpen_f)
                        out_fid.extend(["fine"] * tf_cpu.shape[0])

                    if count_fine:
                        written += tf_cpu.shape[0]
                        do_print()
                        do_flush()

                    if written >= budget:
                        break

    pbar.close() 

    # Final flush
    do_flush(force=True)

    # Final NPZ
    if write_npz and npz_out:
        import numpy as np
        T_all = torch.cat(out_t, dim=0)[:budget]
        R_all = torch.cat(out_r, dim=0)[:budget]
        csdf_all = torch.cat(out_csdf, dim=0)[:budget]
        minsep_all = torch.cat(out_minsep, dim=0)[:budget]
        maxpen_all = torch.cat(out_maxpen, dim=0)[:budget]
        fid_all = np.array(out_fid[:budget], dtype=object)

        contact_band = (csdf_all.abs() <= near_thresh)
        contact_valid = contact_band & (maxpen_all <= epsilon_pen)

        os.makedirs(os.path.dirname(npz_out) or ".", exist_ok=True)
        npz_out = str(npz_out)

        np.savez_compressed(
            npz_out,
            t=np.asarray(T_all.cpu().numpy()),
            r_deg=np.asarray(R_all.cpu().numpy()),
            config_sdf=np.asarray(csdf_all.cpu().numpy()),
            min_separation=np.asarray(minsep_all.cpu().numpy()),
            max_penetration=np.asarray(maxpen_all.cpu().numpy()),
            contact_band=np.asarray(contact_band.cpu().numpy(), dtype=np.uint8),
            contact_valid=np.asarray(contact_valid.cpu().numpy(), dtype=np.uint8),
            fidelity=np.asarray(fid_all, dtype=object),
        )


        print(f"Saved NPZ -> {npz_out}")

    print(f"Done. Total rows written: {written}")
    if csv_out:
        print(f"CSV: {csv_out}")


if __name__ == "__main__":
    main()
