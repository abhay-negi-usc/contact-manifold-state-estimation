# data_generation/kaolin/contact_sampling_voxel_sdf_exhaustive.py
#
# Voxel-SDF-based SE(3) sampler with CONFIG DICT (no argparse).
#
# This version supports **exhaustive (grid) sampling** over SE(3):
#   - Given min/max/step (or num) for each dimension (x,y,z,roll,pitch,yaw),
#     it enumerates *all* poses on that grid.
#   - Optional "frontier-based skipping": if penetration exceeds a threshold
#     along a chosen axis+direction (with all other dims held fixed), it stops
#     sampling further in that axis+direction for that prefix.
#
# Usage:
#   python -m data_generation.kaolin.contact_sampling_voxel_sdf
#   python -m data_generation.kaolin.contact_sampling_voxel_sdf geometry00015   # optional override
#
# Output:
#   Written under:
#     ./data_generation/data/<geometry_name>/sdf_based_contact_data/
#   - contact_sampling_voxel_sdf_config.json
#   - chunk_voxelsdf_0000.npz, ...
#   - samples_voxel_sdf.csv  (created at end)
#
# Requirements: torch, kaolin, numpy, tqdm, h5py

import os
import json
import csv
import glob
import time
import math
import sys
from typing import Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import h5py

from kaolin.io.obj import import_mesh
from kaolin.ops.conversions import trianglemeshes_to_voxelgrids
from kaolin.ops.mesh import sample_points as kal_sample_points


# ----------------------------
# CONFIG (edit here)
# ----------------------------
CFG = {
    "device": "cuda",
    "dtype": "float32",
    "seed": 0,

    # Geometry selection
    "geometry_name": "cylinder_keyway02",
    "mesh_root": "./data_generation/assets/meshes",

    # File naming convention
    "static_suffix": "_hole.obj",
    "moving_suffix": "_peg.obj",

    # SDF build
    "sdf_grid": {
        "resolution": 512,               # 128–256 typical
        "static_surface_samples": 2**12,
        "cdist_chunk": 1_000,            # voxel-centers chunk size for torch.cdist
        "workspace_margin": 0.000,       # extra buffer beyond computed bounds
    },

    # Sampling mode: "exhaustive" or "random"
    "sampling_mode": "exhaustive",

    # Exhaustive pose sampling (grid)
    # Provide either:
    #   - {"min": a, "max": b, "step": s}  (inclusive-ish; see make_grid_1d)
    # OR
    #   - {"min": a, "max": b, "num":  n}  (linspace, inclusive)
    "exhaustive_sampling": {
        "batch_size": 2**10,

        # Order matters for performance and frontier skipping.
        # If frontier_skip is enabled, put the chosen axis as the LAST element.
        "order": ["x", "y", "z", "roll", "pitch", "yaw"],

        "translation": {
            "x": {"min": -0.0005, "max": 0.0005, "step": 0.0001},
            "y": {"min": -0.0005, "max": 0.0005, "step": 0.0001},
            "z": {"min":  0.020, "max": 0.025, "step": 0.00025},
        },

        # rotations in DEGREES (will be converted to radians internally)
        "rotation_rpy_deg": {
            "roll":  {"min": -5.0, "max": 5.0, "step": 0.2},
            "pitch": {"min": -5.0, "max": 5.0, "step": 0.2},
            "yaw":   {"min": -1.0, "max": 1.0, "step": 0.2},
        },
    },

    # Frontier-based skipping:
    # If, for a fixed prefix of all other dims, penetration exceeds threshold at some
    # value along `axis` in `direction`, then further samples in that direction are skipped.
    #
    # NOTES:
    # - This assumes penetration is monotone (or "monotone enough") along that axis+direction
    #   for your local neighborhood. This is typically most reasonable for translation axes
    #   aligned with insertion direction (e.g., +z).
    # - For best effect, set exhaustive_sampling.order so that `axis` is LAST.
    "frontier_skip": {
        "enabled": False,
        "axis": "z",                  # one of: x,y,z,roll,pitch,yaw
        "direction": "+",             # "+", "-", or "both"
        "pen_threshold": 0.002,       # meters (max_penetration threshold)
        "include_threshold_pose": True,
        # internal: which metric triggers skipping; currently only "max_penetration"
        "metric": "max_penetration",
        # evaluate along the axis in chunks of this many samples (<= batch_size)
        "axis_chunk": 256,
    },

    # # Random pose sampling (kept for convenience)
    # "pose_sampling": {
    #     "total_samples": 1_000_000,
    #     "batch_size": 2**8, # increase this if more GPU memory is available
    #     "translation_bounds": {          # uniform bounds
    #         "x": [-0.001, 0.001],
    #         "y": [-0.001, 0.001],
    #         "z": [0.0, 0.025],
    #     },
    #     "rotation_bounds_rpy_deg": {     # uniform in degrees
    #         "roll":  [-5, 5],
    #         "pitch": [-5, 5],
    #         "yaw":   [-5, 5],
    #     },
    # },

    # Moving surface points per pose eval
    "moving_surface": {
        "samples": 2**14,
    },

    # Labels
    "filtering": {
        "near_thresh": 0.0005,
        "epsilon_pen": 0.0003,
    },

    # Output behavior
    "output": {
        "chunk_rows": 10_000_000_000,
        # Fast single-file output options (recommended): "h5" or "npz" or "csv".
        # - "h5": appendable, chunked, compressed; best for 10M+ rows.
        # - "npz": chunked npz files (original behavior)
        # - "csv": very slow for large datasets; kept for small debugging runs
        "format": "h5",

        # HDF5 output
        "h5_name": "exhaustive_samples_voxel_sdf.h5",
        "h5_compression": "lzf",  # "lzf" (fast) or "gzip" (smaller)
        "h5_chunk_rows": 262_144,

        # NPZ output (chunked)
        "npz_prefix": "exhaustive_chunk_voxelsdf",

        # CSV output
        "csv_name": "exhaustive_samples_voxel_sdf.csv",

        "write_config_json": True,
        "delete_npz_after_csv": False,
    }
}


# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)

def to_dtype(dtype_str: str):
    if dtype_str == "float32":
        return torch.float32
    if dtype_str == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype: {dtype_str}")

def load_obj_mesh(path: str, device: torch.device, dtype: torch.dtype):
    mesh = import_mesh(path, with_normals=False, with_materials=False)
    verts = mesh.vertices.to(device=device, dtype=dtype)
    faces = mesh.faces.to(device=device, dtype=torch.long)
    return verts, faces

def mesh_aabb(verts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    vmin = verts.min(dim=0).values
    vmax = verts.max(dim=0).values
    return vmin, vmax

def moving_radius_geometry(verts: torch.Tensor, mode: str = "aabb_center") -> torch.Tensor:
    """
    Radius that measures *geometry size*, not distance from the mesh origin.

    mode:
      - "aabb_center": center = (min+max)/2  (robust for end-origin meshes)
      - "centroid":    center = mean(vertices)
    """
    if mode == "aabb_center":
        vmin = verts.min(dim=0).values
        vmax = verts.max(dim=0).values
        center = 0.5 * (vmin + vmax)
    elif mode == "centroid":
        center = verts.mean(dim=0)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return torch.linalg.norm(verts - center, dim=1).max()

def sample_uniform(bounds: Tuple[float, float], shape, device, dtype):
    lo, hi = bounds
    return lo + (hi - lo) * torch.rand(shape, device=device, dtype=dtype)

def rpy_to_R(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """
    roll, pitch, yaw: (B,)
    returns R: (B,3,3) = Rz(yaw) * Ry(pitch) * Rx(roll)
    """
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)

    R = torch.zeros((roll.shape[0], 3, 3), device=roll.device, dtype=roll.dtype)
    R[:, 0, 0] = cy * cp
    R[:, 0, 1] = cy * sp * sr - sy * cr
    R[:, 0, 2] = cy * sp * cr + sy * sr

    R[:, 1, 0] = sy * cp
    R[:, 1, 1] = sy * sp * sr + cy * cr
    R[:, 1, 2] = sy * sp * cr - cy * sr

    R[:, 2, 0] = -sp
    R[:, 2, 1] = cp * sr
    R[:, 2, 2] = cp * cr
    return R

def world_to_grid_normed(xyz: torch.Tensor, min_xyz: torch.Tensor, max_xyz: torch.Tensor):
    """
    xyz: (B,N,3) in world coords
    returns normalized coords in [-1,1] for grid_sample on (1,1,D,H,W) volume.
    """
    uvw = (xyz - min_xyz) / (max_xyz - min_xyz + 1e-12)
    return uvw * 2.0 - 1.0

def trilinear_query_sdf(sdf_vol: torch.Tensor, coords_normed: torch.Tensor) -> torch.Tensor:
    """
    sdf_vol: (1,1,D,H,W) where D=z,H=y,W=x
    coords_normed: (B,N,3) in [-1,1] order (x,y,z)
    returns sdf: (B,N)
    """
    B, N, _ = coords_normed.shape
    grid = coords_normed.view(B, 1, 1, N, 3)
    vals = F.grid_sample(
        sdf_vol.expand(B, -1, -1, -1, -1),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False
    )
    return vals.view(B, N)

def voxelize_static_occupancy(static_verts: torch.Tensor,
                             static_faces: torch.Tensor,
                             min_xyz: torch.Tensor,
                             max_xyz: torch.Tensor,
                             res: int) -> torch.Tensor:
    """
    Returns occupancy volume (1,1,res,res,res) float in {0,1} where 1 means "inside".
    Kaolin expects:
      vertices: (B,V,3)
      faces:    (F,3)  (UNBATCHED)
    """
    # Normalize verts into [-1, 1] cube of the workspace
    v = static_verts
    v_norm = (v - min_xyz) / (max_xyz - min_xyz + 1e-12)
    v_norm = v_norm * 2.0 - 1.0
    v_norm = v_norm.unsqueeze(0).contiguous()          # (1,V,3)

    faces = static_faces.to(dtype=torch.long).contiguous()  # (F,3), UNBATCHED

    occ = trianglemeshes_to_voxelgrids(v_norm, faces, res)  # (1,res,res,res)
    return occ.float().unsqueeze(1)                          # (1,1,res,res,res)

def make_voxel_centers(min_xyz: torch.Tensor, max_xyz: torch.Tensor, res: int, device, dtype):
    """
    Returns centers in world coords for every voxel:
      centers: (res^3,3) in (x,y,z) ordering.
    """
    spans = (max_xyz - min_xyz)
    vs = spans / res
    xs = min_xyz[0] + (torch.arange(res, device=device, dtype=dtype) + 0.5) * vs[0]
    ys = min_xyz[1] + (torch.arange(res, device=device, dtype=dtype) + 0.5) * vs[1]
    zs = min_xyz[2] + (torch.arange(res, device=device, dtype=dtype) + 0.5) * vs[2]
    X, Y, Z = torch.meshgrid(xs, ys, zs, indexing="ij")
    return torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=-1)  # (Vox,3)

def moving_rotation_radius_about_origin(verts: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(verts, dim=1).max()  # max ||v||

def auto_workspace_from_meshes(static_verts, moving_verts, t_bounds, margin):
    device = static_verts.device
    dtype = static_verts.dtype

    smin, smax = mesh_aabb(static_verts)

    # IMPORTANT: rotation is about moving mesh origin
    r_move = moving_rotation_radius_about_origin(moving_verts)

    tmin = torch.tensor([t_bounds["x"][0], t_bounds["y"][0], t_bounds["z"][0]], device=device, dtype=dtype)
    tmax = torch.tensor([t_bounds["x"][1], t_bounds["y"][1], t_bounds["z"][1]], device=device, dtype=dtype)

    ws_min = smin + torch.minimum(tmin, tmax) - (r_move + margin)
    ws_max = smax + torch.maximum(tmin, tmax) + (r_move + margin)
    return ws_min, ws_max

def build_static_sdf_grid(static_verts: torch.Tensor,
                          static_faces: torch.Tensor,
                          min_xyz: torch.Tensor,
                          max_xyz: torch.Tensor,
                          res: int,
                          surface_samples: int,
                          cdist_chunk: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Builds signed distance field volume for static mesh.
    Returns:
      sdf_vol: (1,1,D,H,W) where D=z,H=y,W=x
      occ_vol: same layout
    """
    device = static_verts.device
    dtype = static_verts.dtype

    # Occupancy
    occ_vol = voxelize_static_occupancy(static_verts, static_faces, min_xyz, max_xyz, res).to(device=device)

    # Sample static surface points
    faces = static_faces.to(dtype=torch.long).contiguous()          # (F,3)
    pts, _ = kal_sample_points(static_verts.unsqueeze(0), faces, surface_samples)
    surf_pts = pts.squeeze(0).contiguous()  # (S,3)

    # Voxel centers
    centers = make_voxel_centers(min_xyz, max_xyz, res, device, dtype)  # (Vox,3)
    Vox = centers.shape[0]

    # Distance-to-surface via chunked cdist
    dmin = torch.empty((Vox,), device=device, dtype=dtype)
    for i in tqdm(range(0, Vox, cdist_chunk), desc="Building distance field (cdist)", unit="chunk"):
        c = centers[i:i + cdist_chunk]      # (C,3)
        d = torch.cdist(c, surf_pts)        # (C,S)
        dmin[i:i + cdist_chunk] = d.min(dim=1).values

    # dist volume initially (x,y,z); permute to (z,y,x) => (D,H,W)
    dist_vol = dmin.view(1, 1, res, res, res).permute(0, 1, 4, 3, 2).contiguous()
    occ_vol = occ_vol.permute(0, 1, 4, 3, 2).contiguous()

    sdf_vol = dist_vol * (1.0 - 2.0 * occ_vol)  # occ=1 => negative
    return sdf_vol, occ_vol


def flush_npz(out_dir: str, npz_prefix: str, chunk_idx: int,
              t_np, r_np, cs_np, ms_np, mp_np, cb_np, cv_np) -> str:
    path = os.path.join(out_dir, f"{npz_prefix}_{chunk_idx:04d}.npz")
    np.savez_compressed(
        path,
        t=t_np.astype(np.float32),
        r=r_np.astype(np.float32),
        config_sdf=cs_np.astype(np.float32),
        min_separation=ms_np.astype(np.float32),
        max_penetration=mp_np.astype(np.float32),
        contact_band=cb_np.astype(np.uint8),
        contact_valid=cv_np.astype(np.uint8),
    )
    return path


def _h5_create_datasets(h5: h5py.File, *, chunk_rows: int, compression: str):
    """Create resizable datasets for fast append."""
    # Use chunking along the first (row) dimension.
    def ds(name, shape_tail, dtype):
        return h5.create_dataset(
            name,
            shape=(0, *shape_tail),
            maxshape=(None, *shape_tail),
            dtype=dtype,
            chunks=(chunk_rows, *shape_tail) if shape_tail else (chunk_rows,),
            compression=compression,
        )

    ds("t", (3,), np.float32)
    ds("r", (3,), np.float32)
    ds("config_sdf", (), np.float32)
    ds("min_separation", (), np.float32)
    ds("max_penetration", (), np.float32)
    ds("contact_band", (), np.uint8)
    ds("contact_valid", (), np.uint8)


def h5_open_for_append(out_path: str, *, chunk_rows: int, compression: str) -> h5py.File:
    """Open (or create) an HDF5 file with appendable datasets."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    exists = os.path.exists(out_path)
    h5 = h5py.File(out_path, "a")
    if (not exists) or ("t" not in h5):
        # Fresh file or missing datasets.
        for k in list(h5.keys()):
            del h5[k]
        _h5_create_datasets(h5, chunk_rows=chunk_rows, compression=compression)
    return h5


def h5_append(
    h5: h5py.File,
    *,
    t_np: np.ndarray,
    r_np: np.ndarray,
    cs_np: np.ndarray,
    ms_np: np.ndarray,
    mp_np: np.ndarray,
    cb_np: np.ndarray,
    cv_np: np.ndarray,
):
    """Append one chunk to the HDF5 datasets."""
    n0 = h5["t"].shape[0]
    n = int(t_np.shape[0])
    n1 = n0 + n

    # Resize once, then write slices (fast).
    h5["t"].resize((n1, 3))
    h5["r"].resize((n1, 3))
    for k in ["config_sdf", "min_separation", "max_penetration", "contact_band", "contact_valid"]:
        h5[k].resize((n1,))

    h5["t"][n0:n1] = t_np.astype(np.float32, copy=False)
    h5["r"][n0:n1] = r_np.astype(np.float32, copy=False)
    h5["config_sdf"][n0:n1] = cs_np.astype(np.float32, copy=False)
    h5["min_separation"][n0:n1] = ms_np.astype(np.float32, copy=False)
    h5["max_penetration"][n0:n1] = mp_np.astype(np.float32, copy=False)
    h5["contact_band"][n0:n1] = cb_np.astype(np.uint8, copy=False)
    h5["contact_valid"][n0:n1] = cv_np.astype(np.uint8, copy=False)

    # Lightweight flush so crashes don't lose too much.
    h5.flush()

def npz_chunks_to_csv(out_dir: str, npz_prefix: str, out_csv: str):
    files = sorted(glob.glob(os.path.join(out_dir, f"{npz_prefix}_*.npz")))
    if not files:
        raise RuntimeError(f"No NPZ chunks found in {out_dir} matching {npz_prefix}_*.npz")

    header = [
        "t1","t2","t3",
        "r1","r2","r3",
        "config_sdf","min_separation","max_penetration",
        "contact_band","contact_valid"
    ]

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)

        for fp in tqdm(files, desc="Converting NPZ -> CSV", unit="file"):
            z = np.load(fp)
            t = z["t"]
            r = z["r"]
            cs = z["config_sdf"]
            ms = z["min_separation"]
            mp = z["max_penetration"]
            cb = z["contact_band"]
            cv = z["contact_valid"]

            for i in range(t.shape[0]):
                w.writerow([
                    float(t[i,0]), float(t[i,1]), float(t[i,2]),
                    float(r[i,0]), float(r[i,1]), float(r[i,2]),
                    float(cs[i]), float(ms[i]), float(mp[i]),
                    int(cb[i]), int(cv[i]),
                ])


def write_chunk_fast(
    *,
    out_dir: str,
    fmt: str,
    chunk_idx: int,
    t_np, r_np, cs_np, ms_np, mp_np, cb_np, cv_np,
    h5_handle: Optional[h5py.File] = None,
) -> Optional[str]:
    """Write one produced chunk in the selected output format."""
    fmt = fmt.lower()
    if fmt == "npz":
        return flush_npz(out_dir, CFG["output"]["npz_prefix"], chunk_idx, t_np, r_np, cs_np, ms_np, mp_np, cb_np, cv_np)
    if fmt == "h5":
        if h5_handle is None:
            raise ValueError("h5_handle is required when output.format == 'h5'")
        h5_append(h5_handle, t_np=t_np, r_np=r_np, cs_np=cs_np, ms_np=ms_np, mp_np=mp_np, cb_np=cb_np, cv_np=cv_np)
        return None
    if fmt == "csv":
        # CSV is only kept for small debugging runs.
        # We still write NPZ chunks and do a single conversion at end.
        return flush_npz(out_dir, CFG["output"]["npz_prefix"], chunk_idx, t_np, r_np, cs_np, ms_np, mp_np, cb_np, cv_np)
    raise ValueError(f"Unknown output.format: {fmt}")


def make_grid_1d(spec: Dict[str, float], *, device, dtype) -> torch.Tensor:
    """
    Create a 1D grid tensor from a spec dict.

    Supported:
      - {"min": a, "max": b, "step": s}  -> arange-like, includes max if within eps
      - {"min": a, "max": b, "num":  n}  -> linspace inclusive

    Returns: (K,) tensor on device/dtype.
    """
    lo = float(spec["min"])
    hi = float(spec["max"])
    if "num" in spec:
        n = int(spec["num"])
        if n <= 1:
            return torch.tensor([lo], device=device, dtype=dtype)
        return torch.linspace(lo, hi, n, device=device, dtype=dtype)

    step = float(spec["step"])
    if step <= 0:
        raise ValueError(f"step must be > 0, got {step}")

    # arange is half-open; we want "inclusive-ish" end
    # include hi if (hi-lo) is an integer multiple of step within tolerance
    n = int(math.floor((hi - lo) / step + 1e-12)) + 1
    vals = lo + step * torch.arange(n, device=device, dtype=dtype)
    # if last value is still < hi by more than tolerance, append hi
    if vals.numel() == 0:
        vals = torch.tensor([lo], device=device, dtype=dtype)
    if (hi - vals[-1]).abs().item() > (1e-9 + 1e-6 * abs(hi)):
        vals = torch.cat([vals, torch.tensor([hi], device=device, dtype=dtype)], dim=0)
    else:
        vals[-1] = torch.tensor(hi, device=device, dtype=dtype)
    return vals


def build_exhaustive_grids(cfg_exh: Dict, *, device, dtype):
    """
    Returns:
      grids: dict dim->(K,) tensor
      order: list[str]
      total_est: int (product of sizes)
    """
    order = list(cfg_exh["order"])

    tcfg = cfg_exh["translation"]
    rcfg = cfg_exh["rotation_rpy_deg"]

    grids = {}
    for k in ["x", "y", "z"]:
        grids[k] = make_grid_1d(tcfg[k], device=device, dtype=dtype)

    # degrees -> radians
    for k in ["roll", "pitch", "yaw"]:
        g_deg = make_grid_1d(rcfg[k], device=device, dtype=dtype)
        grids[k] = torch.deg2rad(g_deg)

    total_est = 1
    for d in order:
        total_est *= int(grids[d].numel())

    return grids, order, total_est


def eval_pose_batch(
    t: torch.Tensor,
    r: torch.Tensor,
    *,
    moving_surf_local: torch.Tensor,
    sdf_vol: torch.Tensor,
    min_xyz: torch.Tensor,
    max_xyz: torch.Tensor,
    near_thresh: float,
    epsilon_pen: float,
):
    """
    t: (B,3)
    r: (B,3) roll,pitch,yaw in radians
    Returns dict of tensors (B,)
    """
    curB = t.shape[0]
    roll, pitch, yaw = r[:, 0], r[:, 1], r[:, 2]

    R = rpy_to_R(roll, pitch, yaw)  # (B,3,3)
    pts = moving_surf_local.unsqueeze(0).expand(curB, -1, -1)                 # (B,Nm,3)
    pts_w = torch.bmm(pts, R.transpose(1, 2)) + t.unsqueeze(1)                # (B,Nm,3)

    coords = world_to_grid_normed(pts_w, min_xyz, max_xyz)                    # (B,Nm,3)
    sdf_vals = trilinear_query_sdf(sdf_vol, coords)                           # (B,Nm)

    config_sdf = sdf_vals.min(dim=1).values                                   # (B,)
    max_pen = torch.clamp(-sdf_vals, min=0.0).max(dim=1).values               # (B,)

    pos_mask = sdf_vals > 0
    min_sep = torch.where(
        pos_mask.any(dim=1),
        torch.where(
            pos_mask,
            sdf_vals,
            torch.tensor(float("inf"), device=sdf_vals.device, dtype=sdf_vals.dtype),
        ).min(dim=1).values,
        torch.zeros((sdf_vals.shape[0],), device=sdf_vals.device, dtype=sdf_vals.dtype),
    )

    contact_band = (torch.abs(config_sdf) <= near_thresh)
    contact_valid = contact_band & (max_pen <= epsilon_pen)

    return {
        "config_sdf": config_sdf,
        "min_separation": min_sep,
        "max_penetration": max_pen,
        "contact_band": contact_band,
        "contact_valid": contact_valid,
    }


# ----------------------------
# Main
# ----------------------------
def main():
    # Optional override without argparse:
    # python -m ... geometry00015
    if len(sys.argv) >= 2:
        CFG["geometry_name"] = sys.argv[1]

    set_seed(int(CFG["seed"]))
    device = torch.device(CFG["device"])
    dtype = to_dtype(CFG["dtype"])

    geom = CFG["geometry_name"]
    mesh_root = CFG["mesh_root"]
    mesh_dir = os.path.join(mesh_root, geom)

    static_obj = os.path.join(mesh_dir, f"{geom}{CFG['static_suffix']}")
    moving_obj = os.path.join(mesh_dir, f"{geom}{CFG['moving_suffix']}")

    if not os.path.exists(static_obj):
        raise FileNotFoundError(static_obj)
    if not os.path.exists(moving_obj):
        raise FileNotFoundError(moving_obj)

    # Output dir
    out_dir = f"./data_generation/data/{geom}/sdf_based_contact_data/"
    os.makedirs(out_dir, exist_ok=True)

    # Save config
    if CFG["output"]["write_config_json"]:
        cfg_path = os.path.join(out_dir, "contact_sampling_voxel_sdf_config.json")
        with open(cfg_path, "w") as f:
            json.dump(CFG, f, indent=2)

    print(f"[Geometry] {geom}")
    print(f"[Load] static: {static_obj}")
    print(f"[Load] moving: {moving_obj}")
    static_verts, static_faces = load_obj_mesh(static_obj, device, dtype)
    moving_verts, moving_faces = load_obj_mesh(moving_obj, device, dtype)

    # Auto workspace (use translation bounds from whichever mode is active)
    margin = float(CFG["sdf_grid"]["workspace_margin"])
    if CFG["sampling_mode"] == "exhaustive":
        tcfg = CFG["exhaustive_sampling"]["translation"]
        t_bounds = {
            "x": (float(tcfg["x"]["min"]), float(tcfg["x"]["max"])),
            "y": (float(tcfg["y"]["min"]), float(tcfg["y"]["max"])),
            "z": (float(tcfg["z"]["min"]), float(tcfg["z"]["max"])),
        }
    else:
        tb = CFG["pose_sampling"]["translation_bounds"]
        t_bounds = {"x": tuple(tb["x"]), "y": tuple(tb["y"]), "z": tuple(tb["z"])}

    min_xyz, max_xyz = auto_workspace_from_meshes(static_verts, moving_verts, t_bounds, margin=margin)

    # Debug: mesh scales + workspace voxel size sanity checks
    smin, smax = mesh_aabb(static_verts)
    mmin, mmax = mesh_aabb(moving_verts)
    m_centroid = moving_verts.mean(dim=0)
    m_aabb_center = 0.5 * (mmin + mmax)
    r_origin = torch.linalg.norm(moving_verts, dim=1).max()
    r_geom = moving_radius_geometry(moving_verts, mode="aabb_center")

    print("[Mesh AABB] static size:", (smax - smin).detach().cpu().numpy().tolist())
    print("[Mesh AABB] moving size:", (mmax - mmin).detach().cpu().numpy().tolist())
    print("[Moving origin offset] centroid:", m_centroid.detach().cpu().numpy().tolist(),
          "aabb_center:", m_aabb_center.detach().cpu().numpy().tolist())
    print("[Moving radius] origin_based:", float(r_origin.detach().cpu()),
          "geom_based:", float(r_geom.detach().cpu()))

    spans = (max_xyz - min_xyz)
    voxel = spans / float(CFG["sdf_grid"]["resolution"])
    print("[Workspace span]:", spans.detach().cpu().numpy().tolist())
    print("[Voxel size (m)]:", voxel.detach().cpu().numpy().tolist())
    print("[Workspace] min:", min_xyz.detach().cpu().numpy().tolist())
    print("[Workspace] max:", max_xyz.detach().cpu().numpy().tolist())

    # Build static SDF volume
    res = int(CFG["sdf_grid"]["resolution"])
    surface_samples = int(CFG["sdf_grid"]["static_surface_samples"])
    cdist_chunk = int(CFG["sdf_grid"]["cdist_chunk"])

    print(f"[SDF] building grid res={res}, static_surface_samples={surface_samples}")
    sdf_vol, _occ_vol = build_static_sdf_grid(
        static_verts, static_faces,
        min_xyz=min_xyz, max_xyz=max_xyz,
        res=res,
        surface_samples=surface_samples,
        cdist_chunk=cdist_chunk,
    )

    # Pre-sample moving surface points in local frame
    mv_samp = int(CFG["moving_surface"]["samples"])
    mfaces = moving_faces.to(dtype=torch.long).contiguous()         # (F,3)
    mv_pts, _ = kal_sample_points(moving_verts.unsqueeze(0), mfaces, mv_samp)
    moving_surf_local = mv_pts.squeeze(0).contiguous()  # (Nm,3)

    near_thresh = float(CFG["filtering"]["near_thresh"])
    epsilon_pen = float(CFG["filtering"]["epsilon_pen"])

    chunk_rows = int(CFG["output"]["chunk_rows"])
    out_fmt = str(CFG["output"].get("format", "npz")).lower()
    npz_prefix = CFG["output"]["npz_prefix"]
    out_csv = os.path.join(out_dir, CFG["output"]["csv_name"])
    out_h5 = os.path.join(out_dir, CFG["output"].get("h5_name", "samples_voxel_sdf.h5"))

    h5_handle: Optional[h5py.File] = None
    if out_fmt == "h5":
        h5_handle = h5_open_for_append(
            out_h5,
            chunk_rows=int(CFG["output"].get("h5_chunk_rows", 262_144)),
            compression=str(CFG["output"].get("h5_compression", "lzf")),
        )

    # Buffers
    buf_t, buf_r, buf_cs, buf_ms, buf_mp, buf_cb, buf_cv = [], [], [], [], [], [], []
    buf_count = 0
    chunk_idx = 0

    def flush():
        nonlocal buf_t, buf_r, buf_cs, buf_ms, buf_mp, buf_cb, buf_cv, buf_count, chunk_idx
        if buf_count == 0:
            return None

        t_np = np.concatenate(buf_t, axis=0)
        r_np = np.concatenate(buf_r, axis=0)
        cs_np = np.concatenate(buf_cs, axis=0)
        ms_np = np.concatenate(buf_ms, axis=0)
        mp_np = np.concatenate(buf_mp, axis=0)
        cb_np = np.concatenate(buf_cb, axis=0).astype(np.uint8)
        cv_np = np.concatenate(buf_cv, axis=0).astype(np.uint8)

        path = write_chunk_fast(
            out_dir=out_dir,
            fmt=out_fmt,
            chunk_idx=chunk_idx,
            t_np=t_np,
            r_np=r_np,
            cs_np=cs_np,
            ms_np=ms_np,
            mp_np=mp_np,
            cb_np=cb_np,
            cv_np=cv_np,
            h5_handle=h5_handle,
        )

        buf_t, buf_r, buf_cs, buf_ms, buf_mp, buf_cb, buf_cv = [], [], [], [], [], [], []
        buf_count = 0
        chunk_idx += 1
        return path

    # -------- Sampling loops --------
    t_start = time.time()

    if CFG["sampling_mode"] == "exhaustive":
        exh = CFG["exhaustive_sampling"]
        B = int(exh["batch_size"])
        grids, order, total_est = build_exhaustive_grids(exh, device=device, dtype=dtype)

        # Name CSV with grid size (optional)
        CFG["output"]["csv_name"] = f"samples_voxel_sdf_{geom}_grid_{total_est}.csv"
        out_csv = os.path.join(out_dir, CFG["output"]["csv_name"])

        fskip = CFG.get("frontier_skip", {})
        fs_enabled = bool(fskip.get("enabled", False))
        fs_axis = str(fskip.get("axis", "z"))
        fs_dir = str(fskip.get("direction", "+"))
        fs_thr = float(fskip.get("pen_threshold", 0.0))
        fs_include = bool(fskip.get("include_threshold_pose", True))
        fs_axis_chunk = int(fskip.get("axis_chunk", B))
        fs_axis_chunk = max(1, min(fs_axis_chunk, B))

        if fs_enabled:
            if fs_axis not in grids:
                raise ValueError(f"frontier_skip.axis must be one of {list(grids.keys())}, got {fs_axis}")
            if fs_dir not in ["+", "-", "both"]:
                raise ValueError(f"frontier_skip.direction must be '+', '-', or 'both', got {fs_dir}")
            if fskip.get("metric", "max_penetration") != "max_penetration":
                raise ValueError("frontier_skip.metric currently supports only 'max_penetration'")
            if order[-1] != fs_axis:
                print(f"[WARN] frontier_skip enabled but order[-1] != axis. "
                      f"Set exhaustive_sampling.order so '{fs_axis}' is LAST for correct pruning. "
                      f"Proceeding WITHOUT pruning.")
                fs_enabled = False

        # Helper to append results to buffers
        def add_to_buffers(t_batch, r_batch, metrics):
            nonlocal buf_count
            buf_t.append(t_batch.detach().cpu().numpy())
            buf_r.append(r_batch.detach().cpu().numpy())
            buf_cs.append(metrics["config_sdf"].detach().cpu().numpy())
            buf_ms.append(metrics["min_separation"].detach().cpu().numpy())
            buf_mp.append(metrics["max_penetration"].detach().cpu().numpy())
            buf_cb.append(metrics["contact_band"].detach().cpu().numpy().astype(np.uint8))
            buf_cv.append(metrics["contact_valid"].detach().cpu().numpy().astype(np.uint8))
            buf_count += int(t_batch.shape[0])

        # Build dim lists for prefix/axis
        dims_prefix = order[:-1]
        dim_axis = order[-1]
        axis_vals = grids[dim_axis]  # (K,)

        # Precompute index ranges for prefix dims
        prefix_sizes = [int(grids[d].numel()) for d in dims_prefix]

        # Progress bar uses total_est (upper bound if skipping)
        pbar = tqdm(total=total_est, desc="Exhaustive sampling", unit="pose")

        # Iterate over all prefixes using an odometer over dims_prefix
        prefix_idx = [0] * len(dims_prefix)
        prefix_done = (len(dims_prefix) == 0)

        def current_prefix_values():
            vals = {}
            for d, i in zip(dims_prefix, prefix_idx):
                vals[d] = grids[d][i]
            return vals

        def increment_prefix():
            nonlocal prefix_done
            if len(prefix_idx) == 0:
                prefix_done = True
                return
            for k in reversed(range(len(prefix_idx))):
                prefix_idx[k] += 1
                if prefix_idx[k] < prefix_sizes[k]:
                    return
                prefix_idx[k] = 0
                if k == 0:
                    prefix_done = True

        produced = 0

        while not prefix_done:
            pref = current_prefix_values()

            # Determine iteration order over the axis (for skipping direction)
            if fs_enabled and fs_dir == "-":
                axis_iter = torch.flip(axis_vals, dims=[0])
            else:
                axis_iter = axis_vals

            # Accumulate poses for the axis line; evaluate in chunks so we can stop early.
            axis_k = int(axis_iter.numel())
            axis_pos = 0
            stop_axis = False

            while axis_pos < axis_k and not stop_axis:
                take = min(fs_axis_chunk if fs_enabled else B, axis_k - axis_pos)
                a_chunk = axis_iter[axis_pos:axis_pos + take]  # (take,)

                # Build t,r for this chunk
                t_list = []
                r_list = []
                for i in range(int(a_chunk.numel())):
                    vals = dict(pref)
                    vals[dim_axis] = a_chunk[i]

                    tx = vals.get("x", torch.tensor(0.0, device=device, dtype=dtype))
                    ty = vals.get("y", torch.tensor(0.0, device=device, dtype=dtype))
                    tz = vals.get("z", torch.tensor(0.0, device=device, dtype=dtype))
                    roll = vals.get("roll", torch.tensor(0.0, device=device, dtype=dtype))
                    pitch = vals.get("pitch", torch.tensor(0.0, device=device, dtype=dtype))
                    yaw = vals.get("yaw", torch.tensor(0.0, device=device, dtype=dtype))

                    t_list.append(torch.stack([tx, ty, tz]))
                    r_list.append(torch.stack([roll, pitch, yaw]))

                t_batch = torch.stack(t_list, dim=0)  # (take,3)
                r_batch = torch.stack(r_list, dim=0)  # (take,3)

                metrics = eval_pose_batch(
                    t_batch, r_batch,
                    moving_surf_local=moving_surf_local,
                    sdf_vol=sdf_vol,
                    min_xyz=min_xyz,
                    max_xyz=max_xyz,
                    near_thresh=near_thresh,
                    epsilon_pen=epsilon_pen,
                )

                # Add all samples by default
                keep_n = int(take)

                if fs_enabled:
                    pen = metrics["max_penetration"]  # (take,)
                    exceed = pen > fs_thr
                    if exceed.any():
                        first = int(torch.nonzero(exceed, as_tuple=False)[0].item())
                        keep_n = first + (1 if fs_include else 0)
                        stop_axis = True

                if keep_n > 0:
                    add_to_buffers(t_batch[:keep_n], r_batch[:keep_n], {
                        "config_sdf": metrics["config_sdf"][:keep_n],
                        "min_separation": metrics["min_separation"][:keep_n],
                        "max_penetration": metrics["max_penetration"][:keep_n],
                        "contact_band": metrics["contact_band"][:keep_n],
                        "contact_valid": metrics["contact_valid"][:keep_n],
                    })

                    produced += keep_n
                    pbar.update(keep_n)

                    if buf_count >= chunk_rows:
                        npz_path = flush()
                        elapsed = time.time() - t_start
                        if npz_path is not None:
                            pbar.set_postfix({
                                "rate": f"{produced/max(elapsed,1e-6):.1f}/s",
                                "chunks": chunk_idx,
                                "last": os.path.basename(npz_path),
                            })

                # If we truncated due to frontier skip, do NOT advance beyond kept part
                if stop_axis:
                    # still "consumed" the chunk up to keep_n; rest is skipped
                    axis_pos = axis_k
                else:
                    axis_pos += take

            # Optional "both" direction: run the opposite direction too, with separate stopping.
            # This executes as a second sweep along the axis line.
            if fs_enabled and fs_dir == "both":
                axis_iter2 = torch.flip(axis_vals, dims=[0])  # opposite direction
                axis_k = int(axis_iter2.numel())
                axis_pos = 0
                stop_axis = False

                while axis_pos < axis_k and not stop_axis:
                    take = min(fs_axis_chunk, axis_k - axis_pos)
                    a_chunk = axis_iter2[axis_pos:axis_pos + take]

                    t_list = []
                    r_list = []
                    for i in range(int(a_chunk.numel())):
                        vals = dict(pref)
                        vals[dim_axis] = a_chunk[i]

                        tx = vals.get("x", torch.tensor(0.0, device=device, dtype=dtype))
                        ty = vals.get("y", torch.tensor(0.0, device=device, dtype=dtype))
                        tz = vals.get("z", torch.tensor(0.0, device=device, dtype=dtype))
                        roll = vals.get("roll", torch.tensor(0.0, device=device, dtype=dtype))
                        pitch = vals.get("pitch", torch.tensor(0.0, device=device, dtype=dtype))
                        yaw = vals.get("yaw", torch.tensor(0.0, device=device, dtype=dtype))

                        t_list.append(torch.stack([tx, ty, tz]))
                        r_list.append(torch.stack([roll, pitch, yaw]))

                    t_batch = torch.stack(t_list, dim=0)
                    r_batch = torch.stack(r_list, dim=0)

                    metrics = eval_pose_batch(
                        t_batch, r_batch,
                        moving_surf_local=moving_surf_local,
                        sdf_vol=sdf_vol,
                        min_xyz=min_xyz,
                        max_xyz=max_xyz,
                        near_thresh=near_thresh,
                        epsilon_pen=epsilon_pen,
                    )

                    keep_n = int(take)
                    pen = metrics["max_penetration"]
                    exceed = pen > fs_thr
                    if exceed.any():
                        first = int(torch.nonzero(exceed, as_tuple=False)[0].item())
                        keep_n = first + (1 if fs_include else 0)
                        stop_axis = True

                    if keep_n > 0:
                        add_to_buffers(t_batch[:keep_n], r_batch[:keep_n], {
                            "config_sdf": metrics["config_sdf"][:keep_n],
                            "min_separation": metrics["min_separation"][:keep_n],
                            "max_penetration": metrics["max_penetration"][:keep_n],
                            "contact_band": metrics["contact_band"][:keep_n],
                            "contact_valid": metrics["contact_valid"][:keep_n],
                        })
                        produced += keep_n
                        pbar.update(keep_n)

                        if buf_count >= chunk_rows:
                            npz_path = flush()
                            elapsed = time.time() - t_start
                            if npz_path is not None:
                                pbar.set_postfix({
                                    "rate": f"{produced/max(elapsed,1e-6):.1f}/s",
                                    "chunks": chunk_idx,
                                    "last": os.path.basename(npz_path),
                                })

                    if stop_axis:
                        axis_pos = axis_k
                    else:
                        axis_pos += take

            increment_prefix()

        pbar.close()

    else:
        # --- random sampling fallback ---
        total = int(CFG["pose_sampling"]["total_samples"])
        B = int(CFG["pose_sampling"]["batch_size"])

        tb = CFG["pose_sampling"]["translation_bounds"]
        rb = CFG["pose_sampling"]["rotation_bounds_rpy_deg"]

        print(f"[Run] mode=random total={total} batch={B} moving_pts={moving_surf_local.shape[0]}")
        produced = 0
        pbar = tqdm(total=total, desc="Random sampling", unit="pose")

        while produced < total:
            curB = min(B, total - produced)

            tx = sample_uniform(tuple(tb["x"]), (curB,), device, dtype)
            ty = sample_uniform(tuple(tb["y"]), (curB,), device, dtype)
            tz = sample_uniform(tuple(tb["z"]), (curB,), device, dtype)
            t = torch.stack([tx, ty, tz], dim=-1)

            roll  = sample_uniform(tuple(np.deg2rad(rb["roll"])),  (curB,), device, dtype)
            pitch = sample_uniform(tuple(np.deg2rad(rb["pitch"])), (curB,), device, dtype)
            yaw   = sample_uniform(tuple(np.deg2rad(rb["yaw"])),   (curB,), device, dtype)
            r = torch.stack([roll, pitch, yaw], dim=-1)

            metrics = eval_pose_batch(
                t, r,
                moving_surf_local=moving_surf_local,
                sdf_vol=sdf_vol,
                min_xyz=min_xyz,
                max_xyz=max_xyz,
                near_thresh=near_thresh,
                epsilon_pen=epsilon_pen,
            )

            buf_t.append(t.detach().cpu().numpy())
            buf_r.append(r.detach().cpu().numpy())
            buf_cs.append(metrics["config_sdf"].detach().cpu().numpy())
            buf_ms.append(metrics["min_separation"].detach().cpu().numpy())
            buf_mp.append(metrics["max_penetration"].detach().cpu().numpy())
            buf_cb.append(metrics["contact_band"].detach().cpu().numpy().astype(np.uint8))
            buf_cv.append(metrics["contact_valid"].detach().cpu().numpy().astype(np.uint8))
            buf_count += curB

            produced += curB
            pbar.update(curB)

            if buf_count >= chunk_rows:
                npz_path = flush()
                elapsed = time.time() - t_start
                if npz_path is not None:
                    pbar.set_postfix({"rate": f"{produced/max(elapsed,1e-6):.1f}/s", "chunks": chunk_idx, "last": os.path.basename(npz_path)})

        pbar.close()

    # Final flush
    if buf_count > 0:
        flush()

    # Finalize outputs
    if out_fmt == "h5":
        assert h5_handle is not None
        h5_handle.flush()
        h5_handle.close()
        print(f"[Done] HDF5 written: {out_h5}")
    elif out_fmt == "csv":
        # CSV is produced via chunked NPZ -> single CSV conversion.
        print("[Convert] NPZ chunks -> CSV ...")
        npz_chunks_to_csv(out_dir, npz_prefix, out_csv)
        print(f"[Done] CSV written: {out_csv}")

        # Optional cleanup
        if CFG["output"]["delete_npz_after_csv"]:
            files = sorted(glob.glob(os.path.join(out_dir, f"{npz_prefix}_*.npz")))
            for fp in tqdm(files, desc="Deleting NPZ chunks", unit="file"):
                os.remove(fp)
    else:
        # NPZ chunk output only
        files = sorted(glob.glob(os.path.join(out_dir, f"{npz_prefix}_*.npz")))
        print(f"[Done] Wrote {len(files)} NPZ chunks under: {out_dir}")


if __name__ == "__main__":
    main()
