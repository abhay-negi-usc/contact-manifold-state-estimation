# data_generation/kaolin/contact_sampling_voxel_sdf.py
#
# Voxel-SDF-based SE(3) sampler with CONFIG DICT (no argparse).
#
# Usage:
#   python -m data_generation.kaolin.contact_sampling_voxel_sdf
#   python -m data_generation.kaolin.contact_sampling_voxel_sdf geometry00015   # optional override
#
# Output:
#   Written to the same directory as the meshes:
#     ./data_generation/assets/meshes/<geometry_name>/
#   - contact_sampling_voxel_sdf_config.json
#   - chunk_voxelsdf_0000.npz, ...
#   - samples_voxel_sdf.csv  (created at end)
#
# Requirements: torch, kaolin, numpy, tqdm

## TODO: 
# - save all SDF files in a subfolder under the mesh dir, and save one large file describing the full SDF also under the mesh dir 
# - add in functionality which checks if SDF file/files already exists and loads them if it exists or if path is provided 
# - create a separate script to visualize the geometry and SDF with a 3D rotating gif, showing the geometry in 

import os
import json
import csv
import glob
import time
import math
import sys
from typing import Tuple, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

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
        "resolution": 256,               # 128–256 typical
        "static_surface_samples": 2**10,
        "cdist_chunk": 1_000,           # voxel-centers chunk size for torch.cdist
        "workspace_margin": 0.000,        # extra buffer beyond computed bounds
    },

    # Pose sampling
    "pose_sampling": {
        "total_samples": 1_000_000,
        "batch_size": 2**8,
        "translation_bounds": {          # uniform bounds
            "x": [-0.001, 0.001],
            "y": [-0.001, 0.001],
            "z": [0.0, 0.025],
        },
        "rotation_bounds_rpy_deg": {     # uniform in degrees
            "roll":  [-5, 5],
            "pitch": [-5, 5],
            "yaw":   [-5, 5],
        },
    },

    # Moving surface points per pose eval
    "moving_surface": {
        "samples": 2**10,
    },

    # Labels
    "filtering": {
        "near_thresh": 0.0005,
        "epsilon_pen": 0.0001,
    },

    # Output behavior
    "output": {
        "chunk_rows": 10_000_000,
        "npz_prefix": "chunk_voxelsdf",
        "csv_name": "samples_voxel_sdf.csv",
        "write_config_json": True,
        "delete_npz_after_csv": False,
    }
}

CFG["output"]["csv_name"] = f"samples_voxel_sdf_{CFG['geometry_name']}_{CFG['pose_sampling']['total_samples']}.csv"


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
    print("static verts:", static_verts.unsqueeze(0).shape, "faces:", faces.shape)
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

            # write rows (I/O heavy, but happens once at end)
            for i in range(t.shape[0]):
                w.writerow([
                    float(t[i,0]), float(t[i,1]), float(t[i,2]),
                    float(r[i,0]), float(r[i,1]), float(r[i,2]),
                    float(cs[i]), float(ms[i]), float(mp[i]),
                    int(cb[i]), int(cv[i]),
                ])


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

    # Output to mesh dir
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

    # Auto workspace
    tb = CFG["pose_sampling"]["translation_bounds"]
    t_bounds = {"x": tuple(tb["x"]), "y": tuple(tb["y"]), "z": tuple(tb["z"])}
    margin = float(CFG["sdf_grid"]["workspace_margin"])
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


    print("[Workspace]")
    print("  min:", min_xyz.detach().cpu().numpy().tolist())
    print("  max:", max_xyz.detach().cpu().numpy().tolist())

    # Build static SDF volume
    res = int(CFG["sdf_grid"]["resolution"])
    surface_samples = int(CFG["sdf_grid"]["static_surface_samples"])
    cdist_chunk = int(CFG["sdf_grid"]["cdist_chunk"])

    print(f"[SDF] building grid res={res}, static_surface_samples={surface_samples}")
    sdf_vol, occ_vol = build_static_sdf_grid(
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

    # Sampling params
    total = int(CFG["pose_sampling"]["total_samples"])
    B = int(CFG["pose_sampling"]["batch_size"])

    rb = CFG["pose_sampling"]["rotation_bounds_rpy_deg"]
    near_thresh = float(CFG["filtering"]["near_thresh"])
    epsilon_pen = float(CFG["filtering"]["epsilon_pen"])

    chunk_rows = int(CFG["output"]["chunk_rows"])
    npz_prefix = CFG["output"]["npz_prefix"]
    out_csv = os.path.join(out_dir, CFG["output"]["csv_name"])

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

        path = flush_npz(out_dir, npz_prefix, chunk_idx, t_np, r_np, cs_np, ms_np, mp_np, cb_np, cv_np)

        buf_t, buf_r, buf_cs, buf_ms, buf_mp, buf_cb, buf_cv = [], [], [], [], [], [], []
        buf_count = 0
        chunk_idx += 1
        return path

    print(f"[Run] total={total} batch={B} moving_pts={moving_surf_local.shape[0]}")
    t_start = time.time()
    produced = 0

    pbar = tqdm(total=total, desc="Sampling", unit="pose")

    # BEGINNING TEST 
    # surface points should have sdf near 0
    pts_s, _ = kal_sample_points(static_verts.unsqueeze(0), static_faces.long(), 2000)
    pts_s = pts_s.squeeze(0)
    coords_s = world_to_grid_normed(pts_s.unsqueeze(0), min_xyz, max_xyz)
    sdf_s = trilinear_query_sdf(sdf_vol, coords_s).squeeze(0)
    print("static surface sdf |min/mean/max|", sdf_s.abs().min().item(), sdf_s.abs().mean().item(), sdf_s.abs().max().item())

    # centroid sign sanity (may be outside if concave / weird)
    cent = static_verts.mean(dim=0, keepdim=True)
    coords_c = world_to_grid_normed(cent.unsqueeze(0), min_xyz, max_xyz)
    sdf_c = trilinear_query_sdf(sdf_vol, coords_c).item()
    print("static centroid sdf", sdf_c)
    # END TEST 

    while produced < total:
        curB = min(B, total - produced)

        # Sample translations
        tx = sample_uniform(tuple(tb["x"]), (curB,), device, dtype)
        ty = sample_uniform(tuple(tb["y"]), (curB,), device, dtype)
        tz = sample_uniform(tuple(tb["z"]), (curB,), device, dtype)
        t = torch.stack([tx, ty, tz], dim=-1)  # (B,3)

        # Sample rotations
        roll  = sample_uniform(tuple(np.deg2rad(rb["roll"])),  (curB,), device, dtype)
        pitch = sample_uniform(tuple(np.deg2rad(rb["pitch"])), (curB,), device, dtype)
        yaw   = sample_uniform(tuple(np.deg2rad(rb["yaw"])),   (curB,), device, dtype)
        r = torch.stack([roll, pitch, yaw], dim=-1)  # (B,3)

        # Transform moving surface points
        R = rpy_to_R(roll, pitch, yaw)  # (B,3,3)
        pts = moving_surf_local.unsqueeze(0).expand(curB, -1, -1)                 # (B,Nm,3)
        pts_w = torch.bmm(pts, R.transpose(1, 2)) + t.unsqueeze(1)                # (B,Nm,3)

        # SDF query
        coords = world_to_grid_normed(pts_w, min_xyz, max_xyz)                    # (B,Nm,3)
        sdf_vals = trilinear_query_sdf(sdf_vol, coords)                           # (B,Nm)

        # Metrics
        config_sdf = sdf_vals.min(dim=1).values                                   # (B,)
        max_pen = torch.clamp(-sdf_vals, min=0.0).max(dim=1).values               # (B,)
        pos_mask = sdf_vals > 0
        # If there are positive distances, take the minimum positive separation; else 0
        min_sep = torch.where(
            pos_mask.any(dim=1),
            torch.where(pos_mask, sdf_vals, torch.tensor(float("inf"), device=sdf_vals.device, dtype=sdf_vals.dtype)).min(dim=1).values,
            torch.zeros((sdf_vals.shape[0],), device=sdf_vals.device, dtype=sdf_vals.dtype),
        )


        contact_band = (torch.abs(config_sdf) <= near_thresh)
        contact_valid = contact_band & (max_pen <= epsilon_pen)

        # Batch -> CPU
        buf_t.append(t.detach().cpu().numpy())
        buf_r.append(r.detach().cpu().numpy())
        buf_cs.append(config_sdf.detach().cpu().numpy())
        buf_ms.append(min_sep.detach().cpu().numpy())
        buf_mp.append(max_pen.detach().cpu().numpy())
        buf_cb.append(contact_band.detach().cpu().numpy().astype(np.uint8))
        buf_cv.append(contact_valid.detach().cpu().numpy().astype(np.uint8))
        buf_count += curB

        produced += curB
        pbar.update(curB)

        if buf_count >= chunk_rows:
            npz_path = flush()
            elapsed = time.time() - t_start
            pbar.set_postfix({"rate": f"{produced/max(elapsed,1e-6):.1f}/s", "chunks": chunk_idx, "last": os.path.basename(npz_path)})

    pbar.close()

    # Final flush
    if buf_count > 0:
        flush()

    # Convert NPZ -> CSV
    print("[Convert] NPZ chunks -> CSV ...")
    npz_chunks_to_csv(out_dir, npz_prefix, out_csv)
    print(f"[Done] CSV written: {out_csv}")

    # Optional cleanup
    if CFG["output"]["delete_npz_after_csv"]:
        files = sorted(glob.glob(os.path.join(out_dir, f"{npz_prefix}_*.npz")))
        for fp in tqdm(files, desc="Deleting NPZ chunks", unit="file"):
            os.remove(fp)


if __name__ == "__main__":
    main()

# Two quick notes (so it behaves well)

# The auto-workspace assumes the moving mesh is “reasonably centered” near its local origin; if your peg mesh has a big offset, you’ll want to recenter it or expand bounds.

# Converting NPZ → CSV at the end will still take time (CSV is slow), but it won’t slow down the sampling loop.