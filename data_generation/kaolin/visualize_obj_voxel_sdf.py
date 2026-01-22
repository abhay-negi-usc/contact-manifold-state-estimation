#!/usr/bin/env python3
"""
Generate a voxel SDF for a single OBJ (same approach as your voxel SDF script),
then create a GIF that sweeps a cross-section plane along world z from +z to -z.

GIF layout (side-by-side):
  Left:  SDF slice in x-y at current z (colored), optional contour at SDF=0.
  Right: Top-down geometry projection in x-y + a z-height gauge with a line showing current slice z.

Outputs are written next to the mesh:
  <mesh_dir>/sdf_voxel/
    - sdf_volume.npz
    - sdf_meta.json
    - sdf_z_sweep.gif
"""

import os
import json
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import imageio.v2 as imageio

from kaolin.io.obj import import_mesh
from kaolin.ops.conversions import trianglemeshes_to_voxelgrids
from kaolin.ops.mesh import sample_points as kal_sample_points


# ----------------------------
# CONFIG
# ----------------------------
CFG = {
    "input": {
        "obj_path": "/home/omey/abhay_ws/contact-manifold-state-estimation/data_generation/assets/meshes/cylinder_keyway02/cylinder_keyway02_peg.obj",   # <-- set this
    },

    "device": "cuda",          # "cuda" or "cpu"
    "dtype": "float32",
    "seed": 0,

    "sdf_grid": {
        "resolution": 2**8,
        "surface_samples": 2**16,
        "cdist_chunk": 2000,
        "workspace_margin": 0.02,
    },

    "viz": {
        "gif_name": "sdf_z_sweep.gif",
        "gif_fps": 24,
        "frame_repeat": 2,   # repeat each frame N times (slower)
        "boomerang": True,   
        "boomerang_hold": 4,    

        # If None, uses all z-slices; otherwise samples this many evenly spaced slices.
        "num_slices": 120,

        # SDF visualization settings
        "max_abs_sdf": None,          # clamp for colors, e.g. 0.02; None => robust auto
        "show_zero_contour": True,    # overlay SDF=0 contour on the slice
        "contour_linewidth": 1.0,

        # Geometry projection settings (right panel)
        "geom_scatter_stride": 1,     # downsample vertices for speed (e.g., 2,4,8)
        "geom_point_size": 0.15,

        # Figure
        "figsize": (10.5, 4.8),
        "dpi": 140,
    },

    "output": {
        "subdir": "sdf_voxel",
        "npz_name": "sdf_volume.npz",
        "meta_name": "sdf_meta.json",
    }
}

CFG["viz"]["gif_name"] = f"{CFG['input']['obj_path'].split('/')[-1].split('.')[0]}_sdf_z_sweep.gif"


# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int):
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

def voxelize_occupancy(verts: torch.Tensor,
                       faces: torch.Tensor,
                       min_xyz: torch.Tensor,
                       max_xyz: torch.Tensor,
                       res: int) -> torch.Tensor:
    """
    Returns occupancy volume (1,1,res,res,res) float in {0,1} where 1 means "inside".
    Kaolin expects vertices normalized to [-1,1].
    """
    v_norm = (verts - min_xyz) / (max_xyz - min_xyz + 1e-12)
    v_norm = v_norm * 2.0 - 1.0
    v_norm = v_norm.unsqueeze(0).contiguous()  # (1,V,3)

    f = faces.to(dtype=torch.long).contiguous()
    occ = trianglemeshes_to_voxelgrids(v_norm, f, res)  # (1,res,res,res)
    return occ.float().unsqueeze(1)  # (1,1,res,res,res)

def make_voxel_centers(min_xyz: torch.Tensor, max_xyz: torch.Tensor, res: int, device, dtype):
    spans = (max_xyz - min_xyz)
    vs = spans / res
    xs = min_xyz[0] + (torch.arange(res, device=device, dtype=dtype) + 0.5) * vs[0]
    ys = min_xyz[1] + (torch.arange(res, device=device, dtype=dtype) + 0.5) * vs[1]
    zs = min_xyz[2] + (torch.arange(res, device=device, dtype=dtype) + 0.5) * vs[2]
    X, Y, Z = torch.meshgrid(xs, ys, zs, indexing="ij")
    centers = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=-1)  # (res^3,3)
    return centers, xs, ys, zs

def build_sdf_grid(verts: torch.Tensor,
                   faces: torch.Tensor,
                   min_xyz: torch.Tensor,
                   max_xyz: torch.Tensor,
                   res: int,
                   surface_samples: int,
                   cdist_chunk: int):
    """
    Build signed distance volume:
      dist(x) = min_{p on surface} ||x-p||
      sdf = dist * (1 - 2*occ)   (occ=1 => negative)

    Returns sdf_vol, occ_vol in layout (1,1,D,H,W) with D=z,H=y,W=x.
    """
    device = verts.device
    dtype = verts.dtype

    # Occupancy (x,y,z) in kaolin -> we'll permute later
    occ_vol = voxelize_occupancy(verts, faces, min_xyz, max_xyz, res).to(device=device)

    # Surface sample points
    f = faces.to(dtype=torch.long).contiguous()
    pts, _ = kal_sample_points(verts.unsqueeze(0), f, surface_samples)
    surf_pts = pts.squeeze(0).contiguous()  # (S,3)

    # Voxel centers in world coords
    centers, xs, ys, zs = make_voxel_centers(min_xyz, max_xyz, res, device, dtype)
    Vox = centers.shape[0]

    # Chunked cdist
    dmin = torch.empty((Vox,), device=device, dtype=dtype)
    for i in tqdm(range(0, Vox, cdist_chunk), desc="Building distance field (cdist)", unit="chunk"):
        c = centers[i:i + cdist_chunk]
        d = torch.cdist(c, surf_pts)
        dmin[i:i + cdist_chunk] = d.min(dim=1).values

    # Reshape to (1,1,x,y,z) then permute to (1,1,z,y,x)
    dist_vol = dmin.view(1, 1, res, res, res).permute(0, 1, 4, 3, 2).contiguous()
    occ_vol = occ_vol.permute(0, 1, 4, 3, 2).contiguous()

    sdf_vol = dist_vol * (1.0 - 2.0 * occ_vol)
    return sdf_vol, occ_vol, xs, ys, zs


# ----------------------------
# Visualization: z-sweep GIF
# ----------------------------
def render_z_sweep_gif(
    out_path: str,
    sdf_vol: np.ndarray,         # (1,1,D,H,W) float32
    xs: np.ndarray,              # (W,) world x centers
    ys: np.ndarray,              # (H,) world y centers
    zs: np.ndarray,              # (D,) world z centers
    min_xyz: np.ndarray,         # (3,) workspace minimum
    max_xyz: np.ndarray,         # (3,) workspace maximum
    verts_xyz: np.ndarray,       # (V,3) vertices (world)
    cfg_viz: dict,
):
    assert sdf_vol.ndim == 5
    D = sdf_vol.shape[2]
    H = sdf_vol.shape[3]
    W = sdf_vol.shape[4]

    # z sweep: +z -> -z
    num_slices = cfg_viz.get("num_slices", None)
    if (num_slices is None) or (num_slices >= D):
        forward = list(range(D - 1, -1, -1))
    else:
        idx = np.linspace(D - 1, 0, int(num_slices)).round().astype(int)
        # de-duplicate in case rounding repeats indices
        forward = idx.tolist()
        forward = [forward[0]] + [forward[i] for i in range(1, len(forward)) if forward[i] != forward[i-1]]

    # boomerang: forward then backward (without repeating endpoints)
    if bool(cfg_viz.get("boomerang", True)) and len(forward) > 1:
        backward = forward[-2:0:-1]  # excludes last and first
        z_indices = forward + backward
    else:
        z_indices = forward

    # optional holds at ends (reduces perceived speed near turnarounds)
    hold = int(cfg_viz.get("boomerang_hold", 0))
    if hold > 0 and len(z_indices) > 2:
        # hold at start and at the first turnaround (end of forward)
        start_idx = z_indices[0]
        end_forward_idx = forward[-1]
        z_indices = ([start_idx] * hold) + z_indices + ([end_forward_idx] * hold)


    # Color clamp (stable)
    max_abs = cfg_viz.get("max_abs_sdf", None)
    if max_abs is None:
        sample = sdf_vol[0, 0, ::max(1, D // 32), ::max(1, H // 32), ::max(1, W // 32)].ravel()
        q = np.quantile(np.abs(sample), 0.98)
        max_abs = float(q) if q > 0 else float(np.max(np.abs(sample)) + 1e-9)
    vmin, vmax = -max_abs, max_abs

    # World extents (fixed) - use actual workspace bounds
    x_min, x_max = float(min_xyz[0]), float(max_xyz[0])
    y_min, y_max = float(min_xyz[1]), float(max_xyz[1])
    z_min, z_max = float(min_xyz[2]), float(max_xyz[2])

    # Geometry projections (fixed / downsampled)
    stride = max(1, int(cfg_viz.get("geom_scatter_stride", 1)))
    V = verts_xyz[::stride]
    Vxy = V[:, [0, 1]]
    Vxz = V[:, [0, 2]]

    pt_size = float(cfg_viz.get("geom_point_size", 0.15))
    show_contour = bool(cfg_viz.get("show_zero_contour", True))
    contour_lw = float(cfg_viz.get("contour_linewidth", 1.0))

    figsize = tuple(cfg_viz.get("figsize", (12.6, 4.8)))
    dpi = int(cfg_viz.get("dpi", 140))
    fps = int(cfg_viz.get("gif_fps", 24))

    images = []

    # Precompute meshgrids for contour (fixed, avoids per-frame allocations)
    Xg, Yg = np.meshgrid(xs, ys, indexing="xy")  # (H,W)

    for zi in tqdm(z_indices, desc="Rendering z-sweep frames", unit="frame"):
        z_world = float(zs[zi])

        # Slice SDF (H,W)
        S = sdf_vol[0, 0, zi, :, :]
        S_clip = np.clip(S, vmin, vmax)

        # --- Fixed, deterministic figure layout (NO tight_layout) ---
        fig = plt.figure(figsize=figsize, dpi=dpi)

        # 3 panels with fixed width ratios
        gs = fig.add_gridspec(
            1, 3,
            width_ratios=[1.25, 1.0, 1.0],
            left=0.05, right=0.94, bottom=0.10, top=0.90, wspace=0.22
        )
        axL  = fig.add_subplot(gs[0, 0])  # SDF slice
        axXY = fig.add_subplot(gs[0, 1])  # XY projection
        axXZ = fig.add_subplot(gs[0, 2])  # XZ projection (right-most)

        # Dedicated colorbar axis with fixed position (prevents resizing jitter)
        cax = fig.add_axes([0.945, 0.17, 0.012, 0.68])  # [left, bottom, width, height]

        # ---- Left: SDF slice in x-y ----
        im = axL.imshow(
            S_clip,
            origin="lower",
            extent=(x_min, x_max, y_min, y_max),
            vmin=vmin, vmax=vmax,
            cmap="viridis",
            interpolation="nearest",
            aspect="equal",
        )

        # Fixed-width title string (prevents tiny layout shifts)
        axL.set_title(f"SDF slice  z={z_world:+.6f}")
        axL.set_xlabel("x")
        axL.set_ylabel("y")
        axL.set_xlim(x_min, x_max)
        axL.set_ylim(y_min, y_max)

        if show_contour:
            axL.contour(Xg, Yg, S, levels=[0.0], colors="white", linewidths=contour_lw)

        cb = fig.colorbar(im, cax=cax)
        cb.set_label("Signed distance")

        # ---- Middle: geometry projection in x-y ----
        axXY.scatter(Vxy[:, 0], Vxy[:, 1], s=pt_size, alpha=0.35)
        axXY.set_title("Geometry (x–y)")
        axXY.set_xlabel("x")
        axXY.set_ylabel("y")
        axXY.set_aspect("equal", adjustable="box")
        axXY.set_xlim(x_min, x_max)
        axXY.set_ylim(y_min, y_max)

        # ---- Right-most: geometry side view in x-z + slice height line ----
        axXZ.scatter(Vxz[:, 0], Vxz[:, 1], s=pt_size, alpha=0.35)
        axXZ.axhline(z_world, color="red", linewidth=2.5)
        axXZ.set_title("Geometry (x–z) + slice height")
        axXZ.set_xlabel("x")
        axXZ.set_ylabel("z")
        axXZ.set_aspect("auto")
        axXZ.set_xlim(x_min, x_max)
        axXZ.set_ylim(z_min, z_max)

        # Render frame to RGB
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        images.append(img)
        repeat = int(cfg_viz.get("frame_repeat", 1))
        for _ in range(max(1, repeat)):
            images.append(img)

        plt.close(fig)

    imageio.mimsave(out_path, images, fps=fps)


# ----------------------------
# Main
# ----------------------------
def main():
    set_seed(int(CFG["seed"]))
    device = torch.device(CFG["device"])
    dtype = to_dtype(CFG["dtype"])

    obj_path = os.path.expanduser(CFG["input"]["obj_path"])
    if not obj_path or not os.path.exists(obj_path):
        raise FileNotFoundError(f"Set CFG['input']['obj_path'] to a valid .obj path. Got: {obj_path}")

    print(f"[Load] {obj_path}")
    verts, faces = load_obj_mesh(obj_path, device=device, dtype=dtype)

    # Workspace
    res = int(CFG["sdf_grid"]["resolution"])
    margin = float(CFG["sdf_grid"]["workspace_margin"])
    vmin, vmax = mesh_aabb(verts)
    min_xyz = vmin - margin
    max_xyz = vmax + margin

    print("[Workspace]")
    print("  min:", min_xyz.detach().cpu().numpy().tolist())
    print("  max:", max_xyz.detach().cpu().numpy().tolist())

    # Build SDF
    surface_samples = int(CFG["sdf_grid"]["surface_samples"])
    cdist_chunk = int(CFG["sdf_grid"]["cdist_chunk"])
    print(f"[SDF] building grid res={res}, surface_samples={surface_samples}")
    sdf_vol_t, occ_vol_t, xs_t, ys_t, zs_t = build_sdf_grid(
        verts, faces, min_xyz, max_xyz, res, surface_samples, cdist_chunk
    )

    # Output dir
    mesh_dir = os.path.dirname(os.path.abspath(obj_path))
    out_dir = os.path.join(mesh_dir, CFG["output"]["subdir"])
    os.makedirs(out_dir, exist_ok=True)

    npz_path = os.path.join(out_dir, CFG["output"]["npz_name"])
    meta_path = os.path.join(out_dir, CFG["output"]["meta_name"])
    gif_path = os.path.join(out_dir, CFG["viz"]["gif_name"])

    meta = {
        "obj_path": os.path.abspath(obj_path),
        "resolution": res,
        "min_xyz": min_xyz.detach().cpu().numpy().tolist(),
        "max_xyz": max_xyz.detach().cpu().numpy().tolist(),
        "layout": "(1,1,D,H,W) with D=z,H=y,W=x",
        "method": "kaolin occupancy + sampled surface points + chunked cdist; signed by occupancy",
        "cfg": CFG,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    np.savez_compressed(
        npz_path,
        sdf_vol=sdf_vol_t.detach().cpu().numpy().astype(np.float32),
        occ_vol=occ_vol_t.detach().cpu().numpy().astype(np.uint8),
        min_xyz=min_xyz.detach().cpu().numpy().astype(np.float32),
        max_xyz=max_xyz.detach().cpu().numpy().astype(np.float32),
        res=np.int32(res),
        xs=xs_t.detach().cpu().numpy().astype(np.float32),
        ys=ys_t.detach().cpu().numpy().astype(np.float32),
        zs=zs_t.detach().cpu().numpy().astype(np.float32),
    )
    print(f"[Write] {npz_path}")
    print(f"[Write] {meta_path}")

    # Prepare arrays for viz
    sdf_vol = sdf_vol_t.detach().cpu().numpy()  # (1,1,D,H,W)
    xs = xs_t.detach().cpu().numpy()
    ys = ys_t.detach().cpu().numpy()
    zs = zs_t.detach().cpu().numpy()

    verts_np = verts.detach().cpu().numpy()
    min_xyz_np = min_xyz.detach().cpu().numpy()
    max_xyz_np = max_xyz.detach().cpu().numpy()
    render_z_sweep_gif(
        out_path=gif_path,
        sdf_vol=sdf_vol,
        xs=xs, ys=ys, zs=zs,
        min_xyz=min_xyz_np,
        max_xyz=max_xyz_np,
        verts_xyz=verts_np,
        cfg_viz=CFG["viz"],
    )

    print(f"[Done] GIF written: {gif_path}")


if __name__ == "__main__":
    main()
