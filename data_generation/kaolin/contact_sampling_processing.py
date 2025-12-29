#!/usr/bin/env python3
"""
Summarize sampling outputs (CSV or NPZ) from kaolin/contact sampling.

This script intentionally avoids argparse; instead it uses a single CONFIG dict.

Outputs:
- total poses
- contact poses (contact_valid)
- coarse/fine counts overall
- pose stats (min/max/mean/median/std) for:
    * all poses
    * contact poses (all contact)
    * contact+coarse
    * contact+fine
- 2x3 histogram figure (tx,ty,tz,rx,ry,rz):
    overlays contact-all vs contact-coarse vs contact-fine
- Optional 3 rotating 3D point-cloud GIFs (translation space) colored by rotation
  dimensions, adapted from data_visualization.py
"""

import os
import sys
from typing import Dict, Any, Tuple, Optional

import numpy as np

# Plot deps
import matplotlib
matplotlib.use("Agg")  # NO GUI, NO Qt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns


# ==========================
# CONFIG (edit here)
# ==========================

CONFIG: Dict[str, Any] = {
    # Input sampling output (.csv or .npz)
    "data_path": "/home/rp/abhay_ws/contact-manifold-state-estimation/data_generation/data/cylinder_simple/cylinder_simple_contact_poses.csv",

    # Histogram options
    "hist": {
        "bins": 60,
        "stat": "density",        # "count" | "probability" | "density"
        "common_norm": False,     # normalize each distribution independently
        "save_fig": True,
        "fig_name_suffix": "contact_hist_2x3.png",
        "fig_dpi": 200,
    },

    # Rotating GIF point-cloud visualization (translation points, colored by rx/ry/rz)
    "gif": {
        "enabled": True,
        # Which poses to visualize:
        #   "all" | "contact_all" | "contact_coarse" | "contact_fine"
        "subset": "contact_fine",

        # Downsample (random) before plotting/animating to keep GIF generation fast
        "max_points": 20_000,
        "random_seed": 42,

        # Output directory (default: alongside data_path)
        "out_dir": None,  # None => same directory as data_path
        "subdir": "visualization",

        # Animation settings (roughly matches data_visualization.py style)
        "frames": 120,
        "deg_per_frame": 3,   # azimuth increment per frame
        "elev": 20,
        "interval_ms": 100,
        "fps": 12,

        # Scatter styling
        "point_size": 1.0,
        "alpha": 0.6,
        "cmap": "viridis",

        # Base filename prefix
        "prefix": None,  # None => uses data file stem
    },
}

POSE_NAMES = ["tx", "ty", "tz", "rx_deg", "ry_deg", "rz_deg"]


# ==========================
# UTILS
# ==========================

def _fmt(x: float) -> str:
    if np.isnan(x):
        return "nan"
    if abs(x) >= 1e4 or (abs(x) > 0 and abs(x) < 1e-3):
        return f"{x:.6e}"
    return f"{x:.6f}"


def stats_1d(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return dict(min=np.nan, max=np.nan, mean=np.nan, median=np.nan, std=np.nan)
    return dict(
        min=float(np.min(x)),
        max=float(np.max(x)),
        mean=float(np.mean(x)),
        median=float(np.median(x)),
        std=float(np.std(x, ddof=0)),
    )


def print_pose_stats_block(title: str, X: np.ndarray) -> None:
    """
    X: (N,6) pose array
    """
    print(f"\n== {title} ==")
    if X.size == 0 or X.shape[0] == 0:
        print("  (no samples)")
        return

    for j, nm in enumerate(POSE_NAMES):
        s = stats_1d(X[:, j])
        print(
            f"  {nm:>6s}: "
            f"min={_fmt(s['min'])}  "
            f"max={_fmt(s['max'])}  "
            f"mean={_fmt(s['mean'])}  "
            f"median={_fmt(s['median'])}  "
            f"std={_fmt(s['std'])}"
        )


def _as_bool_mask(x: Any, n: int) -> np.ndarray:
    arr = np.asarray(x)
    if arr.shape[0] != n:
        raise ValueError(f"Mask length mismatch: {arr.shape[0]} vs {n}")
    if arr.dtype == np.bool_:
        return arr
    return arr.astype(np.int64) != 0


def _as_fidelity_str(fid: Any, n: int) -> np.ndarray:
    fid_arr = np.asarray(fid, dtype=object)
    if fid_arr.shape[0] != n:
        raise ValueError(f"Fidelity length mismatch: {fid_arr.shape[0]} vs {n}")
    return np.array([str(v) for v in fid_arr], dtype=object)


# ==========================
# LOADERS
# ==========================

def _normalize_vec3(x: Any, name: str) -> np.ndarray:
    """
    Accepts either:
      - numeric array of shape (N,3)
      - object array of shape (N,) where each entry is array-like (3,)
    Returns float64 array of shape (N,3)
    """
    x = np.asarray(x)

    # Case 1: already (N,3)
    if x.ndim == 2 and x.shape[1] == 3 and x.dtype != object:
        return x.astype(np.float64, copy=False)

    # Case 2: object array of vectors (N,)
    if x.ndim == 1:
        rows = []
        for i, v in enumerate(x):
            vv = np.asarray(v, dtype=np.float64).reshape(-1)
            if vv.size != 3:
                raise ValueError(f"{name}[{i}] has size {vv.size}, expected 3. Value type={type(v)}")
            rows.append(vv)
        return np.vstack(rows).astype(np.float64, copy=False)

    raise ValueError(f"Unsupported {name} shape/dtype: shape={x.shape}, dtype={x.dtype}")


def load_npz(path: str) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    keys = set(data.files)

    if "t" not in keys or "r_deg" not in keys:
        raise ValueError(f"NPZ must contain 't' and 'r_deg'. Found: {sorted(keys)}")

    t = _normalize_vec3(data["t"], "t")
    r = _normalize_vec3(data["r_deg"], "r_deg")

    if t.shape[0] != r.shape[0]:
        raise ValueError(f"t and r_deg length mismatch: {t.shape[0]} vs {r.shape[0]}")

    pose = np.concatenate([t, r], axis=1)  # (N,6)
    N = pose.shape[0]

    fidelity = data.get("fidelity", np.array(["unknown"] * N, dtype=object))
    contact_valid = data.get("contact_valid", np.zeros((N,), dtype=np.uint8))

    out = dict(
        N=N,
        pose=pose,
        fidelity=_as_fidelity_str(fidelity, N),
        contact_valid=_as_bool_mask(contact_valid, N),
    )

    for k in ["config_sdf", "min_separation", "max_penetration"]:
        if k in keys:
            out[k] = np.asarray(data[k], dtype=float).reshape(-1)

    return out


def load_csv(path: str) -> Dict[str, Any]:
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    arr = np.atleast_1d(arr)
    colnames = set(arr.dtype.names or [])

    required = {
        "tx", "ty", "tz",
        "rx_deg", "ry_deg", "rz_deg",
        "fidelity",
        "contact_valid",
    }
    missing = required - colnames
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}. Found: {sorted(colnames)}")

    pose = np.stack(
        [arr["tx"], arr["ty"], arr["tz"], arr["rx_deg"], arr["ry_deg"], arr["rz_deg"]],
        axis=1,
    ).astype(float)
    N = pose.shape[0]

    out = dict(
        N=N,
        pose=pose,
        fidelity=_as_fidelity_str(arr["fidelity"], N),
        contact_valid=_as_bool_mask(arr["contact_valid"], N),
    )

    for k in ["config_sdf", "min_separation", "max_penetration"]:
        if k in colnames:
            out[k] = np.asarray(arr[k], dtype=float).reshape(-1)

    return out


# ==========================
# ANALYSIS + PLOTTING
# ==========================

def count_fidelity(fid_s: np.ndarray) -> Tuple[int, int, int]:
    coarse = int(np.sum(fid_s == "coarse"))
    fine = int(np.sum(fid_s == "fine"))
    other = int(fid_s.size - coarse - fine)
    return coarse, fine, other


def make_contact_hist_figure(
    pose: np.ndarray,
    contact_all: np.ndarray,
    contact_coarse: np.ndarray,
    contact_fine: np.ndarray,
    out_path: str,
    bins: int,
    stat: str,
    common_norm: bool,
    save_fig: bool,
    dpi: int,
) -> None:
    """
    2x3 subplots: per-dimension histograms.
    Overlays contact-all vs contact-coarse vs contact-fine.
    """
    sns.set_theme(style="whitegrid")

    pose = np.asarray(pose)
    if pose.dtype == object:
        pose = pose.astype(np.float64)

    contact_all = np.asarray(contact_all, dtype=bool)
    contact_coarse = np.asarray(contact_coarse, dtype=bool)
    contact_fine = np.asarray(contact_fine, dtype=bool)

    def _to_1d_float_list(x) -> list[float]:
        x = np.asarray(x).reshape(-1)

        out: list[float] = []
        for v in x:
            vv = np.asarray(v)
            if vv.ndim > 0:
                vv = vv.reshape(-1)
                if vv.size == 0:
                    continue
                if vv.size != 1:
                    raise ValueError(
                        f"Non-scalar element in histogram data: element size={vv.size}, "
                        f"type={type(v)}, example={vv[:5]}"
                    )
                vv = vv[0]

            f = float(vv)
            if np.isfinite(f):
                out.append(f)
        return out

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axes = axes.flatten()

    groups = [
        ("contact_all", contact_all),
        ("contact_coarse", contact_coarse),
        ("contact_fine", contact_fine),
    ]

    for j, (ax, dim_name) in enumerate(zip(axes, POSE_NAMES)):
        base_list = _to_1d_float_list(pose[contact_all, j] if np.any(contact_all) else pose[:, j])

        if len(base_list) == 0:
            ax.set_title(dim_name)
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            continue

        base_np = np.asarray(base_list, dtype=np.float64)
        lo = float(np.min(base_np))
        hi = float(np.max(base_np))
        if not np.isfinite(lo) or not np.isfinite(hi):
            ax.set_title(dim_name)
            ax.text(0.5, 0.5, "non-finite data", ha="center", va="center", transform=ax.transAxes)
            continue
        if lo == hi:
            eps = 1e-9 if abs(lo) < 1 else 1e-9 * abs(lo)
            lo -= eps
            hi += eps

        for label, mask in groups:
            x_list = _to_1d_float_list(pose[mask, j])
            if len(x_list) == 0:
                continue

            sns.histplot(
                x=x_list,
                bins=bins,
                binrange=(lo, hi),
                stat=stat,
                common_norm=common_norm,
                element="step",
                fill=False,
                linewidth=1.5,
                ax=ax,
                label=label,
            )

        ax.set_title(dim_name)
        ax.set_xlabel(dim_name)
        ax.set_ylabel(stat)

    # legend once
    handles, labels = None, None
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        if len(h) > 0:
            handles, labels = h, l
            break
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=True)

    fig.suptitle("Contact pose distributions (all vs coarse vs fine)", y=1.02)

    if save_fig:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"\nSaved histogram figure: {out_path}")

    plt.close(fig)


def _equal_aspect_3d(ax, points_xyz: np.ndarray) -> None:
    """Match data_visualization.py: set equal aspect ratio for 3D scatter."""
    points = np.asarray(points_xyz, dtype=float)
    if points.size == 0:
        return

    max_range = np.array([
        points[:, 0].max() - points[:, 0].min(),
        points[:, 1].max() - points[:, 1].min(),
        points[:, 2].max() - points[:, 2].min(),
    ]).max() / 2.0

    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


def create_rotating_gif(
    points_xyz: np.ndarray,
    colors: np.ndarray,
    color_label: str,
    out_path: str,
    frames: int,
    deg_per_frame: float,
    elev: float,
    interval_ms: int,
    fps: int,
    point_size: float,
    alpha: float,
    cmap: str,
) -> None:
    """
    Adapted from data_visualization.py:
    - Rotating 3D scatter with colorbar
    - Saved as GIF using pillow writer
    """
    points = np.asarray(points_xyz, dtype=float)
    colors = np.asarray(colors, dtype=float).reshape(-1)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_xyz must be (N,3). Got {points.shape}")
    if colors.shape[0] != points.shape[0]:
        raise ValueError(f"colors length mismatch: {colors.shape[0]} vs {points.shape[0]}")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=colors,
        cmap=cmap,
        s=point_size,
        alpha=alpha,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Point cloud (translation) - colored by {color_label}")
    plt.colorbar(sc, ax=ax, label=color_label)

    _equal_aspect_3d(ax, points)

    def animate(frame_idx: int):
        ax.view_init(elev=elev, azim=frame_idx * deg_per_frame)
        return (sc,)

    anim = animation.FuncAnimation(
        fig, animate, frames=frames, interval=interval_ms, blit=False
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"Saving GIF: {out_path}")
    try:
        anim.save(out_path, writer="pillow", fps=fps)
        print(f"Saved GIF: {out_path}")
    except Exception as e:
        print(f"Failed to save GIF {out_path}: {e}")

    plt.close(fig)


def maybe_make_rotation_gifs(
    pose: np.ndarray,
    subset_mask: np.ndarray,
    out_dir: str,
    prefix: str,
    cfg: Dict[str, Any],
) -> None:
    """
    Create 3 GIFs:
      - colored by rx_deg
      - colored by ry_deg
      - colored by rz_deg

    Uses translation coordinates (tx,ty,tz) as the 3D point locations.
    """
    points = pose[subset_mask, :3]
    rots = pose[subset_mask, 3:6]

    if points.shape[0] == 0:
        print("\n[gif] No points in the requested subset. Skipping GIFs.")
        return

    # Downsample (random) for speed
    max_points = int(cfg["max_points"])
    if points.shape[0] > max_points:
        rng = np.random.default_rng(int(cfg["random_seed"]))
        idx = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[idx]
        rots = rots[idx]

    vis_dir = os.path.join(out_dir, str(cfg["subdir"]))
    os.makedirs(vis_dir, exist_ok=True)

    frames = int(cfg["frames"])
    deg_per_frame = float(cfg["deg_per_frame"])
    elev = float(cfg["elev"])
    interval_ms = int(cfg["interval_ms"])
    fps = int(cfg["fps"])
    point_size = float(cfg["point_size"])
    alpha = float(cfg["alpha"])
    cmap = str(cfg["cmap"])

    create_rotating_gif(
        points_xyz=points,
        colors=rots[:, 0],
        color_label="rx_deg",
        out_path=os.path.join(vis_dir, f"{prefix}_pointcloud_rx_deg.gif"),
        frames=frames,
        deg_per_frame=deg_per_frame,
        elev=elev,
        interval_ms=interval_ms,
        fps=fps,
        point_size=point_size,
        alpha=alpha,
        cmap=cmap,
    )
    create_rotating_gif(
        points_xyz=points,
        colors=rots[:, 1],
        color_label="ry_deg",
        out_path=os.path.join(vis_dir, f"{prefix}_pointcloud_ry_deg.gif"),
        frames=frames,
        deg_per_frame=deg_per_frame,
        elev=elev,
        interval_ms=interval_ms,
        fps=fps,
        point_size=point_size,
        alpha=alpha,
        cmap=cmap,
    )
    create_rotating_gif(
        points_xyz=points,
        colors=rots[:, 2],
        color_label="rz_deg",
        out_path=os.path.join(vis_dir, f"{prefix}_pointcloud_rz_deg.gif"),
        frames=frames,
        deg_per_frame=deg_per_frame,
        elev=elev,
        interval_ms=interval_ms,
        fps=fps,
        point_size=point_size,
        alpha=alpha,
        cmap=cmap,
    )


def main() -> None:
    data_path = str(CONFIG["data_path"])
    if not os.path.exists(data_path):
        print(f"ERROR: file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    ext = os.path.splitext(data_path)[1].lower()
    if ext == ".npz":
        data = load_npz(data_path)
    elif ext == ".csv":
        data = load_csv(data_path)
    else:
        raise ValueError("CONFIG['data_path'] must end in .csv or .npz")

    N = int(data["N"])
    pose = np.asarray(data["pose"], dtype=float)
    assert pose.ndim == 2 and pose.shape[1] == 6 and pose.dtype != object, (pose.shape, pose.dtype)
    fid_s = np.asarray(data["fidelity"], dtype=object)
    contact_valid = np.asarray(data["contact_valid"], dtype=bool)

    # fidelity masks
    coarse_mask_all = (fid_s == "coarse")
    fine_mask_all = (fid_s == "fine")

    # contact masks
    contact_all = contact_valid
    contact_coarse = contact_valid & coarse_mask_all
    contact_fine = contact_valid & fine_mask_all

    # counts
    n_contact = int(np.sum(contact_all))
    n_contact_coarse = int(np.sum(contact_coarse))
    n_contact_fine = int(np.sum(contact_fine))
    frac_contact = (n_contact / N) if N > 0 else np.nan
    frac_contact_coarse = (n_contact_coarse / N) if N > 0 else np.nan
    frac_contact_fine = (n_contact_fine / N) if N > 0 else np.nan

    coarse, fine, other = count_fidelity(fid_s)
    frac_coarse = (coarse / N) if N > 0 else np.nan
    frac_fine = (fine / N) if N > 0 else np.nan

    # Contact rates within fidelity groups
    coarse_contact_rate = (n_contact_coarse / coarse) if coarse > 0 else np.nan
    fine_contact_rate = (n_contact_fine / fine) if fine > 0 else np.nan

    print("\n================ SUMMARY ================")
    print(f"File: {data_path}")
    print(f"Total poses sampled: {N}")

    print("\n-- Fidelity (overall) --")
    print(f"  coarse: {coarse} ({frac_coarse:.6f})")
    print(f"  fine:   {fine} ({frac_fine:.6f})")
    print(f"  other:  {other}")

    print("\n-- Contact (contact_valid) --")
    print(f"  contact_all:   {n_contact} ({frac_contact:.6f})")
    print(f"  contact_coarse:{n_contact_coarse} ({frac_contact_coarse:.6f})  | P(contact|coarse)={coarse_contact_rate:.6f}")
    print(f"  contact_fine:  {n_contact_fine} ({frac_contact_fine:.6f})    | P(contact|fine)={fine_contact_rate:.6f}")

    # pose stats
    print_pose_stats_block("Pose stats (ALL poses)", pose)
    print_pose_stats_block("Pose stats (CONTACT poses: all)", pose[contact_all])
    print_pose_stats_block("Pose stats (CONTACT poses: coarse)", pose[contact_coarse])
    print_pose_stats_block("Pose stats (CONTACT poses: fine)", pose[contact_fine])

    # Optional numeric metrics
    for k in ["config_sdf", "min_separation", "max_penetration"]:
        if k in data:
            vals = np.asarray(data[k], dtype=float).reshape(-1)
            print(f"\n== {k} stats ==")
            for name, mask in [
                ("ALL poses", np.ones((N,), dtype=bool)),
                ("CONTACT all", contact_all),
                ("CONTACT coarse", contact_coarse),
                ("CONTACT fine", contact_fine),
            ]:
                s = stats_1d(vals[mask])
                print(
                    f"  {name:>13s}: "
                    f"min={_fmt(s['min'])}  max={_fmt(s['max'])}  mean={_fmt(s['mean'])}  "
                    f"median={_fmt(s['median'])}  std={_fmt(s['std'])}"
                )

    # Histogram figure (contact-only overlays)
    hist_cfg = CONFIG["hist"]
    out_dir = os.path.dirname(os.path.abspath(data_path))
    base = os.path.splitext(os.path.basename(data_path))[0]
    fig_path = os.path.join(out_dir, f"{base}_{hist_cfg['fig_name_suffix']}")

    make_contact_hist_figure(
        pose=pose,
        contact_all=contact_all,
        contact_coarse=contact_coarse,
        contact_fine=contact_fine,
        out_path=fig_path,
        bins=int(hist_cfg["bins"]),
        stat=str(hist_cfg["stat"]),
        common_norm=bool(hist_cfg["common_norm"]),
        save_fig=bool(hist_cfg["save_fig"]),
        dpi=int(hist_cfg["fig_dpi"]),
    )

    # Rotating GIF point-cloud visualization (optional)
    gif_cfg = CONFIG.get("gif", {})
    if bool(gif_cfg.get("enabled", False)):
        subset_name = str(gif_cfg.get("subset", "contact_all"))
        subset_map = {
            "all": np.ones((N,), dtype=bool),
            "contact_all": contact_all,
            "contact_coarse": contact_coarse,
            "contact_fine": contact_fine,
        }
        subset_mask = subset_map.get(subset_name)
        if subset_mask is None:
            raise ValueError(f"Unknown gif.subset='{subset_name}'. Choose from {sorted(subset_map.keys())}")

        gif_out_dir = gif_cfg.get("out_dir", None) or out_dir
        prefix = gif_cfg.get("prefix", None) or base

        maybe_make_rotation_gifs(
            pose=pose,
            subset_mask=subset_mask,
            out_dir=str(gif_out_dir),
            prefix=str(prefix),
            cfg=gif_cfg,
        )

    print("\n========================================")
    print("Done.")


if __name__ == "__main__":
    main()
