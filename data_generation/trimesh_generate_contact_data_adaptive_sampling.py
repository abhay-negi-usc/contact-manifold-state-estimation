#!/usr/bin/env python3
"""
Adaptive Contact Data Generation using Per-Slice AABB Sampling

This script implements an adaptive sampling strategy where:
1. For each z-value (starting from 0), we iterate through all other dimensions
2. For each dimension (x, y, a, b, c), we find the min/max contact bounds while holding other dims at 0
3. We then combinatorially sample within those bounds at the current z-level
4. This is repeated for each z-slice

This sampling method is known as "Axis-Aligned Bounding Box (AABB) Sampling" or 
"Per-Slice Adaptive Range Sampling" - a form of stratified sampling that adapts
the sampling bounds based on the contact manifold at each level.

The advantage is that it focuses sampling effort on regions where contact actually occurs,
potentially providing much higher contact efficiency than uniform grid sampling.
"""
import csv
import sys
import math
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import trimesh

# ====================== CONFIG ======================
config = {
    # Geometry identifier for output filename
    "geometry": "extrusion",
    
    # Fixed paths
    "mesh1": "/home/rp/abhay_ws/contact-manifold-state-estimation/data_generation/assets/meshes/extrusion_hole/extrusion_hole.obj",
    "mesh2": "/home/rp/abhay_ws/contact-manifold-state-estimation/data_generation/assets/meshes/extrusion_peg/extrusion_peg.obj",

    # Pose of mesh1 (static)
    "mesh1_T": np.eye(4).tolist(),   # 4x4 row-major

    # mesh2 pose is sampled across 6-DoF (xyz + abc) using adaptive per-slice sampling
    "sampling": {
        # Units: meters for xyz, radians for abc unless 'degrees'=True
        "xyz": {
            "x": {"min": -0.005, "max": 0.005, "step": 0.0005},
            "y": {"min": -0.005, "max": 0.005, "step": 0.0005},
            "z": {"min": 0.0, "max": 0.0, "step": 0.025},  # z is the primary axis
        },
        "abc": {
            # These are degrees by default (set degrees=False to use radians)
            # Follows scipy convention: R.from_euler('xyz', [c, b, a], degrees=True)
            "a": {"min": -5.0, "max": 5.0, "step": 0.5},
            "b": {"min": -5.0, "max": 5.0, "step": 0.5},
            "c": {"min": -5.0, "max": 5.0, "step": 0.5},
        },
        "degrees": True,
        "inclusive": True,  # include the max endpoint if it lands on the grid
        "adaptive": True,    # enable adaptive per-slice sampling
        "search_margin": 0.1,  # safety margin when searching for contact bounds
        "max_penetration_depth": 0.0005,  # maximum allowed penetration depth in meters
    },

    # Parallel execution (can be used for both traditional and adaptive sampling)
    "parallel": {
        "enabled": True,
        "workers": 20,       # 0 or None => use mp.cpu_count()
        "chunksize": 16,     # tune for throughput; larger = fewer IPC calls
        "adaptive_parallel": True,  # enable parallelization for adaptive sampling
        "z_slice_parallel": True,   # parallelize across z-slices
        "axis_parallel": False,     # parallelize axis bound finding within each z-slice (can cause oversubscription)
        "pose_parallel": True,      # parallelize pose evaluation after adaptive bound finding
    },

    # Output
    "save": {
        "csv_path": "./data_generation/{geometry}_pose_sweep_contacts_adaptive.csv",
        "max_contacts_to_print": 0  # prints none; raise for debug
    },
}
# ====================================================

# ------------------------ Utils ------------------------

def abc_to_matrix(a, b, c, degrees=False):
    """
    Convert angles a, b, c to rotation matrix using scipy convention:
    R.from_euler('xyz', [c, b, a], degrees=True).as_matrix()
    """
    if degrees:
        a, b, c = np.deg2rad([a, b, c])
    
    # Rotation matrices for x, y, z rotations
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cc, sc = np.cos(c), np.sin(c)
    
    # X rotation (angle c)
    Rx = np.array([[1,  0,   0],
                   [0, cc, -sc],
                   [0, sc,  cc]])
    
    # Y rotation (angle b)
    Ry = np.array([[ cb, 0, sb],
                   [  0, 1,  0],
                   [-sb, 0, cb]])
    
    # Z rotation (angle a)
    Rz = np.array([[ca, -sa, 0],
                   [sa,  ca, 0],
                   [ 0,   0, 1]])
    
    # Apply in order: Rz * Ry * Rx (equivalent to 'xyz' order with [c,b,a])
    return Rz @ Ry @ Rx


def make_T_from_xyz_abc(x, y, z, a, b, c, degrees=False):
    T = np.eye(4)
    T[:3, :3] = abc_to_matrix(a, b, c, degrees=degrees)
    T[:3, 3] = [x, y, z]
    return T


def ensure_fcl():
    try:
        import fcl  # noqa: F401
    except ImportError:
        raise ImportError(
            "python-fcl (and FCL) not found. Install with `pip install python-fcl` "
            "and ensure the FCL library is present on your system."
        )


def _clean_mesh(m: trimesh.Trimesh) -> trimesh.Trimesh:
    """Modern, non-deprecated cleanup to avoid IO/tessellation artifacts."""
    m = m.copy()
    if hasattr(m, "nondegenerate_faces"):
        m.update_faces(m.nondegenerate_faces())
    if hasattr(m, "unique_faces"):
        m.update_faces(m.unique_faces())
    m.remove_unreferenced_vertices()
    m.merge_vertices()
    m.remove_infinite_values()
    return m


def build_manager_triangle_mesh(name_prefix, mesh, T):
    """Always use triangle mesh (no convex decomposition)."""
    from trimesh.collision import CollisionManager
    cm = CollisionManager()
    cm.add_object(name_prefix, mesh, transform=T)
    return cm


def contact_points_localized(contact_blob, T1, T2):
    """
    Extract contact depths and approximate contact points in each mesh's local frame.
    Robust to trimesh/python-fcl version differences.
    Returns (depths:list[float], contacts:list[dict]).
    """
    import numpy as np

    def normalize_contact_records(blob):
        if isinstance(blob, (list, tuple)):
            for item in blob:
                yield from normalize_contact_records(item)
            return
        if blob is None:
            return

        rec = {}
        for key in ("depth", "penetration_depth", "points", "point", "normal"):
            if hasattr(blob, key):
                rec[key] = getattr(blob, key)
        if isinstance(blob, dict):
            for key in ("depth", "penetration_depth", "points", "point", "normal"):
                if key in blob and key not in rec:
                    rec[key] = blob[key]

        depth = rec.get("depth", rec.get("penetration_depth", None))
        # vectorized container
        if isinstance(depth, (list, tuple, np.ndarray)):
            depth_arr = np.atleast_1d(np.asarray(depth, float))
            P = rec.get("points", None)
            pt = rec.get("point", None)
            N = rec.get("normal", None)
            P = None if P is None else np.asarray(P, float)
            pt = None if pt is None else np.asarray(pt, float)
            N = None if N is None else np.asarray(N, float)
            for i, d in enumerate(depth_arr):
                out = {"depth": float(d)}
                if P is not None and P.ndim == 3 and P.shape[1] == 2:
                    out["points"] = P[i]
                else:
                    if pt is not None:
                        out["point"] = pt[i] if pt.ndim > 1 else pt
                    if N is not None:
                        out["normal"] = N[i] if N.ndim > 1 else N
                yield out
            return

        if depth is not None:
            try:
                rec["depth"] = float(depth)
            except Exception:
                pass
        yield rec

    def to_local(T_world_obj, p_world):
        p_h = np.ones(4, float); p_h[:3] = p_world
        return (np.linalg.inv(T_world_obj) @ p_h)[:3]

    depths, results = [], []
    for rec in normalize_contact_records(contact_blob):
        d = rec.get("depth", rec.get("penetration_depth", None))
        if d is None:
            continue
        d = float(d)
        P = rec.get("points", None)
        if P is not None:
            P = np.asarray(P, float)
            if P.shape == (2, 3):
                pw1, pw2 = P[0], P[1]
            else:
                pw1 = np.asarray(rec.get("point", [0, 0, 0]), float).reshape(3)
                n = np.asarray(rec.get("normal", [0, 0, 1.0]), float).reshape(3)
                pw2 = pw1 + n * d
        else:
            pw1 = np.asarray(rec.get("point", [0, 0, 0]), float).reshape(3)
            n = np.asarray(rec.get("normal", [0, 0, 1.0]), float).reshape(3)
            pw2 = pw1 + n * d

        p1_local = to_local(T1, pw1)
        p2_local = to_local(T2, pw2)
        depths.append(d)
        results.append({"depth": d,
                        "p1_local": p1_local.tolist(),
                        "p2_local": p2_local.tolist()})
    return depths, results


def build_grid(axis_cfg, inclusive=True):
    """
    Build 1D grid given {"min":..,"max":..,"step":..}
    inclusive=True includes the 'max' if it's on the step grid (within eps).
    """
    lo, hi, st = axis_cfg["min"], axis_cfg["max"], axis_cfg["step"]
    if st <= 0:
        raise ValueError("step must be > 0")
    # number of steps; protect against floating point drift
    n = int(math.floor((hi - lo) / st + 1e-12)) + 1
    arr = lo + np.arange(n) * st
    if inclusive and (abs(arr[-1] - hi) > 1e-9):
        # try to append hi if it's almost on-grid
        if abs(((hi - lo) / st) - round((hi - lo) / st)) < 1e-6:
            arr = np.append(arr, hi)
    # clamp last value
    if inclusive and abs(arr[-1] - hi) < 1e-12:
        arr[-1] = hi
    return arr


def find_contact_bounds_for_axis(axis_name, z_val, cfg):
    """
    Find the min/max values for a given axis where contact occurs at a fixed z value.
    Only considers contacts up to a maximum penetration depth threshold.
    Searches bidirectionally from zero to find both bounds.
    
    Args:
        axis_name: one of 'x', 'y', 'a', 'b', 'c'
        z_val: fixed z value
        cfg: configuration dictionary
    
    Returns:
        (min_val, max_val): tuple of bounds where contact occurs, or (None, None) if no contact
    """
    sampling_cfg = cfg["sampling"]
    deg = sampling_cfg.get("degrees", False)
    margin = sampling_cfg.get("search_margin", 0.1)
    max_penetration = sampling_cfg.get("max_penetration_depth", 0.001)  # default 1mm
    
    # Get the axis configuration
    if axis_name in ['x', 'y']:
        axis_cfg = sampling_cfg["xyz"][axis_name]
    else:
        axis_cfg = sampling_cfg["abc"][axis_name]
    
    # Build search parameters
    search_step = axis_cfg["step"] / 4  # Use finer resolution for searching
    search_min = axis_cfg["min"] - margin * abs(axis_cfg["max"] - axis_cfg["min"])
    search_max = axis_cfg["max"] + margin * abs(axis_cfg["max"] - axis_cfg["min"])
    
    def evaluate_pose_at_val(val):
        """Helper function to create and evaluate pose at a given axis value"""
        if axis_name == 'x':
            pose = (val, 0, z_val, 0, 0, 0)
        elif axis_name == 'y':
            pose = (0, val, z_val, 0, 0, 0)
        elif axis_name == 'a':
            angle_val = np.deg2rad(val) if deg else val
            pose = (0, 0, z_val, angle_val, 0, 0)
        elif axis_name == 'b':
            angle_val = np.deg2rad(val) if deg else val
            pose = (0, 0, z_val, 0, angle_val, 0)
        elif axis_name == 'c':
            angle_val = np.deg2rad(val) if deg else val
            pose = (0, 0, z_val, 0, 0, angle_val)
        
        result = _eval_pose(pose)
        is_contact = result[6] > 0.5
        penetration_depth = result[7] if is_contact else 0.0
        return is_contact, penetration_depth
    
    # Search for bounds by going in both directions from 0
    min_bound = None
    max_bound = None
    
    # Search in positive direction
    pos_vals = np.arange(0, search_max + search_step, search_step)
    for val in pos_vals:
        is_contact, penetration = evaluate_pose_at_val(val)
        if is_contact:
            if penetration <= max_penetration:
                max_bound = val  # Keep updating as long as penetration is acceptable
            else:
                # Penetration exceeds threshold - this becomes the boundary
                max_bound = val
                print(f"      {axis_name}={val:.6f}: penetration {penetration:.6f}m exceeds threshold {max_penetration:.6f}m - setting as positive boundary")
                break
    
    # Search in negative direction
    neg_vals = np.arange(0, search_min - search_step, -search_step)
    for val in neg_vals:
        is_contact, penetration = evaluate_pose_at_val(val)
        if is_contact:
            if penetration <= max_penetration:
                min_bound = val  # Keep updating as long as penetration is acceptable
            else:
                # Penetration exceeds threshold - this becomes the boundary
                min_bound = val
                print(f"      {axis_name}={val:.6f}: penetration {penetration:.6f}m exceeds threshold {max_penetration:.6f}m - setting as negative boundary")
                break
    
    # If no contacts found, return None
    if min_bound is None and max_bound is None:
        return None, None
    
    # If only one bound found, use zero as the other bound
    if min_bound is None:
        min_bound = 0.0
    if max_bound is None:
        max_bound = 0.0
    
    return min_bound, max_bound


def adaptive_sample_at_z(z_val, cfg):
    """
    Perform adaptive sampling at a fixed z value.
    
    1. Find contact bounds for each axis (x, y, a, b, c)
    2. Combinatorially sample within those bounds
    
    Args:
        z_val: fixed z value
        cfg: configuration dictionary
    
    Returns:
        List of pose tuples to evaluate
    """
    sampling_cfg = cfg["sampling"]
    deg = sampling_cfg.get("degrees", False)
    
    print(f"    Finding contact bounds at z={z_val:.6f}")
    
    # Find bounds for each axis
    bounds = {}
    for axis in ['x', 'y', 'a', 'b', 'c']:
        min_val, max_val = find_contact_bounds_for_axis(axis, z_val, cfg)
        if min_val is not None and max_val is not None:
            bounds[axis] = (min_val, max_val)
            print(f"      {axis}: [{min_val:.6f}, {max_val:.6f}]")
        else:
            # Use original range if no contact found
            if axis in ['x', 'y']:
                axis_cfg = sampling_cfg["xyz"][axis]
            else:
                axis_cfg = sampling_cfg["abc"][axis]
            bounds[axis] = (axis_cfg["min"], axis_cfg["max"])
            print(f"      {axis}: no contact found, using full range [{axis_cfg['min']:.6f}, {axis_cfg['max']:.6f}]")
    
    # Build grids for each axis within the found bounds
    def build_bounded_grid(axis_name, min_val, max_val):
        if axis_name in ['x', 'y']:
            step = sampling_cfg["xyz"][axis_name]["step"]
        else:
            step = sampling_cfg["abc"][axis_name]["step"]
        
        n_steps = int(np.ceil((max_val - min_val) / step)) + 1
        grid = np.linspace(min_val, max_val, n_steps)
        return grid
    
    x_grid = build_bounded_grid('x', *bounds['x'])
    y_grid = build_bounded_grid('y', *bounds['y'])
    a_grid = build_bounded_grid('a', *bounds['a'])
    b_grid = build_bounded_grid('b', *bounds['b'])
    c_grid = build_bounded_grid('c', *bounds['c'])
    
    # Convert angle grids to radians if needed
    if deg:
        a_grid = np.deg2rad(a_grid)
        b_grid = np.deg2rad(b_grid)
        c_grid = np.deg2rad(c_grid)
    
    # Generate all combinations
    poses = []
    for x in x_grid:
        for y in y_grid:
            for a in a_grid:
                for b in b_grid:
                    for c in c_grid:
                        poses.append((x, y, z_val, a, b, c))
    
    print(f"    Generated {len(poses)} poses for z={z_val:.6f}")
    return poses


def sample_pose_grid_adaptive(sampling_cfg):
    """
    Adaptive sampling: for each z value, find contact bounds for other axes
    and sample combinatorially within those bounds.
    """
    # Build z grid
    z_grid = build_grid(sampling_cfg["xyz"]["z"], sampling_cfg.get("inclusive", True))
    
    all_poses = []
    for z_val in z_grid:
        poses_at_z = adaptive_sample_at_z(z_val, {"sampling": sampling_cfg})
        all_poses.extend(poses_at_z)
    
    return all_poses


# ---------------------- Parallel worker setup ----------------------

# Globals populated per worker by _worker_init
_G = {}

def _worker_init(cfg):
    """
    Initializer: executed once per worker process.
    Loads meshes and builds managers (triangle meshes only).
    """
    ensure_fcl()
    m1 = trimesh.load(cfg["mesh1"], force="mesh")
    m2 = trimesh.load(cfg["mesh2"], force="mesh")
    if m1.is_empty or m2.is_empty:
        raise RuntimeError("One of the input meshes is empty.")

    # Light cleanup helps robustness
    m1 = _clean_mesh(m1)
    m2 = _clean_mesh(m2)

    T1 = np.array(cfg.get("mesh1_T", np.eye(4)), float)

    # Build triangle-mesh collision managers once per worker
    cm1 = build_manager_triangle_mesh("mesh1", m1, T1)
    cm2 = build_manager_triangle_mesh("mesh2", m2, np.eye(4))

    # Cache object names for the second manager so we can update transforms
    names2 = list(cm2._objs.keys())

    _G["cfg"] = cfg
    _G["T1"] = T1
    _G["cm1"] = cm1
    _G["cm2"] = cm2
    _G["names2"] = names2


def _eval_pose(pose_tuple):
    """
    Worker: evaluate a single pose (x,y,z,a,b,c). 
    Return 8-tuple row: (x_mm, y_mm, z_mm, a_deg, b_deg, c_deg, contact_flag, contact_distance_mm)
    where contact_distance is:
    - (negative) Maximum penetration depth if in contact 
    - (positive) Minimum separation distance if not in contact
    """
    x, y, z, a, b, c = pose_tuple
    cfg = _G["cfg"]
    T1 = _G["T1"]
    cm1 = _G["cm1"]
    cm2 = _G["cm2"]
    names2 = _G["names2"]

    # Build transform for mesh2 and set for all its parts (here just one object)
    T2 = make_T_from_xyz_abc(x, y, z, a, b, c, degrees=False)
    for n in names2:
        cm2.set_transform(n, T2)

    # Query collision
    is_hit, pair_names, cdata = cm1.in_collision_other(cm2, return_names=True, return_data=True)
    if is_hit:
        # Contact detected: use maximum penetration depth as contact_distance
        depths, _contacts = contact_points_localized(cdata, T1, T2)
        max_depth = -1.0 * float(max(depths)) if depths else 0.0
        # Convert to output units: mm for distance, degrees for angles
        return (x * 1000, y * 1000, z * 1000, np.rad2deg(a), np.rad2deg(b), np.rad2deg(c), 1.0, max_depth * 1000)

    # No collision: use minimum separation distance as contact_distance
    dist, names, ddata = cm1.min_distance_other(cm2, return_names=True, return_data=True)
    # Convert to output units: mm for distance, degrees for angles
    return (x * 1000, y * 1000, z * 1000, np.rad2deg(a), np.rad2deg(b), np.rad2deg(c), 0.0, float(dist) * 1000)


# ---------------------- Parallel bound finding ------------------------

def _find_bound_in_direction(args):
    """
    Worker function to find contact bound in a specific direction for an axis.
    
    Args:
        args: tuple of (axis_name, z_val, direction, search_vals, cfg)
              direction: 'positive' or 'negative'
    
    Returns:
        (direction, bound_val): tuple of direction and found bound value
    """
    axis_name, z_val, direction, search_vals, cfg = args
    
    sampling_cfg = cfg["sampling"]
    deg = sampling_cfg.get("degrees", False)
    max_penetration = sampling_cfg.get("max_penetration_depth", 0.001)
    
    def evaluate_pose_at_val(val):
        """Helper function to create and evaluate pose at a given axis value"""
        if axis_name == 'x':
            pose = (val, 0, z_val, 0, 0, 0)
        elif axis_name == 'y':
            pose = (0, val, z_val, 0, 0, 0)
        elif axis_name == 'a':
            angle_val = np.deg2rad(val) if deg else val
            pose = (0, 0, z_val, angle_val, 0, 0)
        elif axis_name == 'b':
            angle_val = np.deg2rad(val) if deg else val
            pose = (0, 0, z_val, 0, angle_val, 0)
        elif axis_name == 'c':
            angle_val = np.deg2rad(val) if deg else val
            pose = (0, 0, z_val, 0, 0, angle_val)
        
        result = _eval_pose(pose)
        is_contact = result[6] > 0.5
        penetration_depth = result[7] if is_contact else 0.0
        return is_contact, penetration_depth
    
    bound = None
    for val in search_vals:
        is_contact, penetration = evaluate_pose_at_val(val)
        if is_contact:
            if penetration <= max_penetration:
                bound = val  # Keep updating as long as penetration is acceptable
            else:
                # Penetration exceeds threshold - this becomes the boundary
                bound = val
                print(f"      {axis_name}={val:.6f}: penetration {penetration:.6f}m exceeds threshold {max_penetration:.6f}m - setting as {direction} boundary")
                break
    
    return direction, bound


def find_contact_bounds_for_axis_parallel(axis_name, z_val, cfg, pool=None):
    """
    Find the min/max values for a given axis where contact occurs at a fixed z value.
    Uses parallel processing to search positive and negative directions simultaneously.
    
    Args:
        axis_name: one of 'x', 'y', 'a', 'b', 'c'
        z_val: fixed z value
        cfg: configuration dictionary
        pool: multiprocessing pool (if None, falls back to serial)
    
    Returns:
        (min_val, max_val): tuple of bounds where contact occurs, or (None, None) if no contact
    """
    sampling_cfg = cfg["sampling"]
    margin = sampling_cfg.get("search_margin", 0.1)
    
    # Get the axis configuration
    if axis_name in ['x', 'y']:
        axis_cfg = sampling_cfg["xyz"][axis_name]
    else:
        axis_cfg = sampling_cfg["abc"][axis_name]
    
    # Build search parameters
    search_step = axis_cfg["step"] / 4  # Use finer resolution for searching
    search_min = axis_cfg["min"] - margin * abs(axis_cfg["max"] - axis_cfg["min"])
    search_max = axis_cfg["max"] + margin * abs(axis_cfg["max"] - axis_cfg["min"])
    
    # Prepare search tasks for parallel execution
    pos_vals = np.arange(0, search_max + search_step, search_step)
    neg_vals = np.arange(0, search_min - search_step, -search_step)
    
    tasks = [
        (axis_name, z_val, 'positive', pos_vals, cfg),
        (axis_name, z_val, 'negative', neg_vals, cfg)
    ]
    
    if pool is not None:
        # Parallel execution
        results = pool.map(_find_bound_in_direction, tasks)
    else:
        # Serial fallback
        results = [_find_bound_in_direction(task) for task in tasks]
    
    # Process results
    min_bound = None
    max_bound = None
    
    for direction, bound in results:
        if direction == 'positive':
            max_bound = bound
        elif direction == 'negative':
            min_bound = bound
    
    # If no contacts found, return None
    if min_bound is None and max_bound is None:
        return None, None
    
    # If only one bound found, use zero as the other bound
    if min_bound is None:
        min_bound = 0.0
    if max_bound is None:
        max_bound = 0.0
    
    return min_bound, max_bound


def _find_axis_bounds_worker(args):
    """
    Worker function to find contact bounds for a single axis at a given z value.
    
    Args:
        args: tuple of (axis_name, z_val, cfg)
    
    Returns:
        (axis_name, min_bound, max_bound): tuple of axis name and bounds
    """
    axis_name, z_val, cfg = args
    
    # Initialize worker globals if not already done
    if not _G:
        _worker_init(cfg)
    
    min_bound, max_bound = find_contact_bounds_for_axis_parallel(axis_name, z_val, cfg, pool=None)
    return axis_name, min_bound, max_bound


def adaptive_sample_at_z_fully_parallel(z_val, cfg, axis_pool=None):
    """
    Perform adaptive sampling at a fixed z value with full parallelization.
    Can parallelize both the bound finding across axes and the directional search.
    
    Args:
        z_val: fixed z value
        cfg: configuration dictionary
        axis_pool: optional multiprocessing pool for axis-level parallelization
    
    Returns:
        List of pose tuples to evaluate
    """
    sampling_cfg = cfg["sampling"]
    deg = sampling_cfg.get("degrees", False)
    
    print(f"    Finding contact bounds at z={z_val:.6f}")
    
    # Prepare tasks for parallel axis bound finding
    axis_tasks = [(axis, z_val, cfg) for axis in ['x', 'y', 'a', 'b', 'c']]
    
    bounds = {}
    
    if axis_pool is not None:
        # Parallel axis bound finding
        axis_results = axis_pool.map(_find_axis_bounds_worker, axis_tasks)
        for axis_name, min_val, max_val in axis_results:
            if min_val is not None and max_val is not None:
                bounds[axis_name] = (min_val, max_val)
                print(f"      {axis_name}: [{min_val:.6f}, {max_val:.6f}]")
            else:
                # Use original range if no contact found
                if axis_name in ['x', 'y']:
                    axis_cfg = sampling_cfg["xyz"][axis_name]
                else:
                    axis_cfg = sampling_cfg["abc"][axis_name]
                bounds[axis_name] = (axis_cfg["min"], axis_cfg["max"])
                print(f"      {axis_name}: no contact found, using full range [{axis_cfg['min']:.6f}, {axis_cfg['max']:.6f}]")
    else:
        # Serial axis bound finding (fallback)
        for axis in ['x', 'y', 'a', 'b', 'c']:
            min_val, max_val = find_contact_bounds_for_axis_parallel(axis, z_val, cfg, pool=None)
            if min_val is not None and max_val is not None:
                bounds[axis] = (min_val, max_val)
                print(f"      {axis}: [{min_val:.6f}, {max_val:.6f}]")
            else:
                # Use original range if no contact found
                if axis in ['x', 'y']:
                    axis_cfg = sampling_cfg["xyz"][axis]
                else:
                    axis_cfg = sampling_cfg["abc"][axis]
                bounds[axis] = (axis_cfg["min"], axis_cfg["max"])
                print(f"      {axis}: no contact found, using full range [{axis_cfg['min']:.6f}, {axis_cfg['max']:.6f}]")
    
    # Build grids for each axis within the found bounds
    def build_bounded_grid(axis_name, min_val, max_val):
        if axis_name in ['x', 'y']:
            step = sampling_cfg["xyz"][axis_name]["step"]
        else:
            step = sampling_cfg["abc"][axis_name]["step"]
        
        n_steps = int(np.ceil((max_val - min_val) / step)) + 1
        grid = np.linspace(min_val, max_val, n_steps)
        return grid
    
    x_grid = build_bounded_grid('x', *bounds['x'])
    y_grid = build_bounded_grid('y', *bounds['y'])
    a_grid = build_bounded_grid('a', *bounds['a'])
    b_grid = build_bounded_grid('b', *bounds['b'])
    c_grid = build_bounded_grid('c', *bounds['c'])
    
    # Convert angle grids to radians if needed
    if deg:
        a_grid = np.deg2rad(a_grid)
        b_grid = np.deg2rad(b_grid)
        c_grid = np.deg2rad(c_grid)
    
    # Generate all combinations
    poses = []
    for x in x_grid:
        for y in y_grid:
            for a in a_grid:
                for b in b_grid:
                    for c in c_grid:
                        poses.append((x, y, z_val, a, b, c))
    
    print(f"    Generated {len(poses)} poses for z={z_val:.6f}")
    return poses


def _adaptive_sample_at_z_fully_parallel_worker(args):
    """
    Worker function for fully parallel z-slice processing.
    Note: This runs inside a worker process, so we cannot create nested pools.
    
    Args:
        args: tuple of (z_val, cfg, use_axis_parallel)
    
    Returns:
        (z_val, poses_at_z): tuple of z value and list of poses
    """
    z_val, cfg, use_axis_parallel = args
    
    # Initialize worker globals if not already done
    if not _G:
        _worker_init(cfg)
    
    # Since we're already in a worker process, we cannot create nested pools
    # So we force use_axis_parallel to False and use serial axis processing
    poses_at_z = adaptive_sample_at_z_fully_parallel(z_val, cfg, axis_pool=None)
    
    return z_val, poses_at_z


# ---------------------- Main sweep ----------------------

def main(cfg):
    sampling_cfg = cfg["sampling"]
    use_adaptive = sampling_cfg.get("adaptive", False)
    used_parallel_adaptive = False  # Initialize this variable
    
    if use_adaptive:
        print("[INFO] Using adaptive per-slice sampling (AABB-style)")
        
        # Build z grid
        z_grid = build_grid(sampling_cfg["xyz"]["z"], sampling_cfg.get("inclusive", True))
        print(f"[INFO] Sampling across {len(z_grid)} z-slices")
        
        # Check if we can use parallelization for adaptive sampling
        par = cfg.get("parallel", {})
        use_parallel_adaptive = bool(par.get("enabled", True)) and bool(par.get("adaptive_parallel", True))
        use_z_slice_parallel = bool(par.get("z_slice_parallel", True))
        use_axis_parallel = bool(par.get("axis_parallel", False))
        workers = int(par.get("workers") or 0) or mp.cpu_count()
        
        # Prevent nested parallelization which causes issues
        if use_z_slice_parallel and use_axis_parallel:
            print("[WARNING] Both z_slice_parallel and axis_parallel are enabled. Disabling axis_parallel to prevent nested pools.")
            use_axis_parallel = False
        
        all_poses = []
        
        if use_parallel_adaptive and use_z_slice_parallel and len(z_grid) > 1:
            print(f"[INFO] Using parallel adaptive sampling with {workers} workers across z-slices")
            if use_axis_parallel:
                print("[INFO] Also using axis-level parallelization within each z-slice")
            
            # Parallel z-slice processing
            ctx = mp.get_context("spawn")
            z_tasks = [(z_val, cfg, use_axis_parallel) for z_val in z_grid]
            
            with ctx.Pool(processes=min(workers, len(z_grid)), initializer=_worker_init, initargs=(cfg,)) as pool:
                z_results = []
                for i, (z_val, poses_at_z) in enumerate(tqdm(
                    pool.imap(_adaptive_sample_at_z_fully_parallel_worker, z_tasks),
                    total=len(z_tasks),
                    desc="Processing z-slices in parallel"
                )):
                    print(f"  Completed z-slice {i+1}/{len(z_grid)}: z={z_val:.6f} with {len(poses_at_z)} poses")
                    all_poses.extend(poses_at_z)
            used_parallel_adaptive = True
        else:
            print("[INFO] Using serial adaptive sampling")
            # Serial adaptive sampling (original approach)
            _worker_init(cfg)  # Initialize worker for bound finding
            
            for i, z_val in enumerate(z_grid):
                print(f"  Processing z-slice {i+1}/{len(z_grid)}: z={z_val:.6f}")
                if use_axis_parallel:
                    # Use axis parallelization even in serial z-slice mode
                    ctx = mp.get_context("spawn")
                    with ctx.Pool(processes=min(5, workers), initializer=_worker_init, initargs=(cfg,)) as axis_pool:
                        poses_at_z = adaptive_sample_at_z_fully_parallel(z_val, cfg, axis_pool=axis_pool)
                else:
                    poses_at_z = adaptive_sample_at_z(z_val, cfg)
                all_poses.extend(poses_at_z)
            used_parallel_adaptive = False
        
        total = len(all_poses)
        print(f"[INFO] Total adaptive samples: {total}")
        pose_iter = iter(all_poses)
    else:
        print("[INFO] Using traditional grid sampling")
        # Traditional sampling
        xs = build_grid(sampling_cfg["xyz"]["x"], sampling_cfg.get("inclusive", True))
        ys = build_grid(sampling_cfg["xyz"]["y"], sampling_cfg.get("inclusive", True))
        zs = build_grid(sampling_cfg["xyz"]["z"], sampling_cfg.get("inclusive", True))

        deg = sampling_cfg.get("degrees", False)
        def to_base(arr):
            return np.deg2rad(arr) if deg else arr

        a_vals = to_base(build_grid(sampling_cfg["abc"]["a"], sampling_cfg.get("inclusive", True)))
        b_vals = to_base(build_grid(sampling_cfg["abc"]["b"], sampling_cfg.get("inclusive", True)))
        c_vals = to_base(build_grid(sampling_cfg["abc"]["c"], sampling_cfg.get("inclusive", True)))
        
        total = len(xs) * len(ys) * len(zs) * len(a_vals) * len(b_vals) * len(c_vals)
        print(f"[INFO] Total samples: {total}")
        
        # Prepare iterator of pose tuples
        pose_iter = ((x, y, z, r, p, yv) for x in xs for y in ys for z in zs
                                      for r in a_vals for p in b_vals for yv in c_vals)

    # Parallel or serial execution for pose evaluation
    par = cfg.get("parallel", {})
    use_parallel_poses = bool(par.get("enabled", True)) and bool(par.get("pose_parallel", True))
    workers = int(par.get("workers") or 0) or mp.cpu_count()
    chunksize = int(par.get("chunksize", 16))
    
    # For adaptive sampling, if we used z-slice parallelization, we can still parallelize pose evaluation
    # but we should avoid oversubscription by using fewer workers
    if use_adaptive and used_parallel_adaptive and use_parallel_poses:
        # We already used parallel processing for bound finding, reduce workers to avoid oversubscription
        workers = max(1, workers // 2)
        print(f"[INFO] Using {workers} workers for pose evaluation (reduced to avoid oversubscription)")
    elif use_adaptive and not use_parallel_poses:
        print("[INFO] Pose parallelization disabled for adaptive sampling")
    
    use_parallel = use_parallel_poses

    rows = []

    if use_parallel:
        # Use "spawn" for safety (fork can inherit non-fork-safe handles).
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers, initializer=_worker_init, initargs=(cfg,)) as pool:
            for out in tqdm(pool.imap(_eval_pose, pose_iter, chunksize=chunksize),
                            total=total, desc="Sampling poses"):
                rows.append(out)
    else:
        # Serial path (debug or adaptive)
        if not use_adaptive:
            _worker_init(cfg)  # initialize globals in this (main) process
        elif use_adaptive and not used_parallel_adaptive:
            # For serial adaptive sampling, globals are already initialized
            pass
        else:
            # For parallel adaptive sampling, we need to reinitialize for pose evaluation
            _worker_init(cfg)
            
        for pose in tqdm(pose_iter, total=total, desc="Sampling poses (serial)"):
            rows.append(_eval_pose(pose))

    # Convert to array and save
    results = np.array(rows, dtype=float)
    csv_path = cfg["save"]["csv_path"].format(geometry=cfg["geometry"])

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "z", "a", "b", "c", "contact", "contact_distance"])
        w.writerow(["units: mm", "mm", "mm", "deg", "deg", "deg", "0/1",
                    "max_penetration_depth_if_contact_else_min_separation_distance_mm"])
        for r in results:
            w.writerow([f"{v:.10g}" for v in r])

    print(f"[INFO] Saved CSV to {csv_path}")
    
    if use_adaptive:
        # Print some statistics about the adaptive sampling
        contact_results = results[results[:, 6] > 0.5]
        print(f"[INFO] Adaptive sampling found {len(contact_results)} contact poses out of {len(results)} total poses")
        if len(contact_results) > 0:
            print(f"[INFO] Contact efficiency: {len(contact_results)/len(results)*100:.1f}%")


if __name__ == "__main__":
    main(config)
