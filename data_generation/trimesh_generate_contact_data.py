#!/usr/bin/env python3
import csv
import sys
import math
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import trimesh

# ====================== CONFIG ======================
config = {
    # Fixed paths
    "mesh1": "/home/rp/abhay_ws/contact-manifold-state-estimation/data_generation/assets/meshes/extrusion_hole/extrusion_hole.obj",
    "mesh2": "/home/rp/abhay_ws/contact-manifold-state-estimation/data_generation/assets/meshes/extrusion_peg/extrusion_peg.obj",

    # Pose of mesh1 (static)
    "mesh1_T": np.eye(4).tolist(),   # 4x4 row-major

    # mesh2 pose is sampled across 6-DoF (xyz + rpy)
    "sampling": {
        # Units: meters for xyz, radians for rpy unless 'degrees'=True
        "xyz": {
            "x": {"min": -0.000, "max": 0.003, "step": 0.0005},
            "y": {"min": -0.000, "max": 0.003, "step": 0.0005},
            "z": {"min": -0.0, "max": 0.025, "step": 0.0025},
        },
        "rpy": {
            # These are degrees by default (set degrees=False to use radians)
            "roll":  {"min": -0.0, "max": 3.0, "step": 0.25},
            "pitch": {"min": -0.0, "max": 3.0, "step": 0.25},
            "yaw":   {"min": -0.0, "max": 3.0, "step": 0.25},
        },
        "degrees": True,
        "inclusive": True,  # include the max endpoint if it lands on the grid
    },

    # Parallel execution (only the pose sampling is parallel)
    "parallel": {
        "enabled": True,
        "workers": 20,       # 0 or None => use mp.cpu_count()
        "chunksize": 16      # tune for throughput; larger = fewer IPC calls
    },

    # Output
    "save": {
        "csv_path": "./data_generation/pose_sweep_contacts.csv",
        "npz_path": "./data_generation/pose_sweep_contacts.npz",
        "max_contacts_to_print": 0  # prints none; raise for debug
    },
}
# ====================================================

# ------------------------ Utils ------------------------

def rpy_to_matrix(roll, pitch, yaw, degrees=False):
    if degrees:
        roll, pitch, yaw = np.deg2rad([roll, pitch, yaw])
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [ 0,   0, 1]])
    Ry = np.array([[ cp, 0, sp],
                   [  0, 1,  0],
                   [-sp, 0, cp]])
    Rx = np.array([[1,  0,   0],
                   [0, cr, -sr],
                   [0, sr,  cr]])
    return Rz @ Ry @ Rx


def make_T_from_xyz_rpy(x, y, z, r, p, yv, degrees=False):
    T = np.eye(4)
    T[:3, :3] = rpy_to_matrix(r, p, yv, degrees=degrees)
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


def sample_pose_grid(sampling_cfg):
    xs = build_grid(sampling_cfg["xyz"]["x"], sampling_cfg.get("inclusive", True))
    ys = build_grid(sampling_cfg["xyz"]["y"], sampling_cfg.get("inclusive", True))
    zs = build_grid(sampling_cfg["xyz"]["z"], sampling_cfg.get("inclusive", True))

    deg = sampling_cfg.get("degrees", False)
    def to_base(arr):
        return np.deg2rad(arr) if deg else arr

    rolls  = to_base(build_grid(sampling_cfg["rpy"]["roll"],  sampling_cfg.get("inclusive", True)))
    pitchs = to_base(build_grid(sampling_cfg["rpy"]["pitch"], sampling_cfg.get("inclusive", True)))
    yaws   = to_base(build_grid(sampling_cfg["rpy"]["yaw"],   sampling_cfg.get("inclusive", True)))
    return xs, ys, zs, rolls, pitchs, yaws


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
    """Worker: evaluate a single pose (x,y,z,r,p,y). Return 8-tuple row."""
    x, y, z, r, p, yv = pose_tuple
    cfg = _G["cfg"]
    T1 = _G["T1"]
    cm1 = _G["cm1"]
    cm2 = _G["cm2"]
    names2 = _G["names2"]

    # Build transform for mesh2 and set for all its parts (here just one object)
    T2 = make_T_from_xyz_rpy(x, y, z, r, p, yv, degrees=False)
    for n in names2:
        cm2.set_transform(n, T2)

    # Query collision
    is_hit, pair_names, cdata = cm1.in_collision_other(cm2, return_names=True, return_data=True)
    if is_hit:
        depths, _contacts = contact_points_localized(cdata, T1, T2)
        max_depth = float(max(depths)) if depths else 0.0
        return (x, y, z, r, p, yv, 1.0, max_depth)

    # No collision -> min separation
    dist, names, ddata = cm1.min_distance_other(cm2, return_names=True, return_data=True)
    return (x, y, z, r, p, yv, 0.0, float(dist))


# ---------------------- Main sweep ----------------------

def main(cfg):
    # Build sampling grids and iterator (don’t materialize huge lists if avoidable)
    xs, ys, zs, rolls, pitchs, yaws = sample_pose_grid(cfg["sampling"])
    total = len(xs) * len(ys) * len(zs) * len(rolls) * len(pitchs) * len(yaws)
    print(f"[INFO] Total samples: {total}")

    # Prepare iterator of pose tuples
    pose_iter = ((x, y, z, r, p, yv) for x in xs for y in ys for z in zs
                                  for r in rolls for p in pitchs for yv in yaws)

    # Parallel or serial execution (only pose sampling is parallelized)
    par = cfg.get("parallel", {})
    use_parallel = bool(par.get("enabled", True))
    workers = int(par.get("workers") or 0) or mp.cpu_count()
    chunksize = int(par.get("chunksize", 16))

    rows = []

    if use_parallel:
        # Use "spawn" for safety (fork can inherit non-fork-safe handles).
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers, initializer=_worker_init, initargs=(cfg,)) as pool:
            for out in tqdm(pool.imap(_eval_pose, pose_iter, chunksize=chunksize),
                            total=total, desc="Sampling poses"):
                rows.append(out)
    else:
        # Serial path (debug)
        _worker_init(cfg)  # initialize globals in this (main) process
        for pose in tqdm(pose_iter, total=total, desc="Sampling poses (serial)"):
            rows.append(_eval_pose(pose))

    # Convert to array and save
    results = np.array(rows, dtype=float)
    csv_path = cfg["save"]["csv_path"]
    npz_path = cfg["save"]["npz_path"]

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "z", "roll", "pitch", "yaw", "contact", "metric"])
        w.writerow(["units: m", "m", "m", "rad", "rad", "rad", "0/1",
                    "depth_if_contact_else_distance"])
        for r in results:
            w.writerow([f"{v:.10g}" for v in r])

    np.savez(npz_path,
             results=results,
             columns=np.array(["x","y","z","roll","pitch","yaw","contact","metric"], dtype=object))

    print(f"[INFO] Saved CSV to {csv_path}")
    print(f"[INFO] Saved NPZ to {npz_path}")


if __name__ == "__main__":
    main(config)
