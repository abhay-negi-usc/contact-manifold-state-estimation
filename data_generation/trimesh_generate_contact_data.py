#!/usr/bin/env python3
import csv
import sys
import math
import numpy as np
import itertools as it
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import trimesh
import importlib
import os, shutil, subprocess, tempfile, glob

# ====================== CONFIG ======================
config = {
    # Fixed paths (as requested)
    "mesh1": "/home/rp/abhay_ws/contact-manifold-state-estimation/data_generation/assets/CAD/cross_hole_real/cross_hole_real.stl",
    "mesh2": "/home/rp/abhay_ws/contact-manifold-state-estimation/data_generation/assets/CAD/cross_peg_real/cross_peg_real.stl",

    # Pose of mesh1 is static (change if needed)
    "mesh1_T": np.eye(4).tolist(),   # 4x4 row-major

    # mesh2 pose is sampled across 6-DoF (xyz + rpy)
    "sampling": {
        # Units: meters for xyz, radians for rpy unless 'degrees'=True
        "xyz": {
            "x": {"min": -0.000, "max": 0.000, "step": 0.0025},
            "y": {"min": -0.000, "max": 0.000, "step": 0.0025},
            "z": {"min": -0.050, "max": 0.050, "step": 0.001},
        },
        "rpy": {
            # For easier config, these are degrees by default (set degrees=False to use radians)
            "roll":  {"min": -0.0, "max": 0.0, "step": 2.5},
            "pitch": {"min": -0.0, "max": 0.0, "step": 2.5},
            "yaw":   {"min": -0.0, "max": 0.0, "step": 2.5},
        },
        "degrees": True,
        "inclusive": True,  # include the max endpoint if it lands on the grid
    },

    # --- Contact modeling strategy ---
    "use_vhacd": True,              # enable VHACD by default (most accurate for non-convex)
    "fallback": "triangle_mesh",    # fallback only if VHACD unavailable

    # VHACD parameters (CLI-first; tuned for accuracy over speed)
    # Notes:
    # - resolution: ↑ => tighter parts (slower)
    # - max_hulls: ↑ => more parts (slower/closer fit)
    # - error: ↓ (% volume error) => stricter (slower)
    # - depth: ↑ => deeper recursion (slower)
    "vhacd": {
        "exe": "/usr/local/bin/vhacd",  # or just "vhacd" if on PATH
        "resolution": 1_000_000_000,
        "max_hulls": 2**11,
        "error": 0.001,        # percent
        "depth": 64,
        # kept for pyVHACD fallback compatibility (ignored by CLI helper):
        # "concavity": 0.001,
        # "maxNumVerticesPerCH": 2**10,
        # "minVolumePerCH": 1e-9,
    },

    # Parallel execution
    "parallel": {
        "enabled": True,
        "workers": 0,        # 0 or None => use mp.cpu_count()
        "chunksize": 16      # tune for throughput; larger = fewer IPC calls
    },

    # Output
    "save": { # FIXME: update save path 
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
    m.rezero()
    return m

def _vhacd_cli_decompose(mesh: trimesh.Trimesh, vhacd_cfg, quiet=True) -> list[trimesh.Trimesh]:
    """
    Decompose using the compiled VHACD CLI (TestVHACD).
    Returns a list of hull meshes (Trimesh). Raises on failure.
    """
    exe = vhacd_cfg.get("exe", "vhacd")
    exe_path = exe if os.path.sep in exe else shutil.which(exe)
    if exe_path is None:
        raise FileNotFoundError(f"VHACD executable not found: {exe}")

    m = _clean_mesh(mesh)

    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "input.obj")
        m.export(in_path)

        cmd = [
            exe_path,
            in_path,
            "-o", "obj",
            "-h", str(vhacd_cfg.get("max_hulls", 128)),
            "-r", str(vhacd_cfg.get("resolution", 400_000)),
            "-e", str(vhacd_cfg.get("error", 0.5)),   # volume error percent
            "-d", str(vhacd_cfg.get("depth", 12)),
            "-g", "false" if quiet else "true",
        ]

        res = subprocess.run(
            cmd, cwd=td, text=True,
            stdout=(subprocess.PIPE if quiet else None),
            stderr=(subprocess.PIPE if quiet else None)
        )
        if res.returncode != 0:
            raise RuntimeError(f"VHACD CLI failed ({res.returncode}). "
                               f"stdout:\n{res.stdout or ''}\nstderr:\n{res.stderr or ''}")

        hull_paths = sorted(glob.glob(os.path.join(td, "*.obj")))
        if not hull_paths:
            raise RuntimeError("VHACD CLI produced no hull files")

        parts = []
        for hp in hull_paths:
            L = trimesh.load(hp, force="mesh")
            if isinstance(L, trimesh.Trimesh):
                parts.append(_clean_mesh(L))
            elif hasattr(L, "geometry") and L.geometry:
                parts.extend([_clean_mesh(g) for g in L.geometry.values() if isinstance(g, trimesh.Trimesh)])
        return parts

def _convexify_all(parts: list[trimesh.Trimesh]) -> list[trimesh.Trimesh]:
    """Guarantee strict convexity (defensive against tiny tessellation errors)."""
    return [p if getattr(p, "is_convex", False) else p.convex_hull for p in parts]


def _vhacd_decompose_pyVHACD_only(mesh: trimesh.Trimesh, debug=False):
    """
    Decompose using pyVHACD builds that expose ONLY `compute_vhacd(V, F)`.
    Strategy:
      1) Try with NumPy arrays (C-contiguous, owned).
      2) On "resize only works on single-segment arrays", retry with nested Python lists.
      3) As last resort, retry with flattened lists.
    Returns list[trimesh.Trimesh] or [].
    """
    spec = importlib.util.find_spec("pyVHACD")
    if spec is None:
        if debug: print("[VHACD] pyVHACD not importable")
        return []

    pyVHACD = importlib.import_module("pyVHACD")
    if not hasattr(pyVHACD, "compute_vhacd") or not callable(pyVHACD.compute_vhacd):
        if debug: print("[VHACD] pyVHACD.compute_vhacd missing/not callable")
        return []

    # --- Clean mesh ---
    m = _clean_mesh(mesh)

    # ---------- Attempt 1: NumPy arrays (owned, C-contiguous) ----------
    V = np.array(m.vertices, dtype=np.float64, order="C", copy=True)
    F = np.array(m.faces,    dtype=np.int32,   order="C", copy=True)
    V = np.require(V, requirements=["O","W","C"])
    F = np.require(F, requirements=["O","W","C"])
    try:
        out = pyVHACD.compute_vhacd(V, F)
        if debug: print("[VHACD] compute_vhacd(V,F[int32]) ok (NumPy)")
        parts = _vhacd_normalize_return(out, debug=debug)
        if parts: return parts
    except Exception as e_np:
        if debug:
            print(f"[VHACD] NumPy call failed: {type(e_np).__name__}: {e_np}")

    # Try uint32 faces quickly
    F_u = np.array(m.faces, dtype=np.uint32, order="C", copy=True)
    F_u = np.require(F_u, requirements=["O","W","C"])
    try:
        out = pyVHACD.compute_vhacd(V, F_u)
        if debug: print("[VHACD] compute_vhacd(V,F[uint32]) ok (NumPy)")
        parts = _vhacd_normalize_return(out, debug=debug)
        if parts: return parts
    except Exception as e_np2:
        if debug:
            print(f"[VHACD] NumPy call (uint32) failed: {type(e_np2).__name__}: {e_np2}")

    # ---------- Attempt 2: nested Python lists (Nx3) ----------
    V_list = m.vertices.tolist()
    F_list32 = m.faces.astype(np.int32, copy=False).tolist()
    try:
        out = pyVHACD.compute_vhacd(V_list, F_list32)
        if debug: print("[VHACD] compute_vhacd(V_list, F_list[int32]) ok (lists)")
        parts = _vhacd_normalize_return(out, debug=debug)
        if parts: return parts
    except Exception as e_list:
        if debug:
            print(f"[VHACD] List call (int32) failed: {type(e_list).__name__}: {e_list}")

    # Try uint32 as lists
    F_listu = m.faces.astype(np.uint32, copy=False).tolist()
    try:
        out = pyVHACD.compute_vhacd(V_list, F_listu)
        if debug: print("[VHACD] compute_vhacd(V_list, F_list[uint32]) ok (lists)")
        parts = _vhacd_normalize_return(out, debug=debug)
        if parts: return parts
    except Exception as e_list2:
        if debug:
            print(f"[VHACD] List call (uint32) failed: {type(e_list2).__name__}: {e_list2}")

    # ---------- Attempt 3: flattened lists ----------
    V_flat = np.array(m.vertices, dtype=np.float64, order="C", copy=True).reshape(-1).tolist()
    F_flat32 = np.array(m.faces, dtype=np.int32, order="C", copy=True).reshape(-1).tolist()
    try:
        out = pyVHACD.compute_vhacd(V_flat, F_flat32)
        if debug: print("[VHACD] compute_vhacd(V_flat, F_flat[int32]) ok (flat lists)")
        parts = _vhacd_normalize_return(out, flat_input=True, debug=debug)
        if parts: return parts
    except Exception as e_flat:
        if debug:
            print(f"[VHACD] Flat list call failed: {type(e_flat).__name__}: {e_flat}")

    if debug: print("[VHACD] All strategies failed for this pyVHACD build.")
    return []


def _vhacd_normalize_return(out, flat_input=False, debug=False):
    """
    Convert pyVHACD results (various formats) to list[trimesh.Trimesh].
    If flat_input=True, expects the function to return hulls that include shape info.
    """
    parts = []

    # (1) tuple: (verts_list, faces_list)
    if isinstance(out, tuple) and len(out) == 2:
        verts_list, faces_list = out
        for v, f in zip(verts_list, faces_list):
            v = np.asarray(v, dtype=np.float64)
            f = np.asarray(f, dtype=np.int32)
            if v.size and f.size:
                parts.append(trimesh.Trimesh(vertices=v, faces=f, process=False))
        return parts

    # (2) list of dicts with "vertices"/"triangles"
    if isinstance(out, list) and out and isinstance(out[0], dict):
        for h in out:
            v = np.asarray(h.get("vertices", []), dtype=np.float64)
            f = np.asarray(h.get("triangles", h.get("faces", [])), dtype=np.int32)
            if v.size and f.size:
                parts.append(trimesh.Trimesh(vertices=v, faces=f, process=False))
        return parts

    # (3) list of hull objects with .points/.triangles
    if isinstance(out, list) and out:
        first = out[0]
        if not isinstance(first, (np.ndarray, list, tuple, dict)):
            for h in out:
                v = np.asarray(getattr(h, "points", getattr(h, "vertices", [])), dtype=np.float64)
                f = np.asarray(getattr(h, "triangles", getattr(h, "faces", [])), dtype=np.int32)
                if v.size and f.size:
                    parts.append(trimesh.Trimesh(vertices=v, faces=f, process=False))
            return parts

    # (4) flat-input special cases:
    # Some bindings return (verts_list, faces_list) even for flat input.
    # If not, there's no universal way to recover shapes here without extra getters.
    if flat_input and isinstance(out, list) and out and isinstance(out[0], dict):
        # already handled above, but keep for clarity
        for h in out:
            v = np.asarray(h.get("vertices", []), dtype=np.float64)
            f = np.asarray(h.get("triangles", h.get("faces", [])), dtype=np.int32)
            if v.size and f.size:
                parts.append(trimesh.Trimesh(vertices=v, faces=f, process=False))
        return parts

    if debug:
        print("[VHACD] Unrecognized return format:", type(out))
    return parts

def decompose_convex_parts(mesh, vhacd_cfg, debug=True):
    # 1) Prefer the CLI you built (robust; avoids NumPy-resize bug)
    try:
        parts = _vhacd_cli_decompose(mesh, vhacd_cfg, quiet=not debug)
        if parts:
            parts = _convexify_all(parts)   # <-- convexify by default
            if debug: print(f"[VHACD] CLI produced {len(parts)} convex parts")
            return parts
    except Exception as e:
        if debug: print(f"[VHACD] CLI failed: {e}")

    # 2) Fallback to pyVHACD bindings (if present)
    try:
        parts = _vhacd_decompose_pyVHACD_only(mesh, debug=debug)
        if parts:
            parts = _convexify_all(parts)   # <-- convexify by default
            if debug: print(f"[VHACD] pyVHACD produced {len(parts)} convex parts")
            return parts
    except Exception as e:
        if debug: print(f"[VHACD] pyVHACD failed: {e}")

    # 3) Decomposition failed
    return None

def build_manager_from_parts(name_prefix, parts, T):
    from trimesh.collision import CollisionManager
    cm = CollisionManager()
    for i, p in enumerate(parts):
        cm.add_object(f"{name_prefix}_{i}", p, transform=T)
    return cm


def build_manager_for_mesh(name_prefix, mesh, T, cfg):
    """
    Strategy: VHACD -> fallback ('triangle_mesh' by default) -> 'convex_hull' if asked.
    Notes:
      - VHACD gives the most reliable penetration depths on non-convex geometry.
      - triangle_mesh is fine for collision/no-collision and distances,
        but penetration depths can be noisier or under-reported.
    """
    if cfg.get("use_vhacd", True):
        parts = decompose_convex_parts(mesh, cfg.get("vhacd", {}))
        if parts is not None:
            return build_manager_from_parts(name_prefix, parts, T), "vhacd"
        
    # Fallback strategy
    print("[WARN] VHACD failed or disabled; using fallback collision strategy.", file=sys.stderr)

    fallback = cfg.get("fallback", "triangle_mesh")
    if fallback == "triangle_mesh":
        from trimesh.collision import CollisionManager
        cm = CollisionManager()
        cm.add_object(name_prefix, mesh, transform=T)
        return cm, "triangle_mesh"
    elif fallback == "convex_hull":
        hull = mesh.convex_hull
        return build_manager_from_parts(name_prefix, [hull], T), "convex_hull"

    else:
        raise ValueError(f"Unknown fallback: {fallback}")


def contact_points_localized(contact_blob, T1, T2):
    """
    Extract contact depths and approximate contact points in each mesh's local frame.
    Robust to trimesh/python-fcl version differences.
    Returns (depths:list[float], contacts:list[dict]).
    """
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
    """Initializer: executed once per worker process; loads meshes and builds managers."""
    ensure_fcl()
    m1 = trimesh.load(cfg["mesh1"], force="mesh")
    m2 = trimesh.load(cfg["mesh2"], force="mesh")
    if m1.is_empty or m2.is_empty:
        raise RuntimeError("One of the input meshes is empty.")

    # Light cleanup helps robustness
    for m in (m1, m2):
        m = _clean_mesh(m)  # ensure consistent topology; avoids deprecated calls


    T1 = np.array(cfg.get("mesh1_T", np.eye(4)), float)

    # Build managers once per worker
    cm1, strat1 = build_manager_for_mesh("mesh1", m1, T1, cfg)
    cm2, strat2 = build_manager_for_mesh("mesh2", m2, np.eye(4), cfg)

    # cache object names for the second manager so we can update transforms
    names2 = list(cm2._objs.keys())

    _G["cfg"] = cfg
    _G["T1"] = T1
    _G["cm1"] = cm1
    _G["cm2"] = cm2
    _G["names2"] = names2
    _G["strat1"] = strat1
    _G["strat2"] = strat2


def _eval_pose(pose_tuple):
    """Worker: evaluate a single pose (x,y,z,r,p,y). Return 8-tuple row."""
    x, y, z, r, p, yv = pose_tuple
    cfg = _G["cfg"]
    T1 = _G["T1"]
    cm1 = _G["cm1"]
    cm2 = _G["cm2"]
    names2 = _G["names2"]

    # Build transform for mesh2 and set for all its parts
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

    # Parallel or serial execution
    par = cfg.get("parallel", {})
    use_parallel = bool(par.get("enabled", True))
    workers = int(par.get("workers") or 0) or mp.cpu_count()
    chunksize = int(par.get("chunksize", 16))

    rows = []

    if use_parallel:
        # Use "spawn" on some platforms for safety (fork can inherit non-fork-safe handles).
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
