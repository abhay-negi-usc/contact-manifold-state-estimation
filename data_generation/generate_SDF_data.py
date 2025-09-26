# pip install trimesh igl numpy
import numpy as np
import trimesh as tm
import igl  # libigl python bindings
import time
from math import radians

# -------------------------- CONFIG: POSE SWEEP --------------------------
POSE_SWEEP_CFG = {
    # Translation ranges in meters: (min, max), resolution (# samples)
    "tx": {"range": (-0.050, 0.050), "res": 10},
    "ty": {"range": (-0.050, 0.050), "res": 10},
    "tz": {"range": (0.0, 0.030), "res": 10},
    # Rotation ranges in degrees (ZYX order): (min, max), resolution
    "roll":  {"range": (-5.0, 5.0), "res": 10},   # X
    "pitch": {"range": (-5.0, 5.0), "res": 10},   # Y
    "yaw":   {"range": (-5.0, 5.0), "res": 10},   # Z
    # If True, prints one line per pose with metrics
    "verbose": False,
}

# -------------------------- UTILITIES --------------------------

def linspace_range(rng, res):
    lo, hi = rng
    if res <= 1:
        return np.array([(lo + hi) / 2.0], dtype=float)
    return np.linspace(lo, hi, int(res), dtype=float)

def euler_zyx_to_matrix(roll_x_deg, pitch_y_deg, yaw_z_deg):
    """Build 3x3 rotation from ZYX Euler angles in degrees."""
    rx, ry, rz = map(radians, (roll_x_deg, pitch_y_deg, yaw_z_deg))
    # R = Rz * Ry * Rx
    cz, sz = np.cos(rz), np.sin(rz)
    cy, sy = np.cos(ry), np.sin(ry)
    cx, sx = np.cos(rx), np.sin(rx)

    Rz = np.array([[cz, -sz, 0.0],
                   [sz,  cz, 0.0],
                   [0.0, 0.0, 1.0]])
    Ry = np.array([[cy, 0.0, sy],
                   [0.0, 1.0, 0.0],
                   [-sy, 0.0, cy]])
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, cx, -sx],
                   [0.0, sx,  cx]])
    return (Rz @ Ry @ Rx)

def se3(tx, ty, tz, roll_deg, pitch_deg, yaw_deg):
    """Return 4x4 SE(3) from translation and ZYX Euler (deg)."""
    T = np.eye(4)
    T[:3, :3] = euler_zyx_to_matrix(roll_deg, pitch_deg, yaw_deg)
    T[:3, 3] = [tx, ty, tz]
    return T

def apply_transform_to_vertices(V: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to Nx3 vertex array."""
    V_h = np.c_[V, np.ones((V.shape[0], 1), dtype=V.dtype)]
    Vt = (T @ V_h.T).T
    return Vt[:, :3]

def load_mesh_as_VF(path: str, T_world_mesh=np.eye(4)):
    m = tm.load(path, force='mesh')
    if not isinstance(m, tm.Trimesh):
        m = tm.util.concatenate(m.dump())  # handle scenes
    V = apply_transform_to_vertices(m.vertices.view(np.ndarray).astype(np.float64), T_world_mesh)
    F = m.faces.view(np.ndarray).astype(np.int32)
    return V, F

def pick_winding_sign_type():
    # Newer enum name
    if hasattr(igl.SignedDistanceType, "FAST_WINDING_NUMBER"):
        return igl.SignedDistanceType.FAST_WINDING_NUMBER
    # Older enum name
    if hasattr(igl.SignedDistanceType, "WINDING_NUMBER"):
        return igl.SignedDistanceType.WINDING_NUMBER
    # Very old constant (not an Enum)
    if hasattr(igl, "SIGNED_DISTANCE_TYPE_WINDING_NUMBER"):
        return igl.SIGNED_DISTANCE_TYPE_WINDING_NUMBER
    # Last resort: default (usually pseudonormal)
    return igl.SignedDistanceType.DEFAULT

def sdf_of_points_to_mesh(P: np.ndarray, V: np.ndarray, F: np.ndarray, sign_type):
    """Signed distance from points P (Nx3) to triangle mesh (V,F)."""
    S, _, _, _ = igl.signed_distance(P.astype(np.float64), V, F, sign_type)
    return S

# -------------------------- MAIN SWEEP LOGIC --------------------------

def main():
    # Base/world transforms (you can change these if you have known poses)
    T_world_A = np.eye(4)  # fixed
    T_world_B_base = np.eye(4)  # base pose for B before sweep

    # Load STL meshes (update paths as needed)
    V_A, F_A = load_mesh_as_VF("./data_generation/assets/CAD/cross_peg_25.8.stl", T_world_A)
    V_B_ref, F_B = load_mesh_as_VF("./data_generation/assets/CAD/cross_hole_26.0.stl", np.eye(4))  # load in local; will transform later

    SIGN_TYPE = pick_winding_sign_type()

    # Precompute translation/rotation grids
    txs = linspace_range(POSE_SWEEP_CFG["tx"]["range"], POSE_SWEEP_CFG["tx"]["res"])
    tys = linspace_range(POSE_SWEEP_CFG["ty"]["range"], POSE_SWEEP_CFG["ty"]["res"])
    tzs = linspace_range(POSE_SWEEP_CFG["tz"]["range"], POSE_SWEEP_CFG["tz"]["res"])
    rolls  = linspace_range(POSE_SWEEP_CFG["roll"]["range"],  POSE_SWEEP_CFG["roll"]["res"])
    pitchs = linspace_range(POSE_SWEEP_CFG["pitch"]["range"], POSE_SWEEP_CFG["pitch"]["res"])
    yaws   = linspace_range(POSE_SWEEP_CFG["yaw"]["range"],   POSE_SWEEP_CFG["yaw"]["res"])

    total_poses = len(txs)*len(tys)*len(tzs)*len(rolls)*len(pitchs)*len(yaws)
    print(f"Sweeping {total_poses} poses...")

    results = []  # each row: (tx,ty,tz,roll,pitch,yaw, min_sdf_B_to_A, frac_B_inside_A, max_penetration_depth)

    t0 = time.time()
    ctr = 0
    verbose = POSE_SWEEP_CFG.get("verbose", True)

    for tz in tzs:
        for ty in tys:
            for tx in txs:
                for roll in rolls:
                    for pitch in pitchs:
                        for yaw in yaws:
                            ctr += 1
                            # Pose for B: base * delta
                            T_delta = se3(tx, ty, tz, roll, pitch, yaw)
                            T_world_B = T_world_B_base @ T_delta

                            # Transform B vertices and compute SDF to A
                            V_B_world = apply_transform_to_vertices(V_B_ref, T_world_B)
                            S_B_to_A = sdf_of_points_to_mesh(V_B_world, V_A, F_A, SIGN_TYPE)

                            min_sdf = float(np.min(S_B_to_A))                 # <0 means some penetration
                            frac_inside = float(np.mean(S_B_to_A < 0.0))      # fraction of B verts inside A
                            max_penetration = float(np.min(S_B_to_A))         # most negative value (same as min_sdf)

                            results.append((tx, ty, tz, roll, pitch, yaw, min_sdf, frac_inside, -max_penetration))

                            if verbose and (ctr % 50 == 0 or min_sdf < 0.0):
                                print(f"[{ctr}/{total_poses}] "
                                      f"tx={tx:+.4f} ty={ty:+.4f} tz={tz:+.4f} "
                                      f"r={roll:+.2f} p={pitch:+.2f} y={yaw:+.2f} deg | "
                                      f"min_sdf={min_sdf:+.5f} m | "
                                      f"frac_in={frac_inside:.3f} | "
                                      f"max_penetration={-max_penetration:.5f} m")

                            # ---- Optional: also compute A→B symmetry (commented)
                            # S_A_to_B = sdf_of_points_to_mesh(V_A, V_B_world, F_B, SIGN_TYPE)
                            # min_sdf_AtoB = float(np.min(S_A_to_B))
                            # You could store/report symmetric metrics if needed.

    elapsed = time.time() - t0
    print(f"Completed {total_poses} poses in {elapsed:.2f}s "
          f"({total_poses/ max(elapsed,1e-9):.1f} poses/s).")

    # Aggregate quick summaries:
    results_np = np.array(results, dtype=float)
    min_overall = results_np[:, 6].min() if results_np.size else np.nan
    any_pen = np.any(results_np[:, 6] < 0.0) if results_np.size else False
    print(f"Overall min SDF (B→A): {min_overall:+.6f} m; any penetration: {any_pen}")

    # Example: find the worst penetration pose
    worst_idx = int(np.argmin(results_np[:, 6])) if results_np.size else -1
    if worst_idx >= 0:
        tx, ty, tz, r, p, y, msdf, frac_in, max_pen = results[worst_idx]
        print("Worst pose (most negative min_sdf):")
        print(f"  tx={tx:+.5f} ty={ty:+.5f} tz={tz:+.5f} m | "
              f"roll={r:+.3f} pitch={p:+.3f} yaw={y:+.3f} deg | "
              f"min_sdf={msdf:+.6f} m | frac_in={frac_in:.3f} | max_penetration={max_pen:.6f} m")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Computation time: {elapsed_time:.6f} seconds")
