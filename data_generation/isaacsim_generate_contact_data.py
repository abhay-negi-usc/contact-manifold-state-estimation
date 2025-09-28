import argparse
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import asyncio, tempfile, os
import torch
import inspect

try:
    import isaacsim
except ImportError:
    pass

from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.utils.types import ArticulationAction
import omni.kit.asset_converter as asset_converter
from omni.kit.asset_converter import AssetConverterContext
from isaacsim.core.prims import SingleGeometryPrim  # single-prim handle

from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf, PhysxSchema

# -------------------------
# HELPER FUNCTIONS
# -------------------------


def solidify_asset_as_rigidbody(stage, root_path: str,
                                collision_approx="convexDecomposition",
                                enable_ccd=True, density=500.0):
    """
    - Applies CollisionAPI on all Mesh prims under root_path.
    - Sets PhysX collision approximation (convexDecomposition/convexHull/triangleMesh).
    - Applies RigidBodyAPI + Mass (density) on the root so it participates in physics.
    - Enables CCD to reduce tunneling.
    """
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        raise RuntimeError(f"Invalid prim: {root_path}")

    # Mark all meshes as colliders
    mesh_count = 0
    for p in Usd.PrimRange(root):
        if p.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(p)
            try:
                pc = PhysxSchema.PhysxCollisionAPI.Apply(p)
                # safer defaults for complex geometry
                pc.CreateContactOffsetAttr(0.005)   # 5 mm
                pc.CreateRestOffsetAttr(0.0)
                pc.CreateSolverPositionIterationCountAttr(8)
                pc.CreateSolverVelocityIterationCountAttr(2)
                if collision_approx:
                    pc.CreateApproximationAttr(collision_approx)
            except Exception:
                pass
            mesh_count += 1

    # Make the root a rigid body
    UsdPhysics.RigidBodyAPI.Apply(root)
    try:
        # Give it mass via density
        ma = UsdPhysics.MassAPI.Apply(root)
        ma.CreateDensityAttr(float(density))
    except Exception:
        pass
    try:
        prb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
        prb.CreateEnableCCDAttr(bool(enable_ccd))
        prb.CreateRigidBodyEnabledAttr(True)
    except Exception:
        pass

    print(f"[solidify] {root_path}: colliders={mesh_count}, rigidbody=on")

def _local_pose_matrix(stage, parent_path: str, child_path: str):
    """Return (t_vec3, q_wxyz) for parent->child using USD matrices (respects scale)."""
    parent = stage.GetPrimAtPath(parent_path)
    child  = stage.GetPrimAtPath(child_path)
    if not (parent and parent.IsValid() and child and child.IsValid()):
        raise RuntimeError(f"Invalid prim(s) for local pose: {parent_path}, {child_path}")

    xf_parent = UsdGeom.Xformable(parent)
    xf_child  = UsdGeom.Xformable(child)

    M_W_P = xf_parent.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    M_W_C = xf_child.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    tf_W_P = np.array(M_W_P).reshape(4,4).T  
    tf_W_C = np.array(M_W_C).reshape(4,4).T

    # renormalize rotation matrix 
    R_W_P = tf_W_P[0:3, 0:3]
    U, _, Vt = np.linalg.svd(R_W_P)
    R_W_P = U @ Vt
    tf_W_P[0:3, 0:3] = R_W_P
    R_W_C = tf_W_C[0:3, 0:3]
    U, _, Vt = np.linalg.svd(R_W_C)
    R_W_C = U @ Vt
    tf_W_C[0:3, 0:3] = R_W_C

    # parent->child in parent's local space (with scale)    
    tf_P_W = np.linalg.inv(tf_W_P)
    tf_P_C = tf_P_W @ tf_W_C 
    t_np = tf_P_C[0:3, 3]
    R_np = tf_P_C[0:3, 0:3]
    q_np = R.from_matrix(R_np).as_quat()  # xyzw
    q_np = np.array([q_np[3], q_np[0], q_np[1], q_np[2]], dtype=np.float64)  # wxyz
    return t_np, q_np

def force_convex_decomposition(stage, root_path: str,
                               contact_offset=0.005, rest_offset=0.0,
                               pos_iters=8, vel_iters=2):
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        raise RuntimeError(f"Invalid prim: {root_path}")

    for p in Usd.PrimRange(root):
        if p.IsA(UsdGeom.Mesh):
            # Make it a collider
            UsdPhysics.CollisionAPI.Apply(p)
            # Set PhysX collision options
            pc = PhysxSchema.PhysxCollisionAPI.Apply(p)
            pc.CreateContactOffsetAttr(float(contact_offset))
            pc.CreateRestOffsetAttr(float(rest_offset))
            pc.CreateSolverPositionIterationCountAttr(int(pos_iters))
            pc.CreateSolverVelocityIterationCountAttr(int(vel_iters))
            # Critical line: request convex decomposition
            pc.CreateApproximationAttr("convexDecomposition")

    print(f"[collision] {root_path}: set approximation=convexDecomposition on all Meshes")


# ---- ATTACH PAYLOAD TO PLANNER AS SPHERES (no Obstacle.mesh) ----

def _quat_wxyz_to_R(q):
    return R.from_quat(q, scalar_first=True).as_matrix()

from pxr import UsdGeom, Usd, Gf
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

def _quat_wxyz_to_R(q):
    return R.from_quat(q, scalar_first=True).as_matrix()

def _usd_aabb_local(stage, prim_path: str):
    """Return (min_xyz, max_xyz) for the prim's LOCAL AABB (respects USD scale & children)."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Invalid prim for AABB: {prim_path}")
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True
    )
    bound = cache.ComputeLocalBound(prim)
    aabb  = bound.ComputeAlignedBox()
    mn, mx = aabb.GetMin(), aabb.GetMax()
    return np.array([mn[0], mn[1], mn[2]], float), np.array([mx[0], mx[1], mx[2]], float)

# -------------------------
# MAIN
# -------------------------

def main():

    # World / Stage
    world = World(stage_units_in_meters=1.0)
    stage = world.stage
    print("[init] world / stage")

    def ensure_physics_scene(stage, path="/World/physicsScene"):
        if not stage.GetPrimAtPath(path):
            scene = stage.DefinePrim(path, "PhysicsScene")
            UsdPhysics.Scene(scene)  # author schema
        return path
    ensure_physics_scene(world.stage)

    # Helper: pack pose as [x,y,z,qw,qx,qy,qz]
    def _pose7_from_prim(prim_path: str):
        t, q = _get_world_pose(stage, prim_path)
        return [float(t[0]), float(t[1]), float(t[2]), float(q[0]), float(q[1]), float(q[2]), float(q[3])]

    stage.SetEditTarget(Usd.EditTarget(stage.GetRootLayer()))

    # # --- Parse 6-tuples
    # p1_t = np.array([0,0,0])
    # p1_q = rpy_deg_to_quat_xyz(np.array([0,0,0]))

    # p2_t = _as_vec3(np.array([0,0,0]))
    # p2_q = rpy_deg_to_quat_xyz(np.array([0,0,0]))

    # # Pactruss #1 (graspable)
    # p1_root = "/World/hole"
    # stage.DefinePrim(p1_root, "Xform").GetReferences().AddReference(assetPath=args.pactruss_usd)
    # set_world_pose_matrix(stage, p1_root, p1_t, p1_q, s_xyz=tuple(args.pactruss_scale))

    # # Pactruss #2 (fixed)
    # p2_root = "/World/peg"
    # stage.DefinePrim(p2_root, "Xform").GetReferences().AddReference(assetPath=args.pactruss_usd)
    # set_world_pose_matrix(stage, p2_root, p2_t, p2_q, s_xyz=tuple(args.pactruss_scale))
    # _fixed_to_world(stage, p2_root)  # keep it pinned

    # # Resolve frames
    # grasp_path = _resolve_child_or_abs(stage, p1_root, args.grasp_frame)
    # cone_path  = _resolve_child_or_abs(stage, p1_root, args.cone_frame)
    # cup_path   = _resolve_child_or_abs(stage, p2_root, args.cup_frame)
    # # NEW: transport target under pactruss_2
    # transport_base_path = _resolve_child_or_abs(stage, p2_root, args.transport_frame)
    # print(f"[init] grasp={grasp_path}  cone={cone_path}  cup={cup_path}  transport={transport_base_path}")

    # # after creating p1_root and p2_root & setting their poses
    # solidify_asset_as_rigidbody(stage, p1_root, collision_approx="convexDecomposition", enable_ccd=True)
    # solidify_asset_as_rigidbody(stage, p2_root, collision_approx="convexDecomposition", enable_ccd=True)


    import isaacsim.core.utils.prims as prims_utils

    hole_prim_path = "/World/hole"
    hole_usd_file_path = "/home/rp/abhay_ws/contact-manifold-state-estimation/data_generation/assets/CAD/cross_hole_real/cross_hole_real.usd"
    peg_prim_path = "/World/peg"
    peg_usd_file_path = "/home/rp/abhay_ws/contact-manifold-state-estimation/data_generation/assets/CAD/cross_peg_real/cross_peg_real.usd"

    hole_prim = prims_utils.create_prim(
        prim_path=hole_prim_path,
        position=Gf.Vec3f(0.0, 0.0, 0.0),
        orientation=np.array([1,0,0,0]),  # wxyz
        scale=Gf.Vec3d(1.0, 1.0, 1.0),  # Use Gf.Vec3d for double precision
        usd_path=hole_usd_file_path,
        semantic_label="hole",
    )
    # wrap the prim in a GeometryPrim object with colliders
    hole_geo_prim = SingleGeometryPrim(
        prim_path=hole_prim_path,
        name=hole_usd_file_path,
        collision=True
    )
    hole_geo_prim.set_collision_approximation("convexDecomposition")
    # force_convex_decomposition(stage, "/World/hole")


    peg_prim = prims_utils.create_prim(
        prim_path=peg_prim_path,
        position=Gf.Vec3f(0.0, 0.0, 0.0),
        orientation=R.from_euler('xyz',[1,0,0],degrees=True).as_quat()[[3,0,1,2]],  # wxyz
        scale=Gf.Vec3d(1.0, 1.0, 1.0),  # Use Gf.Vec3d for double precision
        usd_path=peg_usd_file_path,
        semantic_label="peg",
    )
    # wrap the prim in a GeometryPrim object with colliders
    peg_geo_prim = SingleGeometryPrim(
        prim_path=peg_prim_path,
        name=peg_usd_file_path,
        collision=True
    )
    peg_geo_prim.set_collision_approximation("convexDecomposition")
    # force_convex_decomposition(stage, "/World/peg")

    # --- Make both bodies rigid (colliders already added by SingleGeometryPrim) ---
    solidify_asset_as_rigidbody(stage, hole_prim_path, collision_approx="convexDecomposition", enable_ccd=True, density=1000.0)
    solidify_asset_as_rigidbody(stage, peg_prim_path,  collision_approx="convexDecomposition", enable_ccd=True,  density=1000.0)
    
    # ========================
    # 1) FIXED JOINT: hole → world
    # ========================
    fixed_joint_path = "/World/Joints/hole_fixed_to_world"
    stage.DefinePrim("/World/Joints", "Xform")
    fj = UsdPhysics.FixedJoint.Define(stage, fixed_joint_path)

    # Connect hole as body0; omit body1 to anchor to world
    fj.CreateBody0Rel().AddTarget(Sdf.Path(hole_prim_path))
    # (Body1Rel left empty = world frame)

    # Local anchors (at each body's local origin)
    fj.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    fj.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))  # w,x,y,z
    fj.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    fj.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    # ========================
    # 2) 6-DoF JOINT (D6): peg → world  (via UsdPhysics.Joint)
    # ========================
    peg_joint_path = "/World/Joints/peg_free_6d_world"
    stage.DefinePrim("/World/Joints", "Xform")
    j = UsdPhysics.Joint.Define(stage, peg_joint_path)

    # Attach peg as body0; leave body1 empty → anchored to world
    j.CreateBody0Rel().AddTarget(Sdf.Path(peg_prim_path))
    # j.CreateBody1Rel()  # intentionally omitted → world

    # Joint frames at each body’s local origin (adjust if needed)
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))  # w,x,y,z
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    # Optional: keep collisions between peg and world enabled (default is fine here).
    # If you later joint two bodies, USD joints typically disable pair collisions by default.


    world.play()
    print("[init] play world")

    # ---------------- Simulation loop ----------------
    while simulation_app.is_running():
        world.step(render=True)
        step_idx = world.current_time_step_index

    simulation_app.close()

if __name__ == "__main__":
    main()