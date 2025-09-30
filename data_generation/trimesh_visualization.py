#!/usr/bin/env python3
import trimesh
import numpy as np
from trimesh.collision import CollisionManager

# Paths
hole_mesh_path = "/home/rp/abhay_ws/contact-manifold-state-estimation/data_generation/assets/meshes/extrusion_hole/extrusion_hole.obj"
peg_mesh_path  = "/home/rp/abhay_ws/contact-manifold-state-estimation/data_generation/assets/meshes/extrusion_peg/extrusion_peg.obj"

# Load meshes
hole_mesh = trimesh.load(hole_mesh_path, force="mesh")
peg_mesh  = trimesh.load(peg_mesh_path, force="mesh")

# Build collision managers
cm_hole = CollisionManager()
cm_hole.add_object("hole", hole_mesh, transform=np.eye(4))

cm_peg = CollisionManager()
tf_peg = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,20e-3],[0,0,0,1]])
cm_peg.add_object("peg", peg_mesh, transform=tf_peg)

# Check for collision and list contacts
is_hit, names, data = cm_hole.in_collision_other(cm_peg, return_names=True, return_data=True)

if is_hit:
    print("[INFO] Collision detected!")
    # data may contain penetration depth, contact points, normals depending on FCL build
    if isinstance(data, list):
        for i, c in enumerate(data):
            print(f"  Contact {i}:")
            if hasattr(c, "point"):
                print("   point:", np.asarray(c.point))
            if hasattr(c, "normal"):
                print("   normal:", np.asarray(c.normal))
            if hasattr(c, "depth"):
                print("   depth:", float(c.depth))
    else:
        print("  Contact data:", data)
else:
    dist, names, ddata = cm_hole.min_distance_other(cm_peg, return_names=True, return_data=True)
    print("[INFO] No collision.")
    print(f"  Minimum distance: {dist:.6f} m")

# Optional: visualize
scene = trimesh.Scene([hole_mesh, peg_mesh])
scene.show()
