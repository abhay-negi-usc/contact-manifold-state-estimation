#!/usr/bin/env python3
import argparse
import numpy as np
import trimesh
import matplotlib.pyplot as plt


# ----------------- CLI -----------------
def parse_args():
    p = argparse.ArgumentParser(description="Render silhouettes + feature edges from a given view (compat)")
    p.add_argument("--mesh", required=True, help="Path to mesh (obj, ply, stl, glb, etc.)")
    p.add_argument("--angle-deg", type=float, default=30.0,
                   help="Dihedral angle threshold (degrees) for feature edges")
    p.add_argument("--eye", type=float, nargs=3, default=[0.1, 0.0, 0.1],
                   help="Camera position (world) x y z")
    p.add_argument("--target", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                   help="Look-at target (world) x y z")
    p.add_argument("--up", type=float, nargs=3, default=[0.0, 0.0, 1.0],
                   help="Camera up vector")
    p.add_argument("--width", type=int, default=1024, help="Image width (px)")
    p.add_argument("--height", type=int, default=768, help="Image height (px)")
    p.add_argument("--fov-deg", type=float, default=45.0, help="Vertical field-of-view (degrees)")
    p.add_argument("--out", type=str, default="edges.png", help="Output image path (PNG)")
    return p.parse_args()


# ----------------- Camera & Projection -----------------
def look_at_matrix(eye, target, up):
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(up, dtype=float)

    forward = (eye - target)
    forward = forward / np.linalg.norm(forward)

    right = np.cross(up, forward)
    right = right / np.linalg.norm(right)

    true_up = np.cross(forward, right)

    R = np.stack([right, true_up, forward], axis=0)
    t = -R @ eye.reshape(3, 1)

    view = np.eye(4)
    view[:3, :3] = R
    view[:3, 3] = t.ravel()
    return view


def intrinsics_from_fov(fov_y_deg, width, height):
    fov_y = np.deg2rad(fov_y_deg)
    fy = (height / 2.0) / np.tan(fov_y / 2.0)
    fx = fy
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]], dtype=float)


def project_points(points_cam, K):
    pts = np.asarray(points_cam, dtype=float)
    uv = np.full((pts.shape[0], 2), np.nan, dtype=float)
    z = pts[:, 2]
    valid = z > 1e-8
    if np.any(valid):
        x = pts[valid, 0] / z[valid]
        y = pts[valid, 1] / z[valid]
        homog = np.stack([x, y, np.ones_like(x)], axis=0)
        img = (K @ homog).T[:, :2]
        uv[valid] = img
    return uv


def transform_points(points, T):
    pts = np.asarray(points)
    homog = np.hstack([pts, np.ones((pts.shape[0], 1))])
    out = (T @ homog.T).T[:, :3]
    return out


# ----------------- Edge Computations -----------------
def compute_feature_edges(mesh, angle_deg):
    # Use face adjacency angle threshold
    _ = mesh.face_adjacency  # ensure caches
    angles = mesh.face_adjacency_angles          # (A,)
    adj_edges = mesh.face_adjacency_edges        # (A, 2)
    is_feature = angles >= np.deg2rad(angle_deg)
    return adj_edges[is_feature]


def _boundary_edges_and_face(mesh):
    """
    Compatibility helper: return (boundary_edges (B,2), boundary_face_index (B,))
    Works without edges_boundary / edges_boundary_faces by using:
      - edges_unique (E,2), edges_unique_inverse (3F,)
      - for each unique edge, count occurrences; boundary if count==1
      - record the single face that contributed that edge
    """
    # Unique edges and mapping from per-face edges -> unique edge index
    e_unique = mesh.edges_unique                    # (E, 2)
    e_unique_inv = mesh.edges_unique_inverse        # (3F,)
    F = len(mesh.faces)

    # Count how many times each unique edge occurs
    counts = np.bincount(e_unique_inv, minlength=len(e_unique))

    # We need to find, for each unique edge, one face that uses it.
    # Build a mapping: for each unique edge index, store one face index.
    edge_to_face = np.full(len(e_unique), -1, dtype=int)

    # For each face, it has 3 edges; their unique indices are at 3*i + {0,1,2}
    # Fill edge_to_face if not set.
    for fi in range(F):
        start = 3 * fi
        for k in range(3):
            ue = e_unique_inv[start + k]
            if edge_to_face[ue] == -1:
                edge_to_face[ue] = fi

    # Boundary edges are those that appear exactly once.
    boundary_mask = counts == 1
    boundary_edges = e_unique[boundary_mask]
    boundary_faces = edge_to_face[boundary_mask]
    return boundary_edges, boundary_faces


def compute_silhouette_edges_compat(mesh, front_facing):
    """
    Silhouette edges (compat):
      - interior edges with one face front-facing and the other back-facing
      - boundary edges whose single adjacent face is front-facing
    Works on older trimesh (no edges_unique_faces / edges_boundary*).
    """
    # Interior via face adjacency
    adj_edges = mesh.face_adjacency_edges   # (A,2)
    adj_faces = mesh.face_adjacency         # (A,2)
    ff0 = front_facing[adj_faces[:, 0]]
    ff1 = front_facing[adj_faces[:, 1]]
    silh_interior = adj_edges[ff0 ^ ff1]

    # Boundary via reconstruction
    boundary_edges, boundary_faces = _boundary_edges_and_face(mesh)
    silh_boundary = boundary_edges[front_facing[boundary_faces]]

    # Combine
    if len(silh_interior) == 0:
        return silh_boundary
    if len(silh_boundary) == 0:
        return silh_interior
    return np.vstack([silh_interior, silh_boundary])


# ----------------- Main -----------------
def main():
    args = parse_args()

    # Load mesh
    mesh = trimesh.load(args.mesh, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(
                g for g in mesh.dump().geometry.values() if isinstance(g, trimesh.Trimesh)
            ))
        else:
            raise ValueError("Loaded object is not a mesh or scene of meshes")
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_faces()

    # Camera pose & intrinsics
    view = look_at_matrix(args.eye, args.target, args.up)   # world->cam
    K = intrinsics_from_fov(args.fov_deg, args.width, args.height)

    # Transform verts + face normals into camera
    verts_cam = transform_points(mesh.vertices, view)
    R = view[:3, :3]
    face_normals_cam = (R @ mesh.face_normals.T).T

    # Front-facing if normal points toward camera, given our camera points +Z
    front_facing = face_normals_cam[:, 2] < 0

    # Feature edges
    feat_edges = compute_feature_edges(mesh, args.angle_deg)

    # Silhouette edges (compat)
    silh_edges = compute_silhouette_edges_compat(mesh, front_facing)

    # ---- Project edges ----
    def edges_to_uv_pairs(edges_v_idx):
        v0 = verts_cam[edges_v_idx[:, 0]]
        v1 = verts_cam[edges_v_idx[:, 1]]
        uv0 = project_points(v0, K)
        uv1 = project_points(v1, K)

        def in_bounds(uv):
            u, v = uv[:, 0], uv[:, 1]
            ok = np.isfinite(u) & np.isfinite(v)
            ok &= (u >= 0) & (u < args.width) & (v >= 0) & (v < args.height)
            return ok

        valid = in_bounds(uv0) & in_bounds(uv1)
        return uv0[valid], uv1[valid]

    uv0_feat, uv1_feat = edges_to_uv_pairs(feat_edges) if len(feat_edges) else (np.empty((0,2)), np.empty((0,2)))
    uv0_silh, uv1_silh = edges_to_uv_pairs(silh_edges) if len(silh_edges) else (np.empty((0,2)), np.empty((0,2)))

    # ---- Backdrop (light gray projected triangles) ----
    verts_img = project_points(verts_cam, K)
    faces = mesh.faces
    tri_valid = np.all(np.isfinite(verts_img[faces].reshape(-1, 2)), axis=1)
    faces_ok = faces[tri_valid.reshape(-1, 3).all(axis=1)] if faces.ndim == 2 else faces

    fig = plt.figure(figsize=(args.width / 100.0, args.height / 100.0), dpi=100)
    ax = plt.gca()
    ax.set_xlim(0, args.width)
    ax.set_ylim(args.height, 0)  # image coords
    ax.set_aspect('equal')
    ax.axis('off')

    try:
        import matplotlib.collections as mc
        tris = verts_img[faces_ok]  # (F, 3, 2)
        tri_coll = mc.PolyCollection(tris, facecolors=(0.92, 0.92, 0.92, 1.0), edgecolors='none')
        ax.add_collection(tri_coll)
    except Exception:
        pass

    # Feature edges (thin)
    for p0, p1 in zip(uv0_feat, uv1_feat):
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth=1.2, solid_capstyle='round', color='k')

    # Silhouette edges (thick)
    for p0, p1 in zip(uv0_silh, uv1_silh):
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth=2.0, solid_capstyle='round', color='k')

    plt.subplots_adjust(0, 0, 1, 1)
    fig.canvas.draw()
    fig.savefig(args.out, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    print(f"Saved: {args.out}")
    print(f"Feature edges drawn: {len(uv0_feat)}")
    print(f"Silhouette edges drawn: {len(uv0_silh)}")


if __name__ == "__main__":
    main()
