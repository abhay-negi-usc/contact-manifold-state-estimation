#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import trimesh
from trimesh.ray.ray_triangle import RayMeshIntersector

# =======================
# User config (edit me)
# =======================
CONFIG = {
    # If None, uses a unit cube. Otherwise path to mesh (obj/ply/stl/glb…)
    "mesh_path": "./data_generation/assets/meshes/extrusion_hole/extrusion_hole.obj",

    # Camera & image
    "camera": {
        "eye":    [0.5, 0.0, 0.5],
        "target": [0.0, 0.0, 0.0],
        "up":     [0.0, 0.0, 1.0],
        "fov_deg": 45.0,       # perspective FOV (controls parallax)
    },
    "image": {
        "width":  1024,
        "height": 768,
        "out":    "edges_depth.png"
    },

    # Feature (crease) edges
    "feature": {
        "enable": True,
        "angle_deg": 30.0,     # dihedral >= this is a feature
        "samples_per_edge": 32,
        "line_width": 1.4
    },

    # Silhouette edges (view-dependent + boundary)
    "silhouette": {
        "enable": True,
        "samples_per_edge": 48,
        "line_width": 2.5
    },

    # Appearance
    "backdrop_gray": 0.92,     # 0.0=black, 1.0=white
    "eps_ratio": 1e-4,         # tolerance for depth test (fraction of distance)
}


# =======================
# Math / camera utils
# =======================
def look_at_matrix(eye, target, up):
    """
    Build world->camera matrix with optical axis along +Z.
    Points in front of camera have z_cam > 0.
    """
    eye = np.asarray(eye, float)
    target = np.asarray(target, float)
    up = np.asarray(up, float)

    forward = target - eye
    forward /= np.linalg.norm(forward)

    right = np.cross(forward, up)
    right /= np.linalg.norm(right)

    true_up = np.cross(right, forward)

    R = np.stack([right, true_up, forward], axis=0)  # rows are camera axes
    t = -R @ eye
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def intrinsics_from_fov(fov_y_deg, width, height):
    fov_y = np.deg2rad(fov_y_deg)
    fy = (height / 2.0) / np.tan(fov_y / 2.0)
    fx = fy
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]], dtype=float)


def transform_points(points, T):
    pts = np.asarray(points, float)
    homog = np.hstack([pts, np.ones((pts.shape[0], 1))])
    out = (T @ homog.T).T[:, :3]
    return out


def project_points_persp(points_cam, K):
    pts = np.asarray(points_cam, float)
    uv = np.full((pts.shape[0], 2), np.nan, float)
    z = pts[:, 2]
    valid = z > 1e-8
    if np.any(valid):
        x = pts[valid, 0] / z[valid]
        y = pts[valid, 1] / z[valid]
        homog = np.stack([x, y, np.ones_like(x)], axis=0)
        img = (K @ homog).T[:, :2]
        uv[valid] = img
    return uv


# =======================
# Edge extraction
# =======================
def compute_feature_edges(mesh, angle_deg=30.0):
    """
    Feature edges: dihedral angle between adjacent faces >= angle_deg.
    """
    _ = mesh.face_adjacency  # warm caches
    angles = mesh.face_adjacency_angles          # (A,)
    adj_edges = mesh.face_adjacency_edges        # (A,2)
    is_feature = angles >= np.deg2rad(angle_deg)
    return adj_edges[is_feature]


def _boundary_edges_and_face(mesh):
    """
    Compat for older trimesh: find boundary edges and one adjacent face index.
    """
    e_unique = mesh.edges_unique                # (E,2)
    e_unique_inv = mesh.edges_unique_inverse    # (3F,)
    F = len(mesh.faces)

    counts = np.bincount(e_unique_inv, minlength=len(e_unique))
    edge_to_face = np.full(len(e_unique), -1, int)

    for fi in range(F):
        idx0 = 3 * fi
        for k in range(3):
            ue = e_unique_inv[idx0 + k]
            if edge_to_face[ue] == -1:
                edge_to_face[ue] = fi

    boundary_mask = counts == 1
    return e_unique[boundary_mask], edge_to_face[boundary_mask]


def compute_silhouette_edges(mesh, front_facing):
    """
    Silhouettes = interior edges with one face front-facing and the other back-facing,
                  plus boundary edges whose single face is front-facing.
    """
    adj_edges = mesh.face_adjacency_edges   # (A,2)
    adj_faces = mesh.face_adjacency         # (A,2)
    ff0 = front_facing[adj_faces[:, 0]]
    ff1 = front_facing[adj_faces[:, 1]]
    silh_interior = adj_edges[ff0 ^ ff1]

    be, bf = _boundary_edges_and_face(mesh)
    silh_boundary = be[front_facing[bf]]

    if len(silh_interior) == 0:
        return silh_boundary
    if len(silh_boundary) == 0:
        return silh_interior
    return np.vstack([silh_interior, silh_boundary])


# =======================
# Depth-tested edge drawing
# =======================
def draw_edges_depth_tested(mesh_cam, rmi, edges, K, width, height, ax,
                            samples_per_edge=32, eps_ratio=1e-4, line_w=2.0):
    """
    Draw only *visible* portions of edges using ray casting in camera frame.
    - mesh_cam: Trimesh whose vertices are in camera coords
    - rmi: RayMeshIntersector(mesh_cam)
    """
    if edges is None or len(edges) == 0:
        return

    verts_cam = mesh_cam.vertices.view(np.ndarray)
    for (i0, i1) in edges:
        a_cam = verts_cam[i0]
        b_cam = verts_cam[i1]

        # Sample points along edge in 3D (camera frame)
        ts = np.linspace(0.0, 1.0, samples_per_edge)
        pts = (1 - ts)[:, None] * a_cam + ts[:, None] * b_cam  # (S,3)

        # Rays from camera origin through pts
        dirs = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        origins = np.zeros_like(pts)
        origins += dirs * 1e-6  # tiny bias to avoid self-hit jitter

        # Intersect (first hit)
        locs, idx_ray, _ = rmi.intersects_location(origins, dirs, multiple_hits=False)

        visible = np.zeros(len(ts), dtype=bool)
        if len(idx_ray):
            t_hit = np.linalg.norm(locs - origins[idx_ray], axis=1)
            t_pt  = np.linalg.norm(pts[idx_ray]   - origins[idx_ray], axis=1)
            eps = np.maximum(1.0, t_pt) * eps_ratio
            visible[idx_ray] = t_pt <= t_hit + eps

        # Project sampled points to image
        uvs = project_points_persp(pts, K)

        # Connect consecutive visible samples in image space
        for j in range(len(ts) - 1):
            if not (visible[j] and visible[j + 1]):
                continue
            u0, v0 = uvs[j]
            u1, v1 = uvs[j + 1]
            if (np.isfinite([u0, v0, u1, v1]).all() and
                0 <= u0 < width and 0 <= v0 < height and
                0 <= u1 < width and 0 <= v1 < height):
                ax.plot([u0, u1], [v0, v1],
                        linewidth=line_w, solid_capstyle='round', color='k')


# =======================
# Rendering
# =======================
def render_depth_edges(config):
    # Load mesh
    if config["mesh_path"] is None:
        mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    else:
        mesh = trimesh.load(config["mesh_path"], force='mesh')
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(
                g for g in mesh.dump().geometry.values() if isinstance(g, trimesh.Trimesh)
            ))
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_faces()

    # Camera & intrinsics
    cam = config["camera"]
    img = config["image"]
    eye, target, up = cam["eye"], cam["target"], cam["up"]
    width, height, fov_deg = img["width"], img["height"], cam["fov_deg"]

    view = look_at_matrix(eye, target, up)     # world->camera
    K = intrinsics_from_fov(fov_deg, width, height)

    # Transform mesh to camera frame once; build intersector
    verts_cam = transform_points(mesh.vertices, view)
    mesh_cam = trimesh.Trimesh(vertices=verts_cam, faces=mesh.faces, process=False)
    rmi = RayMeshIntersector(mesh_cam)

    # Compute front-facing flags for silhouettes (robust test)
    R = view[:3, :3]
    face_normals_cam = (R @ mesh.face_normals.T).T
    face_centers_cam = transform_points(mesh.triangles_center, view)
    front_facing = np.einsum("ij,ij->i", face_normals_cam, face_centers_cam) < 0.0

    # Edges
    edges_silh = compute_silhouette_edges(mesh, front_facing) if config["silhouette"]["enable"] else np.empty((0, 2), int)
    edges_feat = compute_feature_edges(mesh, config["feature"]["angle_deg"]) if config["feature"]["enable"] else np.empty((0, 2), int)

    # Prepare figure
    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    ax = plt.gca()
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # image coords
    ax.set_aspect('equal')
    ax.axis('off')

    # Backdrop (light gray projected triangles)
    verts_img = project_points_persp(verts_cam, K)
    faces = mesh.faces
    try:
        import matplotlib.collections as mc
        # filter triangles with valid projections
        tri_ok = np.all(np.isfinite(verts_img[faces].reshape(-1, 2)), axis=1)
        faces_ok = faces[tri_ok.reshape(-1, 3).all(axis=1)] if faces.ndim == 2 else faces
        tris = verts_img[faces_ok]
        g = config["backdrop_gray"]
        tri_coll = mc.PolyCollection(tris, facecolors=(g, g, g, 1.0), edgecolors='none')
        ax.add_collection(tri_coll)
    except Exception:
        pass

    # Draw depth-tested silhouettes (thicker), then features (thinner)
    if len(edges_silh):
        draw_edges_depth_tested(
            mesh_cam, rmi, edges_silh, K, width, height, ax,
            samples_per_edge=config["silhouette"]["samples_per_edge"],
            eps_ratio=config["eps_ratio"],
            line_w=config["silhouette"]["line_width"]
        )
    if len(edges_feat):
        draw_edges_depth_tested(
            mesh_cam, rmi, edges_feat, K, width, height, ax,
            samples_per_edge=config["feature"]["samples_per_edge"],
            eps_ratio=config["eps_ratio"],
            line_w=config["feature"]["line_width"]
        )

    # Save
    out_path = img["out"]
    plt.subplots_adjust(0, 0, 1, 1)
    fig.canvas.draw()
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    print(f"Saved {out_path}")
    print(f"Silhouette edges: {len(edges_silh)}   Feature edges: {len(edges_feat)}")


if __name__ == "__main__":
    render_depth_edges(CONFIG)
