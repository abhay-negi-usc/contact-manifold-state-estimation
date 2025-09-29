#!/usr/bin/env python3
import argparse, os, shutil, subprocess, sys, tempfile, glob
from typing import List, Optional
import numpy as np
import trimesh as tm

# ---------- helpers ----------

def clean_mesh(m: tm.Trimesh) -> tm.Trimesh:
    m = m.copy()
    # replacements for deprecated methods
    m.update_faces(m.nondegenerate_faces())      # was: remove_degenerate_faces()
    m.update_faces(m.unique_faces())             # was: remove_duplicate_faces()
    m.remove_unreferenced_vertices()
    m.merge_vertices()
    m.remove_infinite_values()
    m.rezero()
    return m

def build_concave_mesh() -> tm.Trimesh:
    a = tm.creation.box(extents=(1.0, 0.5, 0.5))
    b = tm.creation.box(extents=(0.5, 1.0, 0.5))
    b.apply_translation([0.25, 0.25, 0.0])
    return clean_mesh(tm.util.concatenate([a, b]))

def vhacd_cli(mesh: tm.Trimesh,
              vhacd_exe: str = "vhacd",
              max_hulls: int = 64,
              resolution: int = 400_000,
              volume_error_percent: float = 0.5,
              depth: int = 12,
              fmt: str = "obj",
              quiet: bool = True) -> List[tm.Trimesh]:

    exe = vhacd_exe if os.path.sep in vhacd_exe else shutil.which(vhacd_exe)
    if exe is None:
        raise FileNotFoundError(f"VHACD executable not found: {vhacd_exe}")

    m = clean_mesh(mesh)

    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "input.obj")
        m.export(in_path)

        cmd = [exe, in_path,
               "-o", fmt, "-h", str(max_hulls), "-r", str(resolution),
               "-e", str(volume_error_percent), "-d", str(depth),
               "-g", "false" if quiet else "true"]

        res = subprocess.run(cmd, cwd=td, text=True,
                             stdout=(subprocess.PIPE if quiet else None),
                             stderr=(subprocess.PIPE if quiet else None))
        if res.returncode != 0:
            raise RuntimeError(f"VHACD CLI failed ({res.returncode}).\n"
                               f"stdout:\n{res.stdout or ''}\n"
                               f"stderr:\n{res.stderr or ''}")

        paths = sorted(glob.glob(os.path.join(td, "*.obj")) + glob.glob(os.path.join(td, "*.stl")))
        if not paths:
            usda = os.path.join(td, "output.usda")
            if os.path.exists(usda):
                scene = tm.load(usda)
                if hasattr(scene, "geometry") and scene.geometry:
                    return [g for g in scene.geometry.values() if isinstance(g, tm.Trimesh)]
            raise RuntimeError("No VHACD hull files found")

        parts: List[tm.Trimesh] = []
        for p in paths:
            L = tm.load(p, force="mesh")
            if isinstance(L, tm.Trimesh):
                parts.append(clean_mesh(L))
            elif hasattr(L, "geometry"):
                parts.extend([clean_mesh(g) for g in L.geometry.values() if isinstance(g, tm.Trimesh)])
        return parts

def verify(parts: List[tm.Trimesh], vol_tol: float = 1e-3) -> dict:
    stats = []
    all_convex_flag = True
    for i, P in enumerate(parts):
        # Trimesh's strict flag:
        strict_flag = bool(P.is_convex)
        # Tolerance-based convexity: volume difference vs its convex hull
        try:
            ch = P.convex_hull
            v_ch = float(ch.volume) if ch.is_volume else 0.0
            v_p  = float(P.volume)  if P.is_volume  else 0.0
            rel_gap = abs(v_ch - v_p) / (v_ch + 1e-12) if v_ch > 0 else 0.0
            tol_flag = (rel_gap <= vol_tol)
        except Exception:
            tol_flag = False
            rel_gap  = float("inf")
        all_convex_flag &= tol_flag
        stats.append({
            "tris": int(P.faces.shape[0]),
            "verts": int(P.vertices.shape[0]),
            "strict_is_convex": strict_flag,
            "rel_volume_gap": rel_gap
        })
    return {"num_parts": len(parts), "all_convex_tol": all_convex_flag, "per_part": stats}

def maybe_convexify(parts: List[tm.Trimesh]) -> List[tm.Trimesh]:
    # If you *require* guaranteed convex parts for physics, take convex hulls:
    return [p.convex_hull if not p.is_convex else p for p in parts]

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser("VHACD CLI self-test with tolerant convexity check")
    ap.add_argument("--vhacd", default="vhacd", help="Path or name of VHACD executable")
    ap.add_argument("--mesh", default=None, help="Optional input mesh file to decompose")
    ap.add_argument("--dump", default=None, help="Optional directory to dump resulting hulls")
    ap.add_argument("--resolution", type=int, default=600_000)
    ap.add_argument("--max-hulls", type=int, default=2**10)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--error", type=float, default=0.5)      # percent
    ap.add_argument("--vol-tol", type=float, default=1e-3, help="Relative volume tolerance for convexity check")
    ap.add_argument("--convexify", action="store_true", help="Replace parts with their convex hulls before checking")
    ap.add_argument("--no-quiet", action="store_true", help="Show VHACD stdout/stderr")
    args = ap.parse_args()

    if args.mesh:
        mesh = tm.load(args.mesh, force="mesh")
        if not isinstance(mesh, tm.Trimesh) and hasattr(mesh, "geometry"):
            mesh = tm.util.concatenate([g for g in mesh.geometry.values() if isinstance(g, tm.Trimesh)])
    else:
        mesh = build_concave_mesh()

    mesh = clean_mesh(mesh)
    print(f"[INFO] Input: {len(mesh.vertices)} verts, {len(mesh.faces)} tris, "
          f"bounds={np.round(mesh.bounds, 3).tolist()}")

    parts = vhacd_cli(mesh,
                      vhacd_exe=args.vhacd,
                      max_hulls=args.max_hulls,
                      resolution=args.resolution,
                      volume_error_percent=args.error,
                      depth=args.depth,
                      fmt="obj",
                      quiet=not args.no_quiet)

    if args.convexify:
        parts = maybe_convexify(parts)

    rep = verify(parts, vol_tol=args.vol_tol)
    print(f"[INFO] VHACD parts: {rep['num_parts']}, all_convex_tol={rep['all_convex_tol']}")
    for i, s in enumerate(rep["per_part"]):
        print(f"  - part[{i}]: tris={s['tris']:4d}, verts={s['verts']:4d}, "
              f"strict={s['strict_is_convex']}, rel_gap={s['rel_volume_gap']:.3e}")

    if args.dump:
        os.makedirs(args.dump, exist_ok=True)
        for i, p in enumerate(parts):
            p.export(os.path.join(args.dump, f"hull_{i:03d}.obj"))
        print(f"[INFO] Wrote {len(parts)} hulls to: {args.dump}")

    # pass/fail: at least 1 part and all convex under tolerance
    if rep["num_parts"] >= 1 and rep["all_convex_tol"]:
        print("[PASS] VHACD CLI working; hulls convex within tolerance.")
        sys.exit(0)
    else:
        print("[FAIL] Some hulls not convex within tolerance.")
        sys.exit(4)

if __name__ == "__main__":
    main()
