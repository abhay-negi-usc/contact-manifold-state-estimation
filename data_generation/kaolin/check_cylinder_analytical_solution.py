#!/usr/bin/env python3
"""
Cylindrical peg-in-hole contact validation: SIDE, END-FACE, and RIM/EDGE contact.

Input CSV columns:
  tx,ty,tz,rx_deg,ry_deg,rz_deg

RPY convention (matches your torch code):
  R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
  roll=rx_deg, pitch=ry_deg, yaw=rz_deg

Contact models checked:
  1) SIDE contact: dist(axis_peg, axis_hole) == c = Rh - Rp
  2) END-FACE contact at each hole end plane: b·(p_end - p_plane) == 0 with fit check
  3) RIM/EDGE contact at each hole rim circle: min_theta dist(point_on_rim_circle, peg_axis) == Rp

Outputs:
  - Per-pose residuals and chosen mode
  - Stats per chosen mode (and optionally per-candidate mode)

Dependencies:
  pip install numpy pandas tqdm
"""

from __future__ import annotations
import math
import os
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# -----------------------------
# CONFIG
# -----------------------------
CONFIG: Dict = {
    "input_csv": "./data_generation/data/cylinder_simple/cylinder_simple_contact_poses.csv",

    # Optional contact flag filter
    "contact_flag_column": "contact_valid",  # "contact_valid" or set to None to disable
    "contact_flag_value": 1,

    # Sampling
    "max_solutions": 10_000,     # e.g. 100000 or None for all
    "random_seed": 7,

    # Geometry (units must match tx,ty,tz)
    "geometry": {
        "hole_radius": 0.011,          # Rh
        "peg_radius": 0.010,           # Rp

        # Hole axis line: p_h + s*b (world frame)
        "hole_axis_point": [0.0, 0.0, 0.0],
        "hole_axis_dir":   [0.0, 0.0, 1.0],

        # Hole end planes are at axis coordinates h0 and h1 along b:
        # plane point = p_h + h*b, plane normal = +/-b
        "hole_h0": 0.025,                 # mouth plane coordinate
        "hole_h1": 0.0,                # bottom plane coordinate (set your hole depth)

        # Peg axis in body frame
        "peg_axis_body": [0.0, 0.0, 1.0],

        # Peg end locations along peg axis in BODY coordinates.
        # If your body origin is at peg center, use +/- peg_length/2.
        # If origin is at one end, set accordingly.
        "peg_end_z_body": [0.0, 0.025],   
    },

    # Thresholds / tolerances
    "thresholds": {
        # mode pass/fail tolerances (meters)
        "side_abs_tol_m": 0.5e-3,
        "end_abs_tol_m": 0.5e-3,
        "rim_abs_tol_m": 0.5e-3,

        # Fit check for end-face: require peg disk fits within hole at that plane
        # i.e. radial offset <= (Rh - Rp) + end_fit_slack
        "end_fit_slack_m": 0.0,

        # near-parallel classification diagnostics only
        "parallel_sin_theta_eps": 1e-3,

        # rotation diagnostics (deg)
        "tilt_tol_deg": 0.5,
    },

    # Rim-circle distance minimization
    "rim_minimization": {
        "num_theta_samples": 720,   # higher = more accurate, still fast
    },

    # Output
    "save_csv_with_residuals": True,
    "output_csv": "pose_contact_check_side_end_rim.csv",
}


# -----------------------------
# Utilities
# -----------------------------
def _as_np3(v) -> np.ndarray:
    return np.asarray(v, dtype=np.float64).reshape(3)

def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return v * 0.0
    return v / n

def project_perp(v: np.ndarray, axis_unit: np.ndarray) -> np.ndarray:
    return v - np.dot(v, axis_unit) * axis_unit

def Rx(roll: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0,  cr, -sr],
                     [0.0,  sr,  cr]], dtype=np.float64)

def Ry(pitch: float) -> np.ndarray:
    cp, sp = math.cos(pitch), math.sin(pitch)
    return np.array([[ cp, 0.0, sp],
                     [0.0, 1.0, 0.0],
                     [-sp, 0.0, cp]], dtype=np.float64)

def Rz(yaw: float) -> np.ndarray:
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([[cy, -sy, 0.0],
                     [sy,  cy, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)

def rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Matches your torch code: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    return Rz(yaw) @ Ry(pitch) @ Rx(roll)

def line_line_distance(p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray,
                       parallel_eps: float = 1e-12) -> float:
    d1u = normalize(d1)
    d2u = normalize(d2)
    n = np.cross(d1u, d2u)
    nn = np.linalg.norm(n)
    w0 = p1 - p2
    if nn > parallel_eps:
        return abs(np.dot(w0, n)) / nn
    # parallel
    perp = w0 - np.dot(w0, d2u) * d2u
    return float(np.linalg.norm(perp))

def orthonormal_basis_perp(b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return u,v orthonormal spanning the plane perpendicular to unit b."""
    b = normalize(b)
    # pick a vector not parallel to b
    a = np.array([1.0, 0.0, 0.0], dtype=np.float64) if abs(b[0]) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = normalize(np.cross(b, a))
    v = normalize(np.cross(b, u))
    return u, v

def point_line_distance(x: np.ndarray, p: np.ndarray, d: np.ndarray) -> float:
    """Distance from point x to line p + s d."""
    d = normalize(d)
    w = x - p
    perp = w - np.dot(w, d) * d
    return float(np.linalg.norm(perp))


# -----------------------------
# Residuals
# -----------------------------
@dataclass
class ModeResiduals:
    # Side
    r_side: float

    # End-face (best over ends)
    r_end_best: float
    end_which: str
    end_fit_ok: bool
    end_radial_offset: float

    # Rim (best over rims)
    r_rim_best: float
    rim_which: str

    # Diagnostics
    tilt_deg: float
    sin_tilt: float


def compute_side_residual(t: np.ndarray, a: np.ndarray, p_h: np.ndarray, b: np.ndarray, clearance: float) -> float:
    d = line_line_distance(t, a, p_h, b)
    return d - clearance


def compute_end_face_residuals(
    t: np.ndarray, a: np.ndarray,
    p_h: np.ndarray, b: np.ndarray,
    hole_h: float,
    peg_end_center: np.ndarray,
    Rh: float, Rp: float,
    end_fit_slack: float
) -> Tuple[float, bool, float]:
    """
    End-face contact at hole plane (axis coordinate hole_h):
      residual = signed distance of peg end center to hole plane along b
      fit check: radial offset of peg axis at that plane <= Rh - Rp + slack
    """
    b = normalize(b)
    plane_point = p_h + hole_h * b
    # signed distance of peg end center to plane
    signed = float(np.dot(b, (peg_end_center - plane_point)))
    r_plane = signed  # want 0

    # radial offset of peg axis from hole axis at this plane:
    # find point on peg axis whose b-coordinate equals hole_h
    denom = float(np.dot(b, a))
    if abs(denom) < 1e-12:
        # peg axis almost perpendicular to hole axis -> undefined; treat as failed fit
        return r_plane, False, float("inf")

    s = (hole_h - float(np.dot(b, (t - p_h)))) / denom
    x_axis_at_plane = t + s * a

    radial_vec = project_perp(x_axis_at_plane - plane_point, b)
    rho = float(np.linalg.norm(radial_vec))  # centerline offset at that plane

    fit_ok = rho <= (Rh - Rp + end_fit_slack)
    return r_plane, fit_ok, rho


def compute_rim_residual(
    t: np.ndarray, a: np.ndarray,
    p_h: np.ndarray, b: np.ndarray,
    hole_h: float,
    Rh: float, Rp: float,
    num_theta: int
) -> float:
    """
    Rim circle at hole plane (axis coordinate hole_h), radius Rh.
    Rim contact with peg side occurs when:
      min_theta distance(point_on_rim(theta), peg_axis_line) == Rp

    residual = min_theta dist(point_on_rim, peg_axis) - Rp
    """
    b = normalize(b)
    u, v = orthonormal_basis_perp(b)
    center = p_h + hole_h * b

    thetas = np.linspace(0.0, 2.0 * math.pi, num_theta, endpoint=False)
    # Evaluate min distance from rim points to peg axis
    min_dist = float("inf")
    for th in thetas:
        x = center + Rh * (math.cos(th) * u + math.sin(th) * v)
        d = point_line_distance(x, t, a)
        if d < min_dist:
            min_dist = d

    return min_dist - Rp


def classify_pose(
    R: np.ndarray, t: np.ndarray,
    geom: Dict, thr: Dict, rim_cfg: Dict
) -> Tuple[str, float, Dict]:
    """
    Compute residuals for SIDE, END (at h0/h1), RIM (at h0/h1) and pick best feasible mode.

    Returns: (mode_name, natural_error_abs_m, details_dict)
    """
    Rh = float(geom["hole_radius"])
    Rp = float(geom["peg_radius"])
    c = Rh - Rp

    p_h = _as_np3(geom["hole_axis_point"])
    b = normalize(_as_np3(geom["hole_axis_dir"]))
    a = normalize(R @ _as_np3(geom["peg_axis_body"]))

    # tilt diagnostics
    cos_th = float(np.clip(np.dot(a, b), -1.0, 1.0))
    tilt_deg = math.degrees(math.acos(cos_th))
    sin_tilt = math.sqrt(max(0.0, 1.0 - cos_th * cos_th))

    # SIDE
    r_side = compute_side_residual(t, a, p_h, b, c)

    # PEG ends
    peg_end_z = geom["peg_end_z_body"]
    peg_end_centers = [t + z * a for z in peg_end_z]

    # END contact at hole_h0 / hole_h1, best over peg ends for each plane
    hole_h0 = float(geom["hole_h0"])
    hole_h1 = float(geom["hole_h1"])
    end_fit_slack = float(thr["end_fit_slack_m"])

    def best_end_for_plane(hole_h: float, tag: str) -> Tuple[float, str, bool, float]:
        best = (float("inf"), "none", False, float("inf"))
        for k, p_end in enumerate(peg_end_centers):
            r_plane, fit_ok, rho = compute_end_face_residuals(
                t, a, p_h, b, hole_h, p_end, Rh, Rp, end_fit_slack
            )
            if abs(r_plane) < abs(best[0]):
                best = (r_plane, f"{tag}_pegEnd{k}", fit_ok, rho)
        return best

    r_end0, end0_tag, end0_fit_ok, end0_rho = best_end_for_plane(hole_h0, "holeH0")
    r_end1, end1_tag, end1_fit_ok, end1_rho = best_end_for_plane(hole_h1, "holeH1")

    # pick best end plane residual among the two planes, but require fit_ok
    end_candidates = []
    end_candidates.append(("end_h0", r_end0, end0_tag, end0_fit_ok, end0_rho))
    end_candidates.append(("end_h1", r_end1, end1_tag, end1_fit_ok, end1_rho))

    # RIM contact at each plane
    num_theta = int(rim_cfg["num_theta_samples"])
    r_rim0 = compute_rim_residual(t, a, p_h, b, hole_h0, Rh, Rp, num_theta)
    r_rim1 = compute_rim_residual(t, a, p_h, b, hole_h1, Rh, Rp, num_theta)

    # Choose best feasible mode:
    # - SIDE always feasible as a pure geometric model
    # - END only feasible if fit_ok
    # - RIM always feasible in geometric sense (but may be irrelevant if far away)
    candidates = []

    candidates.append(("side", abs(r_side), {"r_side": r_side}))

    # end
    for name, r, tag, fit_ok, rho in end_candidates:
        if fit_ok:
            candidates.append((name, abs(r), {"r_end": r, "end_tag": tag, "end_rho": rho, "end_fit_ok": fit_ok}))
        else:
            # still log it, but don't let it win classification
            pass

    # rim
    candidates.append(("rim_h0", abs(r_rim0), {"r_rim": r_rim0}))
    candidates.append(("rim_h1", abs(r_rim1), {"r_rim": r_rim1}))

    # choose the smallest absolute residual
    mode, err_abs, mode_details = min(candidates, key=lambda x: x[1])

    details = {
        "tilt_deg": tilt_deg,
        "sin_tilt": sin_tilt,

        "r_side": r_side,

        "r_end_h0": r_end0,
        "r_end_h0_tag": end0_tag,
        "end_h0_fit_ok": end0_fit_ok,
        "end_h0_rho": end0_rho,

        "r_end_h1": r_end1,
        "r_end_h1_tag": end1_tag,
        "end_h1_fit_ok": end1_fit_ok,
        "end_h1_rho": end1_rho,

        "r_rim_h0": r_rim0,
        "r_rim_h1": r_rim1,
    }
    details.update(mode_details)

    return mode, err_abs, details


# -----------------------------
# Stats
# -----------------------------
def scalar_stats(x: np.ndarray) -> Dict[str, float]:
    if x.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "median": float("nan"),
                "mad": float("nan"), "p95": float("nan"), "max": float("nan")}
    med = float(np.median(x))
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "median": med,
        "mad": float(np.median(np.abs(x - med))),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
    }

def fmt_stats(name: str, st: Dict[str, float]) -> str:
    return (f"{name}: n={st['n']} mean={st['mean']:.6g} std={st['std']:.6g} "
            f"med={st['median']:.6g} mad={st['mad']:.6g} p95={st['p95']:.6g} max={st['max']:.6g}")

def pass_rate(abs_res: np.ndarray, tol: float) -> float:
    if abs_res.size == 0:
        return float("nan")
    return float(np.mean(abs_res <= tol))


# -----------------------------
# Load / main
# -----------------------------
def load_and_sample(cfg: Dict) -> pd.DataFrame:
    path = cfg["input_csv"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input CSV not found: {path}")

    df = pd.read_csv(path)
    required = ["tx", "ty", "tz", "rx_deg", "ry_deg", "rz_deg"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nFound columns: {list(df.columns)}")

    flag_col = cfg.get("contact_flag_column", None)
    if flag_col is not None and flag_col in df.columns:
        df = df[df[flag_col] == cfg.get("contact_flag_value", 1)].copy()

    max_n = cfg.get("max_solutions", None)
    if max_n is not None:
        max_n = int(max_n)
        if len(df) > max_n:
            rng = np.random.default_rng(int(cfg.get("random_seed", 0)))
            idx = rng.choice(len(df), size=max_n, replace=False)
            df = df.iloc[idx].copy()

    df.reset_index(drop=True, inplace=True)
    return df


def main(cfg: Dict) -> None:
    geom = cfg["geometry"]
    Rh = float(geom["hole_radius"])
    Rp = float(geom["peg_radius"])
    if Rh <= Rp:
        raise ValueError(f"Need Rh > Rp for clearance; got Rh={Rh}, Rp={Rp}")
    print("=== Peg-in-hole contact check: SIDE + END + RIM ===")
    print(f"Input: {cfg['input_csv']}")
    print(f"Radii: Rh={Rh}, Rp={Rp}, clearance c={Rh-Rp}")
    print(f"Hole h0={geom['hole_h0']}  h1={geom['hole_h1']}")
    print(f"Peg end z (body): {geom['peg_end_z_body']}")
    print(f"Rim theta samples: {cfg['rim_minimization']['num_theta_samples']}")

    df = load_and_sample(cfg)
    print(f"Checking N={len(df)} poses")

    thr = cfg["thresholds"]
    rim_cfg = cfg["rim_minimization"]

    rows: List[Dict] = []
    for i in tqdm(range(len(df)), desc="Checking poses"):
        row = df.iloc[i]

        t = np.array([row["tx"], row["ty"], row["tz"]], dtype=np.float64)

        roll = math.radians(float(row["rx_deg"]))
        pitch = math.radians(float(row["ry_deg"]))
        yaw = math.radians(float(row["rz_deg"]))
        R = rpy_to_R(roll, pitch, yaw)

        mode, err_abs, details = classify_pose(R, t, geom, thr, rim_cfg)

        rows.append({
            "mode": mode,
            "mode_abs_err_m": err_abs,
            "tilt_deg": details["tilt_deg"],

            # SIDE
            "r_side": details["r_side"],
            "abs_r_side": abs(details["r_side"]),

            # END
            "r_end_h0": details["r_end_h0"],
            "abs_r_end_h0": abs(details["r_end_h0"]),
            "end_h0_fit_ok": details["end_h0_fit_ok"],
            "end_h0_rho": details["end_h0_rho"],

            "r_end_h1": details["r_end_h1"],
            "abs_r_end_h1": abs(details["r_end_h1"]),
            "end_h1_fit_ok": details["end_h1_fit_ok"],
            "end_h1_rho": details["end_h1_rho"],

            # RIM
            "r_rim_h0": details["r_rim_h0"],
            "abs_r_rim_h0": abs(details["r_rim_h0"]),
            "r_rim_h1": details["r_rim_h1"],
            "abs_r_rim_h1": abs(details["r_rim_h1"]),
        })

    res = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

    # Report per chosen mode
    print("\n=== Stats per chosen MODE ===")
    for mode in sorted(res["mode"].unique().tolist()):
        sub = res[res["mode"] == mode]
        print(f"\n--- mode={mode} (n={len(sub)}) ---")

        # natural error for that mode = mode_abs_err_m
        st_mode = scalar_stats(sub["mode_abs_err_m"].to_numpy(dtype=np.float64))
        print(fmt_stats("mode_abs_err_m (m)", st_mode))

        # side residual (useful even if mode not side)
        st_side = scalar_stats(sub["abs_r_side"].to_numpy(dtype=np.float64))
        print(fmt_stats("|r_side| (m)", st_side), f"pass@{thr['side_abs_tol_m']}: {pass_rate(sub['abs_r_side'].to_numpy(), thr['side_abs_tol_m']):.3f}")

        # end residuals
        st_end0 = scalar_stats(sub["abs_r_end_h0"].to_numpy(dtype=np.float64))
        st_end1 = scalar_stats(sub["abs_r_end_h1"].to_numpy(dtype=np.float64))
        print(fmt_stats("|r_end_h0| (m)", st_end0), f"pass@{thr['end_abs_tol_m']}: {pass_rate(sub['abs_r_end_h0'].to_numpy(), thr['end_abs_tol_m']):.3f}")
        print(fmt_stats("|r_end_h1| (m)", st_end1), f"pass@{thr['end_abs_tol_m']}: {pass_rate(sub['abs_r_end_h1'].to_numpy(), thr['end_abs_tol_m']):.3f}")

        # rim residuals
        st_rim0 = scalar_stats(sub["abs_r_rim_h0"].to_numpy(dtype=np.float64))
        st_rim1 = scalar_stats(sub["abs_r_rim_h1"].to_numpy(dtype=np.float64))
        print(fmt_stats("|r_rim_h0| (m)", st_rim0), f"pass@{thr['rim_abs_tol_m']}: {pass_rate(sub['abs_r_rim_h0'].to_numpy(), thr['rim_abs_tol_m']):.3f}")
        print(fmt_stats("|r_rim_h1| (m)", st_rim1), f"pass@{thr['rim_abs_tol_m']}: {pass_rate(sub['abs_r_rim_h1'].to_numpy(), thr['rim_abs_tol_m']):.3f}")

        # rotation
        st_tilt = scalar_stats(sub["tilt_deg"].to_numpy(dtype=np.float64))
        print(fmt_stats("tilt_deg (deg)", st_tilt), f"pass@{thr['tilt_tol_deg']}: {pass_rate(sub['tilt_deg'].to_numpy(), thr['tilt_tol_deg']):.3f}")

    # Overall
    print("\n=== OVERALL ===")
    st_all = scalar_stats(res["mode_abs_err_m"].to_numpy(dtype=np.float64))
    print(fmt_stats("mode_abs_err_m (m)", st_all))

    if cfg.get("save_csv_with_residuals", True):
        out_path = cfg.get("output_csv", "pose_contact_check_side_end_rim.csv")
        res.to_csv(out_path, index=False)
        print(f"\nSaved detailed results to: {out_path}")


if __name__ == "__main__":
    main(CONFIG)
