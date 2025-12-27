#!/usr/bin/env python3
"""
Summarize sampling outputs (CSV or NPZ) from kaolin/contact sampling (NO argparse).

Outputs:
- total poses
- contact poses (contact_valid)
- coarse/fine counts overall
- pose stats (min/max/mean/median/std) for:
    * all poses
    * contact poses (all contact)
    * contact+coarse
    * contact+fine
- 2x3 seaborn histogram figure (tx,ty,tz,rx,ry,rz):
    overlays contact-all vs contact-coarse vs contact-fine
"""

import os
import sys
from typing import Dict, Any, Tuple

import numpy as np

# Plot deps (requested)
import matplotlib
matplotlib.use("Agg")  # NO GUI, NO Qt
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================
# USER CONFIG
# ==========================

DATA_PATH = "/home/rp/abhay_ws/contact-manifold-state-estimation/data_generation/data/cylinder_simple/cylinder_simple_contact_poses.csv"  

# histogram options
HIST_BINS = 60
HIST_STAT = "density"   # "count" or "probability" or "density"
HIST_COMMON_NORM = False  # keep each distribution independently normalized
SAVE_FIG = True

POSE_NAMES = ["tx", "ty", "tz", "rx_deg", "ry_deg", "rz_deg"]
FIG_NAME_SUFFIX = "contact_hist_2x3.png"


# ==========================
# UTILS
# ==========================

def _fmt(x: float) -> str:
    if np.isnan(x):
        return "nan"
    if abs(x) >= 1e4 or (abs(x) > 0 and abs(x) < 1e-3):
        return f"{x:.6e}"
    return f"{x:.6f}"


def stats_1d(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return dict(min=np.nan, max=np.nan, mean=np.nan, median=np.nan, std=np.nan)
    return dict(
        min=float(np.min(x)),
        max=float(np.max(x)),
        mean=float(np.mean(x)),
        median=float(np.median(x)),
        std=float(np.std(x, ddof=0)),
    )


def print_pose_stats_block(title: str, X: np.ndarray) -> None:
    """
    X: (N,6) pose array
    """
    print(f"\n== {title} ==")
    if X.size == 0 or X.shape[0] == 0:
        print("  (no samples)")
        return

    for j, nm in enumerate(POSE_NAMES):
        s = stats_1d(X[:, j])
        print(
            f"  {nm:>6s}: "
            f"min={_fmt(s['min'])}  "
            f"max={_fmt(s['max'])}  "
            f"mean={_fmt(s['mean'])}  "
            f"median={_fmt(s['median'])}  "
            f"std={_fmt(s['std'])}"
        )


def _as_bool_mask(x: Any, n: int) -> np.ndarray:
    arr = np.asarray(x)
    if arr.shape[0] != n:
        raise ValueError(f"Mask length mismatch: {arr.shape[0]} vs {n}")
    if arr.dtype == np.bool_:
        return arr
    return arr.astype(np.int64) != 0


def _as_fidelity_str(fid: Any, n: int) -> np.ndarray:
    fid_arr = np.asarray(fid, dtype=object)
    if fid_arr.shape[0] != n:
        raise ValueError(f"Fidelity length mismatch: {fid_arr.shape[0]} vs {n}")
    return np.array([str(v) for v in fid_arr], dtype=object)


# ==========================
# LOADERS
# ==========================


def _normalize_vec3(x: Any, name: str) -> np.ndarray:
    """
    Accepts either:
      - numeric array of shape (N,3)
      - object array of shape (N,) where each entry is array-like (3,)
    Returns float64 array of shape (N,3)
    """
    x = np.asarray(x)

    # Case 1: already (N,3)
    if x.ndim == 2 and x.shape[1] == 3 and x.dtype != object:
        return x.astype(np.float64, copy=False)

    # Case 2: object array of vectors (N,)
    if x.ndim == 1:
        rows = []
        for i, v in enumerate(x):
            vv = np.asarray(v, dtype=np.float64).reshape(-1)
            if vv.size != 3:
                raise ValueError(f"{name}[{i}] has size {vv.size}, expected 3. Value type={type(v)}")
            rows.append(vv)
        return np.vstack(rows).astype(np.float64, copy=False)

    # Anything else is unsupported
    raise ValueError(f"Unsupported {name} shape/dtype: shape={x.shape}, dtype={x.dtype}")


def load_npz(path: str) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    keys = set(data.files)

    if "t" not in keys or "r_deg" not in keys:
        raise ValueError(f"NPZ must contain 't' and 'r_deg'. Found: {sorted(keys)}")

    # *** Robustly normalize to (N,3) float arrays ***
    t = _normalize_vec3(data["t"], "t")
    r = _normalize_vec3(data["r_deg"], "r_deg")

    if t.shape[0] != r.shape[0]:
        raise ValueError(f"t and r_deg length mismatch: {t.shape[0]} vs {r.shape[0]}")

    pose = np.concatenate([t, r], axis=1)  # (N,6) float64
    N = pose.shape[0]

    fidelity = data.get("fidelity", np.array(["unknown"] * N, dtype=object))
    contact_valid = data.get("contact_valid", np.zeros((N,), dtype=np.uint8))

    out = dict(
        N=N,
        pose=pose,
        fidelity=_as_fidelity_str(fidelity, N),
        contact_valid=_as_bool_mask(contact_valid, N),
    )

    for k in ["config_sdf", "min_separation", "max_penetration"]:
        if k in keys:
            out[k] = np.asarray(data[k], dtype=float).reshape(-1)

    return out


def load_csv(path: str) -> Dict[str, Any]:
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    arr = np.atleast_1d(arr)
    colnames = set(arr.dtype.names or [])

    required = {
        "tx", "ty", "tz",
        "rx_deg", "ry_deg", "rz_deg",
        "fidelity",
        "contact_valid",
    }
    missing = required - colnames
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}. Found: {sorted(colnames)}")

    pose = np.stack(
        [arr["tx"], arr["ty"], arr["tz"], arr["rx_deg"], arr["ry_deg"], arr["rz_deg"]],
        axis=1,
    ).astype(float)
    N = pose.shape[0]

    out = dict(
        N=N,
        pose=pose,
        fidelity=_as_fidelity_str(arr["fidelity"], N),
        contact_valid=_as_bool_mask(arr["contact_valid"], N),
    )

    for k in ["config_sdf", "min_separation", "max_penetration"]:
        if k in colnames:
            out[k] = np.asarray(arr[k], dtype=float).reshape(-1)

    return out


# ==========================
# ANALYSIS + PLOTTING
# ==========================

def count_fidelity(fid_s: np.ndarray) -> Tuple[int, int, int]:
    coarse = int(np.sum(fid_s == "coarse"))
    fine = int(np.sum(fid_s == "fine"))
    other = int(fid_s.size - coarse - fine)
    return coarse, fine, other


def make_contact_hist_figure(
    pose: np.ndarray,
    contact_all: np.ndarray,
    contact_coarse: np.ndarray,
    contact_fine: np.ndarray,
    out_path: str,
) -> None:
    """
    2x3 subplots: per-dimension histograms.
    Overlays contact-all vs contact-coarse vs contact-fine.
    """
    sns.set_theme(style="whitegrid")

    # Ensure pose is a real numeric ndarray (prevents object dtype surprises)
    pose = np.asarray(pose)
    if pose.dtype == object:
        pose = pose.astype(np.float64)

    # Ensure masks are boolean ndarrays
    contact_all = np.asarray(contact_all, dtype=bool)
    contact_coarse = np.asarray(contact_coarse, dtype=bool)
    contact_fine = np.asarray(contact_fine, dtype=bool)

    def _to_1d_float_list(x) -> list[float]:
        """
        Returns a Python list of Python floats.
        Robust to:
        - x being (N,), (N,1), etc.
        - x being object arrays
        - x elements being numpy scalars
        - x elements being 0-d or 1-d arrays (takes scalar if possible)
        """
        x = np.asarray(x)
        x = x.reshape(-1)

        out: list[float] = []
        for v in x:
            # If v is itself an array/list, try to reduce to a scalar
            vv = np.asarray(v)
            if vv.ndim > 0:
                vv = vv.reshape(-1)
                if vv.size == 0:
                    continue
                if vv.size != 1:
                    # If you hit this, your "pose dimension" isn't scalar per sample,
                    # which indicates pose is still nested incorrectly.
                    raise ValueError(
                        f"Non-scalar element encountered in histogram data: element has size {vv.size}, "
                        f"type={type(v)}, value example={vv[:5]}"
                    )
                vv = vv[0]

            # Now vv should be scalar-ish
            try:
                f = float(vv)
            except Exception as e:
                raise ValueError(f"Could not convert element to float. type={type(vv)}, value={vv}") from e

            if np.isfinite(f):
                out.append(f)

        return out

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axes = axes.flatten()

    groups = [
        ("contact_all", contact_all),
        ("contact_coarse", contact_coarse),
        ("contact_fine", contact_fine),
    ]

    for j, (ax, dim_name) in enumerate(zip(axes, POSE_NAMES)):
        # base distribution for shared binrange
        base_list = _to_1d_float_list(pose[contact_all, j] if np.any(contact_all) else pose[:, j])

        if len(base_list) == 0:
            ax.set_title(dim_name)
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            continue

        base_np = np.asarray(base_list, dtype=np.float64)
        lo = float(np.min(base_np))
        hi = float(np.max(base_np))
        # Avoid zero-width binrange
        if not np.isfinite(lo) or not np.isfinite(hi):
            ax.set_title(dim_name)
            ax.text(0.5, 0.5, "non-finite data", ha="center", va="center", transform=ax.transAxes)
            continue
        if lo == hi:
            eps = 1e-9 if abs(lo) < 1 else 1e-9 * abs(lo)
            lo -= eps
            hi += eps

        for label, mask in groups:
            x_list = _to_1d_float_list(pose[mask, j])
            if len(x_list) == 0:
                continue

            # (Optional debug)
            # print(dim_name, label, type(x_list[0]), x_list[0])

            sns.histplot(
                x=x_list,                 # list[float]
                bins=HIST_BINS,           # int (NOT ndarray edges)
                binrange=(lo, hi),        # shared range ensures aligned bins
                stat=HIST_STAT,
                common_norm=HIST_COMMON_NORM,
                element="step",
                fill=False,
                linewidth=1.5,
                ax=ax,
                label=label,
            )

        ax.set_title(dim_name)
        ax.set_xlabel(dim_name)
        ax.set_ylabel(HIST_STAT)

    # Legend once
    handles, labels = None, None
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        if len(h) > 0:
            handles, labels = h, l
            break
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=True)

    fig.suptitle("Contact pose distributions (all vs coarse vs fine)", y=1.02)

    if SAVE_FIG:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"\nSaved histogram figure: {out_path}")

    plt.close(fig)


def main() -> None:
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: file not found: {DATA_PATH}", file=sys.stderr)
        sys.exit(1)

    ext = os.path.splitext(DATA_PATH)[1].lower()
    if ext == ".npz":
        data = load_npz(DATA_PATH)
    elif ext == ".csv":
        data = load_csv(DATA_PATH)
    else:
        raise ValueError("DATA_PATH must end in .csv or .npz")

    N = int(data["N"])
    pose = np.asarray(data["pose"], dtype=float)
    assert pose.ndim == 2 and pose.shape[1] == 6 and pose.dtype != object, (pose.shape, pose.dtype)
    fid_s = np.asarray(data["fidelity"], dtype=object)
    contact_valid = np.asarray(data["contact_valid"], dtype=bool)

    # fidelity masks
    coarse_mask_all = (fid_s == "coarse")
    fine_mask_all = (fid_s == "fine")

    # contact masks (requested: valid is default)
    contact_all = contact_valid
    contact_coarse = contact_valid & coarse_mask_all
    contact_fine = contact_valid & fine_mask_all

    # counts
    n_contact = int(np.sum(contact_all))
    n_contact_coarse = int(np.sum(contact_coarse))
    n_contact_fine = int(np.sum(contact_fine))
    frac_contact = (n_contact / N) if N > 0 else np.nan
    frac_contact_coarse = (n_contact_coarse / N) if N > 0 else np.nan
    frac_contact_fine = (n_contact_fine / N) if N > 0 else np.nan

    coarse, fine, other = count_fidelity(fid_s)
    frac_coarse = (coarse / N) if N > 0 else np.nan
    frac_fine = (fine / N) if N > 0 else np.nan

    # Also compute contact rates within fidelity groups (often useful)
    coarse_contact_rate = (n_contact_coarse / coarse) if coarse > 0 else np.nan
    fine_contact_rate = (n_contact_fine / fine) if fine > 0 else np.nan

    print("\n================ SUMMARY ================")
    print(f"File: {DATA_PATH}")
    print(f"Total poses sampled: {N}")

    print("\n-- Fidelity (overall) --")
    print(f"  coarse: {coarse} ({frac_coarse:.6f})")
    print(f"  fine:   {fine} ({frac_fine:.6f})")
    print(f"  other:  {other}")

    print("\n-- Contact (contact_valid) --")
    print(f"  contact_all:   {n_contact} ({frac_contact:.6f})")
    print(f"  contact_coarse:{n_contact_coarse} ({frac_contact_coarse:.6f})  | P(contact|coarse)={coarse_contact_rate:.6f}")
    print(f"  contact_fine:  {n_contact_fine} ({frac_contact_fine:.6f})    | P(contact|fine)={fine_contact_rate:.6f}")

    # pose stats
    print_pose_stats_block("Pose stats (ALL poses)", pose)
    print_pose_stats_block("Pose stats (CONTACT poses: all)", pose[contact_all])
    print_pose_stats_block("Pose stats (CONTACT poses: coarse)", pose[contact_coarse])
    print_pose_stats_block("Pose stats (CONTACT poses: fine)", pose[contact_fine])

    # Optional numeric metrics (all + contact splits)
    for k in ["config_sdf", "min_separation", "max_penetration"]:
        if k in data:
            vals = np.asarray(data[k], dtype=float).reshape(-1)
            # print scalar stats for all/contact/coarse/fine contact subsets
            print(f"\n== {k} stats ==")
            for name, mask in [
                ("ALL poses", np.ones((N,), dtype=bool)),
                ("CONTACT all", contact_all),
                ("CONTACT coarse", contact_coarse),
                ("CONTACT fine", contact_fine),
            ]:
                s = stats_1d(vals[mask])
                print(
                    f"  {name:>13s}: "
                    f"min={_fmt(s['min'])}  max={_fmt(s['max'])}  mean={_fmt(s['mean'])}  "
                    f"median={_fmt(s['median'])}  std={_fmt(s['std'])}"
                )

    # histogram figure (contact-only overlays)
    out_dir = os.path.dirname(os.path.abspath(DATA_PATH))
    base = os.path.splitext(os.path.basename(DATA_PATH))[0]
    fig_path = os.path.join(out_dir, f"{base}_{FIG_NAME_SUFFIX}")

    make_contact_hist_figure(
        pose=pose,
        contact_all=contact_all,
        contact_coarse=contact_coarse,
        contact_fine=contact_fine,
        out_path=fig_path,
    )

    print("\n========================================")
    print("Done.")


if __name__ == "__main__":
    main()
