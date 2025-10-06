#!/usr/bin/env python3
"""
compare_models_estimation.py

Compare multiple contact manifold models on hole-pose-only estimation across many observation trials.

This script is based on utilities defined in state_estimator.py (estimate_holePose, set_seed, etc.).
It will:
- Discover observation trials under a directory (e.g., files like extrusion_pose_H_P_0.npy, ...).
- Load a list of contact manifold models (PyTorch checkpoints) into ContactPoseManifold.
- For each model, run hole-pose-only estimation for each trial using the same random seed per-trial
  so initial conditions are identical across models (fair comparison).
- Aggregate results (pose errors when ground-truth is available, otherwise proxy metrics like final loss),
  print summary tables, and save figures/CSVs.

USAGE (example):
    python compare_models_estimation.py \\
        --observations_dir /path/to/obs \\
        --obs-type h_p \\
        --model /path/to/checkpoints/runA.pth \\
        --model /path/to/checkpoints/runB.pth \\
        --device cuda \\
        --seed 42 \\
        --max-it 5000 \\\
        --lr 1e-2 \\\
        --save-dir ./cmp_out

Ground truth (optional):
    If you have ground-truth hole pose offsets (tf_H_h_true) per trial, place matching files in --gt-dir
    with any of these name patterns (uses the same numeric index as the observation file):
        * *_H_h_<index>.npy
        * *_hole_offset_<index>.npy
        * *_true_H_h_<index>.npy
    Each ground-truth file can be a 4x4 transform or 6D pose [x,y,z,rx,ry,rz] (radians).

Observations:
    The script supports two observation types via --obs-type:
      - h_p : files encode tf_h_p (4x4) or pose_h_p (6D). Preferred.
      - H_P : files encode tf_H_P (4x4) or pose_H_P (6D). If you use H_P, the estimator is still
              driven via pose_h_p internally. We approximate by setting tf_h_p := tf_H_P, which is only
              exact if the hole offset and in-hand offset are identities at observation generation.
              Prefer saving h_p observations if possible.

Outputs (in --save-dir):
    - results_per_trial.csv : per trial & model metrics
    - summary_by_model.csv  : aggregate stats per model
    - barplots_[translation|rotation|loss].png : quick visual comparisons

NOTE:
    This script does not estimate in-hand pose. It uses state_estimator.estimate_holePose().

Author: ChatGPT
"""

#!/usr/bin/env python3
"""
compare_models_estimation.py (config-driven)

Same as before, but uses a CONFIG dictionary for reproducibility
and easier integration in pipelines or notebooks.
"""

from state_estimation.state_estimator import (
    set_seed, estimate_holePose, read_observation,
    torch_matrix_to_pose_xyzabc, torch_pose_xyzabc_to_matrix,
    offset_observation
)
from cicp.icp_manifold import ContactPoseManifold
import os, re, glob, json, numpy as np, torch, pandas as pd, matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from scipy.spatial.transform import Rotation as R

# ---------------- CONFIG -----------------
config = {
    "observations_dir": '/home/rp/abhay_ws/cicp/data/extrusion_observations/extrusion_timed_icp_10/hole_frame/',
    "obs_type": "h_p",
    "glob_pattern": "*.npy",
    "limit_trials": 1, 
    "model_paths": [
        '/home/rp/abhay_ws/cicp/checkpoints/extrusion_run_4_best_NN_model_xyzabc.pth', 
        '/home/rp/abhay_ws/contact-manifold-state-estimation/model_training/checkpoints/extrusion_run_2_best_NN_model_xyzabc.pth'
    ],
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "base_seed": 42,
    "max_it": 250,
    "lr": 1e-1,
    "optimizer_type": "adam",
    "lr_decay_type": "exponential",
    "lr_decay_rate": 0.98,
    "lr_decay_step": 100,
    "max_samples": None,
    "gt_dir": None,
    "save_dir": "./state_estimation/compare_models_out",
    "layer_sizes": [6, 4096, 4096, 4096, 4096, 6],
    # Offset observation parameters
    "use_offset_observations": True,
    "max_hole_pose_offsets": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
    "max_in_hand_pose_offsets": [0, 0, 0, 0, 0, 0],  # No in-hand offsets for hole-only estimation
    "set_max_offsets": True,  # Use maximum offsets instead of random
}
# ------------------------------------------



# Import utilities from state_estimator.py (assumed to be importable)
from state_estimation.state_estimator import (
    set_seed,
    estimate_holePose,
    read_observation,            # reads tf_H_P (4x4) but is generic np.load
    torch_matrix_to_pose_xyzabc,
    torch_pose_xyzabc_to_matrix,
)
from cicp.icp_manifold import ContactPoseManifold


# ----------------------------- Helpers -----------------------------

def natural_sort_key(s: str):
    """Natural sort key so '2' < '10' (alphanumeric)."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\\d+)', s)]


def list_npys(observations_dir: str, pattern: str = '*.npy', limit: Optional[int] = None) -> List[str]:
    files = glob.glob(os.path.join(observations_dir, pattern))
    files.sort(key=natural_sort_key)
    if limit is not None and limit > 0:
        files = files[:limit]
    return files


def load_pose_6d_from_array(arr: np.ndarray) -> np.ndarray:
    """
    Return pose as shape (B,6) np.float32 from either 4x4 or 6-vector (or (1,6)).
    """
    if arr.ndim == 2 and arr.shape == (4, 4):
        pose = torch_matrix_to_pose_xyzabc(torch.tensor(arr.reshape(1, 4, 4), dtype=torch.float32)).cpu().numpy()
        return pose.astype(np.float32)
    elif arr.ndim == 1 and arr.shape[0] == 6:
        return arr.reshape(1, 6).astype(np.float32)
    elif arr.ndim == 2 and arr.shape[1] == 6:
        return arr.astype(np.float32)
    else:
        raise ValueError(f"Unsupported array shape for pose conversion: {arr.shape}")

def pose_errors_from_mats(tf_true: np.ndarray, tf_est: np.ndarray) -> Tuple[float, float]:
    """
    Compute translation (m) and rotation (deg) error between two 4x4 transforms.
    """
    if tf_true is None:
        return np.nan, np.nan
    tf_err = tf_true @ np.linalg.inv(tf_est)
    pose_err = torch_matrix_to_pose_xyzabc(torch.tensor(tf_err, dtype=torch.float32)).cpu().numpy().reshape(-1)
    trans_err = float(np.linalg.norm(pose_err[:3], ord=2))  # meters
    rot_err_deg = float(np.linalg.norm(pose_err[3:], ord=2))  # degrees
    return trans_err, rot_err_deg


def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


@dataclass
class TrialResult:
    model_name: str
    model_path: str
    trial_file: str
    trial_idx: int
    seed: int
    iters_run: int
    final_loss: float
    trans_err_m: float
    rot_err_deg: float


# -------------------------- Core Routine ---------------------------

def run_comparison(
    observations_dir: str,
    model_paths: List[str],
    device: str = 'cuda',
    obs_type: str = 'h_p',
    glob_pattern: str = '*.npy',
    limit_trials: Optional[int] = None,
    base_seed: int = 42,
    max_it: int = 10_000,
    lr: float = 1e-1,
    optimizer_type: str = 'adam',
    lr_decay_type: str = 'exponential',
    lr_decay_rate: float = 0.98,
    lr_decay_step: int = 100,
    max_samples: Optional[int] = None,
    gt_dir: Optional[str] = None,
    save_dir: str = './cmp_out',
    layer_sizes: Optional[List[int]] = None,
    use_offset_observations: bool = False,
    max_hole_pose_offsets: Optional[List[float]] = None,
    max_in_hand_pose_offsets: Optional[List[float]] = None,
    set_max_offsets: bool = True
):
    ensure_dir(save_dir)
    # 1) Discover trials
    obs_files = list_npys(observations_dir, glob_pattern, limit_trials)
    if len(obs_files) == 0:
        raise FileNotFoundError(f"No observation files found in {observations_dir} with pattern {glob_pattern}")

    # 2) Prepare models
    manifolds = []
    for mp in model_paths:
        cpm = ContactPoseManifold(geometry="extrusion")
        # allow override of layer sizes, else default to 6-4096-... per your training scripts
        ls = layer_sizes if layer_sizes is not None else [6, 4096, 4096, 4096, 4096, 6]
        cpm.load_model_from_path(mp, layer_sizes=ls)
        manifolds.append((os.path.basename(mp), mp, cpm))

    # 3) Loop trials & models
    results: List[TrialResult] = []

    for trial_idx, f in enumerate(obs_files):

        print(f"Trial {trial_idx+1}/{len(obs_files)}: {os.path.basename(f)}")

        # Set per-trial seed once so it's identical across all models
        trial_seed = base_seed + trial_idx
        set_seed(trial_seed)

        # Load observation
        tf_H_P = np.load(f, allow_pickle=True)

        
        
        # Handle offset observations if enabled
        if use_offset_observations:
            # Generate offset observations with known ground truth
            if max_hole_pose_offsets is None:
                max_hole_pose_offsets = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
            if max_in_hand_pose_offsets is None:
                max_in_hand_pose_offsets = [0, 0, 0, 0, 0, 0]
            
            
        # else:
        #     # Use observations as-is
        #     tf_H_h_true = None  # No ground truth available
        #     if obs_type.lower() == 'h_p':
        #         pose_h_p = load_pose_6d_from_array(arr)  # (N,6)
        #     elif obs_type.lower() == 'h_p_pose':
        #         pose_h_p = load_pose_6d_from_array(arr)
        #     elif obs_type.lower() == 'h_p_mat':
        #         pose_h_p = load_pose_6d_from_array(arr)
        #     elif obs_type.lower() == 'h_p_or_H_P_auto':
        #         # Try to infer from filename; fallback to treating as h_p
        #         if re.search(r'(H_P|Hp|H-P)', os.path.basename(f)):
        #             # Best-effort: treat as tf_H_P but approximate tf_h_p := tf_H_P
        #             pose_h_p = load_pose_6d_from_array(arr)
        #         else:
        #             pose_h_p = load_pose_6d_from_array(arr)
        #     elif obs_type.lower() == 'H_P'.lower():
        #         # Best-effort fallback: pass H_P as if it were h_p (see header note)
        #         pose_h_p = load_pose_6d_from_array(arr)
        #     else:
        #         raise ValueError(f"Unsupported --obs-type '{obs_type}'. Use 'h_p' or 'H_P'.")

        # Shared config for estimator
        config = {'device': device}

        for model_name, model_path, cpm in manifolds:
            # Ensure estimator uses the same seed for this trial+model
            # (Calling set_seed again is okay; we want identical initial guess across models)
            set_seed(trial_seed)

            # FIXME: this is a bandaid solution for now 
            if 'cicp' in model_path.lower():
                sim_data_model = False 
                print("Assuming model trained on real data")
            else: 
                sim_data_model = True 
                print("Assuming model trained on sim data")

            if sim_data_model: 
                # Apply transform to account for different reference frames in sim data
                for i, pose_H_P_i in enumerate(tf_H_P): 
                    pose_H_P_i = torch.tensor(pose_H_P_i, dtype=torch.float32)
                    tf_H_P_i = torch_pose_xyzabc_to_matrix(pose_H_P_i.unsqueeze(0)).squeeze(0)
                    transform_peg = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,-25],[0,0,0,1]], dtype=torch.float32) # 25mm offset in z
                    transform_hole = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,-25],[0,0,0,1]], dtype=torch.float32) # 25mm offset in z
                    transform_hole_inv = torch.inverse(transform_hole)
                    tf_H_P_transformed = transform_hole_inv @ tf_H_P_i @ transform_peg
                    pose_H_P_transformed = torch_matrix_to_pose_xyzabc(tf_H_P_transformed.unsqueeze(0)).cpu().numpy()
                    tf_H_P[i] = pose_H_P_transformed.flatten()

            tf_h_p, tf_H_h_true, tf_P_p = offset_observation(
                max_hole_pose_offsets=np.array(max_hole_pose_offsets),
                max_in_hand_pose_offsets=np.array(max_in_hand_pose_offsets),
                observation=tf_H_P,
                set_max_offsets=set_max_offsets,
                seed=trial_seed
            )
            
            # Convert to pose format for estimation
            pose_h_p = torch_matrix_to_pose_xyzabc(torch.tensor(tf_h_p, dtype=torch.float32)).cpu().numpy()



            # Run estimation
            ret = estimate_holePose(
                contact_model=cpm,
                observations=pose_h_p,
                config=config,
                max_samples=max_samples,
                max_it=max_it,
                lr=lr,
                optimizer_type=optimizer_type,
                seed=trial_seed,
                lr_decay_type=lr_decay_type,
                lr_decay_rate=lr_decay_rate,
                lr_decay_step=lr_decay_step
            )
            tf_H_h_est, pose_H_h_est, pose_H_h_history, loss_history, lr_history = ret

            # Compute errors if GT exists; else NaN
            trans_err_m, rot_err_deg = pose_errors_from_mats(tf_H_h_true, tf_H_h_est)

            results.append(TrialResult(
                model_name=model_name,
                model_path=model_path,
                trial_file=os.path.basename(f),
                trial_idx=trial_idx,
                seed=trial_seed,
                iters_run=len(loss_history),
                final_loss=float(loss_history[-1]) if len(loss_history) else float('nan'),
                trans_err_m=trans_err_m,
                rot_err_deg=rot_err_deg
            ))

    # 4) Aggregate & analyze
    df = pd.DataFrame([asdict(r) for r in results])
    df_path = os.path.join(save_dir, 'results_per_trial.csv')
    df.to_csv(df_path, index=False)

    # Summary per model
    agg_funcs = {
        'final_loss': ['mean', 'std', 'median'],
        'trans_err_m': ['mean', 'std', 'median'],
        'rot_err_deg': ['mean', 'std', 'median'],
        'iters_run': ['mean', 'std', 'median']
    }
    summary = df.groupby('model_name').agg(agg_funcs)
    # Flatten columns
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary_path = os.path.join(save_dir, 'summary_by_model.csv')
    summary.to_csv(summary_path)

    # 5) Print tables
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 160)
    print("\\nPer-trial results:")
    print(df)
    print("\\nSummary by model:")
    print(summary)

    # 6) Plots
    def barplot_metric(metric: str, ylabel: str, filename: str):
        fig = plt.figure(figsize=(8, 5))
        ax = fig.gca()
        # Use means with error bars (std). If GT missing, values may be NaN; drop them.
        means = df.groupby('model_name')[metric].mean(numeric_only=True)
        stds = df.groupby('model_name')[metric].std(numeric_only=True)
        means = means.dropna()
        stds = stds.reindex(means.index).fillna(0.0)
        means.plot(kind='bar', yerr=stds, capsize=4, ax=ax)
        ax.set_ylabel(ylabel)
        ax.set_title(f'Model comparison: {metric}')
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        out_path = os.path.join(save_dir, filename)
        plt.savefig(out_path, dpi=150)
        plt.close(fig)

    ensure_dir(save_dir)
    barplot_metric('trans_err_m', 'Translation error (m, mean ± std)', 'barplot_translation.png')
    barplot_metric('rot_err_deg', 'Rotation error (deg, mean ± std)', 'barplot_rotation.png')
    barplot_metric('final_loss', 'Final loss (mean ± std)', 'barplot_loss.png')

    # 7) Save a JSON manifest for reproducibility
    manifest = {
        'observations_dir': observations_dir,
        'model_paths': model_paths,
        'device': device,
        'obs_type': obs_type,
        'glob_pattern': glob_pattern,
        'limit_trials': limit_trials,
        'base_seed': base_seed,
        'max_it': max_it,
        'lr': lr,
        'optimizer_type': optimizer_type,
        'lr_decay_type': lr_decay_type,
        'lr_decay_rate': lr_decay_rate,
        'lr_decay_step': lr_decay_step,
        'max_samples': max_samples,
        'gt_dir': gt_dir,
        'save_dir': save_dir,
        'use_offset_observations': use_offset_observations,
        'max_hole_pose_offsets': max_hole_pose_offsets,
        'max_in_hand_pose_offsets': max_in_hand_pose_offsets,
        'set_max_offsets': set_max_offsets
    }
    with open(os.path.join(save_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\\nSaved per-trial CSV: {df_path}")
    print(f"Saved summary CSV   : {summary_path}")
    print(f"Saved plots to      : {save_dir}")

# ----------------------------- CLI ------------------------------

# (the rest of the script stays same as the one I wrote earlier,
# except we remove parse_args() and replace main() with:)
def main():
    run_comparison(**config)

if __name__ == "__main__":
    main()