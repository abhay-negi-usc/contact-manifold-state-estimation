import numpy as np 
import pandas as pd 
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from utils.pose_utils import torch_pose_xyzabc_to_matrix

#--- CONFIG ---# 
# filepath_sim_data = "/media/rp/Elements1/abhay_ws/contact-manifold-state-generation/data/cross_data/cross_data_merged/processed_data/cross_contact_poses_mujoco.csv"
# filepath_real_data = "/home/rp/dhanush_ws/sunrise-wrapper/contact_maps/cross_round_tight/CASE_cross_round_tight_aut_map_4.pkl" 
filepath_sim_data = "/home/rp/abhay_ws/contact-manifold-state-estimation/data_generation/extrusion_sim_data_with_logmaps_no_units_row.csv"
filepath_real_data = "/home/rp/dhanush_ws/sunrise-wrapper/contact_maps/extrusion_rounded/zfilt_extrusion_rounded_aut_map_1.pkl" 

#--- LOAD DATA ---# 
df_sim = pd.read_csv(filepath_sim_data) 
df_real = pd.read_pickle(filepath_real_data) 
print(f"Loaded {len(df_sim)} rows of sim data from {filepath_sim_data}")
print(f"Loaded {len(df_real)} rows of real data from {filepath_real_data}")
print("\n")

# NOTE: FIXME 
# delete row 0 and reset index
df_sim = df_sim.drop(index=0).reset_index(drop=True) # remove units row 
# convert from strings to floats 
df_sim[['x', 'y', 'z', 'a', 'b', 'c']] = df_sim[['x', 'y', 'z', 'a', 'b', 'c']].astype(float)
df_sim[['x', 'y', 'z']] = df_sim[['x', 'y', 'z']] * 1000.0 # m to mm
df_sim[['a', 'b', 'c']] = df_sim[['a', 'b', 'c']] #* (180.0 / np.pi) # rad to deg
print(f"\n\nConverted sim data to mm and degrees.\n\n")
# END OF FIXME 

# truncate real data
N_real_max = 100_000
if len(df_real) > N_real_max:
    df_real = df_real.sample(N_real_max, random_state=42).reset_index(drop=True)
    print(f"Truncated real data to {N_real_max} random samples for faster processing.")
print(f"Using {len(df_real)} rows of real data for comparison.")

#--- PROCESS DATA ---#

# create kdtrees for both sim and real data 
kdtree_sim = cKDTree(df_sim[['x', 'y', 'z', 'a', 'b', 'c']].to_numpy())
kdtree_real = cKDTree(df_real[['x', 'y', 'z', 'a', 'b', 'c']].to_numpy())
print(f"KDTree of sim data with {len(df_sim)} points created.")
print(f"KDTree of real data with {len(df_real)} points created.")
print("\n")

# convert each row of df_sim to a 4x4 transformation matrix
transforms_sim_torch = torch_pose_xyzabc_to_matrix(torch.tensor(df_sim[['x', 'y', 'z', 'a', 'b', 'c']].to_numpy(), dtype=torch.float32))
transforms_real_torch = torch_pose_xyzabc_to_matrix(torch.tensor(df_real[['x', 'y', 'z', 'a', 'b', 'c']].to_numpy(), dtype=torch.float32))  
print(f"Converted sim and real data to transformation matrices.")
print("\n")

# fix real data offset 
transform_sim_peg_fix = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0.025],[0,0,0,1]], dtype=torch.float32) # 25mm offset in z
transform_sim_hole_fix = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0.025],[0,0,0,1]], dtype=torch.float32) # 25mm offset in z
transform_sim_hole_fix_inv = torch.inverse(transform_sim_hole_fix)
transforms_sim_torch = transform_sim_hole_fix_inv @ transforms_sim_torch @ transform_sim_peg_fix

#--- COMPARE DATA ---#

def compute_pose_errors(source_transforms, target_transforms, indices, desc="Computing errors"):
    """Compute 6D pose errors between source and target transforms."""
    pose_errors_6d = np.zeros((len(indices), 6))  # [x, y, z, rx, ry, rz]
    t_diffs = []
    angle_diffs = []
    
    for i, idx in enumerate(tqdm(indices, desc=desc)):
        tf_source = source_transforms[i]
        tf_target = target_transforms[idx]
        tf_diff = torch.inverse(tf_source) @ tf_target
        t_diff = tf_diff[:3, 3].numpy() # already in mm 
        r_diff = R.from_matrix(tf_diff[:3, :3].numpy())
        angle_diff = r_diff.magnitude() * (180.0 / np.pi) # in degrees
        
        # Store individual components for 6D pose error visualization
        pose_errors_6d[i, :3] = t_diff  # x, y, z translation errors in mm
        r_euler = r_diff.as_euler('xyz', degrees=True)  # rotation errors in degrees
        pose_errors_6d[i, 3:] = r_euler  # rx, ry, rz rotation errors in degrees
        
        t_diffs.append(np.linalg.norm(t_diff))
        angle_diffs.append(angle_diff)
    
    return pose_errors_6d, np.array(t_diffs), np.array(angle_diffs)

# Case 1: Real to Sim (find nearest sim point for each real point)
print("=== CASE 1: Real to Sim Nearest Neighbors ===")
_, indices_real_to_sim = kdtree_sim.query(df_real[['x', 'y', 'z', 'a', 'b', 'c']].to_numpy(), k=1)
print(f"Found nearest neighbors in sim data for each real data point.")

pose_errors_6d_real_to_sim, t_diffs_real_to_sim, angle_diffs_real_to_sim = compute_pose_errors(
    transforms_real_torch, transforms_sim_torch, indices_real_to_sim, 
    "Computing Real->Sim transformation differences"
)

# Case 2: Sim to Real (find nearest real point for each sim point)
print("\n=== CASE 2: Sim to Real Nearest Neighbors ===")
_, indices_sim_to_real = kdtree_real.query(df_sim[['x', 'y', 'z', 'a', 'b', 'c']].to_numpy(), k=1)
print(f"Found nearest neighbors in real data for each sim data point.")

pose_errors_6d_sim_to_real, t_diffs_sim_to_real, angle_diffs_sim_to_real = compute_pose_errors(
    transforms_sim_torch, transforms_real_torch, indices_sim_to_real, 
    "Computing Sim->Real transformation differences"
)

print(f"\nCompleted transformation difference computations for both cases.")
print("\n")# print statistics of differences for both cases
print("=== CASE 1: Real to Sim Statistics ===")
print(f"Translation difference (mm): mean={t_diffs_real_to_sim.mean():.2f}, std={t_diffs_real_to_sim.std():.2f}, max={t_diffs_real_to_sim.max():.2f}")
print(f"Rotation difference (deg): mean={angle_diffs_real_to_sim.mean():.2f}, std={angle_diffs_real_to_sim.std():.2f}, max={angle_diffs_real_to_sim.max():.2f}")

print("\n=== CASE 2: Sim to Real Statistics ===")
print(f"Translation difference (mm): mean={t_diffs_sim_to_real.mean():.2f}, std={t_diffs_sim_to_real.std():.2f}, max={t_diffs_sim_to_real.max():.2f}")
print(f"Rotation difference (deg): mean={angle_diffs_sim_to_real.mean():.2f}, std={angle_diffs_sim_to_real.std():.2f}, max={angle_diffs_sim_to_real.max():.2f}")

#--- VISUALIZATION ---#

def plot_6d_pose_errors(pose_errors_6d, title_suffix, filename_suffix):
    """Create 6D pose error histogram plots."""
    # Create 2x3 subplot figure for 6D pose error histograms
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'6D Pose Errors: {title_suffix}', fontsize=16, fontweight='bold')

    # Labels and units for each component
    labels = ['X Translation', 'Y Translation', 'Z Translation', 'X Rotation', 'Y Rotation', 'Z Rotation']
    units = ['(mm)', '(mm)', '(mm)', '(deg)', '(deg)', '(deg)']

    # Create histograms for each pose component
    for i in range(6):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # Plot histogram
        data = pose_errors_6d[:, i]
        n_bins = 50
        ax.hist(data, bins=n_bins, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
        
        # Add statistics text
        mean_val = np.mean(data)
        mae_val = np.mean(np.abs(data))
        std_val = np.std(data)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(mae_val, color='green', linestyle='-.', linewidth=2, label=f'MAE: {mae_val:.2f}')
        ax.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=1.5, alpha=0.8, label=f'±1σ: {std_val:.2f}')
        ax.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=1.5, alpha=0.8)
        
        # Formatting
        ax.set_title(f'{labels[i]} {units[i]}', fontsize=12, fontweight='bold')
        ax.set_xlabel(f'Error {units[i]}', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Add statistics box
        stats_text = f'Mean: {mean_val:.2f}\nMAE: {mae_val:.2f}\nStd: {std_val:.2f}\nMin: {np.min(data):.2f}\nMax: {np.max(data):.2f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()
    
    # Save the figure
    filename = f'6d_pose_errors_histogram_{filename_suffix}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Figure saved as '{filename}'")
    plt.show()
    
    return labels, units

# Plot for Case 1: Real to Sim
labels, units = plot_6d_pose_errors(pose_errors_6d_real_to_sim, "Real to Sim Nearest Neighbors", "real_to_sim")

# Plot for Case 2: Sim to Real  
plot_6d_pose_errors(pose_errors_6d_sim_to_real, "Sim to Real Nearest Neighbors", "sim_to_real")

def print_detailed_statistics(pose_errors_6d, labels, units, case_name):
    """Print detailed statistics for each component."""
    print(f"\n=== Detailed 6D Pose Error Statistics - {case_name} ===")
    for i, (label, unit) in enumerate(zip(labels, units)):
        data = pose_errors_6d[:, i]
        mae_val = np.mean(np.abs(data))
        print(f"{label} {unit}:")
        print(f"  Mean: {np.mean(data):.3f}, MAE: {mae_val:.3f}, Std: {np.std(data):.3f}")
        print(f"  Min: {np.min(data):.3f}, Max: {np.max(data):.3f}")
        print(f"  Median: {np.median(data):.3f}, 95th percentile: {np.percentile(data, 95):.3f}")
        print()

# Print detailed statistics for both cases
print_detailed_statistics(pose_errors_6d_real_to_sim, labels, units, "Real to Sim")
print_detailed_statistics(pose_errors_6d_sim_to_real, labels, units, "Sim to Real")

#--- COMPARISON SUMMARY ---#
print("\n" + "="*60)
print("COMPARISON SUMMARY")
print("="*60)

print(f"\nDataset sizes:")
print(f"  Real data points: {len(df_real):,}")
print(f"  Sim data points: {len(df_sim):,}")

print(f"\nOverall Translation Errors (mm):")
print(f"  Real→Sim: Mean={t_diffs_real_to_sim.mean():.2f}, Std={t_diffs_real_to_sim.std():.2f}, Max={t_diffs_real_to_sim.max():.2f}")
print(f"  Sim→Real: Mean={t_diffs_sim_to_real.mean():.2f}, Std={t_diffs_sim_to_real.std():.2f}, Max={t_diffs_sim_to_real.max():.2f}")

print(f"\nOverall Rotation Errors (deg):")
print(f"  Real→Sim: Mean={angle_diffs_real_to_sim.mean():.2f}, Std={angle_diffs_real_to_sim.std():.2f}, Max={angle_diffs_real_to_sim.max():.2f}")
print(f"  Sim→Real: Mean={angle_diffs_sim_to_real.mean():.2f}, Std={angle_diffs_sim_to_real.std():.2f}, Max={angle_diffs_sim_to_real.max():.2f}")

# Determine which case has better coverage
real_to_sim_mean_error = t_diffs_real_to_sim.mean()
sim_to_real_mean_error = t_diffs_sim_to_real.mean()

print(f"\nInterpretation:")
if real_to_sim_mean_error < sim_to_real_mean_error:
    print(f"  Real→Sim has lower average error ({real_to_sim_mean_error:.2f} mm vs {sim_to_real_mean_error:.2f} mm)")
    print(f"  This suggests the simulation dataset provides better coverage of real-world poses.")
else:
    print(f"  Sim→Real has lower average error ({sim_to_real_mean_error:.2f} mm vs {real_to_sim_mean_error:.2f} mm)")
    print(f"  This suggests the real dataset provides better coverage of simulation poses.")

print(f"\nNote: Lower errors in Real→Sim indicate good sim coverage of real poses.")
print(f"      Lower errors in Sim→Real indicate good real coverage of sim poses.")
print("="*60)