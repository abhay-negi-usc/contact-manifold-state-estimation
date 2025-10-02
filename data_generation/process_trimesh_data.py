import numpy as np 
import pandas as pd 
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from utils.pose_utils import torch_pose_xyzabc_to_matrix

def rpy_to_matrix(roll, pitch, yaw, degrees=False):
    if degrees:
        roll, pitch, yaw = np.deg2rad([roll, pitch, yaw])
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [ 0,   0, 1]])
    Ry = np.array([[ cp, 0, sp],
                   [  0, 1,  0],
                   [-sp, 0, cp]])
    Rx = np.array([[1,  0,   0],
                   [0, cr, -sr],
                   [0, sr,  cr]])
    return Rz @ Ry @ Rx

df = pd.read_csv('/home/rp/abhay_ws/contact-manifold-state-estimation/data_generation/pose_sweep_contacts.csv')
df_contacts = df[df['contact']==1]
df_valid_contacts = df_contacts[df_contacts['metric']<1e-3] 

print(len(df))
print(len(df_contacts))
print(len(df_valid_contacts))

# fix rotations 
for idx, row in df_valid_contacts.iterrows():
    df_valid_contacts.at[idx, 'x'] *= 1e3
    df_valid_contacts.at[idx, 'y'] *= 1e3
    df_valid_contacts.at[idx, 'z'] *= 1e3
    rotmat = rpy_to_matrix(row['roll'], pitch=row['pitch'], yaw=row['yaw'])
    c, b, a = R.from_matrix(rotmat).as_euler('xyz', degrees=True)
    df_valid_contacts.at[idx, 'a'] = a
    df_valid_contacts.at[idx, 'b'] = b
    df_valid_contacts.at[idx, 'c'] = c


# save data 
df_valid_contacts.to_csv('/home/rp/abhay_ws/contact-manifold-state-estimation/data_generation/trimesh_contact_data.csv', index=False)

#--- LOAD REAL DATA FOR COMPARISON ---#
filepath_real_data = "/home/rp/dhanush_ws/sunrise-wrapper/contact_maps/extrusion_rounded/zfilt_extrusion_rounded_aut_map_1.pkl"
df_real = pd.read_pickle(filepath_real_data)
print(f"Loaded {len(df_real)} rows of real data from {filepath_real_data}")
print("\n")

#--- PROCESS DATA FOR COMPARISON ---#

# create kdtree of real data rows 
kdtree_real = cKDTree(df_real[['x', 'y', 'z', 'a', 'b', 'c']].to_numpy())
print(f"KDTree of real data with {len(df_real)} points created.")
print("\n")

# convert each row to 4x4 transformation matrices
transforms_sim_torch = torch_pose_xyzabc_to_matrix(torch.tensor(df_valid_contacts[['x', 'y', 'z', 'a', 'b', 'c']].to_numpy(dtype=np.float32), dtype=torch.float32))
transforms_real_torch = torch_pose_xyzabc_to_matrix(torch.tensor(df_real[['x', 'y', 'z', 'a', 'b', 'c']].to_numpy(), dtype=torch.float32))  
print(f"Converted sim and real data to transformation matrices.")
print("\n")

# fix sim data offset to match real data coordinate frame
transform_sim_peg_fix = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0.025],[0,0,0,1]], dtype=torch.float32) # 25mm offset in z
transform_sim_hole_fix = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0.025],[0,0,0,1]], dtype=torch.float32) # 25mm offset in z
transform_sim_hole_fix_inv = torch.inverse(transform_sim_hole_fix)
transforms_sim_torch = transform_sim_hole_fix_inv @ transforms_sim_torch @ transform_sim_peg_fix

#--- COMPARE SIM TO REAL DATA ---#
# for each row in df_valid_contacts (sim data), find nearest neighbor in df_real using kdtree
_, indices = kdtree_real.query(df_valid_contacts[['x', 'y', 'z', 'a', 'b', 'c']].to_numpy(), k=1)
print(f"Found nearest neighbors in real data for each sim data point.")

# Initialize arrays to store 6D pose error components
pose_errors_6d = np.zeros((len(df_valid_contacts), 6))  # [x, y, z, rx, ry, rz]

# compute transformation difference between each sim data point and its nearest real neighbor
for i, idx in enumerate(tqdm(indices, desc="Computing transformation differences")):
    tf_sim = transforms_sim_torch[i]
    tf_real = transforms_real_torch[idx]
    tf_diff = torch.inverse(tf_sim) @ tf_real
    t_diff = tf_diff[:3, 3].numpy() # already in mm 
    r_diff = R.from_matrix(tf_diff[:3, :3].numpy())
    angle_diff = r_diff.magnitude() * (180.0 / np.pi) # in degrees
    
    # Store individual components for 6D pose error visualization
    pose_errors_6d[i, :3] = t_diff  # x, y, z translation errors in mm
    r_euler = r_diff.as_euler('xyz', degrees=True)  # rotation errors in degrees
    pose_errors_6d[i, 3:] = r_euler  # rx, ry, rz rotation errors in degrees
    
    df_valid_contacts.at[df_valid_contacts.index[i], 'real_index'] = idx
    df_valid_contacts.at[df_valid_contacts.index[i], 'real_t_diff'] = np.linalg.norm(t_diff)
    df_valid_contacts.at[df_valid_contacts.index[i], 'real_angle_diff'] = angle_diff

print(f"Computed transformation differences between sim data and nearest real data.")
print("\n")

# print statistics of differences
print(f"Translation difference (mm): mean={df_valid_contacts['real_t_diff'].mean():.2f}, std={df_valid_contacts['real_t_diff'].std():.2f}, max={df_valid_contacts['real_t_diff'].max():.2f}")
print(f"Rotation difference (deg): mean={df_valid_contacts['real_angle_diff'].mean():.2f}, std={df_valid_contacts['real_angle_diff'].std():.2f}, max={df_valid_contacts['real_angle_diff'].max():.2f}")      

#--- VISUALIZATION ---#
# Create 2x3 subplot figure for 6D pose error histograms
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('6D Pose Errors Between Trimesh Simulation and Real Data', fontsize=16, fontweight='bold')

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
plt.show()

# Save the figure
plt.savefig('trimesh_6d_pose_errors_histogram.png', dpi=300, bbox_inches='tight')
print("Figure saved as 'trimesh_6d_pose_errors_histogram.png'")
print("\n")

# Print detailed statistics for each component
print("=== Detailed 6D Pose Error Statistics (Trimesh Sim vs Real) ===")
for i, (label, unit) in enumerate(zip(labels, units)):
    data = pose_errors_6d[:, i]
    mae_val = np.mean(np.abs(data))
    print(f"{label} {unit}:")
    print(f"  Mean: {np.mean(data):.3f}, MAE: {mae_val:.3f}, Std: {np.std(data):.3f}")
    print(f"  Min: {np.min(data):.3f}, Max: {np.max(data):.3f}")
    print(f"  Median: {np.median(data):.3f}, 95th percentile: {np.percentile(data, 95):.3f}")
    print()

# Save updated dataframe with comparison results
df_valid_contacts.to_csv('/home/rp/abhay_ws/contact-manifold-state-estimation/data_generation/trimesh_contact_data_with_real_comparison.csv', index=False)
print("Updated dataframe with real data comparison saved as 'trimesh_contact_data_with_real_comparison.csv'")

import pdb; pdb.set_trace() 
