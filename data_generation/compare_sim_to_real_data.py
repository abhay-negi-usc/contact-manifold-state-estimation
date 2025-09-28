import numpy as np 
import pandas as pd 
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
from tqdm import tqdm
import torch
from utils.pose_utils import torch_pose_xyzabc_to_matrix

#--- CONFIG ---# 
filepath_sim_data = "/media/rp/Elements1/abhay_ws/contact-manifold-state-generation/data/cross_data/cross_data_merged/processed_data/cross_contact_poses_mujoco.csv"
filepath_real_data = "/home/rp/dhanush_ws/sunrise-wrapper/contact_maps/cross_round_tight/CASE_cross_round_tight_aut_map_4.pkl" 

#--- LOAD DATA ---# 
df_sim = pd.read_csv(filepath_sim_data) 
df_real = pd.read_pickle(filepath_real_data) 
print(f"Loaded {len(df_sim)} rows of sim data from {filepath_sim_data}")
print(f"Loaded {len(df_real)} rows of real data from {filepath_real_data}")
print("\n")

# truncate real data
N_real_max = 100_000
if len(df_real) > N_real_max:
    df_real = df_real.sample(N_real_max, random_state=42).reset_index(drop=True)
    print(f"Truncated real data to {N_real_max} random samples for faster processing.")
print(f"Using {len(df_real)} rows of real data for comparison.")

#--- PROCESS DATA ---#

# create kdtree of sim data rows 
kdtree_sim = cKDTree(df_sim[['x', 'y', 'z', 'a', 'b', 'c']].to_numpy())
print(f"KDTree of sim data with {len(df_sim)} points created.")
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
# for each row in df_real, find nearest neighbor in df_sim using kdtree
_, indices = kdtree_sim.query(df_real[['x', 'y', 'z', 'a', 'b', 'c']].to_numpy(), k=1)
print(f"Found nearest neighbors in sim data for each real data point.")
# compute transformation difference between each real data point and its nearest sim neighbor
for i, idx in enumerate(tqdm(indices, desc="Computing transformation differences")):
    tf_real = transforms_real_torch[i]
    tf_sim = transforms_sim_torch[idx]
    tf_diff = torch.inverse(tf_real) @ tf_sim
    t_diff = tf_diff[:3, 3].numpy() # already in mm 
    r_diff = R.from_matrix(tf_diff[:3, :3].numpy())
    angle_diff = r_diff.magnitude() * (180.0 / np.pi) # in degrees
    df_real.at[i, 'sim_index'] = idx
    df_real.at[i, 'sim_t_diff'] = np.linalg.norm(t_diff)
    df_real.at[i, 'sim_angle_diff'] = angle_diff

print(f"Computed transformation differences between real data and nearest sim data.")
print("\n")

# print statistics of differences
print(f"Translation difference (mm): mean={df_real['sim_t_diff'].mean():.2f}, std={df_real['sim_t_diff'].std():.2f}, max={df_real['sim_t_diff'].max():.2f}")
print(f"Rotation difference (deg): mean={df_real['sim_angle_diff'].mean():.2f}, std={df_real['sim_angle_diff'].std():.2f}, max={df_real['sim_angle_diff'].max():.2f}")      







