import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import glob 
import os 
import pickle 
from scipy.spatial.transform import Rotation as R 
import random 

# go through each timestep of each trial of data 

geometry = "cross_real" 

dir_results = f"/media/rp/Elements1/abhay_ws/contact-manifold-state-generation/data/{geometry}_data/" 
dir_pkl = dir_results + "/pkl" 
pkl_files = sorted(glob.glob(os.path.join(dir_pkl, "*.pkl")), key=os.path.getmtime)
output_file = f"{geometry}_contact_poses_mujoco.csv"

dir_save = dir_pkl.removesuffix("pkl") + "processed_data"
if not os.path.exists(dir_save): 
    os.makedirs(dir_save)

# list of all contact state history 
N_trials = 250_000 # FIXME: program gets killed if N_trials is 1_000_000 
N_trials = len(pkl_files) if len(pkl_files) < N_trials else N_trials
# N_max = 1_000_000_000 
N_max = 10_000 
pkl_files = [pkl_files[i] for i in random.sample(range(len(pkl_files)), N_trials)]
pose_boundary_list = [] 

hole_length = 0.025 # m 
hole_diameter = 0.026 # m
upper_z_limit = hole_length + 0.001 # m, 1mm above the hole top surface
xy_limit = hole_diameter/2 + 0.0001 # m, 0.1mm outside the hole diameter 

for i, pkl_file in enumerate(pkl_files): 

    # Read the pickle file
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    # unpack data 
    state_hist = data['state_hist'] 
    contact_pos = data['contact_pos'] 

    # check if there is contact within the hole area, if so, add the pose to the list 
    for j, contact_pos_j in enumerate(contact_pos): # iterate through each time step 
        if len(contact_pos_j) > 0: # if there is contact 
            for k, contact_pos_hole_frame in enumerate(contact_pos_j): # iterate through each contact at current time step 
                if geometry == "cross_real":
                    if contact_pos_hole_frame[2] < upper_z_limit and max(abs(contact_pos_hole_frame[:2])) < xy_limit: # if contact is below the surface and within hole area 
                        peg_pose = state_hist[j, 1:8] 
                        pose_boundary_list.append(peg_pose) 
                        print(f"Added pose at j: {j}, k: {k}, contact_pos: {contact_pos_hole_frame}, peg_pose: {peg_pose}")
                        
                        # debug: checking where contact is occurring 
                        if peg_pose[0] == 0 and peg_pose[1] == 0 and peg_pose[3] == 1 and peg_pose[4] == 0 and peg_pose[5] == 0 and peg_pose[6] == 0: 
                            print("Contact at center of hole, likely at bottom surface contact.")
                            import pdb; pdb.set_trace() 

                        continue # don't need to check pose again 
    # print progress rate every 1% of total iterations 
    if (i+1) % np.floor(len(pkl_files)/100) == 0: 
        print(f"Completion Progress: {i+1}/{len(pkl_files)}")  

    if len(pose_boundary_list) > N_max: 
        print(f"Reached maximum data points of {N_max}. Stopping further processing.")
        break

import pdb; pdb.set_trace() 

# convert list to dataframe 
pose_boundary_df = pd.DataFrame(pose_boundary_list, columns=['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz']) 

# drop columns with NaN values 
pose_boundary_df.dropna(inplace=True) 

# convert quaternion to euler angles 
quaternions = pose_boundary_df[['qw', 'qx', 'qy', 'qz']].values 
euler_angles = R.from_quat(quaternions, scalar_first=True).as_euler("xyz", degrees=True) 
pose_boundary_df['a'] = euler_angles[:,2]
pose_boundary_df['b'] = euler_angles[:,1]
pose_boundary_df['c'] = euler_angles[:,0] 

# convert position from meters to millimeters 
pose_boundary_df[['x', 'y', 'z']] *= 1000 

# save the dataframe 
if not os.path.exists(dir_save): 
    os.makedirs(dir_save)

pose_boundary_df.to_pickle(os.path.join(dir_save, output_file.removesuffix(".csv") + ".pkl")) 
pose_boundary_df.to_csv(os.path.join(dir_save, output_file), index=False)  

print(f"Pose boundary data saved to {os.path.join(dir_save, output_file)}.") 
print("Number of data points: ", len(pose_boundary_df)) 
