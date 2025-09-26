import mujoco 
import mediapy as media 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy.spatial.transform import Rotation as R 
import pickle 
import os 
import cv2
import argparse
import time
import random 

# Parse command line arguments for parallel execution
def parse_args():
    parser = argparse.ArgumentParser(description='Generate contact data with support for parallel execution')
    parser.add_argument('--worker-id', type=int, default=0, 
                       help='Unique worker ID for parallel execution (default: 0)')
    parser.add_argument('--num-workers', type=int, default=1,
                       help='Total number of parallel workers (default: 1)')
    parser.add_argument('--trials-per-worker', type=int, default=None,
                       help='Number of trials per worker (default: auto-split total trials)')
    parser.add_argument('--output-suffix', type=str, default='',
                       help='Additional suffix for output directory (default: empty)')
    return parser.parse_args()

args = parse_args()

# hyperparameters 
geometry = "cross_real" # "extrusion" "cross" "plug_3_pin" 

# Create unique output directory for this worker
# base_output_dir = f"./data/{geometry}_data"
base_output_dir = f"/media/rp/Elements1/abhay_ws/contact-manifold-state-generation/data/{geometry}_data/{geometry}_data"
if args.num_workers > 1:
    # Use worker-specific subdirectory for parallel execution
    worker_suffix = f"_worker_{args.worker_id}"
    if args.output_suffix:
        worker_suffix += f"_{args.output_suffix}"
    dir_results = base_output_dir + worker_suffix + "/"
else:
    # Use base directory for single worker
    if args.output_suffix:
        dir_results = base_output_dir + f"_{args.output_suffix}/"
    else:
        dir_results = base_output_dir + "/"

xml_path = f"./data_generation/mujoco_environments/{geometry}_env.xml" 
peg_length = 0.025 

# Calculate trials for this worker
total_trials = 250_000  # Total trials across all workers
if args.trials_per_worker is not None:
    num_trials = args.trials_per_worker
    # For explicitly set trials per worker, start from worker_id * trials_per_worker
    trial_start_idx = args.worker_id * args.trials_per_worker
else:
    # Auto-split trials among workers
    trials_per_worker = total_trials // args.num_workers
    remaining_trials = total_trials % args.num_workers
    
    if args.worker_id < remaining_trials:
        num_trials = trials_per_worker + 1
        trial_start_idx = args.worker_id * (trials_per_worker + 1)
    else:
        num_trials = trials_per_worker
        trial_start_idx = remaining_trials * (trials_per_worker + 1) + (args.worker_id - remaining_trials) * trials_per_worker

# Set random seed based on worker ID and current time for reproducibility while avoiding conflicts
random_seed = int(time.time() * 1000) % 10000 + args.worker_id * 10000
np.random.seed(random_seed)
random.seed(random_seed)

print(f"Worker {args.worker_id}: Processing {num_trials} trials")
print(f"Worker {args.worker_id}: Output directory: {dir_results}")
print(f"Worker {args.worker_id}: Random seed: {random_seed}") 

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
flag_show_video = False
flag_save_video = False     

os.makedirs(dir_results + "/pkl", exist_ok=True)
if flag_save_video: 
    os.makedirs(dir_results + "/vid", exist_ok=True)

n_frames = 2500    
height = 720 
width = 960
frames = []

# visualize contact frames and forces, make body transparent
options = mujoco.MjvOption()
mujoco.mjv_defaultOption(options)
options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False 
options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

# tweak scales of contact visualization elements
model.vis.scale.contactwidth = 0.02
model.vis.scale.contactheight = 0.02
model.vis.scale.forcewidth = 0.05
model.vis.map.force = 0.3
model.opt.gravity = (0,0,0)

for idx_trial in range(num_trials): 
    # Calculate global trial index for unique naming
    if args.num_workers > 1:
        global_trial_idx = trial_start_idx + idx_trial
    else:
        global_trial_idx = idx_trial 

    # define initial conditions 
    x0 = 0
    y0 = 0
    z0 = np.random.uniform(0.000, 0.010)  
    a0 = 0  
    b0 = 0  
    c0 = 0 

    mujoco.mj_resetData(model, data)
    data.qpos = np.array([x0, y0, z0, a0, b0, c0]) 
    data.qvel = np.zeros(6) 
    mujoco.mj_forward(model, data)

    # initialize data structures 
    frames = []
    state_hist = np.zeros((n_frames,1+3+4+6))   
    contact_hist = [] 
    contact_num = [] 
    contact_geom1 = [] 
    contact_geom2 = [] 
    contact_dist = [] 
    contact_pos = [] 
    contact_frame = [] 
    ctrl_hist = np.zeros((n_frames,1+6))   
    sensor_hist = np.zeros((n_frames,13))

    # define controller parameters  
    z_step = 1e-4
    # initialize random signs 
    random_signs = np.random.choice([-1, 1], size=5)

    if flag_save_video: 
        # Initialize video writer with unique filename
        video_path = dir_results + f"/vid/trial_{global_trial_idx}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        video_writer = cv2.VideoWriter(video_path, fourcc, 30, (width, height))

    # Simulate and display video.
    with mujoco.Renderer(model, height, width) as renderer:
        for i in range(n_frames): 
            while data.time < i/(30.0*4): #1/4x real time
                mujoco.mj_step(model, data)
            if flag_show_video or flag_save_video: 
                renderer.update_scene(data, "track", options)
                frame = renderer.render()
                frames.append(frame)

                # Convert frame to BGR format for OpenCV
                if flag_save_video:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    video_writer.write(frame_bgr)

            # save data 
            state_hist[i,:] = np.concatenate([np.array([data.time]), data.xpos[2], data.xquat[2], data.qvel]) 
            contact_num.append(len(data.contact.geom1)) 
            contact_geom1.append(np.array(data.contact.geom1)) 
            contact_geom2.append(np.array(data.contact.geom2)) 
            contact_dist.append(np.array(data.contact.dist))  
            contact_pos.append(np.array(data.contact.pos)) 
            contact_frame.append(np.array(data.contact.frame)) 
            ctrl_hist[i,:] = np.concatenate([np.array([data.time]), data.ctrl])  
            sensor_hist[i,:] = data.sensordata  

            # controller update 
            peg_x_axis = data.xmat[2].reshape(3,3)[:,0] 
            peg_y_axis = data.xmat[2].reshape(3,3)[:,1] 
            peg_z_axis = data.xmat[2].reshape(3,3)[:,2]  
            angle_step = 1.0 * np.pi/180 
            ii = i * angle_step 
            amplitudes = np.array([3,5,7]) + np.random.normal(0, 1, 3)   
            np.random.shuffle(amplitudes) 
            theta = amplitudes[0] * ii 
            phi = amplitudes[1] * ii 
            psi = amplitudes[2] * ii 
            tau = ii 
            r = 1.0e-9 
            delta_x = r * np.cos(tau) * peg_x_axis * random_signs[0]
            delta_y = r * np.sin(tau) * peg_y_axis * random_signs[1] 
            delta_z = +1 * z_step * peg_z_axis 
            delta_pos = delta_x + delta_y + delta_z 
            delta_a = 5.0 * np.sin(theta) * random_signs[2] 
            delta_b = 5.0 * np.sin(phi) * random_signs[3] 
            delta_c = 5.0 * np.sin(psi) * random_signs[4] 
            delta_angle = np.array([delta_a, delta_b, delta_c]) * np.pi/180 
            delta_pose_tool = np.concatenate([delta_pos, delta_angle])
            noise = np.concatenate([np.random.normal(0, 1.0, 2)*1e-3, np.random.normal(0, 1.0, 1)*1e-3, np.random.normal(0, 1, 3)*np.pi/180])  
            data.ctrl = data.qpos + delta_pose_tool + noise 
                
        if flag_show_video: 
            media.show_video(frames, fps=30)

        if flag_save_video:
            # Release video writer
            video_writer.release() 

        data_dict = {
            'state_hist': state_hist, 
            'contact_num': contact_num,   
            'contact_geom1': contact_geom1,
            'contact_geom2': contact_geom2,
            'contact_dist': contact_dist,
            'contact_pos': contact_pos,
            'contact_frame': contact_frame,
            'ctrl_hist': ctrl_hist,
            'sensor_hist': sensor_hist,
        }

        # save data as pkl file with unique filename
        with open(dir_results + f"/pkl/trial_{global_trial_idx}.pkl", 'wb') as f: 
            pickle.dump(data_dict, f) 
        
        print(f"Worker {args.worker_id}: Trial {idx_trial}/{num_trials} (global: {global_trial_idx}) complete.") 