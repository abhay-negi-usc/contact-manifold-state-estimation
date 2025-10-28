import numpy as np 
import os 

from cicp.icp_manifold import ContactPoseManifold
import cicp.cicp_util as SU

model_path = "./model_training/checkpoints/extrusion_mujoco_run_1_best_NN_model_xyzabc.pth"
geometry = "extrusion_mujoco"
observations_path = "/media/rp/Elements1/abhay_ws/ICP Data from Dhanush Google Drive/extrusion_data/debug_obs_set/"

# # read all npy files in the observations_path
# observation_files = [f for f in os.listdir(observations_path) if f.endswith('.npy')]
# observation_file = observation_files[0]  # pick the first file for testing
observation_file = "observations_spiral_tilt_0_1_PI.npy"
observation_pose_B_P = np.load(os.path.join(observations_path, observation_file))

T_B_H_file = "T_B_H_spiral_tilt_0_1_PI.npy" 
true_pose_B_H = np.load(os.path.join(observations_path, T_B_H_file))
true_pose_H_B = SU.batch_invert_poses_xyzabc(true_pose_B_H.reshape(1,6))

cpm = ContactPoseManifold(geometry=geometry)
cpm.load_model_from_path(model_path) 
cpm.set_true_hole_pose(true_pose_B_H.squeeze(axis=0))

cpm.set_observation(
    observation=observation_pose_B_P, 
    sample_size=1_000,  
)

# initial_guess_pose_h_B = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])  # shape (1, 6)
initial_guess_pose_h_B = true_pose_H_B

cpm.set_initial_guess(initial_guess_pose_h_B.squeeze(axis=0))
# cpm.set_random_initial_guess(offset_range=[5, 5, 0, 5, 5, 5], verbose=False) 
final_pose_h_B, pose_guesses_h_B = cpm.estimate_pose(
    n_iter=1_000, 
    random_initial_guess=False, 
)

print("True pose (h_B): ", true_pose_H_B)
print("Final estimated pose (h_B): ", final_pose_h_B)

# plot results
cpm.offset_range = [5, 5, 5, 5, 5, 5]  # set offset range for plotting
cpm.plot_results()