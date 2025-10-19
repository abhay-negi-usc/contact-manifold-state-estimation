import numpy as np
import pandas as pd
from cicp.icp import ContactPoseMap
from cicp.cicp_util import *
import cicp.cicp_util as CU
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import os


geometry = "bnc_path"
 
# map_path = "/media/rp/Elements/abhay_ws/mujoco_contact_graph_generation/results/cross_rounded_data/perturb_v3/processed_data/cross_rounded_peg_contact_map_sim_with_normals_neighbors_10.csv"
# map_path = "/media/rp/Elements1/abhay_ws/real_contact_data/MAP1_GEAR_JAN16_PEG_filtered.csv"
map_path = "/home/rp/abhay_ws/contact-manifold-state-estimation/real_data/BNC_real_MAP_1_smoothed_path.csv"
map_df = pd.read_csv(map_path)
map_df.rename(columns={'FK_X':'x','FK_Y':'y','FK_Z':'z','FK_A':'a','FK_B':'b','FK_C':'c'}, inplace=True) 
map_df.columns = map_df.columns.str.lower()
map_df = map_df[map_df['z'] < 24.5] # REMOVE ME 
map_data = map_df[["x", "y", "z", "a", "b", "c"]].values

# transform map data 
delta = np.array([5,0,0,0,0,0]) 
transformed_map_data = CU.batch_apply_delta_poses(map_data, np.repeat(delta.reshape(1,6), map_data.shape[0], axis=0))

# cpm = ContactPoseMap(pose_data=transformed_map_data, numpy_seed=np.random.randint(1_000_000_000), flag_batch=True)
# cpm.downsample_map(10_000)

# Create output directory if it doesn't exist
os.makedirs('./real_data/visualization', exist_ok=True)

# Extract x, y, z coordinates and rotation data
points_3d = transformed_map_data[:, :3]  # First 3 columns are x, y, z
rotations = transformed_map_data[:, 3:6]  # Assuming columns 3, 4, 5 are rotation axes

def create_rotating_gif(points, colors, color_label, filename):
    """Create a rotating 3D plot and save as GIF"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create scatter plot
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                        c=colors, cmap='viridis', s=1, alpha=0.6)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'6D Point Cloud - Colored by {color_label}')
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax, label=color_label)
    
    # Set equal aspect ratio
    max_range = np.array([points[:,0].max()-points[:,0].min(), 
                          points[:,1].max()-points[:,1].min(),
                          points[:,2].max()-points[:,2].min()]).max() / 2.0
    mid_x = (points[:,0].max()+points[:,0].min()) * 0.5
    mid_y = (points[:,1].max()+points[:,1].min()) * 0.5
    mid_z = (points[:,2].max()+points[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Animation function
    def animate(frame):
        ax.view_init(elev=20, azim=frame * 3)  # 3 degrees per frame
        return [scatter]
    
    # Create animation (360 degrees rotation, 120 frames)
    frames = 120
    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=100, blit=False)
    
    # Save as GIF
    print(f"Saving {filename}...")
    try:
        anim.save(f'./real_data/visualization/{filename}', writer='pillow', fps=12)
        print(f"Animation saved as './real_data/{filename}'")
    except Exception as e:
        print(f"Failed to save {filename}: {e}")
    
    plt.close(fig)

# Create three GIFs with different colorings
print("Creating rotating point cloud visualizations...")

# GIF 1: Colored by X-axis rotation
create_rotating_gif(points_3d, rotations[:, 0], 'Z-axis Rotation', f'{geometry}_pointcloud_z_rotation.gif')

# GIF 2: Colored by Y-axis rotation  
create_rotating_gif(points_3d, rotations[:, 1], 'Y-axis Rotation', f'{geometry}_pointcloud_y_rotation.gif')

# GIF 3: Colored by Z-axis rotation
create_rotating_gif(points_3d, rotations[:, 2], 'X-axis Rotation', f'{geometry}_pointcloud_x_rotation.gif')

print("All animations completed!")

