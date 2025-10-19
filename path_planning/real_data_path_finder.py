import numpy as np 
import pandas as pd
from scipy.spatial.transform import Rotation as R
from utils.pose_utils import mean_rotation_from_rotvecs 
from tqdm import tqdm

real_data_path = "./real_data/BNC_real_MAP_1.csv" 
real_data_df = pd.read_csv(real_data_path)

# order rows by z value
real_data_df = real_data_df.sort_values(by='z')

# for moving window of z, find mean values of x,y,a,b,c
dz = 0.25 # step resolution in mm 
n_waypoints = int((real_data_df['z'].max() - real_data_df['z'].min()) / dz) + 1
window_size = len(real_data_df) // n_waypoints
# window_size = len(real_data_df) // 100 # moving window of 0.1% of data size - end up with 1000 points 
z_values = real_data_df['z'].values
x_values = real_data_df['x'].values
y_values = real_data_df['y'].values 
rotvec_values = real_data_df[['wx', 'wy', 'wz']].values

# Pre-allocate array for better performance
num_windows = len(z_values) - window_size + 1
smoothed_data = np.zeros((num_windows, 9))

# Vectorized window operations using lambda functions
get_window_means = lambda arr, i: np.mean(arr[i:i+window_size])

print(f"Processing {num_windows} windows with window size {window_size}")
print(f"Target waypoints: {n_waypoints}, Step resolution: {dz}mm")

# Add tqdm progress bar
for i in tqdm(range(num_windows), desc="Smoothing trajectory", unit="windows"):
    # Vectorized mean calculations
    smoothed_data[i, 0] = get_window_means(x_values, i)  # mean_x
    smoothed_data[i, 1] = get_window_means(y_values, i)  # mean_y  
    smoothed_data[i, 2] = get_window_means(z_values, i)  # mean_z

    # Rotation calculations (can't easily vectorize due to mean_rotation_from_rotvecs)
    R_mean, w_mean = mean_rotation_from_rotvecs(rotvec_values[i:i+window_size], return_rotvec=True) 
    smoothed_data[i, 6:9] = w_mean  # mean_wx, mean_wy, mean_wz
    smoothed_data[i, 5], smoothed_data[i, 4], smoothed_data[i, 3] = R.from_matrix(R_mean).as_euler('xyz', degrees=True)  # mean_c, mean_b, mean_a

# Create DataFrame using lambda for column mapping
column_mapper = lambda cols: ['x', 'y', 'z', 'a', 'b', 'c', 'wx', 'wy', 'wz']
smoothed_df = pd.DataFrame(smoothed_data, columns=column_mapper(None))

# Reorder by z column in descending order
print("Sorting data by z-value...")
smoothed_df = smoothed_df.sort_values(by='z', ascending=False)

# Save to CSV
smoothed_csv_path = "./real_data/BNC_real_MAP_1_smoothed_path.csv"
print(f"Saving smoothed trajectory to {smoothed_csv_path}")
smoothed_df.to_csv(smoothed_csv_path, index=False)
print(f"Saved {len(smoothed_df)} waypoints")

# Plot trajectory in 2x3 figure using lambda for labels
import matplotlib.pyplot as plt
get_dimension_label = lambda i: ['x (mm)', 'y (mm)', 'z (mm)', 'a (deg)', 'b (deg)', 'c (deg)'][i]
get_subplot_pos = lambda i: (i // 3, i % 3)

print("Generating plots...")
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
list(map(lambda i: (
    axs[get_subplot_pos(i)].plot(smoothed_data[:, i]),
    axs[get_subplot_pos(i)].set_xlabel('Point Index'),
    axs[get_subplot_pos(i)].set_ylabel(get_dimension_label(i)),
    axs[get_subplot_pos(i)].set_title(f'Trajectory of {get_dimension_label(i)}')
), range(6)))

plt.tight_layout()
plt.show()
print("Processing complete!")
