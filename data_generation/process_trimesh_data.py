import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

# --- Config ---
BATCH_SIZE = 100_000  # tune based on your RAM/CPU
INPUT_CSV  = "/home/rp/abhay_ws/contact-manifold-state-estimation/data_generation/extrusion_sim_data.csv"
OUTPUT_CSV = INPUT_CSV.replace(".csv", "_with_logmaps.csv")

# --- Progress bar (tqdm) ---
try:
    from tqdm import tqdm
except ImportError:
    # lightweight fallback if tqdm isn't installed
    def tqdm(x, **kwargs): return x

# --- Load data ---
df = pd.read_csv(INPUT_CSV)
# remove units row if present (as in your script)
translation_units = df.loc[0]['x']
rotation_units = df.loc[0]['a']
df = df.drop(index=0).reset_index(drop=True)

# pull Euler angles (degrees) as an (N,3) array
if rotation_units == 'rad':
    df[['a', 'b', 'c']] = df[['a', 'b', 'c']].astype(float) * (180.0 / np.pi) # rad to deg
elif rotation_units == 'deg':
    df[['a', 'b', 'c']] = df[['a', 'b', 'c']].astype(float)
else:
    raise ValueError(f"Unknown rotation units: {rotation_units}")
angles_deg = df[['a', 'b', 'c']].to_numpy(dtype=float)
N = angles_deg.shape[0]

# --- Compute log-maps in batches with a progress bar ---
logmaps = np.empty((N, 3), dtype=float)

for start in tqdm(range(0, N, BATCH_SIZE), total=(N + BATCH_SIZE - 1) // BATCH_SIZE,
                  desc="Converting Euler→logmap"):
    end = min(start + BATCH_SIZE, N)
    batch = angles_deg[start:end]

    # Use SciPy's vectorized conversion:
    # from_euler(...).as_rotvec() returns axis-angle vector (the so(3) log map)
    rotvec = R.from_euler('xyz', batch, degrees=True).as_rotvec()  # shape (B,3), in radians
    logmaps[start:end] = rotvec

# --- Attach to dataframe with correct naming order ---
df[['wx', 'wy', 'wz']] = logmaps

# add back in the units row 
units_row = {
    'x': translation_units,
    'y': translation_units,
    'z': translation_units,
    'a': 'deg',
    'b': 'deg',
    'c': 'deg',
    'wx': 'rad',
    'wy': 'rad',
    'wz': 'rad',
    'contact': '1=contact',
    'contact_distance': translation_units
}
df = pd.concat([pd.DataFrame([units_row]), df], ignore_index=True)

# --- Save ---
df.to_csv(OUTPUT_CSV, index=False)
print(f"Wrote: {OUTPUT_CSV}")
