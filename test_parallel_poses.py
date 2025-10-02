#!/usr/bin/env python3
"""
Test script for parallel pose evaluation with adaptive sampling
"""

import sys
import os
sys.path.append('/home/rp/abhay_ws/contact-manifold-state-estimation')

from data_generation.trimesh_generate_contact_data_adaptive_sampling import main, config

# Create a test configuration for parallel pose evaluation
test_config = config.copy()
test_config["sampling"]["xyz"]["z"]["max"] = 0.010  # Small z range
test_config["sampling"]["xyz"]["z"]["step"] = 0.005  # 3 z steps
test_config["sampling"]["xyz"]["x"]["step"] = 0.002  # Coarser sampling for faster test
test_config["sampling"]["xyz"]["y"]["step"] = 0.002  # Coarser sampling for faster test
test_config["sampling"]["abc"]["a"]["step"] = 2.0   # Coarser angular sampling
test_config["sampling"]["abc"]["b"]["step"] = 2.0
test_config["sampling"]["abc"]["c"]["step"] = 2.0
test_config["sampling"]["adaptive"] = True  # Enable adaptive sampling
test_config["sampling"]["max_penetration_depth"] = 0.0005  # 0.5mm threshold for testing

# Test parallel configuration
test_config["parallel"]["enabled"] = True  # Enable parallel processing
test_config["parallel"]["adaptive_parallel"] = True  # Enable parallel adaptive sampling
test_config["parallel"]["z_slice_parallel"] = True  # Enable z-slice parallelization
test_config["parallel"]["axis_parallel"] = False  # Disable axis parallelization to avoid nested pools
test_config["parallel"]["pose_parallel"] = True  # Enable pose parallelization (this is what we're testing!)
test_config["parallel"]["workers"] = 4  # Use fewer workers for testing

test_config["save"]["csv_path"] = "./test_parallel_poses_results.csv"
test_config["save"]["npz_path"] = "./test_parallel_poses_results.npz"

if __name__ == "__main__":
    print("Testing parallel pose evaluation with adaptive sampling...")
    try:
        main(test_config)
        print("Test completed successfully!")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
