#!/usr/bin/env python3
"""
Test script to verify the new rotation convention matches scipy's R.from_euler('xyz', [c, b, a], degrees=True)
"""
import numpy as np

# Import the new rotation function from our script
import sys
sys.path.append('/home/rp/abhay_ws/contact-manifold-state-estimation/data_generation')
from trimesh_generate_contact_data import abc_to_matrix

# Test with scipy if available
try:
    from scipy.spatial.transform import Rotation as R
    
    # Test angles
    a, b, c = 10, 20, 30  # degrees
    
    # Our implementation
    our_matrix = abc_to_matrix(a, b, c, degrees=True)
    
    # Scipy implementation 
    scipy_matrix = R.from_euler('xyz', [c, b, a], degrees=True).as_matrix()
    
    print("Test angles: a=10°, b=20°, c=30°")
    print("\nOur rotation matrix:")
    print(our_matrix)
    print("\nScipy rotation matrix:")
    print(scipy_matrix)
    print("\nDifference (should be near zero):")
    print(np.abs(our_matrix - scipy_matrix))
    print("\nMax absolute difference:", np.max(np.abs(our_matrix - scipy_matrix)))
    
    if np.allclose(our_matrix, scipy_matrix, atol=1e-10):
        print("✅ SUCCESS: Our implementation matches scipy!")
    else:
        print("❌ ERROR: Our implementation doesn't match scipy!")
        
except ImportError:
    print("Scipy not available, showing our rotation matrix only:")
    a, b, c = 10, 20, 30  # degrees
    our_matrix = abc_to_matrix(a, b, c, degrees=True)
    print("Test angles: a=10°, b=20°, c=30°")
    print("\nOur rotation matrix:")
    print(our_matrix)
