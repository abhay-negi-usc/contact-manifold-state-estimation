#!/usr/bin/env python3
"""
Test script to verify the new rotation convention matches scipy's R.from_euler('xyz', [c, b, a], degrees=True)
"""
import numpy as np

def abc_to_matrix(a, b, c, degrees=False):
    """
    Convert angles a, b, c to rotation matrix using scipy convention:
    R.from_euler('xyz', [c, b, a], degrees=True).as_matrix()
    """
    if degrees:
        a, b, c = np.deg2rad([a, b, c])
    
    # Rotation matrices for x, y, z rotations
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cc, sc = np.cos(c), np.sin(c)
    
    # X rotation (angle c)
    Rx = np.array([[1,  0,   0],
                   [0, cc, -sc],
                   [0, sc,  cc]])
    
    # Y rotation (angle b)
    Ry = np.array([[ cb, 0, sb],
                   [  0, 1,  0],
                   [-sb, 0, cb]])
    
    # Z rotation (angle a)
    Rz = np.array([[ca, -sa, 0],
                   [sa,  ca, 0],
                   [ 0,   0, 1]])
    
    # Apply in order: Rz * Ry * Rx (equivalent to 'xyz' order with [c,b,a])
    return Rz @ Ry @ Rx

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
