#!/usr/bin/env python3
"""
Utility script for managing checkpoints in the contact data generation script.
"""
import os
import pickle
import time
import argparse

def check_checkpoint_status(checkpoint_path):
    """Check the status of a checkpoint file."""
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at: {checkpoint_path}")
        return
    
    try:
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        completed_poses = checkpoint_data.get('completed_poses', set())
        all_poses = checkpoint_data.get('all_poses', [])
        results = checkpoint_data.get('results', [])
        remaining_poses = checkpoint_data.get('remaining_poses', None)
        timestamp = checkpoint_data.get('timestamp', 0)
        
        print(f"Checkpoint Status:")
        print(f"  File: {checkpoint_path}")
        print(f"  Created: {time.ctime(timestamp)}")
        print(f"  Progress: {len(completed_poses)}/{len(all_poses)} poses completed ({len(completed_poses)/len(all_poses)*100:.1f}%)")
        print(f"  Results collected: {len(results)}")
        if remaining_poses is not None:
            print(f"  Remaining poses: {len(remaining_poses)}")
        else:
            remaining_count = len(all_poses) - len(completed_poses)
            print(f"  Remaining poses: {remaining_count} (estimated)")
        
        if len(results) > 0:
            # Show some statistics
            import numpy as np
            results_array = np.array(results, dtype=float)
            contact_count = np.sum(results_array[:, 6] > 0.5)  # column 6 is contact flag
            print(f"  Contact poses found: {contact_count}/{len(results)} ({contact_count/len(results)*100:.1f}%)")
        
    except Exception as e:
        print(f"Error reading checkpoint: {e}")

def clean_checkpoint(checkpoint_path):
    """Remove a checkpoint file."""
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Removed checkpoint: {checkpoint_path}")
    else:
        print(f"No checkpoint found at: {checkpoint_path}")

def list_checkpoints(directory="."):
    """List all checkpoint files in a directory."""
    checkpoint_files = []
    for filename in os.listdir(directory):
        if filename.endswith("_checkpoint.pkl"):
            checkpoint_files.append(os.path.join(directory, filename))
    
    if not checkpoint_files:
        print(f"No checkpoint files found in {directory}")
        return
    
    print(f"Found {len(checkpoint_files)} checkpoint files:")
    for cp_file in sorted(checkpoint_files):
        print(f"  {cp_file}")
        check_checkpoint_status(cp_file)
        print()

def main():
    parser = argparse.ArgumentParser(description="Manage contact data generation checkpoints")
    parser.add_argument("action", choices=["status", "clean", "list"], 
                       help="Action to perform")
    parser.add_argument("--checkpoint", "-c", 
                       help="Path to checkpoint file (for status/clean)")
    parser.add_argument("--directory", "-d", default=".",
                       help="Directory to search for checkpoints (for list)")
    
    args = parser.parse_args()
    
    if args.action == "status":
        if not args.checkpoint:
            parser.error("--checkpoint required for status action")
        check_checkpoint_status(args.checkpoint)
    
    elif args.action == "clean":
        if not args.checkpoint:
            parser.error("--checkpoint required for clean action")
        clean_checkpoint(args.checkpoint)
    
    elif args.action == "list":
        list_checkpoints(args.directory)

if __name__ == "__main__":
    main()
