#!/usr/bin/env python3
"""
Helper script to merge output from multiple parallel workers into a single directory.
"""

import os
import shutil
import argparse
import glob
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Merge output from parallel workers')
    parser.add_argument('--base-dir', type=str, default='./data',
                       help='Base directory containing worker outputs (default: ./data)')
    parser.add_argument('--output-dir', type=str, default='./data/cross_real_data_merged',
                       help='Output directory for merged results (default: ./data/cross_real_data_merged)')
    parser.add_argument('--worker-pattern', type=str, default='*_worker_*',
                       help='Pattern to match worker directories (default: *_worker_*)')
    parser.add_argument('--copy-files', action='store_true',
                       help='Copy files instead of moving them (default: move)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print operations without executing them')
    return parser.parse_args()

def merge_directories(args):
    base_path = Path(args.base_dir)
    output_path = Path(args.output_dir)
    
    # Find all worker directories
    worker_dirs = list(base_path.glob(args.worker_pattern))
    
    if not worker_dirs:
        print(f"No worker directories found matching pattern: {args.worker_pattern}")
        print(f"Looking in: {base_path}")
        return
    
    print(f"Found {len(worker_dirs)} worker directories:")
    for worker_dir in sorted(worker_dirs):
        print(f"  {worker_dir}")
    print()
    
    if not args.dry_run:
        # Create output directories
        (output_path / "pkl").mkdir(parents=True, exist_ok=True)
        (output_path / "vid").mkdir(parents=True, exist_ok=True)
    
    total_files_moved = 0
    
    # Process each worker directory
    for worker_dir in sorted(worker_dirs):
        print(f"Processing {worker_dir.name}...")
        
        # Process pkl files
        pkl_dir = worker_dir / "pkl"
        if pkl_dir.exists():
            pkl_files = list(pkl_dir.glob("*.pkl"))
            print(f"  Found {len(pkl_files)} pkl files")
            
            for pkl_file in pkl_files:
                dst_file = output_path / "pkl" / pkl_file.name
                if args.dry_run:
                    action = "copy" if args.copy_files else "move"
                    print(f"    Would {action}: {pkl_file} -> {dst_file}")
                else:
                    if args.copy_files:
                        shutil.copy2(pkl_file, dst_file)
                    else:
                        shutil.move(str(pkl_file), str(dst_file))
                    total_files_moved += 1
        
        # Process vid files
        vid_dir = worker_dir / "vid"
        if vid_dir.exists():
            vid_files = list(vid_dir.glob("*.mp4"))
            print(f"  Found {len(vid_files)} video files")
            
            for vid_file in vid_files:
                dst_file = output_path / "vid" / vid_file.name
                if args.dry_run:
                    action = "copy" if args.copy_files else "move"
                    print(f"    Would {action}: {vid_file} -> {dst_file}")
                else:
                    if args.copy_files:
                        shutil.copy2(vid_file, dst_file)
                    else:
                        shutil.move(str(vid_file), str(dst_file))
                    total_files_moved += 1
    
    if args.dry_run:
        print("\nDry run complete. Add --dry-run=false to actually merge the files.")
    else:
        print(f"\nMerge complete! Moved {total_files_moved} files to {output_path}")
        
        # Optionally clean up empty worker directories
        if not args.copy_files:
            print("\nCleaning up empty worker directories...")
            for worker_dir in worker_dirs:
                try:
                    # Remove empty subdirectories
                    for subdir in ["pkl", "vid"]:
                        subdir_path = worker_dir / subdir
                        if subdir_path.exists() and not any(subdir_path.iterdir()):
                            subdir_path.rmdir()
                            print(f"  Removed empty directory: {subdir_path}")
                    
                    # Remove worker directory if empty
                    if not any(worker_dir.iterdir()):
                        worker_dir.rmdir()
                        print(f"  Removed empty worker directory: {worker_dir}")
                
                except OSError as e:
                    print(f"  Could not remove {worker_dir}: {e}")

def main():
    args = parse_args()
    merge_directories(args)

if __name__ == "__main__":
    main()
