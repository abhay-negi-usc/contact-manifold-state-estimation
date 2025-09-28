#!/usr/bin/env python3
"""
Helper script to run multiple parallel instances of contact data generation.
"""

import subprocess
import argparse
import time
import os
import signal
import sys
import threading
import json
from pathlib import Path
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Run parallel contact data generation')
    parser.add_argument('--num-workers', type=int, default=1,
                       help='Number of parallel workers to launch (default: 4)')
    parser.add_argument('--total-trials', type=int, default=250000,
                       help='Total number of trials across all workers (default: 250000)')
    parser.add_argument('--output-suffix', type=str, default='',
                       help='Additional suffix for output directories (default: empty)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print commands without executing them')
    return parser.parse_args()

def monitor_progress(progress_file, total_trials, pbar, stop_event):
    """Monitor progress from all workers and update the progress bar."""
    last_total = 0
    
    while not stop_event.is_set():
        try:
            if os.path.exists(progress_file):
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
                
                # Sum up progress from all workers
                current_total = sum(progress_data.values())
                
                # Update progress bar
                if current_total > last_total:
                    pbar.update(current_total - last_total)
                    last_total = current_total
                    
                # Check if we're done
                if current_total >= total_trials:
                    break
                    
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            # Progress file doesn't exist yet or is corrupted, continue waiting
            pass
        
        time.sleep(0.5)  # Update every 0.5 seconds

def main():
    args = parse_args()
    
    # Calculate trials per worker
    trials_per_worker = args.total_trials // args.num_workers
    remaining_trials = args.total_trials % args.num_workers
    
    print(f"Launching {args.num_workers} parallel workers")
    print(f"Total trials: {args.total_trials}")
    print(f"Base trials per worker: {trials_per_worker}")
    if remaining_trials > 0:
        print(f"First {remaining_trials} workers will process {trials_per_worker + 1} trials")
    print()
    
    # Create progress tracking file
    script_dir = Path(__file__).parent
    progress_file = script_dir / "worker_progress.json"
    
    # Initialize progress file
    if not args.dry_run:
        with open(progress_file, 'w') as f:
            json.dump({}, f)
    
    processes = []
    
    try:
        # Launch worker processes
        for worker_id in range(args.num_workers):
            # Calculate exact number of trials for this worker
            if worker_id < remaining_trials:
                worker_trials = trials_per_worker + 1
            else:
                worker_trials = trials_per_worker
            
            cmd = [
                'python', 'generate_contact_data.py',
                '--worker-id', str(worker_id),
                '--num-workers', str(args.num_workers),
                '--trials-per-worker', str(worker_trials),
                '--progress-file', str(progress_file)
            ]
            
            if args.output_suffix:
                cmd.extend(['--output-suffix', args.output_suffix])
            
            print(f"Worker {worker_id}: {worker_trials} trials")
            print(f"Command: {' '.join(cmd)}")
            
            if not args.dry_run:
                # Change to the data_generation directory
                process = subprocess.Popen(cmd, cwd=script_dir)
                processes.append((worker_id, process))
                time.sleep(1)  # Small delay between launches
            print()
        
        if args.dry_run:
            print("Dry run complete. Use --dry-run=false to actually run the processes.")
            return
        
        print(f"All {args.num_workers} workers launched.")
        print("Press Ctrl+C to terminate all workers")
        print()
        
        # Create progress bar
        with tqdm(total=args.total_trials, desc="Overall Progress", unit="trials") as pbar:
            # Start progress monitoring thread
            stop_event = threading.Event()
            progress_thread = threading.Thread(
                target=monitor_progress,
                args=(progress_file, args.total_trials, pbar, stop_event)
            )
            progress_thread.daemon = True
            progress_thread.start()
            
            # Wait for all processes to complete
            while processes:
                for i, (worker_id, process) in enumerate(processes):
                    if process.poll() is not None:
                        # Process has finished
                        return_code = process.returncode
                        if return_code == 0:
                            print(f"\nWorker {worker_id} completed successfully")
                        else:
                            print(f"\nWorker {worker_id} failed with return code {return_code}")
                        processes.pop(i)
                        break
                
                time.sleep(1)  # Check every second
            
            # Stop progress monitoring
            stop_event.set()
            progress_thread.join(timeout=1)
            
            print("\nAll workers completed!")
        
        # Clean up progress file
        if progress_file.exists():
            progress_file.unlink()
        
    except KeyboardInterrupt:
        print("\n\nReceived interrupt signal. Terminating all workers...")
        
        # Stop progress monitoring if it exists
        try:
            stop_event.set()
        except:
            pass
            
        for worker_id, process in processes:
            try:
                process.terminate()
                print(f"Terminated worker {worker_id}")
            except:
                pass
        
        # Wait a bit for graceful termination
        time.sleep(2)
        
        # Force kill any remaining processes
        for worker_id, process in processes:
            try:
                if process.poll() is None:
                    process.kill()
                    print(f"Force killed worker {worker_id}")
            except:
                pass
        
        # Clean up progress file
        script_dir = Path(__file__).parent
        progress_file = script_dir / "worker_progress.json"
        if progress_file.exists():
            try:
                progress_file.unlink()
            except:
                pass
        
        sys.exit(1)

if __name__ == "__main__":
    main()
