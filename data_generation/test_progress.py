#!/usr/bin/env python3
"""
Test script to demonstrate the progress bar functionality
"""

import subprocess
import time
import json
import os
import threading
from pathlib import Path
from tqdm import tqdm

def test_worker(worker_id, num_trials, progress_file):
    """Simulate a worker that updates progress"""
    for i in range(num_trials):
        # Simulate some work
        time.sleep(0.1)
        
        # Update progress
        try:
            if os.path.exists(progress_file):
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
            else:
                progress_data = {}
            
            progress_data[str(worker_id)] = i + 1
            
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f)
        except:
            pass

def monitor_progress(progress_file, total_trials, pbar, stop_event):
    """Monitor progress from all workers and update the progress bar."""
    last_total = 0
    
    while not stop_event.is_set():
        try:
            if os.path.exists(progress_file):
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
                
                current_total = sum(progress_data.values())
                
                if current_total > last_total:
                    pbar.update(current_total - last_total)
                    last_total = current_total
                    
                if current_total >= total_trials:
                    break
                    
        except:
            pass
        
        time.sleep(0.1)

def main():
    num_workers = 3
    total_trials = 20
    trials_per_worker = total_trials // num_workers
    
    progress_file = Path(__file__).parent / "test_progress.json"
    
    # Initialize progress file
    with open(progress_file, 'w') as f:
        json.dump({}, f)
    
    print(f"Testing progress bar with {num_workers} workers, {total_trials} total trials")
    print(f"Each worker will simulate {trials_per_worker} trials")
    print()
    
    # Create progress bar
    with tqdm(total=total_trials, desc="Test Progress", unit="trials") as pbar:
        # Start progress monitoring thread
        stop_event = threading.Event()
        progress_thread = threading.Thread(
            target=monitor_progress,
            args=(progress_file, total_trials, pbar, stop_event)
        )
        progress_thread.daemon = True
        progress_thread.start()
        
        # Start worker threads
        worker_threads = []
        for worker_id in range(num_workers):
            thread = threading.Thread(
                target=test_worker,
                args=(worker_id, trials_per_worker, progress_file)
            )
            thread.start()
            worker_threads.append(thread)
        
        # Wait for all workers to complete
        for thread in worker_threads:
            thread.join()
        
        # Stop progress monitoring
        stop_event.set()
        progress_thread.join(timeout=1)
    
    print("\nTest completed!")
    
    # Clean up
    if progress_file.exists():
        progress_file.unlink()

if __name__ == "__main__":
    main()
