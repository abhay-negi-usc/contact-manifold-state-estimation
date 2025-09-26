#!/usr/bin/env python3
"""
Benchmark script to find optimal number of workers for your system.
"""

import subprocess
import time
import argparse
import os
from pathlib import Path

def run_benchmark(num_workers, trials_per_test=20):
    """Run a benchmark with specified number of workers."""
    print(f"\n=== Benchmarking {num_workers} workers ===")
    
    # Clean up any existing benchmark data
    cleanup_cmd = "rm -rf ./data/*benchmark*"
    subprocess.run(cleanup_cmd, shell=True, cwd=".", capture_output=True)
    
    start_time = time.time()
    
    # Run the parallel generation
    cmd = [
        "python", "run_parallel_generation.py",
        "--num-workers", str(num_workers),
        "--total-trials", str(trials_per_test),
        "--output-suffix", "benchmark"
    ]
    
    try:
        result = subprocess.run(cmd, cwd=".", capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            end_time = time.time()
            duration = end_time - start_time
            trials_per_second = trials_per_test / duration
            
            print(f"✓ Completed {trials_per_test} trials in {duration:.1f}s")
            print(f"✓ Rate: {trials_per_second:.2f} trials/second")
            
            return duration, trials_per_second
        else:
            print(f"✗ Failed: {result.stderr}")
            return None, None
            
    except subprocess.TimeoutExpired:
        print("✗ Timeout (>5 minutes)")
        return None, None
    finally:
        # Clean up benchmark data
        cleanup_cmd = "rm -rf ./data/*benchmark*"
        subprocess.run(cleanup_cmd, shell=True, cwd=".", capture_output=True)

def main():
    parser = argparse.ArgumentParser(description='Benchmark parallel workers')
    parser.add_argument('--trials-per-test', type=int, default=8,
                       help='Number of trials per benchmark test (default: 8)')
    parser.add_argument('--min-workers', type=int, default=1,
                       help='Minimum number of workers to test (default: 1)')
    parser.add_argument('--max-workers', type=int, default=20,
                       help='Maximum number of workers to test (default: 20)')
    parser.add_argument('--step', type=int, default=2,
                       help='Step size for worker count (default: 2)')
    args = parser.parse_args()
    
    print("=== Worker Performance Benchmark ===")
    print(f"Testing {args.min_workers} to {args.max_workers} workers")
    print(f"Each test runs {args.trials_per_test} trials")
    print("This will take several minutes...")
    
    results = []
    
    # Test different worker counts
    for num_workers in range(args.min_workers, args.max_workers + 1, args.step):
        duration, rate = run_benchmark(num_workers, args.trials_per_test)
        if duration is not None:
            results.append((num_workers, duration, rate))
    
    # Display results
    print("\n=== BENCHMARK RESULTS ===")
    print("Workers | Duration (s) | Trials/sec | Efficiency")
    print("--------|--------------|------------|----------")
    
    best_rate = 0
    best_workers = 1
    
    for num_workers, duration, rate in results:
        # Calculate efficiency relative to single worker
        baseline_rate = results[0][2] if results else rate
        efficiency = (rate / baseline_rate) / num_workers * 100
        
        print(f"{num_workers:7d} | {duration:10.1f} | {rate:8.2f} | {efficiency:7.1f}%")
        
        if rate > best_rate:
            best_rate = rate
            best_workers = num_workers
    
    print("\n=== RECOMMENDATIONS ===")
    print(f"🏆 Best performance: {best_workers} workers ({best_rate:.2f} trials/sec)")
    
    # Find the sweet spot (good performance with reasonable efficiency)
    sweet_spot_workers = 1
    best_efficiency_ratio = 0
    
    for num_workers, duration, rate in results:
        baseline_rate = results[0][2] if results else rate
        efficiency_ratio = rate / best_rate * (1.0 / num_workers)
        
        if efficiency_ratio > best_efficiency_ratio and rate > best_rate * 0.8:
            best_efficiency_ratio = efficiency_ratio
            sweet_spot_workers = num_workers
    
    print(f"⚖️  Sweet spot: {sweet_spot_workers} workers (good performance/efficiency balance)")
    
    # Memory usage estimate
    max_memory_per_worker = 0.5  # GB estimate
    estimated_memory = best_workers * max_memory_per_worker
    print(f"💾 Estimated memory usage at {best_workers} workers: ~{estimated_memory:.1f}GB")
    
    if estimated_memory > 20:  # Conservative limit
        safe_workers = int(20 / max_memory_per_worker)
        print(f"⚠️  Consider limiting to {safe_workers} workers to avoid memory issues")

if __name__ == "__main__":
    main()
