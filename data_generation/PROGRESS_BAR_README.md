# Progress Bar Enhancement for Parallel Data Generation

## Overview

The `run_parallel_generation.py` script has been enhanced with an overall progress bar using `tqdm` that tracks the combined progress of all worker processes.

## Features

- **Real-time Progress Tracking**: Shows overall progress across all workers
- **Live Updates**: Progress bar updates every 0.5 seconds
- **Worker Coordination**: Uses a shared JSON file to coordinate progress between workers
- **Graceful Cleanup**: Automatically removes temporary progress tracking files
- **Interrupt Handling**: Properly handles Ctrl+C interrupts and cleans up resources

## Usage

### Basic Usage
```bash
python run_parallel_generation.py --num-workers 8 --total-trials 100000
```

### With Output Suffix
```bash
python run_parallel_generation.py --num-workers 16 --total-trials 250000 --output-suffix "experiment_1"
```

### Dry Run (Test Commands)
```bash
python run_parallel_generation.py --dry-run --num-workers 4 --total-trials 1000
```

## How It Works

1. **Progress File**: Creates a temporary `worker_progress.json` file to track progress
2. **Worker Updates**: Each worker updates its progress every 10 trials (or at completion)
3. **Progress Monitoring**: A separate thread monitors the progress file and updates the tqdm bar
4. **Real-time Display**: Shows trials/second, estimated time remaining, and percentage complete
5. **Cleanup**: Automatically removes the progress file when done or interrupted

## Progress Bar Features

The progress bar displays:
- Current percentage complete
- Number of completed trials vs total trials
- Processing speed (trials/second)
- Estimated time remaining
- Visual progress indicator

Example output:
```
Overall Progress:  45%|████████████████████▌                           | 45000/100000 [02:30<03:15, 287.43trials/s]
```

## Notes

- Progress updates every 10 trials per worker for performance optimization
- Console output from workers is reduced to every 100 trials to keep display clean
- The progress bar thread runs in daemon mode for clean shutdown
- Handles worker failures gracefully without breaking progress tracking
