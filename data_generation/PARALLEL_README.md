# Parallel Contact Data Generation

This directory contains scripts for generating contact data with support for parallel execution to avoid file conflicts.

## Files

- `generate_contact_data.py` - Main data generation script (modified for parallel execution)
- `run_parallel_generation.py` - Helper script to launch multiple parallel workers
- `merge_worker_outputs.py` - Helper script to merge outputs from parallel workers

## Usage

### Single Worker (Original Behavior)
```bash
python generate_contact_data.py
```

### Parallel Execution

#### Option 1: Manual Worker Launch
Launch individual workers manually:
```bash
# Worker 0 of 4
python generate_contact_data.py --worker-id 0 --num-workers 4

# Worker 1 of 4  
python generate_contact_data.py --worker-id 1 --num-workers 4

# ... and so on
```

#### Option 2: Automatic Parallel Launch (Recommended)
Use the helper script to launch all workers automatically:
```bash
# Launch 4 parallel workers
python run_parallel_generation.py --num-workers 4

# Launch 8 workers with custom total trials
python run_parallel_generation.py --num-workers 8 --total-trials 100000

# Dry run to see what would be executed
python run_parallel_generation.py --num-workers 4 --dry-run
```

### Merging Worker Outputs
After parallel execution, merge the outputs from all workers:
```bash
# Merge all worker outputs into a single directory
python merge_worker_outputs.py

# Dry run to see what would be merged
python merge_worker_outputs.py --dry-run

# Custom paths
python merge_worker_outputs.py \
    --base-dir ./data/cross_real_data \
    --output-dir ./data/cross_real_data_final
```

## How Parallel Execution Works

1. **Unique Output Directories**: Each worker creates its own output directory (`cross_real_data_worker_0`, `cross_real_data_worker_1`, etc.)

2. **Non-Overlapping Trial Ranges**: The total number of trials is automatically split among workers, ensuring no trial ID conflicts

3. **Independent Random Seeds**: Each worker uses a different random seed based on worker ID and current time

4. **Global Trial Indexing**: Trial files are numbered globally (e.g., worker 0 handles trials 0-62499, worker 1 handles 62500-124999, etc.)

## Example Workflow

```bash
# 1. Generate data with 4 parallel workers
python run_parallel_generation.py --num-workers 4 --total-trials 1000

# 2. Merge all worker outputs
python merge_worker_outputs.py

# 3. Clean up (optional) - remove individual worker directories
rm -rf ./data/cross_real_data_worker_*
```

## Command Line Arguments

### generate_contact_data.py
- `--worker-id`: Unique worker ID (default: 0)
- `--num-workers`: Total number of workers (default: 1) 
- `--trials-per-worker`: Trials per worker (default: auto-split)
- `--output-suffix`: Additional suffix for output directory

### run_parallel_generation.py  
- `--num-workers`: Number of parallel workers (default: 4)
- `--total-trials`: Total trials across all workers (default: 250000)
- `--output-suffix`: Additional suffix for output directories
- `--dry-run`: Print commands without executing

### merge_worker_outputs.py
- `--base-dir`: Base directory with worker outputs 
- `--output-dir`: Output directory for merged results
- `--worker-pattern`: Pattern to match worker directories
- `--copy-files`: Copy instead of move files
- `--dry-run`: Print operations without executing
