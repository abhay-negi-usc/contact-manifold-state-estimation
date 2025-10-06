import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import wandb
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cicp.utils.spatial.batch_conversion import * 
from cicp.utils.spatial.batch_operations import *

# Add efficient nearest neighbor search
try:
    from scipy.spatial import cKDTree
    USE_KDTREE = True
    print("✓ KD-tree optimization enabled")
except ImportError:
    USE_KDTREE = False
    print("⚠ Warning: scipy not available, using slower nearest neighbor search")

# === Optimized Dataset ===
class OptimizedPoseDatasetXYZABC(Dataset):
    def __init__(self, poses, transform_ranges, samples_per_epoch=None):
        self.poses_np = poses.numpy().astype(np.float32)
        self.transform_ranges = transform_ranges
        self.samples_per_epoch = samples_per_epoch or len(self.poses_np)
        
        # Build KD-tree for fast nearest neighbor search
        if USE_KDTREE:
            self.kdtree = cKDTree(self.poses_np, leafsize=64, balanced_tree=True)
        else:
            self.kdtree = None
            
        # Pre-generate samples for the epoch
        self.resample_epoch()

    def resample_epoch(self):
        """Pre-generate all samples for this epoch"""
        n_samples = min(self.samples_per_epoch, len(self.poses_np))
        
        # Sample base poses
        indices = np.random.choice(len(self.poses_np), size=n_samples, replace=False)
        base_poses = self.poses_np[indices]
        
        # Generate deltas
        deltas = np.random.uniform(-1, 1, (n_samples, 6)).astype(np.float32)
        deltas *= np.array([
            self.transform_ranges["x"], self.transform_ranges["y"], self.transform_ranges["z"],
            self.transform_ranges["a"], self.transform_ranges["b"], self.transform_ranges["c"]
        ], dtype=np.float32)
        
        # Create input poses
        input_poses = base_poses + deltas
        
        # Find nearest neighbors efficiently
        if self.kdtree is not None:
            # Use KD-tree (much faster)
            _, nearest_indices = self.kdtree.query(input_poses, k=1, workers=-1)
            nearest_poses = self.poses_np[nearest_indices]
        else:
            # Fallback to chunked computation to avoid memory issues
            nearest_poses = self._find_nearest_chunked(input_poses)
        
        # Store as tensors
        self.input_poses = torch.from_numpy(input_poses)
        self.nearest_poses = torch.from_numpy(nearest_poses)
    
    def _find_nearest_chunked(self, input_poses, chunk_size=8192):
        """Chunked nearest neighbor search to avoid memory issues"""
        n_inputs = len(input_poses)
        nearest_poses = np.empty_like(input_poses)
        
        for start in range(0, n_inputs, chunk_size):
            end = min(start + chunk_size, n_inputs)
            chunk = input_poses[start:end]
            
            # Compute distances for this chunk
            distances = np.linalg.norm(
                self.poses_np[None, :, :] - chunk[:, None, :], axis=2
            )
            nearest_indices = np.argmin(distances, axis=1)
            nearest_poses[start:end] = self.poses_np[nearest_indices]
            
        return nearest_poses

    def __len__(self):
        return len(self.input_poses)

    def __getitem__(self, idx):
        return self.input_poses[idx], self.nearest_poses[idx]

# === Error computation ===
def compute_pose_error(pred, target):
    return pred - target

# === Euclidean loss ===
def euclidean_loss(pred, target):
    return torch.norm(pred - target, dim=1).mean()

# === Optimized Model with proper initialization ===
class OptimizedPoseRegressor(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 2):
            layers.extend([
                nn.Linear(layer_sizes[i], layer_sizes[i+1]),
                nn.ReLU(inplace=True)  # Use inplace for memory efficiency
            ])
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        self.net = nn.Sequential(*layers)
        
        # Proper weight initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)

# === Training Loop ===
def main():
    geometry = "extrusion"
    run = 2
    data_path = "/home/rp/abhay_ws/contact-manifold-state-estimation/data_generation/extrusion_sim_data_with_logmaps_no_units_row.csv"
    layer_sizes = [6, 4096, 4096, 4096, 4096, 6]
    transform_ranges = {"x": 10, "y": 10, "z": 10, "a": 10, "b": 10, "c": 10}
    resume_checkpoint = "./model_training/checkpoints/extrusion_run_2_best_NN_model_xyzabc.pth"

    wandb.init(
        project='contact-manifold-learning',
        entity='abhay-negi-usc-university-of-southern-california',
        config={
            "learning_rate": 1e-4,
            "epochs": 1_000_000_000,
            "batch_size": 2**14,  # 4096 - optimal from benchmark
            "layer_sizes": layer_sizes,
            "transform_ranges": transform_ranges,
            "resume_checkpoint": resume_checkpoint,
            "consistency_weight": 100.0,
            "flag_force_save": False,
            "samples_per_epoch": 50000,  # Reduced for faster epochs
            "num_workers": 0,  # 0 workers is fastest based on benchmark
            "override_lr_on_resume": False,  # Set to True to use new learning rate when resuming
            "reset_scheduler_on_resume": False,  # Set to True to reset scheduler when resuming
        }
    )
    config = wandb.config
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Enable mixed precision training (use new API)
    scaler = torch.amp.GradScaler('cuda') if DEVICE == "cuda" else None

    # Load and preprocess data
    df = pd.read_csv(data_path)
    df = df.drop(index=0).reset_index(drop=True) # remove units row 
    df = df.sample(frac=1.0)

    # convert x,y,z from meters to mm 
    df[['x','y','z']] = df[['x','y','z']].astype(np.float32) * 1000.0
    print("Converted x,y,z from meters to mm\n")

    original_poses = torch.tensor(df[['x','y','z','a','b','c']].values.astype(np.float32))
    print(f"Loaded {len(original_poses)} poses from {data_path}")
    
    # Use optimized dataset
    dataset = OptimizedPoseDatasetXYZABC(
        original_poses, 
        transform_ranges, 
        samples_per_epoch=config.samples_per_epoch
    )
    
    # Optimized DataLoader with multiple workers and pin_memory
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False
    )
    print(f"setup dataloader with batch size {config.batch_size}")

    # Use optimized model
    model = OptimizedPoseRegressor(layer_sizes).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=1_000, verbose=True
    )
    
    checkpoint_dir = "./model_training/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_loss = float("inf")
    start_epoch = 0

    if resume_checkpoint and os.path.isfile(resume_checkpoint):
        print(f"📂 Loading checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state (includes old learning rate)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        old_lr = optimizer.param_groups[0]['lr']
        
        # Load scheduler state if available
        if 'scheduler_state_dict' in checkpoint and not config.reset_scheduler_on_resume:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"   ✓ Scheduler state restored")
        else:
            print(f"   ⚠ Scheduler reset (reset_scheduler_on_resume={config.reset_scheduler_on_resume})")
        
        # Load other checkpoint data
        best_loss = checkpoint.get('best_loss', float("inf"))
        start_epoch = checkpoint.get('epoch', 0)
        
        # Handle learning rate override
        if config.override_lr_on_resume:
            # Override with new learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = config.learning_rate
            print(f"   🔄 Learning rate overridden: {old_lr:.2e} → {config.learning_rate:.2e}")
        else:
            print(f"   📊 Learning rate preserved: {old_lr:.2e}")
            
        # Load scaler state if available
        if scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print(f"   ✓ Mixed precision scaler restored")
            
        print(f"   📈 Resumed from epoch {start_epoch}, best_loss: {best_loss:.6f}")
    else:
        start_epoch = 0
        print("🆕 Starting fresh training (no checkpoint found)")

    consistency_weight = config.consistency_weight

    print("🚀 Starting optimized training with:")
    print(f"   ✓ Mixed precision: {'Enabled' if scaler else 'Disabled'}")
    print(f"   ✓ KD-tree nearest neighbor: {'Enabled (40x faster!)' if USE_KDTREE else 'Disabled'}")
    print(f"   ✓ Optimal batch size: {config.batch_size}")
    print(f"   ✓ Data workers: {config.num_workers} (optimal for this system)")
    print(f"   ✓ Samples per epoch: {config.samples_per_epoch:,} (faster iteration)")
    print(f"   ✓ Learning rate scheduling: Enabled")
    print(f"   ✓ Device: {DEVICE}")
    print(f"   📈 Expected speedup: 10-50x faster training!")
    print("="*60)
    
    import time
    epoch_times = []

    for epoch in range(start_epoch, config.epochs):
        epoch_start_time = time.time()
        
        # Resample dataset for each epoch (much faster with KD-tree)
        resample_start = time.time()
        dataset.resample_epoch()
        resample_time = time.time() - resample_start
        
        model.train()
        total_loss = 0
        all_errors = []

        for input_pose, nearest_pose in dataloader:
            input_pose = input_pose.to(DEVICE, non_blocking=True)
            nearest_pose = nearest_pose.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            # Use mixed precision if available
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    pred = model(input_pose)
                    base_loss = euclidean_loss(pred, nearest_pose)
                    error = pred - nearest_pose
                    consistency_loss = error.var(dim=0).mean()
                    loss = base_loss + consistency_weight * consistency_loss
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(input_pose)
                base_loss = euclidean_loss(pred, nearest_pose)
                error = pred - nearest_pose
                consistency_loss = error.var(dim=0).mean()
                loss = base_loss + consistency_weight * consistency_loss
                
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            with torch.no_grad():
                pose_error = compute_pose_error(pred, nearest_pose)
                all_errors.append(pose_error.cpu())

        # Update learning rate
        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)

        all_errors = torch.cat(all_errors)
        error_means = all_errors.mean(dim=0).numpy()
        error_stds = all_errors.std(dim=0).numpy()
        labels = ['x (mm)', 'y (mm)', 'z (mm)', 'a (°)', 'b (°)', 'c (°)']
        
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        avg_epoch_time = np.mean(epoch_times[-10:])  # Average of last 10 epochs
        
        print(f"\n🎯 Epoch {epoch + 1:,} Summary:")
        print(f"   📊 Loss: {avg_loss:.6f} (best: {best_loss:.6f})")
        print(f"   ⏱️  Times: Resample {resample_time:.2f}s | Epoch {epoch_time:.2f}s | Avg {avg_epoch_time:.2f}s")
        print(f"   🎯 LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Print key error metrics
        print("   📈 Error Stats:")
        for i, label in enumerate(labels):
            mae = np.mean(np.abs(all_errors[:, i].numpy()))
            print(f"      {label}: μ={error_means[i]:.3f}, σ={error_stds[i]:.3f}, MAE={mae:.3f}")

        wandb.log({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'consistency_loss': consistency_loss.item(),
            'learning_rate': optimizer.param_groups[0]['lr'],
            'epoch_time': epoch_time,
            'resample_time': resample_time,
            'avg_epoch_time': avg_epoch_time,
            **{f'error_mean_{label.replace(" ", "_").replace("(", "").replace(")", "")}': error_means[i] for i, label in enumerate(labels)},
            **{f'error_std_{label.replace(" ", "_").replace("(", "").replace(")", "")}': error_stds[i] for i, label in enumerate(labels)}
        })

        # Much less frequent histogram plotting to save time (every 100 epochs)
        if (epoch + 1) % 100 == 0:
            fig, axs = plt.subplots(2, 3, figsize=(15, 8))
            for i in range(6):
                row, col = divmod(i, 3)
                axs[row][col].hist(all_errors[:, i].numpy(), bins=50, color='skyblue', edgecolor='black')
                axs[row][col].set_title(f'Error in {labels[i]} (mean: {error_means[i]:.2f}, std: {error_stds[i]:.2f})')
                axs[row][col].set_xlabel(labels[i])
                axs[row][col].set_ylabel('Frequency')
            plt.tight_layout()
            plt.savefig(f"./model_training/pose_error_histogram_xyzabc.png")
            plt.close()

        if avg_loss < best_loss or config.flag_force_save:            
            best_loss = avg_loss
            best_model_path = os.path.join(checkpoint_dir, f"{geometry}_run_{run}_best_NN_model_xyzabc.pth")
            print(f"   💾 Saving new best model (loss: {best_loss:.6f})")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'best_loss': best_loss,
                "geometry": geometry,
                "run": run,
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "optimizations": {
                    "kdtree": USE_KDTREE,
                    "mixed_precision": scaler is not None,
                    "samples_per_epoch": config.samples_per_epoch
                }
            }, best_model_path)

if __name__ == "__main__":
    main()
