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

# === Dataset ===
class PoseDatasetXYZABC(Dataset):
    def __init__(self, poses, transform_ranges):
        self.poses = poses.numpy()
        self.transform_ranges = transform_ranges

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        base_pose = self.poses[idx]
        delta = np.random.uniform(-1, 1, 6) * np.array([
            self.transform_ranges["x"], self.transform_ranges["y"], self.transform_ranges["z"],
            self.transform_ranges["a"], self.transform_ranges["b"], self.transform_ranges["c"]
        ])
        input_pose = batch_apply_delta_poses(base_pose.reshape((1,6)), delta.reshape((1,6)))[0]  

        nearest_idx = np.argmin(np.linalg.norm(self.poses - input_pose, axis=1))
        nearest_pose = self.poses[nearest_idx]

        return (
            torch.tensor(input_pose, dtype=torch.float32),
            torch.tensor(nearest_pose, dtype=torch.float32)
        )

# === Error computation ===
def compute_pose_error(pred, target):
    return pred - target

# === Euclidean loss ===
def euclidean_loss(pred, target):
    return torch.norm(pred - target, dim=1).mean()

# === Model ===
class PoseRegressor(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 2):
            layers.extend([
                nn.Linear(layer_sizes[i], layer_sizes[i+1]),
                nn.ReLU()
            ])
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# === Training Loop ===
def main():
    geometry = "extrusion"
    run = 1
    data_path = "/home/rp/abhay_ws/contact-manifold-state-estimation/data_generation/extrusion_sim_data_with_logmaps_no_units_row.csv"
    layer_sizes = [6, 2048, 2048, 2048, 2048, 6]
    transform_ranges = {"x": 10, "y": 10, "z": 10, "a": 10, "b": 10, "c": 10}
    resume_checkpoint = "./model_training/checkpoints/extrusion_run_1_best_NN_model_xyzabc.pth"

    wandb.init(
        project='contact-manifold-learning',
        entity='abhay-negi-usc-university-of-southern-california',
        config={
            "learning_rate": 1e-4,
            "epochs": 1_000_000_000,
            "batch_size": 2**10,
            "layer_sizes": layer_sizes,
            "transform_ranges": transform_ranges,
            "resume_checkpoint": resume_checkpoint,
            "consistency_weight": 100.0, # Log the consistency weight in wandb config
            "flag_force_save": False, 
        }
    )
    config = wandb.config
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # df = pd.read_pickle(data_path)
    df = pd.read_csv(data_path)
    df = df.drop(index=0).reset_index(drop=True) # remove units row 
    df = df.sample(frac=0.01, random_state=np.random.randint(1024))
    original_poses = torch.tensor(df[['x','y','z','a','b','c']].values.astype(np.float32))
    print(f"Loaded {len(original_poses)} poses from {data_path}")
    
    dataset = PoseDatasetXYZABC(original_poses, transform_ranges)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    print(f"setup dataloader with batch size {config.batch_size}")

    model = PoseRegressor(layer_sizes).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    checkpoint_dir = "./model_training/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_loss = float("inf")
    start_epoch = 0

    if resume_checkpoint and os.path.isfile(resume_checkpoint):
        checkpoint = torch.load(resume_checkpoint, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_loss = checkpoint.get('best_loss', float("inf"))
        start_epoch = checkpoint.get('epoch', 0)

    consistency_weight = config.consistency_weight  # Load consistency weight from config

    for epoch in range(start_epoch, config.epochs):
        model.train()
        total_loss = 0
        all_errors = []

        for input_pose, nearest_pose in dataloader:
            input_pose = input_pose.to(DEVICE)
            nearest_pose = nearest_pose.to(DEVICE)

            optimizer.zero_grad()
            pred = model(input_pose)

            # Compute base Euclidean loss
            base_loss = euclidean_loss(pred, nearest_pose)

            # Compute consistency penalty (variance across batch)
            error = pred - nearest_pose
            consistency_loss = error.var(dim=0).mean()

            loss = base_loss + consistency_weight * consistency_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            with torch.no_grad():
                pose_error = compute_pose_error(pred, nearest_pose)
                all_errors.append(pose_error.cpu())

        all_errors = torch.cat(all_errors)
        error_means = all_errors.mean(dim=0).numpy()
        error_stds = all_errors.std(dim=0).numpy()
        labels = ['x (mm)', 'y (mm)', 'z (mm)', 'a (°)', 'b (°)', 'c (°)']

        print(f"Epoch {epoch + 1} Error Stats:")
        for i, label in enumerate(labels):
            print(f"  {label}: mean = {error_means[i]:.4f}, std = {error_stds[i]:.4f}, mae = {np.mean(np.abs(all_errors[:, i].numpy())):.4f}")

        wandb.log({
            'epoch': epoch + 1,
            'loss': total_loss / len(dataloader),
            'consistency_loss': consistency_loss.item(),
            **{f'error_mean_{label}': error_means[i] for i, label in enumerate(labels)},
            **{f'error_std_{label}': error_stds[i] for i, label in enumerate(labels)}
        })

        if (epoch + 1) % 10 == 0:
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

        if total_loss < best_loss or config.flag_force_save:            
            best_loss = total_loss
            best_model_path = os.path.join(checkpoint_dir, f"{geometry}_run_{run}_best_NN_model_xyzabc.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                "geometry": geometry,
                "run": run,
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
            }, best_model_path)

if __name__ == "__main__":
    main()
