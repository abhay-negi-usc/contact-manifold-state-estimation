# train_wxyz.py
import os
import math
import csv
import numpy as np
import pandas as pd
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import wandb

# ===== Utilities =====
def to_rad(deg): return deg * math.pi / 180.0
def to_deg(rad): return rad * 180.0 / math.pi

def chunked_nn_argmin(inputs: np.ndarray, corpus: np.ndarray, chunk: int = 16384) -> np.ndarray:
    """
    Find nearest neighbor indices in 'corpus' for each row in 'inputs'.
    Works in chunks to reduce peak memory.
    Returns [len(inputs)] indices.
    """
    N = inputs.shape[0]
    out = np.empty(N, dtype=np.int64)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        block = inputs[s:e]                    # [b,6]
        # [b,N] distances
        dists = np.linalg.norm(corpus[None, :, :] - block[:, None, :], axis=2)
        out[s:e] = dists.argmin(axis=1)
    return out

@dataclass
class TrainConfig:
    resume: bool = True
    resume_path: str = "./model_training/checkpoints/best_model_wxyz.pth"

    data_path: str = "./data_generation/extrusion_sim_data_with_logmaps_no_units_row.csv"
    data_units = {
        "x": "m", "y": "m", "z": "m",
        "wx": "rad", "wy": "rad", "wz": "rad"
    }
    layer_sizes: tuple = (6, 2048, 2048, 2048, 2048, 6)
    lr: float = 1e-5
    epochs: int = 10_000_000
    batch_size: int = 2**18
    num_workers: int = 0
    pin_memory: bool = True

    # use full dataset by default
    sample_frac: float = 0.25

    # perturbation ranges: x,y,z in data units (e.g., mm); w* given in degrees
    transform_ranges = {"x": 5.0, "y": 5.0, "z": 5.0, "wx_deg": 5.0, "wy_deg": 5.0, "wz_deg": 5.0}

    # loss weights
    trans_w: float = 1.0
    rot_w: float = 1000.0

    # how many perturbed samples to generate per epoch (None => use full corpus length)
    train_epoch_samples: int | None = None

    # NN backend: "ckdtree" | "faiss" | "numpy"
    nn_backend: str = "ckdtree"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: str = "./model_training/checkpoints"
    save_name: str = "best_model_wxyz.pth"

    # evaluation
    eval_every: int = 1
    eval_subset_frac: float = 0.25
    eval_batch_size: int = 8192
    metrics_csv: str = "./model_training/metrics_wxyz.csv"

    # wandb configuration
    wandb_project: str = "contact-manifold-learning"
    wandb_entity: str = "abhay-negi-usc-university-of-southern-california"
    wandb_enabled: bool = True

# ---- Nearest Neighbor Index (KD-tree primary) ----
class NNIndex:
    def __init__(self, data_np: np.ndarray, backend: str = "ckdtree"):
        """
        data_np: [N, D] float32
        backend: "ckdtree" (default), "faiss", or "numpy" (chunked scan)
        """
        self.data = np.ascontiguousarray(data_np.astype(np.float32))
        self.backend = backend
        self.D = self.data.shape[1]

        if backend == "ckdtree":
            try:
                from scipy.spatial import cKDTree
                # leafsize=64 is a good default; tweak if memory/time is tight
                self.index = cKDTree(self.data, leafsize=64, balanced_tree=True, compact_nodes=True)
                self.query = self._query_ckdtree
            except Exception:
                print("[NNIndex] cKDTree not available; falling back to chunked NumPy.")
                self.backend = "numpy"
                self.query = self._query_numpy

        elif backend == "faiss":
            try:
                import faiss
                self.index = faiss.IndexFlatL2(self.D)
                self.index.add(self.data)  # CPU; to use GPU: faiss.index_cpu_to_all_gpus(self.index)
                self.query = self._query_faiss
            except Exception:
                print("[NNIndex] FAISS not available; falling back to cKDTree/NumPy.")
                self.backend = "ckdtree"
                return self.__init__(self.data, backend="ckdtree")

        else:
            self.query = self._query_numpy

    def _query_ckdtree(self, X: np.ndarray) -> np.ndarray:
        # X: [M, D] -> returns argmin indices [M]
        _, idx = self.index.query(np.ascontiguousarray(X, dtype=np.float32), k=1, workers=-1)
        return idx.astype(np.int64)

    def _query_faiss(self, X: np.ndarray) -> np.ndarray:
        import faiss
        Xc = np.ascontiguousarray(X.astype(np.float32))
        # For very large M, you can process in chunks to avoid temporary allocations
        D, I = self.index.search(Xc, k=1)  # D: distances, I: indices
        return I.reshape(-1).astype(np.int64)

    def _query_numpy(self, X: np.ndarray, corpus_chunk: int = 262144) -> np.ndarray:
        # memory-safe chunked scan across the corpus
        M = X.shape[0]
        Xb = np.ascontiguousarray(X.astype(np.float32))
        out = np.empty(M, dtype=np.int64)
        best = np.full(M, np.inf, dtype=np.float64)
        for c0 in range(0, self.data.shape[0], corpus_chunk):
            c1 = min(c0 + corpus_chunk, self.data.shape[0])
            corp = self.data[c0:c1]  # [c, D]
            # [M, c] distances (if M*c is large, split X too)
            for s0 in range(0, M, 8192):
                s1 = min(s0 + 8192, M)
                block = Xb[s0:s1]  # [b, D]
                d = np.linalg.norm(block[:, None, :] - corp[None, :, :], axis=2)  # [b, c]
                loc = d.argmin(axis=1)
                val = d[np.arange(d.shape[0]), loc]
                mask = val < best[s0:s1]
                best[s0:s1][mask] = val[mask]
                out[s0:s1][mask] = (c0 + loc[mask]).astype(np.int64)
        return out


# ===== Dataset that re-samples each epoch =====
class EpochPerturbedDataset(Dataset):
    def __init__(self, poses: torch.Tensor, transform_ranges: dict, epoch_samples: int, nn_index: NNIndex):
        self.poses_np = np.ascontiguousarray(poses.numpy().astype(np.float32))
        self.tr = transform_ranges
        self.nn_index = nn_index
        self.epoch_samples = int(min(epoch_samples, len(self.poses_np)))
        self.inputs = None
        self.targets = None
        self.resample()

    def resample(self):
        S = self.epoch_samples
        N = len(self.poses_np)
        idxs = np.random.choice(N, size=S, replace=False)
        base = self.poses_np[idxs]

        rot_range_rad = np.array([to_rad(self.tr["wx_deg"]), to_rad(self.tr["wy_deg"]), to_rad(self.tr["wz_deg"])], dtype=np.float32)
        delta = np.random.uniform(-1.0, 1.0, size=(S, 6)).astype(np.float32)
        delta[:, :3] *= np.array([self.tr["x"], self.tr["y"], self.tr["z"]], dtype=np.float32)[None, :]
        delta[:, 3:] *= rot_range_rad[None, :]

        inputs = base + delta
        nn_idx = self.nn_index.query(inputs)  # <<< KD-tree lookup
        targets = self.poses_np[nn_idx]

        self.inputs = torch.from_numpy(inputs)
        self.targets = torch.from_numpy(targets)

    def __len__(self): return 0 if self.inputs is None else self.inputs.shape[0]
    def __getitem__(self, i): return self.inputs[i], self.targets[i]

# ===== Model =====
class PoseRegressor(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 2):
            layers += [nn.Linear(layer_sizes[i], layer_sizes[i+1]), nn.ReLU()]
        layers += [nn.Linear(layer_sizes[-2], layer_sizes[-1])]
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # x: [B,6]
        return self.net(x)

# ===== Loss =====
def weighted_pose_loss(pred, target, trans_w=1.0, rot_w=100.0):
    """
    Weighted sum of Euclidean distances:
      trans: ||pred[:,:3] - target[:,:3]||_2
      rot:   ||pred[:,3:] - target[:,3:]||_2   (wx,wy,wz space; radians)
    """
    trans_err = torch.norm(pred[:, :3] - target[:, :3], dim=1)   # [B]
    rot_err   = torch.norm(pred[:, 3:] - target[:, 3:], dim=1)   # [B], radians
    loss = trans_w * trans_err.mean() + rot_w * rot_err.mean()
    return loss, trans_err, rot_err

# ===== Eval helpers =====
def sample_perturbed_batch_with_index(poses_np: np.ndarray, tr: dict, n_samples: int, nn_index: NNIndex):
    N = poses_np.shape[0]
    S = min(n_samples, N)
    idxs = np.random.choice(N, size=S, replace=False)
    base = poses_np[idxs].astype(np.float32)

    rot_range_rad = np.array([to_rad(tr["wx_deg"]), to_rad(tr["wy_deg"]), to_rad(tr["wz_deg"])], dtype=np.float32)
    delta = np.random.uniform(-1.0, 1.0, size=(S, 6)).astype(np.float32)
    delta[:, :3] *= np.array([tr["x"], tr["y"], tr["z"]], dtype=np.float32)[None, :]
    delta[:, 3:] *= rot_range_rad[None, :]

    inputs = base + delta
    nn_idx = nn_index.query(inputs)             # <<< KD-tree / FAISS / NumPy
    targets = poses_np[nn_idx].astype(np.float32)
    return torch.from_numpy(inputs), torch.from_numpy(targets)

@torch.no_grad()
def evaluate(model, poses: torch.Tensor, tr: dict, device: str,
             eval_batch_size: int, subset_frac: float, nn_index):
    model.eval()
    poses_np = np.ascontiguousarray(poses.cpu().numpy().astype(np.float32))
    n_samples = max(1, int(len(poses_np) * subset_frac))

    in_all, tgt_all = sample_perturbed_batch_with_index(poses_np, tr, n_samples, nn_index)
    # keep on CPU for now; move per-batch to avoid OOM
    B = eval_batch_size

    trans_mag, rot_comp_deg, rot_norm_deg = [], [], []
    for s in range(0, in_all.shape[0], B):
        e = min(s + B, in_all.shape[0])

        # >>> Move to the SAME device as the model here <<<
        inp = in_all[s:e].to(device, non_blocking=True)
        tgt = tgt_all[s:e].to(device, non_blocking=True)

        pred = model(inp)  # now both model and inputs are on `device`

        # compute errors (rotation reported in degrees)
        _, trans_err, rot_err = weighted_pose_loss(pred, tgt, trans_w=1.0, rot_w=1.0)
        rot_comp = (pred[:, 3:] - tgt[:, 3:]).abs().cpu().numpy()
        rot_comp = rot_comp * (180.0 / np.pi)  # to degrees
        rot_norm = (rot_err.detach().cpu().numpy()) * (180.0 / np.pi)
        t_mag    = trans_err.detach().cpu().numpy()

        trans_mag.append(t_mag)
        rot_comp_deg.append(rot_comp)
        rot_norm_deg.append(rot_norm)

    trans_mag = np.concatenate(trans_mag) if trans_mag else np.array([0.0])
    rot_comp_deg = np.concatenate(rot_comp_deg) if rot_comp_deg else np.zeros((1,3))
    rot_norm_deg = np.concatenate(rot_norm_deg) if rot_norm_deg else np.array([0.0])

    return {
        "eval_trans_mean": float(trans_mag.mean()),
        "eval_trans_std": float(trans_mag.std()),
        "eval_trans_mae": float(np.abs(trans_mag).mean()),
        "eval_wx_mean_deg": float(rot_comp_deg[:,0].mean()),
        "eval_wx_std_deg": float(rot_comp_deg[:,0].std()),
        "eval_wx_mae_deg": float(np.abs(rot_comp_deg[:,0]).mean()),
        "eval_wy_mean_deg": float(rot_comp_deg[:,1].mean()),
        "eval_wy_std_deg": float(rot_comp_deg[:,1].std()),
        "eval_wy_mae_deg": float(np.abs(rot_comp_deg[:,1]).mean()),
        "eval_wz_mean_deg": float(rot_comp_deg[:,2].mean()),
        "eval_wz_std_deg": float(rot_comp_deg[:,2].std()),
        "eval_wz_mae_deg": float(np.abs(rot_comp_deg[:,2]).mean()),
        "eval_rot_norm_mean_deg": float(rot_norm_deg.mean()),
        "eval_rot_norm_std_deg": float(rot_norm_deg.std()),
        "eval_rot_norm_mae_deg": float(np.abs(rot_norm_deg).mean()),
        "eval_samples": int(n_samples),
    }

# ===== Train / Eval Driver =====
def main():
    cfg = TrainConfig()

    # Initialize wandb
    if cfg.wandb_enabled:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            config={
                "learning_rate": cfg.lr,
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "layer_sizes": cfg.layer_sizes,
                "transform_ranges": cfg.transform_ranges,
                "trans_w": cfg.trans_w,
                "rot_w": cfg.rot_w,
                "sample_frac": cfg.sample_frac,
                "nn_backend": cfg.nn_backend,
                "train_epoch_samples": cfg.train_epoch_samples,
                "eval_every": cfg.eval_every,
                "eval_subset_frac": cfg.eval_subset_frac,
                "resume": cfg.resume,
                "data_units": cfg.data_units,
            }
        )

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("medium")
    except Exception:
        pass

    # --- Load data
    df = pd.read_csv(cfg.data_path)
    # filter df such that contact flag is 1
    if 'contact' in df.columns:
        df = df[df['contact'] == 1].reset_index(drop=True)    

    expected = ["x","y","z","wx","wy","wz"]
    for col in expected:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' not found. Provide wx,wy,wz (log-map, radians).")

    if cfg.data_units["x"] == "m":
        df[["x","y","z"]] *= 1000.0  # convert to mm
        print("[INFO] Converted x,y,z from meters to millimeters.")

    if cfg.sample_frac < 1.0:
        df = df.sample(frac=cfg.sample_frac).reset_index(drop=True)

    poses = torch.tensor(df[["x","y","z","wx","wy","wz"]].values.astype(np.float32))
    corpus_np = np.ascontiguousarray(poses.numpy().astype(np.float32))

    # Build the NN index ONCE for the full corpus
    nn_index = NNIndex(corpus_np, backend=cfg.nn_backend)  # "ckdtree" by default

    # Epoch dataset: use full corpus length if train_epoch_samples is None
    epoch_size = len(poses) if (cfg.train_epoch_samples is None) else int(min(cfg.train_epoch_samples, len(poses)))

    dataset = EpochPerturbedDataset(poses, cfg.transform_ranges, epoch_size, nn_index)
    loader = DataLoader(
        dataset,
        batch_size=min(cfg.batch_size, len(dataset)) or 1,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False
    )

    # ---- Model/optim
    model = PoseRegressor(cfg.layer_sizes).to(cfg.device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scaler = torch.amp.GradScaler(device=cfg.device, enabled=(cfg.device == "cuda"))

    # metrics CSV header
    new_csv = not os.path.exists(cfg.metrics_csv)
    if new_csv:
        with open(cfg.metrics_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_loss",
                "train_trans_mean", "train_trans_std", "train_trans_mae",
                "train_wx_mean_deg", "train_wx_std_deg", "train_wx_mae_deg",
                "train_wy_mean_deg", "train_wy_std_deg", "train_wy_mae_deg",
                "train_wz_mean_deg", "train_wz_std_deg", "train_wz_mae_deg",
                "train_rot_norm_mean_deg", "train_rot_norm_std_deg", "train_rot_norm_mae_deg",
                "eval_trans_mean", "eval_trans_std", "eval_trans_mae",
                "eval_wx_mean_deg", "eval_wx_std_deg", "eval_wx_mae_deg",
                "eval_wy_mean_deg", "eval_wy_std_deg", "eval_wy_mae_deg",
                "eval_wz_mean_deg", "eval_wz_std_deg", "eval_wz_mae_deg",
                "eval_rot_norm_mean_deg", "eval_rot_norm_std_deg", "eval_rot_norm_mae_deg",
                "eval_samples"
            ])

    best_loss = float("inf")

    # ==== Load checkpoint if available ====
    ckpt_path = os.path.join(cfg.checkpoint_dir, cfg.save_name)
    if cfg.resume and os.path.exists(cfg.resume_path):
        print(f"[INFO] Loading checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=cfg.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_loss = checkpoint.get("best_loss", best_loss)
        start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"[INFO] Resumed from epoch {start_epoch-1}, best_loss={best_loss:.6f}")
    else:
        start_epoch = 0
        print("[INFO] No checkpoint found, starting fresh.")

    # ==== Training loop ====
    for epoch in range(start_epoch, cfg.epochs + 1):
        # >>>>>>> RESAMPLE TRAINING DATA FOR THIS EPOCH <<<<<<<
        dataset.resample()

        model.train()
        running_loss = 0.0
        train_trans_errs, train_rot_comp_deg, train_rot_norm_deg = [], [], []

        for (inp_cpu, tgt_cpu) in loader:
            inp = inp_cpu.to(cfg.device, non_blocking=True)
            tgt = tgt_cpu.to(cfg.device, non_blocking=True)

            with torch.amp.autocast(device_type=cfg.device,enabled=(cfg.device == "cuda")):
                pred = model(inp)
                loss, trans_err, rot_err = weighted_pose_loss(
                    pred, tgt, trans_w=cfg.trans_w, rot_w=cfg.rot_w
                )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.detach())
            with torch.no_grad():
                rot_comp_deg = to_deg((pred[:, 3:] - tgt[:, 3:]).abs()).detach().cpu().numpy()
                rot_norm_deg = to_deg(rot_err.detach().cpu().numpy())
                t_mag = trans_err.detach().cpu().numpy()

                train_trans_errs.append(t_mag)
                train_rot_comp_deg.append(rot_comp_deg)
                train_rot_norm_deg.append(rot_norm_deg)

        # ---- Train epoch summary
        N_batches = max(1, len(loader))
        epoch_loss = running_loss / N_batches
        train_trans_errs = np.concatenate(train_trans_errs) if train_trans_errs else np.array([0.0])
        train_rot_comp_deg = np.concatenate(train_rot_comp_deg) if train_rot_comp_deg else np.zeros((1,3))
        train_rot_norm_deg = np.concatenate(train_rot_norm_deg) if train_rot_norm_deg else np.array([0.0])

        print(f"\nEpoch {epoch:04d} | Train Loss: {epoch_loss:.6f}")
        print(f"  Train ||e_t|| ({cfg.data_units['x']}): mean={train_trans_errs.mean():.4f}, std={train_trans_errs.std():.4f}")
        print(f"  Train ||e_w|| (deg): mean={train_rot_norm_deg.mean():.4f}, std={train_rot_norm_deg.std():.4f}")

        # ---- Fresh evaluation
        if (epoch % cfg.eval_every) == 0:
            eval_metrics = evaluate(
                model=model,
                poses=poses,
                tr=cfg.transform_ranges,
                device=cfg.device,
                eval_batch_size=cfg.eval_batch_size,
                subset_frac=cfg.eval_subset_frac,
                nn_index=nn_index,  # pass index here too
            )
        else:
            eval_metrics = {k: float("nan") for k in [
                "eval_trans_mean","eval_trans_std","eval_trans_mae",
                "eval_wx_mean_deg","eval_wx_std_deg","eval_wx_mae_deg",
                "eval_wy_mean_deg","eval_wy_std_deg","eval_wy_mae_deg",
                "eval_wz_mean_deg","eval_wz_std_deg","eval_wz_mae_deg",
                "eval_rot_norm_mean_deg","eval_rot_norm_std_deg","eval_rot_norm_mae_deg","eval_samples"
            ]}
            eval_metrics["eval_samples"] = 0

        # ---- Log to wandb
        if cfg.wandb_enabled:
            wandb_metrics = {
                'epoch': epoch,
                'train_loss': epoch_loss,
                'train_trans_mean': float(train_trans_errs.mean()),
                'train_trans_std': float(train_trans_errs.std()),
                'train_trans_mae': float(np.abs(train_trans_errs).mean()),
                'train_wx_mean_deg': float(train_rot_comp_deg[:,0].mean()),
                'train_wx_std_deg': float(train_rot_comp_deg[:,0].std()),
                'train_wx_mae_deg': float(np.abs(train_rot_comp_deg[:,0]).mean()),
                'train_wy_mean_deg': float(train_rot_comp_deg[:,1].mean()),
                'train_wy_std_deg': float(train_rot_comp_deg[:,1].std()),
                'train_wy_mae_deg': float(np.abs(train_rot_comp_deg[:,1]).mean()),
                'train_wz_mean_deg': float(train_rot_comp_deg[:,2].mean()),
                'train_wz_std_deg': float(train_rot_comp_deg[:,2].std()),
                'train_wz_mae_deg': float(np.abs(train_rot_comp_deg[:,2]).mean()),
                'train_rot_norm_mean_deg': float(train_rot_norm_deg.mean()),
                'train_rot_norm_std_deg': float(train_rot_norm_deg.std()),
                'train_rot_norm_mae_deg': float(np.abs(train_rot_norm_deg).mean()),
            }
            
            # Add eval metrics if they are available (not NaN)
            if (epoch % cfg.eval_every) == 0:
                wandb_metrics.update(eval_metrics)
            
            wandb.log(wandb_metrics)

        # ---- Write CSV row
        with open(cfg.metrics_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, f"{epoch_loss:.8f}",
                f"{train_trans_errs.mean():.8f}", f"{train_trans_errs.std():.8f}", f"{np.abs(train_trans_errs).mean():.8f}",
                f"{train_rot_comp_deg[:,0].mean():.8f}", f"{train_rot_comp_deg[:,0].std():.8f}", f"{np.abs(train_rot_comp_deg[:,0]).mean():.8f}",
                f"{train_rot_comp_deg[:,1].mean():.8f}", f"{train_rot_comp_deg[:,1].std():.8f}", f"{np.abs(train_rot_comp_deg[:,1]).mean():.8f}",
                f"{train_rot_comp_deg[:,2].mean():.8f}", f"{train_rot_comp_deg[:,2].std():.8f}", f"{np.abs(train_rot_comp_deg[:,2]).mean():.8f}",
                f"{train_rot_norm_deg.mean():.8f}", f"{train_rot_norm_deg.std():.8f}", f"{np.abs(train_rot_norm_deg).mean():.8f}",
                f"{eval_metrics['eval_trans_mean']:.8f}", f"{eval_metrics['eval_trans_std']:.8f}", f"{eval_metrics['eval_trans_mae']:.8f}",
                f"{eval_metrics['eval_wx_mean_deg']:.8f}", f"{eval_metrics['eval_wx_std_deg']:.8f}", f"{eval_metrics['eval_wx_mae_deg']:.8f}",
                f"{eval_metrics['eval_wy_mean_deg']:.8f}", f"{eval_metrics['eval_wy_std_deg']:.8f}", f"{eval_metrics['eval_wy_mae_deg']:.8f}",
                f"{eval_metrics['eval_wz_mean_deg']:.8f}", f"{eval_metrics['eval_wz_std_deg']:.8f}", f"{eval_metrics['eval_wz_mae_deg']:.8f}",
                f"{eval_metrics['eval_rot_norm_mean_deg']:.8f}", f"{eval_metrics['eval_rot_norm_std_deg']:.8f}", f"{eval_metrics['eval_rot_norm_mae_deg']:.8f}",
                eval_metrics["eval_samples"],
            ])

        # ---- Save best (by train loss; switch to eval if you prefer)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_loss": best_loss,
                "cfg": cfg.__dict__,
            }
            
            # Add wandb run info to checkpoint if wandb is enabled
            if cfg.wandb_enabled and wandb.run is not None:
                checkpoint_data["wandb_run_id"] = wandb.run.id
                checkpoint_data["wandb_run_name"] = wandb.run.name
            
            torch.save(
                checkpoint_data,
                os.path.join(cfg.checkpoint_dir, cfg.save_name),
            )

    # Close wandb run
    if cfg.wandb_enabled:
        wandb.finish()

if __name__ == "__main__":
    main()
