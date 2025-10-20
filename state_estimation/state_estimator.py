import numpy as np 
from scipy.spatial.transform import Rotation as R
import torch 
from utils.pose_utils import torch_matrix_to_pose_xyzabc, torch_pose_xyzabc_to_matrix
from cicp.icp_manifold import ContactPoseManifold 
import matplotlib.pyplot as plt
import time
import random

def set_seed(seed=42):
    """Set random seeds for reproducible results across numpy, torch, and random modules."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    # For deterministic behavior in CUDA operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed} for reproducible results")

class ForwardModel(torch.nn.Module):
    def __init__(self, contact_model):
        super().__init__()
        self.contact_manifold = contact_model

    def forward(self, pose_H_h, pose_h_p, pose_p_P):
        """
        Args:
            pose_H_h: (B, 6) tensor of poses of hole prior wrt hole posterior 
            pose_h_E: (B, 6) tensor of poses of end effector wrt hole prior  
            pose_E_P: (B, 6) tensor of poses of peg wrt end effector 
        Returns:
            loss: differentiable scalar loss
        """
        B = pose_h_p.shape[0] # batch size 
        dtype = pose_h_p.dtype
        device = pose_h_p.device 

        # Convert pose to matrix form
        tf_H_h = torch_pose_xyzabc_to_matrix(pose_H_h).to(device) 
        tf_h_p = torch_pose_xyzabc_to_matrix(pose_h_p).to(device) 
        tf_p_P = torch_pose_xyzabc_to_matrix(pose_p_P).to(device)
        
        if tf_H_h.shape[0] == 1:
            tf_H_h = tf_H_h.expand(B, -1, -1)

        if tf_p_P.shape[0] == 1:
            tf_p_P = tf_p_P.expand(B, -1, -1)

        tf_h_P = torch.bmm(tf_h_p, tf_p_P)

        tf_H_P = torch.bmm(tf_H_h, tf_h_P)

        pose_H_P = torch_matrix_to_pose_xyzabc(tf_H_P).to(device)  

        pose_H_P = pose_H_P.to(next(self.contact_manifold.model.parameters()).device)
        pose_H_P_proj = self.contact_manifold.model(pose_H_P)

        loss_regularization = torch.mean(torch.abs(pose_H_h)) + torch.mean(torch.abs(pose_p_P))
        loss = torch.mean(torch.abs(pose_H_P_proj - pose_H_P)) + 1e-3 * loss_regularization # FIXME: adjust the regularization term weight as needed
        loss_position = torch.mean(torch.abs(pose_H_P_proj[:, :3] - pose_H_P[:, :3])) 
        loss_rotation = torch.mean(torch.abs(pose_H_P_proj[:, 3:] - pose_H_P[:, 3:]))
        # loss = torch.mean(torch.abs(pose_H_P_proj[:,:3] - pose_H_P[:,:3]))
        # loss = torch.mean(torch.sum((pose_H_P_proj[:,:3] - pose_H_P[:,:3]) ** 2, dim=1)) 

        return loss, loss_position, loss_rotation 
    
def estimate_holePose_and_inHandPose(
        contact_model, 
        observations, 
        config, 
        max_it=10_000, 
        lr=1e-5, 
        optimizer_type='adam', 
        gradient_noise_std=0.02, 
        max_samples=None, 
        seed=None,
        lr_decay_type='none',
        lr_decay_rate=0.95,
        lr_decay_step=100
    ):
    """
    Estimate both the hole pose offset and in-hand pose simultaneously.
    
    Args:
        contact_model: The contact manifold model
        observations: Pose observations for estimation
        config: Configuration dictionary with device settings
        max_it: Maximum number of iterations
        lr: Initial learning rate
        optimizer_type: Type of optimizer ('adam' or 'sgd')
        gradient_noise_std: Standard deviation for gradient noise
        max_samples: Maximum number of samples to use
        seed: Random seed for reproducibility
        lr_decay_type: Type of learning rate decay ('none', 'exponential', 'step', 'cosine')
        lr_decay_rate: Decay rate for exponential/step decay (typically 0.9-0.99)
        lr_decay_step: Step size for step decay (number of iterations between decay)
    """

    device = config['device']
    
    # Set seed for reproducibility if provided
    if seed is not None:
        set_seed(seed)
        
    pose_h_p = torch.tensor(observations, dtype=torch.float32, device=device) if isinstance(observations, np.ndarray) else observations.to(device)
    initial_hole_pose = np.zeros((1,6), dtype=np.float32) # tf_H_h 
    pose_H_h = torch.nn.Parameter(torch.tensor(initial_hole_pose, dtype=torch.float32, device=device)) # tf_H_h 
    initial_inhand_yxrx = np.zeros((1,3), dtype=np.float32) # yzrx

    # Optimizable part: y, z, rx (3D)
    pose_P_p_y_optim = torch.nn.Parameter(torch.tensor(initial_inhand_yxrx[0,0], dtype=torch.float32).reshape(1,1).to(device))
    pose_P_p_z_optim = torch.nn.Parameter(torch.tensor(initial_inhand_yxrx[0,1], dtype=torch.float32).reshape(1,1).to(device))
    pose_P_p_rx_optim = torch.nn.Parameter(torch.tensor(initial_inhand_yxrx[0,2], dtype=torch.float32).reshape(1,1).to(device))


    # Non-optimizable parts
    pose_P_p_prefix = torch.tensor([0.0], dtype=torch.float32, device=device).reshape(1,1)
    pose_P_p_suffix = torch.tensor([0.0, 0.0], dtype=torch.float32, device=device).reshape(1,2)
    pose_P_p = torch.cat([pose_P_p_prefix, pose_P_p_y_optim, pose_P_p_z_optim, pose_P_p_suffix, pose_P_p_rx_optim], dim=-1)

    B = pose_h_p.shape[0] # batch size 
    
    if max_samples is not None and max_samples < B:
        selected_idx = torch.randperm(pose_h_p.shape[0])[:max_samples]
        pose_h_p = pose_h_p[selected_idx]
    B = pose_h_p.shape[0] # batch size 

    parameters = [pose_H_h, pose_P_p_y_optim, pose_P_p_z_optim, pose_P_p_rx_optim]

    FM = ForwardModel(contact_model=contact_model).to(device)

    if optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=lr, betas=(0.0, 0.0), eps=1e-12, weight_decay=0.0)
    elif optimizer_type.lower() == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=lr)

    # Set up learning rate scheduler
    scheduler = None
    if lr_decay_type.lower() == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay_rate)
    elif lr_decay_type.lower() == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_rate)
    elif lr_decay_type.lower() == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_it)

    pose_H_h_history = [pose_H_h.detach().clone()]  # keep on device
    pose_P_p_history = [pose_H_h.detach().clone()]  # keep on device
    loss_history = []
    lr_history = []  # Track learning rate over iterations
    lr_history = []  # Track learning rate history

    use_combined_loss = False

    for iter in range(max_it):
        optimizer.zero_grad()
        loss_total, loss_position, loss_rotation = FM.forward(pose_H_h=pose_H_h, pose_h_p=pose_h_p, pose_p_P=pose_P_p)
        loss = loss_total if use_combined_loss else loss_position
        loss.backward()

        if gradient_noise_std > 0:
            for p in parameters:
                if p.grad is not None:
                    p.grad += torch.randn_like(p.grad) * gradient_noise_std

        optimizer.step()
        
        # Apply learning rate decay
        if scheduler is not None:
            if lr_decay_type.lower() == 'step' and (iter + 1) % lr_decay_step == 0:
                scheduler.step()
            elif lr_decay_type.lower() in ['exponential', 'cosine']:
                scheduler.step()
        
        pose_P_p = torch.cat([pose_P_p_prefix, pose_P_p_y_optim, pose_P_p_z_optim, pose_P_p_suffix, pose_P_p_rx_optim], dim=-1)

        pose_H_h_history.append(pose_H_h.detach().clone())
        pose_P_p_history.append(pose_P_p.detach().clone())
        loss_history.append(loss.item())
        lr_history.append(optimizer.param_groups[0]['lr'])  # Track current learning rate

        # Record the current learning rate
        for param_group in optimizer.param_groups:
            lr_history.append(param_group['lr'])

        if loss.item() < 1e-6:
            print(f"Early stopping at iteration {iter}, loss: {loss.item():.6f}")
            break

    # Move to CPU only once at the end
    pose_H_h_history_cpu = [p.cpu().numpy() for p in pose_H_h_history]

    final_hole_pose = pose_H_h_history[-1].detach().cpu().numpy()
    pose_H_h_est = final_hole_pose  # or any processing if needed
    tf_H_h_est = torch_pose_xyzabc_to_matrix(torch.tensor(final_hole_pose, dtype=torch.float32)).cpu().numpy()  # Convert to matrix form

    final_inhand_pose = pose_P_p_history[-1].detach().cpu().numpy()
    pose_P_p_est = final_inhand_pose  # or any processing if needed
    tf_P_p_est = torch_pose_xyzabc_to_matrix(torch.tensor(final_inhand_pose, dtype=torch.float32)).cpu().numpy()  # Convert to matrix form

    return tf_H_h_est, pose_H_h_est, pose_H_h_history, tf_P_p_est, pose_P_p_est, pose_P_p_history, loss_history, lr_history

def estimate_holePose(
        contact_model, 
        observations, 
        config, 
        max_it=10_000, 
        lr=1e-5, 
        optimizer_type='adam', 
        gradient_noise_std=0.02, 
        max_samples=None, 
        save_results=False,
        convergence_tolerance=1e-4,
        convergence_window=25,
        param_change_tolerance=1e-3,
        seed=None,
        lr_decay_type='none',
        lr_decay_rate=0.95,
        lr_decay_step=100
    ):
    """
    Estimate only the hole pose offset (tf_H_h) without optimizing the in-hand pose.
    The peg frame remains fixed relative to the end effector.
    
    Args:
        convergence_tolerance: Threshold for loss change to consider convergence
        convergence_window: Number of iterations to check for convergence
        param_change_tolerance: Threshold for parameter change to consider convergence
        seed: Random seed for reproducibility
        lr_decay_type: Type of learning rate decay ('none', 'exponential', 'step', 'cosine')
        lr_decay_rate: Decay rate for exponential/step decay (typically 0.9-0.99)
        lr_decay_step: Step size for step decay (number of iterations between decay)
    """
    device = config['device']
    
    # Set seed for reproducibility if provided
    if seed is not None:
        set_seed(seed)
        
    pose_h_p = torch.tensor(observations, dtype=torch.float32, device=device) if isinstance(observations, np.ndarray) else observations.to(device)
    
    # Initialize hole pose offset
    initial_hole_pose = np.zeros((1,6), dtype=np.float32) # tf_H_h 
    pose_H_h = torch.nn.Parameter(torch.tensor(initial_hole_pose, dtype=torch.float32, device=device)) # tf_H_h 
    
    # Fixed in-hand pose (no optimization)
    fixed_inhand_pose = np.zeros((1,6), dtype=np.float32) # tf_p_P (identity transform)
    pose_p_P = torch.tensor(fixed_inhand_pose, dtype=torch.float32, device=device)

    B = pose_h_p.shape[0] # batch size 
    
    if max_samples is not None and max_samples < B:
        selected_idx = torch.randperm(pose_h_p.shape[0])[:max_samples]
        pose_h_p = pose_h_p[selected_idx]
    B = pose_h_p.shape[0] # batch size 

    parameters = [pose_H_h]  # Only optimize hole pose

    FM = ForwardModel(contact_model=contact_model).to(device)

    if optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=lr, betas=(0.0, 0.0), eps=1e-12, weight_decay=0.0)
    elif optimizer_type.lower() == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=lr)

    # Set up learning rate scheduler
    scheduler = None
    if lr_decay_type.lower() == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay_rate)
    elif lr_decay_type.lower() == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_rate)
    elif lr_decay_type.lower() == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_it)

    pose_H_h_history = [pose_H_h.detach().clone()]  # keep on device
    loss_history = []
    lr_history = []  # Track learning rate over iterations
    lr_history = []  # Track learning rate history

    use_combined_loss = False
    
    # Convergence tracking variables
    convergence_met = False
    
    for iter in range(max_it):
        # Store previous parameters for convergence check
        prev_pose_H_h = pose_H_h.detach().clone()
        
        optimizer.zero_grad()
        loss_total, loss_position, loss_rotation = FM.forward(pose_H_h=pose_H_h, pose_h_p=pose_h_p, pose_p_P=pose_p_P)
        loss = loss_total if use_combined_loss else loss_position
        loss.backward()

        if gradient_noise_std > 0:
            for p in parameters:
                if p.grad is not None:
                    p.grad += torch.randn_like(p.grad) * gradient_noise_std

        optimizer.step()

        # Apply learning rate decay
        if scheduler is not None:
            if lr_decay_type.lower() == 'step' and (iter + 1) % lr_decay_step == 0:
                scheduler.step()
            elif lr_decay_type.lower() in ['exponential', 'cosine']:
                scheduler.step()

        pose_H_h_history.append(pose_H_h.detach().clone())
        loss_history.append(loss.item())
        lr_history.append(optimizer.param_groups[0]['lr'])  # Track current learning rate

        # Record the current learning rate
        for param_group in optimizer.param_groups:
            lr_history.append(param_group['lr'])

        # Early stopping based on absolute loss threshold
        if loss.item() < 1e-6:
            print(f"Early stopping at iteration {iter}, loss: {loss.item():.6f}")
            convergence_met = True
            break
        
        # Convergence check based on loss change and parameter change
        if iter >= convergence_window:
            # Check loss convergence over the window
            recent_losses = loss_history[-convergence_window:]
            loss_std = np.std(recent_losses)
            loss_change = abs(recent_losses[-1] - recent_losses[0])
            
            # Check parameter change
            param_change = torch.norm(pose_H_h - prev_pose_H_h).item()
            
            # Check if both loss and parameters have converged
            if (loss_std < convergence_tolerance and 
                loss_change < convergence_tolerance and 
                param_change < param_change_tolerance):
                print(f"Convergence achieved at iteration {iter}")
                print(f"  Loss std over {convergence_window} iterations: {loss_std:.2e}")
                print(f"  Loss change: {loss_change:.2e}")
                print(f"  Parameter change: {param_change:.2e}")
                convergence_met = True
                break
    
    if not convergence_met:
        print(f"Maximum iterations ({max_it}) reached without convergence")
        if len(loss_history) >= convergence_window:
            recent_losses = loss_history[-convergence_window:]
            loss_std = np.std(recent_losses)
            loss_change = abs(recent_losses[-1] - recent_losses[0])
            param_change = torch.norm(pose_H_h - pose_H_h_history[-2]).item() if len(pose_H_h_history) >= 2 else 0
            print(f"  Final loss std: {loss_std:.2e}")
            print(f"  Final loss change: {loss_change:.2e}")
            print(f"  Final parameter change: {param_change:.2e}")

    # Move to CPU only once at the end
    pose_H_h_history_cpu = [p.cpu().numpy() for p in pose_H_h_history]

    final_hole_pose = pose_H_h_history[-1].detach().cpu().numpy()
    pose_H_h_est = final_hole_pose  # or any processing if needed
    tf_H_h_est = torch_pose_xyzabc_to_matrix(torch.tensor(final_hole_pose, dtype=torch.float32)).cpu().numpy()  # Convert to matrix form

    return tf_H_h_est, pose_H_h_est, pose_H_h_history, loss_history, lr_history

def read_observation(observation_path):
    tf_H_P = np.load(observation_path, allow_pickle=True)
    return tf_H_P 

def offset_observation(max_hole_pose_offsets, max_in_hand_pose_offsets, observation, set_max_offsets=False, seed=None):
    """
    Generate offset observations for testing.
    
    Args:
        max_hole_pose_offsets: Maximum offsets for hole pose
        max_in_hand_pose_offsets: Maximum offsets for in-hand pose
        observation: Original observation
        set_max_offsets: Whether to use maximum offsets or random offsets
        seed: Random seed for reproducibility
    """
    
    # Set seed for reproducibility if provided
    if seed is not None:
        set_seed(seed)

    tf_H_h = np.eye(4) 
    if set_max_offsets:
        # set offset to +/- max_offsets
        offset_signs = np.random.choice([-1, 1], size=6)
        offset = offset_signs * max_hole_pose_offsets
        tf_H_h[:3, 3] = offset[:3]
        tf_H_h[:3, :3] = R.from_euler('xyz', offset[3:][::-1], degrees=True).as_matrix()
    else:
        # set offset to random values within the range of max_offsets
        tf_H_h[:3, 3] = np.random.uniform(-max_hole_pose_offsets[:3], max_hole_pose_offsets[:3])
        tf_H_h[:3, :3] = R.from_euler('xyz', np.random.uniform(-max_hole_pose_offsets, +max_hole_pose_offsets), degrees=True).as_matrix()
    tf_h_H = np.linalg.inv(tf_H_h)  # Inverse of the transformation

    tf_H_P = torch_pose_xyzabc_to_matrix(torch.tensor(observation, dtype=torch.float32)) if isinstance(observation, np.ndarray) else observation
    tf_H_P = tf_H_P.cpu().numpy() if isinstance(tf_H_P, torch.Tensor) else tf_H_P
    tf_h_P = np.matmul(tf_h_H, tf_H_P) 

    tf_P_p = np.eye(4)
    if set_max_offsets:
        # set offset to +/- max_offsets
        offset_signs = np.random.choice([-1, 1], size=6)
        offset = offset_signs * max_in_hand_pose_offsets
        tf_P_p[:3, 3] = offset[:3]
        tf_P_p[:3, :3] = R.from_euler('xyz', offset[3:][::-1], degrees=True).as_matrix()
    else:
        # set offset to random values within the range of max_offsets
        tf_P_p[:3, 3] = np.random.uniform(-max_in_hand_pose_offsets[:3], max_in_hand_pose_offsets[:3])
        tf_P_p[:3, :3] = R.from_euler('xyz', np.random.uniform(-max_in_hand_pose_offsets, +max_in_hand_pose_offsets), degrees=True).as_matrix()
    tf_h_p = np.matmul(tf_h_P, tf_P_p)  # Apply the in-hand pose offset

    return tf_h_p, tf_H_h, tf_P_p  

def plot_history(pose_H_h_error_history, pose_P_p_error_history, loss_history):

    # Plot the loss history
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss History')
    plt.legend()
    plt.grid()
    plt.show()

    pose_H_h_history = np.array(pose_H_h_error_history)
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    axs[0, 0].plot(pose_H_h_history[:, 0], label='Hole Pose - X')
    axs[0,0].axhline(0, color='gray', linestyle='--')
    axs[0, 1].plot(pose_H_h_history[:, 1], label='Hole Pose - Y')
    axs[0,1].axhline(0, color='gray', linestyle='--')
    axs[0, 2].plot(pose_H_h_history[:, 2], label='Hole Pose - Z')
    axs[0,2].axhline(0, color='gray', linestyle='--')
    axs[1, 0].plot(pose_H_h_history[:, 3], label='Hole Pose - Theta Z')
    axs[1,0].axhline(0, color='gray', linestyle='--')
    axs[1, 1].plot(pose_H_h_history[:, 4], label='Hole Pose - Theta Y')
    axs[1,1].axhline(0, color='gray', linestyle='--')
    axs[1, 2].plot(pose_H_h_history[:, 5], label='Hole Pose - Theta X') 
    axs[1,2].axhline(0, color='gray', linestyle='--')
    axs[2, 0].plot(pose_P_p_error_history[:, 1], label='In-Hand Pose - Y')
    axs[2,0].axhline(0, color='gray', linestyle='--')
    axs[2, 1].plot(pose_P_p_error_history[:, 2], label='In-Hand Pose - Z')
    axs[2,1].axhline(0, color='gray', linestyle='--')
    axs[2, 2].plot(pose_P_p_error_history[:, 5], label='In-Hand Pose - Theta X')
    axs[2,2].axhline(0, color='gray', linestyle='--')

    for ax in axs.flat:
        ax.legend()
        ax.grid()

    plt.tight_layout()
    plt.show()

def compute_errors(tf_H_h, pose_H_h_history, tf_H_h_est, tf_P_p, pose_P_p_history, tf_P_p_est):
    tf_H_h_error = tf_H_h @ np.linalg.inv(tf_H_h_est)  
    pose_H_h_error = torch_matrix_to_pose_xyzabc(torch.tensor(tf_H_h_error, dtype=torch.float32)).cpu().numpy()  # Convert to pose representation
    tf_H_h_history = [] 
    tf_H_h_error_history = [] 
    pose_H_h_error_history = [] 
    for pose in pose_H_h_history:
        tf_H_h_history.append(torch_pose_xyzabc_to_matrix(torch.tensor(pose.reshape(1,6), dtype=torch.float32)).cpu().numpy()) 
        tf_H_h_error_history.append(tf_H_h @ np.linalg.inv(tf_H_h_history[-1]))
        pose_H_h_error_history.append(torch_matrix_to_pose_xyzabc(torch.tensor(tf_H_h_error_history[-1], dtype=torch.float32)).cpu().numpy())
    tf_H_h_history = np.array(tf_H_h_history).reshape(-1, 4, 4)
    tf_H_h_error_history = np.array(tf_H_h_error_history).reshape(-1, 4, 4)
    pose_H_h_error_history = np.array(pose_H_h_error_history).reshape(-1, 6)

    tf_P_p_error = tf_P_p @ np.linalg.inv(tf_P_p_est)
    pose_P_p_error = torch_matrix_to_pose_xyzabc(torch.tensor(tf_P_p_error, dtype=torch.float32)).cpu().numpy()  # Convert to pose representation
    tf_P_p_history = []
    tf_P_p_error_history = []
    pose_P_p_error_history = []
    for pose in pose_P_p_history:
        tf_P_p_history.append(torch_pose_xyzabc_to_matrix(torch.tensor(pose.reshape(1,6), dtype=torch.float32)).cpu().numpy()) 
        tf_P_p_error_history.append(tf_P_p @ np.linalg.inv(tf_P_p_history[-1]))
        pose_P_p_error_history.append(torch_matrix_to_pose_xyzabc(torch.tensor(tf_P_p_error_history[-1], dtype=torch.float32)).cpu().numpy())
    tf_P_p_history = np.array(tf_P_p_history).reshape(-1, 4, 4)
    tf_P_p_error_history = np.array(tf_P_p_error_history).reshape(-1, 4, 4)
    pose_P_p_error_history = np.array(pose_P_p_error_history).reshape(-1, 6)    
    
    return tf_H_h_error, pose_H_h_error, tf_H_h_history, tf_H_h_error_history, pose_H_h_error_history, tf_P_p_error, pose_P_p_error, tf_P_p_history, tf_P_p_error_history, pose_P_p_error_history, 

def plot_pose_estimates_2x3(pose_H_h_history, pose_P_p_history=None, title_prefix="Pose Estimates"):
    """
    Plot pose estimates vs iteration in a 2x3 subplot layout.
    
    Args:
        pose_H_h_history: List of hole pose estimates over iterations (N, 6)
        pose_P_p_history: Optional list of in-hand pose estimates over iterations (N, 6)
        title_prefix: Prefix for the plot title
    """
    # Convert history to numpy array for easier indexing
    if isinstance(pose_H_h_history[0], torch.Tensor):
        pose_H_h_array = np.array([p.cpu().numpy().flatten() for p in pose_H_h_history])
    else:
        pose_H_h_array = np.array([p.flatten() for p in pose_H_h_history])
    
    iterations = np.arange(len(pose_H_h_array))
    
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'{title_prefix} vs Iteration', fontsize=16)
    
    # Pose component labels
    pose_labels = ['X', 'Y', 'Z', 'Rx', 'Ry', 'Rz']
    pose_units = ['[m]', '[m]', '[m]', '[rad]', '[rad]', '[rad]']
    
    # Plot hole pose estimates (6 components in 2x3 layout)
    for i in range(6):
        row = i // 3
        col = i % 3
        axs[row, col].plot(iterations, pose_H_h_array[:, i], 'b-', linewidth=2, label='Hole Pose')
        axs[row, col].set_xlabel('Iteration')
        axs[row, col].set_ylabel(f'{pose_labels[i]} {pose_units[i]}')
        axs[row, col].set_title(f'Hole Pose {pose_labels[i]}')
        axs[row, col].grid(True, alpha=0.3)
        
        # Add horizontal line at zero for reference
        axs[row, col].axhline(0, color='gray', linestyle='--', alpha=0.5)
        
        # If in-hand pose history is provided, plot it as well
        if pose_P_p_history is not None:
            if isinstance(pose_P_p_history[0], torch.Tensor):
                pose_P_p_array = np.array([p.cpu().numpy().flatten() for p in pose_P_p_history])
            else:
                pose_P_p_array = np.array([p.flatten() for p in pose_P_p_history])
            
            axs[row, col].plot(iterations, pose_P_p_array[:, i], 'r--', linewidth=2, label='In-Hand Pose')
        
        axs[row, col].legend()
    
    plt.tight_layout()
    plt.show()

def plot_pose_errors_2x3(pose_H_h_error_history, pose_P_p_error_history=None, title_prefix="Pose Errors"):
    """
    Plot pose errors vs iteration in a 2x3 subplot layout.
    
    Args:
        pose_H_h_error_history: Array of hole pose errors over iterations (N, 6)
        pose_P_p_error_history: Optional array of in-hand pose errors over iterations (N, 6)
        title_prefix: Prefix for the plot title
    """
    pose_H_h_array = np.array(pose_H_h_error_history)
    iterations = np.arange(len(pose_H_h_array))
    
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'{title_prefix} vs Iteration', fontsize=16)
    
    # Pose component labels
    pose_labels = ['X', 'Y', 'Z', 'RZ', 'RY', 'RX']
    pose_units = ['[m]', '[m]', '[m]', '[deg]', '[deg]', '[deg]']
    
    # Plot hole pose errors (6 components in 2x3 layout)
    for i in range(6):
        row = i // 3
        col = i % 3
        axs[row, col].plot(iterations, pose_H_h_array[:, i], 'b-', linewidth=2, label='Hole Pose Error')
        axs[row, col].set_xlabel('Iteration')
        axs[row, col].set_ylabel(f'{pose_labels[i]} Error {pose_units[i]}')
        axs[row, col].set_title(f'Hole Pose {pose_labels[i]} Error')
        axs[row, col].grid(True, alpha=0.3)
        
        # Add horizontal line at zero for reference
        axs[row, col].axhline(0, color='gray', linestyle='--', alpha=0.5)
        
        # If in-hand pose error history is provided, plot it as well
        if pose_P_p_error_history is not None:
            pose_P_p_array = np.array(pose_P_p_error_history)
            axs[row, col].plot(iterations, pose_P_p_array[:, i], 'r--', linewidth=2, label='In-Hand Pose Error')
        
        axs[row, col].legend()
    
    plt.tight_layout()
    plt.show()

def plot_pose_estimates_and_errors_combined(pose_history, pose_error_history, pose_P_p_history=None, pose_P_p_error_history=None, title_prefix="Pose Analysis"):
    """
    Plot both pose estimates and errors in a combined 4x3 subplot layout.
    Top 2 rows show estimates, bottom 2 rows show errors.
    
    Args:
        pose_history: List/array of pose estimates over iterations (N, 6)
        pose_error_history: Array of pose errors over iterations (N, 6)
        pose_P_p_history: Optional in-hand pose estimates (N, 6)
        pose_P_p_error_history: Optional in-hand pose errors (N, 6)
        title_prefix: Prefix for the plot title
    """
    # Convert history to numpy array for easier indexing
    if isinstance(pose_history[0], torch.Tensor):
        pose_array = np.array([p.cpu().numpy().flatten() for p in pose_history])
    else:
        pose_array = np.array([p.flatten() for p in pose_history])
    
    pose_error_array = np.array(pose_error_history)
    iterations = np.arange(len(pose_array))
    
    fig, axs = plt.subplots(4, 3, figsize=(18, 16))
    fig.suptitle(f'{title_prefix} - Estimates and Errors vs Iteration', fontsize=16)
    
    # Pose component labels
    pose_labels = ['X', 'Y', 'Z', 'Rx', 'Ry', 'Rz']
    pose_units = ['[m]', '[m]', '[m]', '[rad]', '[rad]', '[rad]']
    
    # Plot estimates and errors for each component
    for i in range(6):
        col = i % 3
        
        # Estimates (top 2 rows)
        est_row = i // 3
        axs[est_row, col].plot(iterations, pose_array[:, i], 'b-', linewidth=2, label='Hole Pose Estimate')
        axs[est_row, col].set_xlabel('Iteration')
        axs[est_row, col].set_ylabel(f'{pose_labels[i]} {pose_units[i]}')
        axs[est_row, col].set_title(f'Hole Pose {pose_labels[i]} Estimate')
        axs[est_row, col].grid(True, alpha=0.3)
        axs[est_row, col].axhline(0, color='gray', linestyle='--', alpha=0.5)
        
        # Add in-hand pose estimates if provided
        if pose_P_p_history is not None:
            if isinstance(pose_P_p_history[0], torch.Tensor):
                pose_P_p_array = np.array([p.cpu().numpy().flatten() for p in pose_P_p_history])
            else:
                pose_P_p_array = np.array([p.flatten() for p in pose_P_p_history])
            axs[est_row, col].plot(iterations, pose_P_p_array[:, i], 'r--', linewidth=2, label='In-Hand Pose Estimate')
        
        axs[est_row, col].legend()
        
        # Errors (bottom 2 rows)
        err_row = est_row + 2
        axs[err_row, col].plot(iterations, pose_error_array[:, i], 'b-', linewidth=2, label='Hole Pose Error')
        axs[err_row, col].set_xlabel('Iteration')
        axs[err_row, col].set_ylabel(f'{pose_labels[i]} Error {pose_units[i]}')
        axs[err_row, col].set_title(f'Hole Pose {pose_labels[i]} Error')
        axs[err_row, col].grid(True, alpha=0.3)
        axs[err_row, col].axhline(0, color='gray', linestyle='--', alpha=0.5)
        
        # Add in-hand pose errors if provided
        if pose_P_p_error_history is not None:
            pose_P_p_error_array = np.array(pose_P_p_error_history)
            axs[err_row, col].plot(iterations, pose_P_p_error_array[:, i], 'r--', linewidth=2, label='In-Hand Pose Error')
        
        axs[err_row, col].legend()
    
    plt.tight_layout()
    plt.show()

def plot_learning_rate_history(lr_history, title_prefix="Learning Rate"):
    """
    Plot learning rate vs iteration.
    
    Args:
        lr_history: List of learning rates over iterations
        title_prefix: Prefix for the plot title
    """
    plt.figure(figsize=(10, 6))
    plt.plot(lr_history, linewidth=2, marker='o', markersize=3)
    plt.xlabel('Iteration')
    plt.ylabel('Learning Rate')
    plt.title(f'{title_prefix} vs Iteration')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()

def main(seed=1): 
    # Set global seed for reproducibility
    set_seed(seed)
    print(f"Running main() with seed={seed} for reproducible testing")
    
    # example usage 
    config = {
        # 'contact_model_path': '/home/rp/abhay_ws/cicp/checkpoints/extrusion_run_4_best_NN_model_xyzabc.pth', 
        # 'contact_model_path': '/home/rp/abhay_ws/contact-manifold-state-estimation/model_training/checkpoints/extrusion_run_2_best_NN_model_xyzabc.pth', 
        'contact_model_path': "./model_training/checkpoints/BNC_run_1_best_NN_model_xyzabc.pth", 
        # 'observation_path': '/home/rp/abhay_ws/cicp/data/extrusion_observations/extrusion_timed_icp_10/hole_frame/extrusion_pose_H_P_0.npy',
        'observation_path': '/home/rp/dhanush_ws/sunrise-wrapper/data/BNC/noisy_trajectories/Oct18_2118/exec_1_traj_H_P.npy',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu', 
        'sim_data': False, 
    }

    tf_H_P = read_observation(config['observation_path'])  

    # sample 
    tf_H_P = tf_H_P[20000:22000]  
    tf_H_P = tf_H_P[::10]  # downsample for faster testing

    plot_pose_errors_2x3(tf_H_P, title_prefix="Observation Trajectory")


    if config['sim_data']:
        # updating frame of observations from base of peg to tip of peg 
        for i, pose_H_P_i in enumerate(tf_H_P): 
            pose_H_P_i = torch.tensor(pose_H_P_i, dtype=torch.float32)
            tf_H_P_i = torch_pose_xyzabc_to_matrix(pose_H_P_i.unsqueeze(0)).squeeze(0)
            transform_peg = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,-25],[0,0,0,1]], dtype=torch.float32) # 25mm offset in z
            transform_hole = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,-25],[0,0,0,1]], dtype=torch.float32) # 25mm offset in z
            transform_hole_inv = torch.inverse(transform_hole)
            tf_H_P_transformed = transform_hole_inv @ tf_H_P_i @ transform_peg
            pose_H_P_transformed = torch_matrix_to_pose_xyzabc(tf_H_P_transformed.unsqueeze(0)).cpu().numpy()
            tf_H_P[i] = pose_H_P_transformed.flatten()

    tf_h_p, tf_H_h, tf_P_p = offset_observation(
        max_hole_pose_offsets=np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0]), 
        # max_in_hand_pose_offsets=np.array([0, 10.0, 10.0, 0, 0, 10.0]),
        max_in_hand_pose_offsets=np.array([0, 0, 0, 0, 0, 0]),
        observation=tf_H_P, 
        set_max_offsets=True,
        seed=seed  # Pass seed for reproducible offset generation
    )  # Offset the observation
    pose_H_h = torch_matrix_to_pose_xyzabc(torch.tensor(tf_H_h.reshape(1,4,4), dtype=torch.float32)).cpu().numpy()      
    pose_P_p = torch_matrix_to_pose_xyzabc(torch.tensor(tf_P_p.reshape(1,4,4), dtype=torch.float32)).cpu().numpy()  

    print(f"Initial pose_H_h: {pose_H_h}")
    print(f"Initial pose_P_p: {pose_P_p}")
    pose_h_p = torch_matrix_to_pose_xyzabc(torch.tensor(tf_h_p, dtype=torch.float32)).cpu().numpy()
    
    cpm = ContactPoseManifold(geometry="gear") 
    cpm.load_model_from_path(config['contact_model_path'], layer_sizes=[6, 4096, 4096, 4096, 4096, 6])

    # print("\n" + "="*50)
    # print("Testing estimate_holePose_and_inHandPose() - joint estimation")
    # print("="*50)
    
    # ret = estimate_holePose_and_inHandPose(
    #     contact_model=cpm,  
    #     observations=pose_h_p,
    #     config=config,
    #     max_samples=10000,
    #     max_it=1000, 
    #     lr=1e-2, 
    #     optimizer_type='adam',
    #     seed=seed,  # Pass seed for reproducible estimation
    #     lr_decay_type='step',  # Enable step learning rate decay
    #     lr_decay_rate=0.9,  # Reduce LR by 10% at each decay step
    #     lr_decay_step=200  # Decay every 200 iterations
    # ) 
    # tf_H_h_est, pose_H_h_est, pose_H_h_history, tf_P_p_est, pose_P_p_est, pose_P_p_history, loss_history, lr_history = ret 
    # tf_H_h_error, pose_H_h_error, tf_H_h_history, tf_H_h_error_history, pose_H_h_error_history, tf_P_p_error, pose_P_p_error, tf_P_p_history, tf_P_p_error_history, pose_P_p_error_history = compute_errors(tf_H_h, pose_H_h_history, tf_H_h_est, tf_P_p, pose_P_p_history, tf_P_p_est) 
    # print(f"Joint estimation - Initial hole pose error: {pose_H_h}")
    # print(f"Joint estimation - Final hole pose est: {pose_H_h_est}")
    # print(f"Joint estimation - Final pose error: {pose_H_h_error}") 
    # print(f"Joint estimation - Initial in-hand pose error: {pose_P_p}")
    # print(f"Joint estimation - Final in-hand pose est: {pose_P_p_est}")
    # print(f"Joint estimation - Final in-hand pose error: {pose_P_p_error}")
    
    # # Plot joint estimation results
    # plot_pose_estimates_2x3(pose_H_h_history, pose_P_p_history, title_prefix="Joint Pose Estimates (Hole + In-Hand)")
    # plot_pose_errors_2x3(pose_H_h_error_history, pose_P_p_error_history, title_prefix="Joint Pose Errors (Hole + In-Hand)")
    
    # # Plot combined estimates and errors in 4x3 subplots
    # plot_pose_estimates_and_errors_combined(pose_H_h_history, pose_H_h_error_history, pose_P_p_history, pose_P_p_error_history, title_prefix="Joint Estimation (Hole + In-Hand)")
    
    # # Plot learning rate history
    # plot_learning_rate_history(lr_history, title_prefix="Joint Estimation Learning Rate")
    
    # plot_history(pose_H_h_error_history, pose_P_p_error_history, loss_history) 

    print("\n" + "="*50)
    print("Testing estimate_holePose() - hole pose only estimation")
    print("="*50)
    
    # Test the new estimate_holePose function (only hole pose estimation)
    time_start = time.time()
    ret_hole_only = estimate_holePose(
        contact_model=cpm,  
        observations=pose_h_p,
        config=config,
        max_samples=None,
        max_it=10000, 
        lr=1e-3, 
        optimizer_type='adam',
        # convergence_tolerance=1e-2,
        # convergence_window=10,
        # param_change_tolerance=1e-3,
        seed=seed,  # Pass seed for reproducible estimation
        lr_decay_type='exponential',  # Enable exponential learning rate decay
        lr_decay_rate=1.0,  # Decay rate (reduce LR by 2% each step)
        lr_decay_step=100000  # Decay every 10 iterations (only used for 'step' decay)
    )     
    tf_H_h_est_hole_only, pose_H_h_est_hole_only, pose_H_h_history_hole_only, loss_history_hole_only, lr_history_hole_only = ret_hole_only
    time_end = time.time()
    print(f"Hole-only estimation took {time_end - time_start:.2f} seconds")
    
    # Compute errors for hole-only estimation
    tf_H_h_error_hole_only = tf_H_h @ np.linalg.inv(tf_H_h_est_hole_only)  
    pose_H_h_error_hole_only = torch_matrix_to_pose_xyzabc(torch.tensor(tf_H_h_error_hole_only, dtype=torch.float32)).cpu().numpy()
    
    # Compute error history for hole-only estimation
    tf_H_h_history_hole_only = []
    tf_H_h_error_history_hole_only = []
    pose_H_h_error_history_hole_only = []
    for pose in pose_H_h_history_hole_only:
        tf_H_h_history_hole_only.append(torch_pose_xyzabc_to_matrix(torch.tensor(pose.reshape(1,6), dtype=torch.float32)).cpu().numpy()) 
        tf_H_h_error_history_hole_only.append(tf_H_h @ np.linalg.inv(tf_H_h_history_hole_only[-1]))
        pose_H_h_error_history_hole_only.append(torch_matrix_to_pose_xyzabc(torch.tensor(tf_H_h_error_history_hole_only[-1], dtype=torch.float32)).cpu().numpy())
    tf_H_h_history_hole_only = np.array(tf_H_h_history_hole_only).reshape(-1, 4, 4)
    tf_H_h_error_history_hole_only = np.array(tf_H_h_error_history_hole_only).reshape(-1, 4, 4)
    pose_H_h_error_history_hole_only = np.array(pose_H_h_error_history_hole_only).reshape(-1, 6)
    
    print(f"Hole-only estimation - Initial hole pose error: {pose_H_h}")
    print(f"Hole-only estimation - Final hole pose est: {pose_H_h_est_hole_only}")
    print(f"Hole-only estimation - Final hole pose error: {pose_H_h_error_hole_only}")
    
    # # Plot pose estimates vs iteration in 2x3 subplots
    # plot_pose_estimates_2x3(pose_H_h_history_hole_only, title_prefix="Hole Pose Estimates (Hole-Only)")
    
    # # Plot learning rate history
    # plot_learning_rate_history(lr_history_hole_only, title_prefix="Hole-Only Estimation Learning Rate")
    
    # Plot loss history
    plt.figure(figsize=(8, 6))
    plt.plot(loss_history_hole_only, label='Hole-only estimation', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Hole-only Estimation Loss History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot pose errors vs iteration in 2x3 subplots
    plot_pose_errors_2x3(pose_H_h_error_history_hole_only, title_prefix="Hole Pose Errors (Hole-Only)")

    # # Combined plots for estimates and errors
    # plot_pose_estimates_and_errors_combined(pose_H_h_history_hole_only, pose_H_h_error_history_hole_only, title_prefix="Combined Pose Analysis (Hole-Only)")
    
if __name__ == "__main__":
    # Default seed for reproducible testing
    import sys
    
    # Normal operation with specified seed
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    print(f"Using seed: {seed}")
    main(seed=seed)