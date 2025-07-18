import numpy as np 
from scipy.spatial.transform import Rotation as R
import torch 
from utils.pose_utils import torch_matrix_to_pose_xyzabc, torch_pose_xyzabc_to_matrix
from cicp.icp_manifold import ContactPoseManifold 
import matplotlib.pyplot as plt

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
        save_results=False
    ):

    device = config['device']
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

    pose_H_h_history = [pose_H_h.detach().clone()]  # keep on device
    pose_P_p_history = [pose_H_h.detach().clone()]  # keep on device
    loss_history = []

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
        pose_P_p = torch.cat([pose_P_p_prefix, pose_P_p_y_optim, pose_P_p_z_optim, pose_P_p_suffix, pose_P_p_rx_optim], dim=-1)

        pose_H_h_history.append(pose_H_h.detach().clone())
        pose_P_p_history.append(pose_P_p.detach().clone())
        loss_history.append(loss.item())

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

    return tf_H_h_est, pose_H_h_est, pose_H_h_history, tf_P_p_est, pose_P_p_est, pose_P_p_history, loss_history

def read_observation(observation_path):
    tf_H_P = np.load(observation_path, allow_pickle=True)
    return tf_H_P 

def offset_observation(max_hole_pose_offsets, max_in_hand_pose_offsets, observation, set_max_offsets=False):

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

def main(): 
    # example usage 
    config = {
        'contact_model_path': '/home/rp/abhay_ws/cicp/checkpoints/gear_best_NN_model_xyzabc (copy).pth', 
        'observation_path': '/home/rp/abhay_ws/cicp/data/gear_observations/GEAR_TIMED_ICP_10_B3/hole_frame/gear_pose_H_P_0.npy',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu', 
    }

    tf_H_P = read_observation(config['observation_path'])  
    tf_h_p, tf_H_h, tf_P_p = offset_observation(
        max_hole_pose_offsets=np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0]), 
        max_in_hand_pose_offsets=np.array([0, 10.0, 10.0, 0, 0, 10.0]),
        observation=tf_H_P, 
        set_max_offsets=True
    )  # Offset the observation
    pose_H_h = torch_matrix_to_pose_xyzabc(torch.tensor(tf_H_h.reshape(1,4,4), dtype=torch.float32)).cpu().numpy()      
    pose_P_p = torch_matrix_to_pose_xyzabc(torch.tensor(tf_P_p.reshape(1,4,4), dtype=torch.float32)).cpu().numpy()  

    print(f"Initial pose_H_h: {pose_H_h}")
    pose_h_p = torch_matrix_to_pose_xyzabc(torch.tensor(tf_h_p, dtype=torch.float32)).cpu().numpy()  

    cpm = ContactPoseManifold(geometry="gear") 
    cpm.load_model_from_path(config['contact_model_path'])

    ret = estimate_holePose_and_inHandPose(
        contact_model=cpm,  
        observations=pose_h_p,
        config=config,
        max_samples=10000,
        max_it=1000, 
        lr=1e-2, 
        optimizer_type='adam', 
    ) 
    tf_H_h_est, pose_H_h_est, pose_H_h_history, tf_P_p_est, pose_P_p_est, pose_P_p_history, loss_history = ret 

    tf_H_h_error, pose_H_h_error, tf_H_h_history, tf_H_h_error_history, pose_H_h_error_history, tf_P_p_error, pose_P_p_error, tf_P_p_history, tf_P_p_error_history, pose_P_p_error_history = compute_errors(tf_H_h, pose_H_h_history, tf_H_h_est, tf_P_p, pose_P_p_history, tf_P_p_est) 

    print(f"Initial hole pose error: {pose_H_h}")
    print(f"Final hole pose est: {pose_H_h_est}")
    print(f"Final pose error: {pose_H_h_error}") 


    print(f"Initial in-hand pose error: {pose_P_p}")
    print(f"Final in-hand pose est: {pose_P_p_est}")
    print(f"Final in-hand pose error: {pose_P_p_error}")

    plot_history(pose_H_h_error_history, pose_P_p_error_history, loss_history) 

if __name__ == "__main__":
    main() 