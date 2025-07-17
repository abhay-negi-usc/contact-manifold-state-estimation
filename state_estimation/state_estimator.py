import numpy as np 
from scipy.spatial.transform import Rotation as R
import torch 
from utils.pose_utils import torch_matrix_to_pose_xyzabc, torch_pose_xyzabc_to_matrix
from cicp.icp_manifold import ContactPoseManifold 

class ForwardModel(torch.nn.Module):
    def __init__(self, contact_model):
        super().__init__()
        self.contact_manifold = contact_model

    def forward(self, pose_H_h, pose_h_E, pose_E_P):
        """
        Args:
            pose_H_h: (B, 6) tensor of poses of hole prior wrt hole posterior 
            pose_h_E: (B, 6) tensor of poses of end effector wrt hole prior  
            pose_E_P: (B, 6) tensor of poses of peg wrt end effector 
        Returns:
            loss: differentiable scalar loss
        """
        B = pose_h_E.shape[0] # batch size 
        dtype = pose_h_E.dtype
        device = pose_h_E.device 

        # Convert pose to matrix form
        tf_H_h = torch_pose_xyzabc_to_matrix(pose_H_h).to(device) 
        tf_h_E = torch_pose_xyzabc_to_matrix(pose_h_E).to(device) 
        tf_E_P = torch_pose_xyzabc_to_matrix(pose_E_P).to(device)
        
        if tf_H_h.shape[0] == 1:
            tf_H_h = tf_H_h.expand(B, -1, -1)

        if tf_E_P.shape[0] == 1:
            tf_E_P = tf_E_P.expand(B, -1, -1)

        tf_h_P = torch.bmm(tf_h_E, tf_E_P)

        tf_H_P = torch.bmm(tf_H_h, tf_h_P)

        pose_H_P = torch_matrix_to_pose_xyzabc(tf_H_P).to(device)  

        pose_H_P = pose_H_P.to(next(self.contact_manifold.model.parameters()).device)
        pose_H_P_proj = self.contact_manifold.model(pose_H_P)

        loss = torch.mean(torch.abs(pose_H_P_proj - pose_H_P))
        loss_position = torch.mean(torch.abs(pose_H_P_proj[:, :3] - pose_H_P[:, :3])) 
        loss_rotation = torch.mean(torch.abs(pose_H_P_proj[:, 3:] - pose_H_P[:, 3:]))
        # loss = torch.mean(torch.abs(pose_H_P_proj[:,:3] - pose_H_P[:,:3]))
        # loss = torch.mean(torch.sum((pose_H_P_proj[:,:3] - pose_H_P[:,:3]) ** 2, dim=1)) 

        return loss, loss_position, loss_rotation 
    
def estimate_holePose(contact_model, observations, config, max_it=10_000, lr=1e-5, optimizer_type='adam', gradient_noise_std=0.02, max_samples=None, save_results=False):

    device = config['device']
    pose_h_P = torch.tensor(observations, dtype=torch.float32, device=device) if isinstance(observations, np.ndarray) else observations.to(device)
    initial_hole_pose = np.zeros((1,6), dtype=np.float32) # tf_H_h 
    pose_H_h = torch.nn.Parameter(torch.tensor(initial_hole_pose, dtype=torch.float32, device=device)) # tf_H_h 
    if max_samples is not None and max_samples < B:
        selected_idx = torch.randperm(pose_h_P.shape[0])[:max_samples]
        pose_h_P = pose_h_P[selected_idx]

    B = pose_h_P.shape[0] # batch size 

    pose_E_P = torch.zeros((B, 6), dtype=torch.float32, device=device)  # Assuming peg frame is aligned with end effector frame

    parameters = [pose_H_h]

    FM = ForwardModel(contact_model=contact_model).to(device)

    if optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=lr, betas=(0.0, 0.0), eps=1e-12, weight_decay=0.0)
    elif optimizer_type.lower() == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=lr)

    hole_pose_history, loss_history = [], [] 
    hole_pose_history.append(pose_H_h.detach().cpu().numpy().copy()) 

    use_combined_loss = False

    for iter in range(max_it):
        optimizer.zero_grad()
        loss_total, loss_position, loss_rotation = FM.forward(pose_H_h=pose_H_h, pose_h_E=pose_h_P, pose_E_P=pose_E_P)
        loss = loss_total if use_combined_loss else loss_position
        loss.backward()

        if gradient_noise_std > 0:
            for p in parameters:
                if p.grad is not None:
                    p.grad += torch.randn_like(p.grad) * gradient_noise_std

        optimizer.step()

        hole_pose_history.append(pose_H_h.detach().cpu().numpy().copy())
        loss_history.append(loss.item())

        if loss.item() < 1e-6:
            print(f"Early stopping at iteration {iter}, loss: {loss.item():.6f}")
            break

    opt_pose_H_h = pose_H_h.detach().cpu().numpy()

    return opt_pose_H_h, hole_pose_history, loss_history 

def read_observation(observation_path):
    tf_H_P = np.load(observation_path, allow_pickle=True)
    return tf_H_P 

def offset_observation(max_offsets, observation):

    tf_H_h = np.eye(4) 
    tf_H_h[:3, 3] = np.random.uniform(-max_offsets[:3], max_offsets[:3])
    tf_H_h[:3, :3] = R.from_euler('xyz', np.random.uniform(-max_offsets[3:], max_offsets[3:])).as_matrix()
    tf_h_H = np.linalg.inv(tf_H_h)  # Inverse of the transformation

    tf_H_P = torch_pose_xyzabc_to_matrix(torch.tensor(observation, dtype=torch.float32)) if isinstance(observation, np.ndarray) else observation
    tf_H_P = tf_H_P.cpu().numpy() if isinstance(tf_H_P, torch.Tensor) else tf_H_P
    tf_h_P = np.matmul(tf_h_H, tf_H_P)
    return tf_h_P, tf_H_h

def main(): 
    # example usage 
    config = {
        'contact_model_path': '/home/rp/abhay_ws/cicp/checkpoints/gear_best_NN_model_xyzabc (copy).pth', 
        'observation_path': '/home/rp/abhay_ws/cicp/data/gear_observations/GEAR_TIMED_ICP_10_B3/hole_frame/gear_pose_H_P_0.npy',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu', 
    }

    tf_H_P = read_observation(config['observation_path'])  
    tf_h_P, tf_H_h = offset_observation(max_offsets=np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0]), observation=tf_H_P) 
    pose_h_P = torch_matrix_to_pose_xyzabc(torch.tensor(tf_h_P, dtype=torch.float32)).cpu().numpy()  # Convert to pose representation

    contact_model = torch.load(config['contact_model_path'], map_location=config['device'])  # Load your contact model here

    cpm = ContactPoseManifold(geometry="gear") 
    cpm.load_model_from_path(config['contact_model_path'])

    tf_H_h_est, tf_H_h_history, loss_history = estimate_holePose(
        contact_model=cpm,  
        observations=pose_h_P,
        config=config
    ) 

    tf_H_h_error = tf_H_h @ np.linalg.inv(tf_H_h_est)  # Calculate the error in the estimated pose 

    print(f"Error in Estimated Pose: {tf_H_h_error}")

if __name__ == "__main__":
    main() 