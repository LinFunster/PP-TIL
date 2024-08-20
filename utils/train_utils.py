import torch
import logging
import glob
import random
import numpy as np
from torch.utils.data import Dataset
from torch.nn import functional as F

def initLogging(log_file: str, level: str = "INFO"):
    logging.basicConfig(filename=log_file, filemode='w',
                        level=getattr(logging, level, None),
                        format='[%(levelname)s %(asctime)s] %(message)s',
                        datefmt='%m-%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler())

# def set_seed(CUR_SEED):
#     random.seed(CUR_SEED)
#     np.random.seed(CUR_SEED)
#     torch.manual_seed(CUR_SEED)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

def fixed_seed(seed, if_torch=True, if_tf=False, if_np=True, if_random=True, if_torch_cuda=True):
    if if_np:
        import numpy as np
        np.random.seed(seed)
    if if_random:
        import random
        random.seed(seed)
    if if_torch:
        import torch as t
        t.manual_seed(seed)
    if if_torch_cuda:
        import torch as t
        t.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
        # t.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
        # True:根据网络结构进行提速，测试采用最快的卷积算法，不适用于动态的网络结构。
        t.backends.cudnn.benchmark = False
        # Ture:将这个flag置为True的话，每次返回的卷积算法将是确定的，即默认算法。
        t.backends.cudnn.deterministic = True
    if if_tf:
        import tensorflow as tf
        tf.random.set_seed(seed)

class DrivingData(Dataset):
    def __init__(self, data_dir):
        self.data_list = glob.glob(data_dir)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = np.load(self.data_list[idx])
        ego = data['ego']
        neighbors = data['neighbors']
        ref_line = data['ref_line'] 
        map_lanes = data['map_lanes']
        map_crosswalks = data['map_crosswalks']
        gt_future_states = data['gt_future_states']

        return ego, neighbors, map_lanes, map_crosswalks, ref_line, gt_future_states

def motion_metrics(plan_trajectory, prediction_trajectories, ground_truth_trajectories, weights):
    prediction_trajectories = prediction_trajectories * weights
    plan_distance = torch.norm(plan_trajectory[:, :, :2] - ground_truth_trajectories[:, 0, :, :2], dim=-1)
    prediction_distance = torch.norm(prediction_trajectories[:, :, :, :2] - ground_truth_trajectories[:, 1:, :, :2], dim=-1)

    # planning
    plannerADE = torch.mean(plan_distance)
    plannerFDE = torch.mean(plan_distance[:, -1])
    
    # prediction
    predictorADE = torch.mean(prediction_distance, dim=-1)
    predictorADE = torch.masked_select(predictorADE, weights[:, :, 0, 0])
    predictorADE = torch.mean(predictorADE)
    predictorFDE = prediction_distance[:, :, -1]
    predictorFDE = torch.masked_select(predictorFDE, weights[:, :, 0, 0])
    predictorFDE = torch.mean(predictorFDE)

    return plannerADE.item(), plannerFDE.item(), predictorADE.item(), predictorFDE.item()

def project_to_frenet_frame(traj, ref_line):
    distance_to_ref = torch.cdist(traj[:, :, :2], ref_line[:, :, :2])
    k = torch.argmin(distance_to_ref, dim=-1).view(-1, traj.shape[1], 1).expand(-1, -1, 3)
    ref_points = torch.gather(ref_line, 1, k)
    x_r, y_r, theta_r = ref_points[:, :, 0], ref_points[:, :, 1], ref_points[:, :, 2] 
    x, y = traj[:, :, 0], traj[:, :, 1]
    s = 0.1 * (k[:, :, 0] - 200)
    l = torch.sign((y-y_r)*torch.cos(theta_r)-(x-x_r)*torch.sin(theta_r)) * torch.sqrt(torch.square(x-x_r)+torch.square(y-y_r))
    sl = torch.stack([s, l], dim=-1)

    return sl

def project_to_cartesian_frame(traj, ref_line):
    k = (10 * traj[:, :, 0] + 200).long()
    k = torch.clip(k, 0, 1200-1)
    ref_points = torch.gather(ref_line, 1, k.view(-1, traj.shape[1], 1).expand(-1, -1, 3))
    x_r, y_r, theta_r = ref_points[:, :, 0], ref_points[:, :, 1], ref_points[:, :, 2] 
    x = x_r - traj[:, :, 1] * torch.sin(theta_r)
    y = y_r + traj[:, :, 1] * torch.cos(theta_r)
    xy = torch.stack([x, y], dim=-1)

    return xy

def bicycle_model_train(control, current_state):
    dt = 0.1 # discrete time period [s]
    max_delta = 0.6 # vehicle's steering limits [rad]
    max_a = 5 # vehicle's accleration limits [m/s^2]

    x_0 = current_state[:, 0] # vehicle's x-coordinate [m]
    y_0 = current_state[:, 1] # vehicle's y-coordinate [m]
    theta_0 = current_state[:, 2] # vehicle's heading [rad]
    v_0 = torch.hypot(current_state[:, 3], current_state[:, 4]) # vehicle's velocity [m/s]
    L = 3.089 # vehicle's wheelbase [m]
    a = control[:, :, 0].clamp(-max_a, max_a) # vehicle's accleration [m/s^2]
    delta = control[:, :, 1].clamp(-max_delta, max_delta) # vehicle's steering [rad]

    # speed
    v = v_0.unsqueeze(1) + torch.cumsum(a * dt, dim=1)
    v = torch.clamp(v, min=0)

    # angle
    d_theta = v * delta / L # use delta to approximate tan(delta)
    theta = theta_0.unsqueeze(1) + torch.cumsum(d_theta * dt, dim=-1)
    theta = torch.fmod(theta, 2*torch.pi)
    
    # x and y coordniate
    x = x_0.unsqueeze(1) + torch.cumsum(v * torch.cos(theta) * dt, dim=-1)
    y = y_0.unsqueeze(1) + torch.cumsum(v * torch.sin(theta) * dt, dim=-1)
    
    # output trajectory
    traj = torch.stack([x, y, theta, v * torch.cos(theta), v * torch.sin(theta)], dim=-1)

    return traj

def bicycle_model(control, current_state):
    dt = 0.1 # discrete time period [s]
    max_delta = 0.6 # vehicle's steering limits [rad]
    max_a = 5 # vehicle's accleration limits [m/s^2]

    x_0 = current_state[:, 0] # vehicle's x-coordinate [m]
    y_0 = current_state[:, 1] # vehicle's y-coordinate [m]
    theta_0 = current_state[:, 2] # vehicle's heading [rad]
    v_0 = torch.hypot(current_state[:, 3], current_state[:, 4]) # vehicle's velocity [m/s]
    L = 3.089 # vehicle's wheelbase [m]
    a = control[:, :, 0].clamp(-max_a, max_a) # vehicle's accleration [m/s^2]
    delta = control[:, :, 1].clamp(-max_delta, max_delta) # vehicle's steering [rad]

    # speed
    v = v_0.unsqueeze(1) + torch.cumsum(a * dt, dim=1)
    v = torch.clamp(v, min=0)

    # angle
    d_theta = v * delta / L # use delta to approximate tan(delta)
    theta = theta_0.unsqueeze(1) + torch.cumsum(d_theta * dt, dim=-1)
    theta = torch.fmod(theta, 2*torch.pi)
    
    # x and y coordniate
    x = x_0.unsqueeze(1) + torch.cumsum(v * torch.cos(theta) * dt, dim=-1)
    y = y_0.unsqueeze(1) + torch.cumsum(v * torch.sin(theta) * dt, dim=-1)
    
    # output trajectory
    traj = torch.stack([x, y, theta, v], dim=-1)

    return traj

def physical_model(control, current_state, dt=0.1):
    dt = 0.1 # discrete time period [s]
    max_d_theta = 0.5 # vehicle's change of angle limits [rad/s]
    max_a = 5 # vehicle's accleration limits [m/s^2]

    x_0 = current_state[:, 0] # vehicle's x-coordinate
    y_0 = current_state[:, 1] # vehicle's y-coordinate
    theta_0 = current_state[:, 2] # vehicle's heading [rad]
    v_0 = torch.hypot(current_state[:, 3], current_state[:, 4]) # vehicle's velocity [m/s]
    a = control[:, :, 0].clamp(-max_a, max_a) # vehicle's accleration [m/s^2]
    d_theta = control[:, :, 1].clamp(-max_d_theta, max_d_theta) # vehicle's heading change rate [rad/s]

    # speed
    v = v_0.unsqueeze(1) + torch.cumsum(a * dt, dim=1)
    v = torch.clamp(v, min=0)

    # angle
    theta = theta_0.unsqueeze(1) + torch.cumsum(d_theta * dt, dim=-1)
    theta = torch.fmod(theta, 2*torch.pi)
    
    # x and y coordniate
    x = x_0.unsqueeze(1) + torch.cumsum(v * torch.cos(theta) * dt, dim=-1)
    y = y_0.unsqueeze(1) + torch.cumsum(v * torch.sin(theta) * dt, dim=-1)

    # output trajectory
    traj = torch.stack([x, y, theta, v], dim=-1)

    return traj
