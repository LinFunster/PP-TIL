#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import torch
from torch import nn
import numpy as np
import theseus as th
from utils.train_utils import project_to_frenet_frame
import logging

def select_future_dg(plans, predictions, scores):
    best_mode = torch.argmax(scores, dim=-1)
    plan = torch.stack([plans[i, m] for i, m in enumerate(best_mode)])
    prediction = torch.stack([predictions[i, m] for i, m in enumerate(best_mode)])

    return plan, prediction

def show_metrics(epoch_metrics,str="raw"):
    epoch_metrics = np.array(epoch_metrics)
    plannerADE1, plannerFDE1 = np.mean(epoch_metrics[:, 0]), np.mean(epoch_metrics[:, 3])
    plannerADE2, plannerFDE2 = np.mean(epoch_metrics[:, 1]), np.mean(epoch_metrics[:, 4])
    plannerADE3, plannerFDE3 = np.mean(epoch_metrics[:, 2]), np.mean(epoch_metrics[:, 5])
    epoch_metrics = [plannerADE1, plannerADE2, plannerADE3, plannerFDE1, plannerFDE2, plannerFDE3]
    logging.info(str + f'-plannerADE1: {plannerADE1:.4f}, ' + str + f'-plannerFDE1: {plannerFDE1:.4f}')
    logging.info(str + f'-plannerADE2: {plannerADE2:.4f}, ' + str + f'-plannerFDE2: {plannerFDE2:.4f}')
    logging.info(str + f'-plannerADE3: {plannerADE3:.4f}, ' + str + f'-plannerFDE3: {plannerFDE3:.4f}')
    return epoch_metrics

def msirl_metrics(plan_trajectory, ground_truth_trajectories):
    plan_distance = torch.norm(plan_trajectory[:, :, :2] - ground_truth_trajectories[:, 0, :, :2], dim=-1)
    # planning
    plannerADE1 = torch.mean(plan_distance[:, :10])
    plannerADE2 = torch.mean(plan_distance[:,:30])
    plannerADE3 = torch.mean(plan_distance[:,:50])
    plannerFDE1 = torch.mean(plan_distance[:, 9])
    plannerFDE2 = torch.mean(plan_distance[:, 29])
    plannerFDE3 = torch.mean(plan_distance[:, 49])
    return plannerADE1.item(),plannerADE2.item(),plannerADE3.item(),plannerFDE1.item(),plannerFDE2.item(),plannerFDE3.item()

class RewardFunctionRaw(nn.Module):
    def __init__(self, feature_len=9): # 0.1s / step 
        super(RewardFunctionRaw, self).__init__()
        self.cost = nn.Sequential(nn.Linear(1, 128), nn.ReLU(), nn.Linear(128, feature_len), nn.Softmax(dim=-1))
        self.register_buffer('scale', torch.tensor([1, 1, 1, 1, 1, 10, 100])) # 10, 100
        self.register_buffer('constraint', torch.tensor([[10, 10]])) # 10,10

    def forward(self):
        dummy = torch.ones(1, 1).to(self.cost[0].weight.device)
        cost_function_weights = torch.cat([self.cost(dummy)[:, :7] * self.scale, self.constraint], dim=-1)

        return cost_function_weights

class RewardFunction2(nn.Module):
    def __init__(self, feature_len=9): # 0.1s / step 
        super(RewardFunction2, self).__init__()
        self.cost = nn.Sequential(nn.Linear(1, 128), nn.ReLU(), nn.Linear(128, feature_len), nn.Softmax(dim=-1))
        self.register_buffer('scale', torch.tensor([1, 1, 1, 1, 1, 1, 1])) # 10, 100
        self.register_buffer('constraint', torch.tensor([[1, 10]])) # 10,10

    def forward(self):
        dummy = torch.ones(1, 1).to(self.cost[0].weight.device)
        cost_function_weights = torch.cat([self.cost(dummy)[:, :7] * self.scale, self.constraint], dim=-1)

        return cost_function_weights
 
def cal_traj_features_train(traj, ref_line, prediction, current_state): # 先横再纵：十字顺序
    """base element"""
    # lat_speed, lon_speed = torch.diff(traj[:, :, 0]) / 0.1, torch.diff(traj[:, :, 1]) / 0.1 # dim 49
    # lat_speed, lon_speed = torch.clamp(lat_speed, min=-10., max=50.), torch.clamp(lon_speed, min=-20., max=40.)
    lat_speed, lon_speed = traj[:, :, 3], traj[:, :, 4]
    speed = traj[:, :, 3] / torch.cos(traj[:, :, 2])
    """acc"""
    lat_acc, lon_acc = torch.diff(lat_speed) / 0.1, torch.diff(lon_speed) / 0.1, # dim 48
    f_lat_acc, f_lon_acc = torch.clamp(lat_acc[:, :47], min=-200., max=200.), torch.clamp(lon_acc[:, :47], min=-200., max=200.)
    """jerk"""
    lat_jerk, lon_jerk = torch.diff(lat_speed, n=2) / 0.01, torch.diff(lon_speed, n=2) / 0.01 # dim 47
    f_lat_jerk, f_lon_jerk = torch.clamp(lat_jerk[:, :47], min=-4000., max=2000.), torch.clamp(lon_jerk[:, :47], min=-4000., max=2000.)
    """curvature"""
    # # print((f_lon_acc * lat_speed[:, :47] - f_lat_acc * lon_speed[:, :47]).mean(dim=1).data)
    # # print(((lat_speed[:, :47] ** 2 + lon_speed[:, :47] ** 2).pow(3 / 2)).mean(dim=1).data)
    # f_k_up = (f_lon_acc * lat_speed[:, :47] - f_lat_acc * lon_speed[:, :47])* 0.01
    # f_k_down =  ((lat_speed[:, :47] ** 2 + lon_speed[:, :47] ** 2).pow(3 / 2))*0.0001
    # # print(f_k.mean(dim=1).data)
    # f_k = f_k_up / (f_k_down + 1e-3)
    # f_k2 = f_k**2  # dim 47
    """speed efficiency"""
    speed_limit = torch.max(ref_line[:, :, -1], dim=-1, keepdim=True)[0]
    abs_speed_limit = torch.abs(speed_limit - speed[:,:47])
    f_efficiency = torch.mean(abs_speed_limit, dim=1) # dim 1
    """lane"""
    distance_to_ref = torch.cdist(traj[:, :, :2], ref_line[:, :, :2])  # L2范数
    k = torch.argmin(distance_to_ref, dim=-1).view(-1, traj.shape[1], 1).expand(-1, -1, 3)
    ref_points = torch.gather(ref_line, 1, k)
    f_lane_error = torch.hypot(traj[:, :47, 0]-ref_points[:, :47, 0], traj[:, :47, 1]-ref_points[:, :47, 1]) # dim47
    """collision avoidance"""
    neighbors = prediction.permute(0, 2, 1, 3)
    actor_mask = torch.ne(current_state, 0)[:, 1:, -1]
    ego_current_state = current_state[:, 0]
    ego_len, ego_width = ego_current_state[:, -3], ego_current_state[:, -2]
    neighbors_current_state = current_state[:, 1:]
    neighbors_len, neighbors_width = neighbors_current_state[..., -3], neighbors_current_state[..., -2]
    l_eps = (ego_width.unsqueeze(1) + neighbors_width)/2 + 0.5
    frenet_neighbors = torch.stack([project_to_frenet_frame(neighbors[:, :, i].detach(), ref_line) for i in range(neighbors.shape[2])], dim=2)
    frenet_ego = project_to_frenet_frame(traj.detach(), ref_line)

    safe_error = []
    for t in [0, 2, 5, 9, 14, 19, 24, 29, 39, 46]: # key frames
        # find objects of interest
        l_distance = torch.abs(frenet_ego[:, t, 1].unsqueeze(1) - frenet_neighbors[:, t, :, 1])
        s_distance = frenet_neighbors[:, t, :, 0] - frenet_ego[:, t, 0].unsqueeze(-1)
        interactive = torch.logical_and(s_distance > 0, l_distance < l_eps) * actor_mask

        # find closest object
        distances = torch.norm(traj[:, t, :2].unsqueeze(1) - neighbors[:, t, :, :2], dim=-1).squeeze(1)
        distances = torch.masked_fill(distances, torch.logical_not(interactive), 100)
        distance, index = torch.min(distances, dim=1)
        s_eps = (ego_len + torch.index_select(neighbors_len, 1, index)[:, 0])/2 + 5

        # calculate cost
        error = (s_eps - distance) * (distance < s_eps)
        safe_error.append(error) # dim 47

    f_safe_error = torch.stack(safe_error, dim=1)

    # features
    features = torch.abs(torch.stack([
                torch.sum((0.008*f_lat_acc),dim=1), 
                torch.sum((0.008*f_lon_acc),dim=1), 
                torch.sum((0.004*f_lat_jerk),dim=1), 
                torch.sum((0.01*f_lon_jerk),dim=1), 
                # torch.sum((0.0001*f_k2),dim=1), 
                0.004*f_efficiency, 
                torch.sum((0.0005*f_lane_error),dim=1), 
                torch.sum((0.01*f_safe_error),dim=1)
                ],dim=1))
    
    return features

def cal_traj_features(traj, ref_line, prediction, current_state): # 先横再纵：十字顺序
    """base element"""
    # lat_speed, lon_speed = torch.diff(traj[:, :, 0]) / 0.1, torch.diff(traj[:, :, 1]) / 0.1 # dim 49
    # lat_speed, lon_speed = torch.clamp(lat_speed, min=-10., max=50.), torch.clamp(lon_speed, min=-20., max=40.)
    # speed = torch.hypot(lat_speed, lon_speed) 
    lat_speed, lon_speed = traj[:, :, 3], traj[:, :, 4]
    speed = traj[:, :, 3] / torch.cos(traj[:, :, 2])
    """acc"""
    lat_acc, lon_acc = torch.diff(lat_speed) / 0.1, torch.diff(lon_speed) / 0.1, # dim 48
    f_lat_acc, f_lon_acc = torch.clamp(lat_acc[:, :47], min=-200., max=200.), torch.clamp(lon_acc[:, :47], min=-200., max=200.)
    """jerk"""
    lat_jerk, lon_jerk = torch.diff(lat_speed, n=2) / 0.01, torch.diff(lon_speed, n=2) / 0.01 # dim 47
    f_lat_jerk, f_lon_jerk = torch.clamp(lat_jerk[:, :47], min=-4000., max=2000.), torch.clamp(lon_jerk[:, :47], min=-4000., max=2000.)
    """curvature"""
    # # print((f_lon_acc * lat_speed[:, :47] - f_lat_acc * lon_speed[:, :47]).mean(dim=1).data)
    # # print(((lat_speed[:, :47] ** 2 + lon_speed[:, :47] ** 2).pow(3 / 2)).mean(dim=1).data)
    # f_k_up = (f_lon_acc * lat_speed[:, :47] - f_lat_acc * lon_speed[:, :47])* 0.01
    # f_k_down =  ((lat_speed[:, :47] ** 2 + lon_speed[:, :47] ** 2).pow(3 / 2))*0.0001
    # # print(f_k.mean(dim=1).data)
    # f_k = f_k_up / (f_k_down + 1e-3)
    # f_k2 = f_k**2  # dim 47
    """speed efficiency"""
    speed_limit = torch.max(ref_line[:, :, -1], dim=-1, keepdim=True)[0]
    abs_speed_limit = torch.abs(speed_limit - speed[:,:47])
    f_efficiency = torch.mean(abs_speed_limit, dim=1) # dim 1
    """lane"""
    distance_to_ref = torch.cdist(traj[:, :, :2], ref_line[:, :, :2])  # L2范数
    k = torch.argmin(distance_to_ref, dim=-1).view(-1, traj.shape[1], 1).expand(-1, -1, 3)
    ref_points = torch.gather(ref_line, 1, k)
    f_lane_error = torch.hypot(traj[:, :47, 0]-ref_points[:, :47, 0], traj[:, :47, 1]-ref_points[:, :47, 1]) # dim47
    """collision avoidance"""
    neighbors = prediction.permute(0, 2, 1, 3)
    actor_mask = torch.ne(current_state, 0)[:, 1:, -1]
    ego_current_state = current_state[:, 0]
    ego_len, ego_width = ego_current_state[:, -3], ego_current_state[:, -2]
    neighbors_current_state = current_state[:, 1:]
    neighbors_len, neighbors_width = neighbors_current_state[..., -3], neighbors_current_state[..., -2]
    l_eps = (ego_width.unsqueeze(1) + neighbors_width)/2 + 0.5
    frenet_neighbors = torch.stack([project_to_frenet_frame(neighbors[:, :, i].detach(), ref_line) for i in range(neighbors.shape[2])], dim=2)
    frenet_ego = project_to_frenet_frame(traj.detach(), ref_line)

    safe_error = []
    for t in [0, 2, 5, 9, 14, 19, 24, 29, 39, 46]: # key frames
        # find objects of interest
        l_distance = torch.abs(frenet_ego[:, t, 1].unsqueeze(1) - frenet_neighbors[:, t, :, 1])
        s_distance = frenet_neighbors[:, t, :, 0] - frenet_ego[:, t, 0].unsqueeze(-1)
        interactive = torch.logical_and(s_distance > 0, l_distance < l_eps) * actor_mask

        # find closest object
        distances = torch.norm(traj[:, t, :2].unsqueeze(1) - neighbors[:, t, :, :2], dim=-1).squeeze(1)
        distances = torch.masked_fill(distances, torch.logical_not(interactive), 100)
        distance, index = torch.min(distances, dim=1)
        s_eps = (ego_len + torch.index_select(neighbors_len, 1, index)[:, 0])/2 + 5

        # calculate cost
        error = (s_eps - distance) * (distance < s_eps)
        safe_error.append(error) # dim 47

    f_safe_error = torch.stack(safe_error, dim=1)

    # features
    features = torch.abs(torch.stack([
                torch.sum((0.008*f_lat_acc),dim=1), 
                torch.sum((0.008*f_lon_acc),dim=1), 
                torch.sum((0.004*f_lat_jerk),dim=1), 
                torch.sum((0.01*f_lon_jerk),dim=1), 
                # torch.sum((0.0001*f_k2),dim=1), 
                0.004*f_efficiency, 
                torch.sum((0.0005*f_lane_error),dim=1), 
                torch.sum((0.01*f_safe_error),dim=1)
                ],dim=1))
    
    return features

class MotionPlanner:
    def __init__(self, trajectory_len, feature_len, device, max_iterations=2, step_size=0.4, test=False):
        self.device = device

        # define cost function
        cost_function_weights = [th.ScaleCostWeight(th.Variable(torch.rand(1), name=f'cost_function_weight_{i+1}')) for i in range(feature_len)]

        # define control variable
        control_variables = th.Vector(dof=100, name="control_variables")
        
        # define prediction variable
        predictions = th.Variable(torch.empty(1, 10, trajectory_len, 3), name="predictions")

        # define ref_line_info
        ref_line_info = th.Variable(torch.empty(1, 1200, 5), name="ref_line_info")
        
        # define current state
        current_state = th.Variable(torch.empty(1, 11, 8), name="current_state")

        # set up objective
        objective = th.Objective()
        self.objective = cost_function(objective, control_variables, current_state, predictions, ref_line_info, cost_function_weights)

        # set up optimizer
        if test:
            self.optimizer = th.GaussNewton(objective, th.CholeskyDenseSolver, vectorize=False, max_iterations=50, step_size=0.2, abs_err_tolerance=1e-2)
        else:
            self.optimizer = th.GaussNewton(objective, th.LUDenseSolver, vectorize=False, max_iterations=max_iterations, step_size=step_size, abs_err_tolerance=1e-2)

        # set up motion planner
        self.layer = th.TheseusLayer(self.optimizer, vectorize=False) 
        self.layer.to(device=device)

# model
def bicycle_model_diff(control, current_state): 
    dt = 0.1 # discrete time period [s]
    max_a = 5 # vehicle's accleration limits [m/s^2]
    max_d = 0.5 # vehicle's steering limits [rad]
    L = 3.089 # vehicle's wheelbase [m]
    
    x_0 = current_state[:, 0] # vehicle's x-coordinate [m]
    y_0 = current_state[:, 1] # vehicle's y-coordinate [m]
    theta_0 = current_state[:, 2] # vehicle's heading [rad]
    v_0 = torch.hypot(current_state[:, 3], current_state[:, 4]) # vehicle's velocity [m/s]
    a = control[:, :, 0].clamp(-max_a, max_a) # vehicle's accleration [m/s^2]
    delta = control[:, :, 1].clamp(-max_d, max_d) # vehicle's steering [rad]

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

# cost functions
def acceleration(optim_vars, aux_vars):
    control = optim_vars[0].tensor.view(-1, 50, 2)
    acc = control[:, :, 0]
    
    return acc

def jerk(optim_vars, aux_vars):
    control = optim_vars[0].tensor.view(-1, 50, 2)
    acc = control[:, :, 0]
    jerk = torch.diff(acc) / 0.1
    
    return jerk

def steering(optim_vars, aux_vars):
    control = optim_vars[0].tensor.view(-1, 50, 2)
    steering = control[:, :, 1]

    return steering 

def steering_change(optim_vars, aux_vars):
    control = optim_vars[0].tensor.view(-1, 50, 2)
    steering = control[:, :, 1]
    steering_change = torch.diff(steering) / 0.1

    return steering_change

def speed(optim_vars, aux_vars):
    control = optim_vars[0].tensor.view(-1, 50, 2)
    current_state = aux_vars[1].tensor[:, 0]
    velocity = torch.hypot(current_state[:, 3], current_state[:, 4]) 
    dt = 0.1

    acc = control[:, :, 0]
    speed = velocity.unsqueeze(1) + torch.cumsum(acc * dt, dim=1)
    speed = torch.clamp(speed, min=0)
    speed_limit = torch.max(aux_vars[0].tensor[:, :, -1], dim=-1, keepdim=True)[0]
    speed_error = speed - speed_limit

    return speed_error

def lane_xy(optim_vars, aux_vars):
    control = optim_vars[0].tensor.view(-1, 50, 2)
    ref_line = aux_vars[0].tensor
    current_state = aux_vars[1].tensor[:, 0]
    
    traj = bicycle_model_diff(control, current_state)
    distance_to_ref = torch.cdist(traj[:, :, :2], ref_line[:, :, :2])  # L2范数
    k = torch.argmin(distance_to_ref, dim=-1).view(-1, traj.shape[1], 1).expand(-1, -1, 3)
    ref_points = torch.gather(ref_line, 1, k)
    lane_error = torch.cat([traj[:, 1::2, 0]-ref_points[:, 1::2, 0], traj[:, 1::2, 1]-ref_points[:, 1::2, 1]], dim=1)
    
    return lane_error

def lane_theta(optim_vars, aux_vars):
    control = optim_vars[0].tensor.view(-1, 50, 2)
    ref_line = aux_vars[0].tensor
    current_state = aux_vars[1].tensor[:, 0]

    traj = bicycle_model_diff(control, current_state)
    distance_to_ref = torch.cdist(traj[:, :, :2], ref_line[:, :, :2])
    k = torch.argmin(distance_to_ref, dim=-1).view(-1, traj.shape[1], 1).expand(-1, -1, 3)
    ref_points = torch.gather(ref_line, 1, k)
    theta = traj[:, :, 2]
    lane_error = theta[:, 1::2] - ref_points[:, 1::2, 2]
    
    return lane_error

def red_light_violation(optim_vars, aux_vars):
    control = optim_vars[0].tensor.view(-1, 50, 2)
    current_state = aux_vars[1].tensor[:, 0]
    ref_line = aux_vars[0].tensor
    red_light = ref_line[..., -1]
    dt = 0.1

    velocity = torch.hypot(current_state[:, 3], current_state[:, 4])
    acc = control[:, :, 0]
    speed = velocity.unsqueeze(1) + torch.cumsum(acc * dt, dim=1)
    speed = torch.clamp(speed, min=0)
    s = torch.cumsum(speed * dt, dim=-1)

    stop_point = torch.max(red_light[:, 200:]==0, dim=-1)[1] * 0.1
    stop_distance = stop_point.view(-1, 1) - 3
    red_light_error = (s - stop_distance) * (s > stop_distance) * (stop_point.unsqueeze(-1) != 0)

    return red_light_error

def safety(optim_vars, aux_vars):
    control = optim_vars[0].tensor.view(-1, 50, 2)
    neighbors = aux_vars[0].tensor.permute(0, 2, 1, 3)
    current_state = aux_vars[1].tensor
    ref_line = aux_vars[2].tensor

    actor_mask = torch.ne(current_state, 0)[:, 1:, -1]
    ego_current_state = current_state[:, 0]
    ego = bicycle_model_diff(control, ego_current_state)
    ego_len, ego_width = ego_current_state[:, -3], ego_current_state[:, -2]
    neighbors_current_state = current_state[:, 1:]
    neighbors_len, neighbors_width = neighbors_current_state[..., -3], neighbors_current_state[..., -2]

    l_eps = (ego_width.unsqueeze(1) + neighbors_width)/2 + 0.5
    frenet_neighbors = torch.stack([project_to_frenet_frame(neighbors[:, :, i].detach(), ref_line) for i in range(neighbors.shape[2])], dim=2)
    frenet_ego = project_to_frenet_frame(ego.detach(), ref_line)
    
    safe_error = []
    for t in [0, 2, 5, 9, 14, 19, 24, 29, 39, 49]: # key frames
        # find objects of interest
        l_distance = torch.abs(frenet_ego[:, t, 1].unsqueeze(1) - frenet_neighbors[:, t, :, 1])
        s_distance = frenet_neighbors[:, t, :, 0] - frenet_ego[:, t, 0].unsqueeze(-1)
        interactive = torch.logical_and(s_distance > 0, l_distance < l_eps) * actor_mask

        # find closest object
        distances = torch.norm(ego[:, t, :2].unsqueeze(1) - neighbors[:, t, :, :2], dim=-1).squeeze(1)
        distances = torch.masked_fill(distances, torch.logical_not(interactive), 100)
        distance, index = torch.min(distances, dim=1)
        s_eps = (ego_len + torch.index_select(neighbors_len, 1, index)[:, 0])/2 + 5

        # calculate cost
        error = (s_eps - distance) * (distance < s_eps)
        safe_error.append(error)

    safe_error = torch.stack(safe_error, dim=1)

    return safe_error

def cost_function(objective, control_variables, current_state, predictions, ref_line, cost_function_weights, vectorize=True):
    # travel efficiency
    speed_cost = th.AutoDiffCostFunction([control_variables], speed, 50, cost_function_weights[0], aux_vars=[ref_line, current_state], autograd_vectorize=vectorize, name="speed")
    objective.add(speed_cost)

    # comfort
    acc_cost = th.AutoDiffCostFunction([control_variables], acceleration, 50, cost_function_weights[1], autograd_vectorize=vectorize, name="acceleration")
    objective.add(acc_cost)
    jerk_cost = th.AutoDiffCostFunction([control_variables], jerk, 49, cost_function_weights[2], autograd_vectorize=vectorize, name="jerk")
    objective.add(jerk_cost)
    steering_cost = th.AutoDiffCostFunction([control_variables], steering, 50, cost_function_weights[3], autograd_vectorize=vectorize, name="steering")
    objective.add(steering_cost)
    steering_change_cost = th.AutoDiffCostFunction([control_variables], steering_change, 49, cost_function_weights[4], autograd_vectorize=vectorize, name="steering_change")
    objective.add(steering_change_cost)
    
    # lane departure
    lane_xy_cost = th.AutoDiffCostFunction([control_variables], lane_xy, 50, cost_function_weights[5], aux_vars=[ref_line, current_state], autograd_vectorize=vectorize, name="lane_xy")
    objective.add(lane_xy_cost)
    lane_theta_cost = th.AutoDiffCostFunction([control_variables], lane_theta, 25, cost_function_weights[6], aux_vars=[ref_line, current_state], autograd_vectorize=vectorize, name="lane_theta")
    objective.add(lane_theta_cost)

    # traffic rules
    red_light_cost = th.AutoDiffCostFunction([control_variables], red_light_violation, 50, cost_function_weights[7], aux_vars=[ref_line, current_state], autograd_vectorize=vectorize, name="red_light")
    objective.add(red_light_cost)
    safety_cost = th.AutoDiffCostFunction([control_variables], safety, 10, cost_function_weights[8], aux_vars=[predictions, current_state, ref_line], autograd_vectorize=vectorize, name="safety")
    objective.add(safety_cost)

    return objective


if __name__ == "__main__":
    pass