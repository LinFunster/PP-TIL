#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
from torch import nn
from model.NN_modules import *
import theseus as th
from utils.train_utils import project_to_frenet_frame

class Pipeline:
    # def __init__(self) -> None:
    #     super(Pipeline,self).__init__()
    @staticmethod
    def MFMA_loss(plans, predictions, scores, ground_truth, weights):
        global best_mode

        predictions = predictions * weights.unsqueeze(1)
        prediction_distance = torch.norm(predictions[:, :, :, 9::10, :2] - ground_truth[:, None, 1:, 9::10, :2], dim=-1)
        plan_distance = torch.norm(plans[:, :, 9::10, :2] - ground_truth[:, None, 0, 9::10, :2], dim=-1)
        prediction_distance = prediction_distance.mean(-1).sum(-1)
        plan_distance = plan_distance.mean(-1)

        best_mode = torch.argmin(plan_distance+prediction_distance, dim=-1) 
        score_loss = F.cross_entropy(scores, best_mode)
        best_mode_plan = torch.stack([plans[i, m] for i, m in enumerate(best_mode)])
        best_mode_prediction = torch.stack([predictions[i, m] for i, m in enumerate(best_mode)])
        # best_mode_plan,best_mode_prediction = select_best_future_traj(plans, predictions, best_mode)
        prediction = torch.cat([best_mode_plan.unsqueeze(1), best_mode_prediction], dim=1)

        prediction_loss: torch.tensor = 0
        for i in range(prediction.shape[1]):
            prediction_loss += F.smooth_l1_loss(prediction[:, i], ground_truth[:, i, :, :3])
            prediction_loss += F.smooth_l1_loss(prediction[:, i, -1], ground_truth[:, i, -1, :3])
            
        return 0.5 * prediction_loss + score_loss
    
    @staticmethod
    def select_future(plans, predictions, scores):
        # best_mode = torch.argmin(scores, dim=-1) 
        plan = torch.stack([plans[i, m] for i, m in enumerate(best_mode)])
        prediction = torch.stack([predictions[i, m] for i, m in enumerate(best_mode)])

        return plan, prediction

class AVDecoder(nn.Module):
    def __init__(self, future_steps=50, feature_len=9): # 0.1s / step
        super(AVDecoder, self).__init__()
        self._future_steps = future_steps
        self.planer = nn.Sequential(nn.Dropout(0.1), nn.Linear(512, 256), nn.ELU(), nn.Linear(256, future_steps*2))
        self.cost = nn.Sequential(nn.Linear(1, 128), nn.ReLU(), nn.Linear(128, feature_len), nn.Softmax(dim=-1))
        self.register_buffer('scale', torch.tensor([1, 1, 1, 1, 1, 10, 100]))
        self.register_buffer('constraint', torch.tensor([[10, 10]]))

    def forward(self, agent_map, agent_agent):
        feature = torch.cat([agent_map, agent_agent.unsqueeze(1).repeat(1, 3, 1)], dim=-1)
        trajs = self.planer(feature).view(-1, 3, self._future_steps, 2)
        dummy = torch.ones(1, 1).to(self.cost[0].weight.device)
        cost_function_weights = torch.cat([self.cost(dummy)[:, :7] * self.scale, self.constraint], dim=-1)

        return trajs, cost_function_weights
    

class Score(nn.Module):
    def __init__(self):
        super(Score, self).__init__()
        self.reduce = nn.Sequential(nn.Dropout(0.1), nn.Linear(512, 256), nn.ELU())
        self.decode = nn.Sequential(nn.Dropout(0.1), nn.Linear(512, 128), nn.ELU(), nn.Linear(128, 1))

    def forward(self, map_feature, agent_agent, agent_map):
        # pooling
        map_feature = map_feature.view(map_feature.shape[0], -1, map_feature.shape[-1])
        map_feature = torch.max(map_feature, dim=1)[0]
        agent_agent = torch.max(agent_agent, dim=1)[0]
        agent_map = torch.max(agent_map, dim=2)[0]

        feature = torch.cat([map_feature, agent_agent], dim=-1)
        feature = self.reduce(feature.detach())
        feature = torch.cat([feature.unsqueeze(1).repeat(1, 3, 1), agent_map.detach()], dim=-1)
        scores = self.decode(feature).squeeze(-1)

        return scores

# Build predictor
class Predictor(nn.Module):
    def __init__(self, future_steps):
        super(Predictor, self).__init__()
        self._future_steps = future_steps

        # agent layer
        self.vehicle_encoder = AgentEncoder()
        self.pedestrian_encoder = AgentEncoder()
        self.cyclist_encoder = AgentEncoder()

        # map layer
        self.lane_encoder = LaneEncoder()
        self.crosswalk_encoder = CrosswalkEncoder()

        # attention layers
        self.agent_map_transformer = Agent2Map()
        self.agent_agent_transformer = Agent2Agent()

        # decode layers
        self.plan_decoder = AVDecoder(self._future_steps)
        self.predict_decoder = AgentDecoder(self._future_steps)
        self.score_net = Score()

    def forward(self, ego, neighbors, map_lanes, map_crosswalks):
        # actors
        ego_actor = self.vehicle_encoder(ego)
        vehicles = torch.stack([self.vehicle_encoder(neighbors[:, i]) for i in range(10)], dim=1) 
        pedestrians = torch.stack([self.pedestrian_encoder(neighbors[:, i]) for i in range(10)], dim=1) 
        cyclists = torch.stack([self.cyclist_encoder(neighbors[:, i]) for i in range(10)], dim=1)
        neighbor_actors = torch.where(neighbors[:, :, -1, -1].unsqueeze(2)==2, pedestrians, vehicles)
        neighbor_actors = torch.where(neighbors[:, :, -1, -1].unsqueeze(2)==3, cyclists, neighbor_actors)
        actors = torch.cat([ego_actor.unsqueeze(1), neighbor_actors], dim=1)
        actor_mask = torch.eq(torch.cat([ego.unsqueeze(1), neighbors[..., :-1]], dim=1), 0)[:, :, -1, -1]
        
        # maps
        lane_feature = self.lane_encoder(map_lanes)
        crosswalk_feature = self.crosswalk_encoder(map_crosswalks)
        lane_mask = torch.eq(map_lanes, 0)[:, :, :, 0, 0]
        crosswalk_mask = torch.eq(map_crosswalks, 0)[:, :, :, 0, 0]
        map_mask = torch.cat([lane_mask, crosswalk_mask], dim=2)
        map_mask[:, :, 0] = False # prevent nan
        
        # actor to actor
        agent_agent = self.agent_agent_transformer(actors, actor_mask)
        

        # map to actor
        map_feature = []
        agent_map = []
        # print("ego",ego.shape)
        # print("neighbors",neighbors.shape)
        # print("actors",actors.shape)
        # print("agent_agent",agent_agent.shape)
        # print("lane_feature",lane_feature.shape)
        # print("crosswalk_feature",crosswalk_feature.shape)
        # print("map_mask",map_mask.shape)  
        """
        ego torch.Size([1, 20, 8])
        neighbors torch.Size([1, 10, 20, 9])
        actors torch.Size([1, 11, 256])
        agent_agent torch.Size([1, 11, 256])
        lane_feature torch.Size([1, 11, 6, 100, 256])
        crosswalk_feature torch.Size([1, 11, 4, 100, 256])
        map_mask torch.Size([1, 11, 10])
        """
        for i in range(actors.shape[1]):
            output = self.agent_map_transformer(agent_agent[:, i], lane_feature[:, i], crosswalk_feature[:, i], map_mask[:, i])
            map_feature.append(output[0])
            agent_map.append(output[1])

        map_feature = torch.stack(map_feature, dim=1)
        agent_map = torch.stack(agent_map, dim=2)
        
        # print("ego_actor",ego_actor.shape)
        # print("vehicles",vehicles.shape)
        # print("pedestrians",pedestrians.shape)
        # print("cyclists",cyclists.shape)
        # print("neighbor_actors",neighbor_actors.shape)
        # print("actors",actors.shape)
        # print("agent_agent",agent_agent.shape)
        # print("map_feature",map_feature.shape)
        # print("agent_map",agent_map.shape)
        """
        ego_actor torch.Size([32, 256])
        vehicles torch.Size([32, 10, 256])
        pedestrians torch.Size([32, 10, 256])
        cyclists torch.Size([32, 10, 256])
        neighbor_actors torch.Size([32, 10, 256])
        actors torch.Size([32, 11, 256])
        agent_agent torch.Size([32, 11, 256])
        map_feature torch.Size([32, 11, 10, 256])
        agent_map torch.Size([32, 3, 11, 256])
        """

        # plan + prediction 
        plans, cost_function_weights = self.plan_decoder(agent_map[:, :, 0], agent_agent[:, 0])
        predictions = self.predict_decoder(agent_map[:, :, 1:], agent_agent[:, 1:], neighbors[:, :, -1])
        scores = self.score_net(map_feature, agent_agent, agent_map)
        
        return plans, predictions, scores, cost_function_weights


class MotionPlanner:
    def __init__(self, trajectory_len, feature_len, device, test=False):
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
            self.optimizer = th.GaussNewton(objective, th.LUDenseSolver, vectorize=False, max_iterations=2, step_size=0.4)

        # set up motion planner
        self.layer = th.TheseusLayer(self.optimizer, vectorize=False)
        self.layer.to(device=device)

# model
def bicycle_model(control, current_state):
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
    
    traj = bicycle_model(control, current_state)
    distance_to_ref = torch.cdist(traj[:, :, :2], ref_line[:, :, :2])
    k = torch.argmin(distance_to_ref, dim=-1).view(-1, traj.shape[1], 1).expand(-1, -1, 3)
    ref_points = torch.gather(ref_line, 1, k)
    lane_error = torch.cat([traj[:, 1::2, 0]-ref_points[:, 1::2, 0], traj[:, 1::2, 1]-ref_points[:, 1::2, 1]], dim=1)

    return lane_error

def lane_theta(optim_vars, aux_vars):
    control = optim_vars[0].tensor.view(-1, 50, 2)
    ref_line = aux_vars[0].tensor
    current_state = aux_vars[1].tensor[:, 0]

    traj = bicycle_model(control, current_state)
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
    ego = bicycle_model(control, ego_current_state)
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

    # lane
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
    # set up model
    model = Predictor(50)
    print(model)
    print('Model Params:', sum(p.numel() for p in model.parameters()))
