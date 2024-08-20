
import torch
import argparse
import os
import logging
import time
from utils.test_utils import *
from model.DataManager import *
from torch.utils.data import DataLoader
import sys
from utils.Kmeans import K_Means
from utils.train_utils import fixed_seed,project_to_frenet_frame
from model.L_IRL import cal_traj_features
# from utils.train_utils import *
# 简化：把所有数据当成一个batch来处理

def bicycle_model_plan2traj(control, current_state):
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
    d_theta = v * torch.tan(delta) / L
    theta = theta_0.unsqueeze(1) + torch.cumsum(d_theta * dt, dim=-1)
    theta = torch.fmod(theta, 2*torch.pi)
    
    # x and y coordniate
    x = x_0.unsqueeze(1) + torch.cumsum(v * torch.cos(theta) * dt, dim=-1)
    y = y_0.unsqueeze(1) + torch.cumsum(v * torch.sin(theta) * dt, dim=-1)
    
    # output trajectory
    traj = torch.stack([x, y, theta, v * torch.cos(theta), v * torch.sin(theta)], dim=-1)

    return traj


def check_traj_info(traj, efficiency_cost, risky_cost, lane_cost):
    lat_speed, lon_speed = torch.diff(traj[:, :, 0]) / 0.1, torch.diff(traj[:, :, 1]) / 0.1
    lat_acc, lon_acc = torch.diff(lat_speed) / 0.1, torch.diff(lon_speed) / 0.1
    lat_jerk, lon_jerk = torch.diff(lat_speed, n=2) / 0.01, torch.diff(lon_speed, n=2) / 0.01
    f_k_up = (lon_acc[:, :47] * lat_speed[:, :47] - lat_acc[:, :47] * lon_speed[:, :47]) * 0.01
    f_k_down =((lat_speed[:, :47] ** 2 + lon_speed[:, :47] ** 2).pow(3 / 2)) * 0.0001
    f_k = f_k_up/(f_k_down+1e-3)
    f_k2 = f_k**2
    logging.info(f"lat_speed: {lat_speed.min():.4f}, {lat_speed.max():.4f}")
    logging.info(f"lon_speed: {lon_speed.min():.4f}, {lon_speed.max():.4f}")
    logging.info(f"lat_acc: {lat_acc.min():.4f}, {lat_acc.max():.4f}")
    logging.info(f"lon_acc: {lon_acc.min():.4f}, {lon_acc.max():.4f}")
    logging.info(f"lat_jerk: {lat_jerk.min():.4f}, {lat_jerk.max():.4f}")
    logging.info(f"lon_jerk: {lon_jerk.min():.4f}, {lon_jerk.max():.4f}")
    logging.info(f"f_k_up: {f_k_up.min():.4f}, {f_k_up.max():.4f}")
    logging.info(f"f_k_down: {f_k_down.min():.4f}, {f_k_down.max():.4f}")
    logging.info(f"f_k: {f_k.min():.4f}, {f_k.max():.4f}")
    logging.info(f"f_k2: {f_k2.min():.4f}, {f_k2.max():.4f}")
    logging.info(f"speed_efficiency: {efficiency_cost.min():.4f}, {efficiency_cost.max():.4f}")
    logging.info(f"lane_mean_error: {lane_cost.min():.4f}, {lane_cost.max():.4f}")
    logging.info(f"risky_cost: {risky_cost.min():.4f}, {risky_cost.max():.4f}")

def check_feature_min_max_info(features):
    logging.info(f"lat_acc: {features[:,0].min():.4f}, {features[:,0].max():.4f}, {features[:,0].mean():.4f}")
    logging.info(f"lon_acc: {features[:,1].min():.4f}, {features[:,1].max():.4f}, {features[:,1].mean():.4f}")
    logging.info(f"lat_jerk: {features[:,2].min():.4f}, {features[:,2].max():.4f}, {features[:,2].mean():.4f}")
    logging.info(f"lon_jerk: {features[:,3].min():.4f}, {features[:,3].max():.4f}, {features[:,3].mean():.4f}")
    # logging.info(f"f_k2: {features[:,4].min():.4f}, {features[:,4].max():.4f}, {features[:,4].mean():.4f}")
    logging.info(f"speed_efficiency: {features[:,4].min():.4f}, {features[:,4].max():.4f}, {features[:,4].mean():.4f}")
    logging.info(f"lane_mean_error: {features[:,5].min():.4f}, {features[:,5].max():.4f}, {features[:,5].mean():.4f}")
    logging.info(f"risky_cost: {features[:,6].min():.4f}, {features[:,6].max():.4f}, {features[:,6].mean():.4f}")

# StyleN_Each_behavior = 3
def cal_comfort_cost(traj):
    traj = np.array(traj)
    v_x, v_y, theta = np.diff(traj[:, :, 0]) / 0.1, np.diff(traj[:, :, 1]) / 0.1, traj[:, 1:, 2]
    lon_speed = v_x * np.cos(theta) + v_y * np.sin(theta)
    lat_speed = v_y * np.cos(theta) - v_x * np.sin(theta)
    acc = np.diff(lon_speed) / 0.1
    jerk = np.diff(lon_speed, n=2) / 0.01
    lat_acc = np.diff(lat_speed) / 0.1
    
    # return np.mean(np.abs(acc),axis=1), np.mean(np.abs(jerk),axis=1), np.mean(np.abs(lat_acc),axis=1)
    acc = np.clip(np.abs(acc),0.,10.)
    jerk = np.clip(np.abs(jerk),0.,10.)
    lat_acc = np.clip(np.abs(lat_acc),0.,10.)

    Acc,Jerk,Lat_Acc = np.mean(lat_acc,axis=1), np.mean(jerk,axis=1), np.mean(lat_acc, axis=1)
    comfort_cost = Acc + Jerk + Lat_Acc
    return comfort_cost

def cal_efficiency_cost(traj, ref_line):
    speed = traj[:, :, -1]
    speed = torch.clamp(speed, min=0)
    speed_limit = torch.max(ref_line[:, :, -1], dim=-1, keepdim=True)[0]
    speed_mean_error = torch.mean(torch.abs(speed - speed_limit), dim=1)
    return speed_mean_error

def cal_risky_cost(plan, prediction, current_state, ref_line):
    prediction = prediction.permute(0, 2, 1, 3)

    actor_mask = torch.ne(current_state, 0)[:, 1:, -1]
    ego_current_state = current_state[:, 0]
    
    ego_len, ego_width = ego_current_state[:, -3], ego_current_state[:, -2]
    neighbors_current_state = current_state[:, 1:]
    neighbors_len, neighbors_width = neighbors_current_state[..., -3], neighbors_current_state[..., -2]

    l_eps = (ego_width.unsqueeze(1) + neighbors_width)/2 + 0.5
    frenet_neighbors = torch.stack([project_to_frenet_frame(prediction[:, :, i].detach(), ref_line) for i in range(prediction.shape[2])], dim=2)
    frenet_ego = project_to_frenet_frame(plan.detach(), ref_line)
    
    safe_error = []
    for t in [0, 2, 5, 9, 14, 19, 24, 29, 39, 49]: # key frames
        # find objects of interest
        l_distance = torch.abs(frenet_ego[:, t, 1].unsqueeze(1) - frenet_neighbors[:, t, :, 1])
        s_distance = frenet_neighbors[:, t, :, 0] - frenet_ego[:, t, 0].unsqueeze(-1)
        interactive = torch.logical_and(s_distance > 0, l_distance < l_eps) * actor_mask

        # find closest object
        distances = torch.norm(plan[:, t, :2].unsqueeze(1) - prediction[:, t, :, :2], dim=-1).squeeze(1)
        distances = torch.masked_fill(distances, torch.logical_not(interactive), 100)
        distance, index = torch.min(distances, dim=1)
        s_eps = (ego_len + torch.index_select(neighbors_len, 1, index)[:, 0])/2 + 5 # 安全距离 为5 单位是？

        # calculate cost
        error = (s_eps - distance) * (distance < s_eps)
        safe_error.append(error)

    # print(torch.stack(safe_error, dim=1).shape) # (7000, 10)
    safe_error_mean = torch.mean(torch.stack(safe_error, dim=1), dim=1)
    # print(safe_error_mean.shape) # (7000, 1)
    return safe_error_mean

def cal_lane_error_cost(plan_traj, ref_line, current_state):
    current_state = current_state[:, 0]
    
    distance_to_ref = torch.cdist(plan_traj[:, :, :2], ref_line[:, :, :2])  # L2范数
    k = torch.argmin(distance_to_ref, dim=-1).view(-1, plan_traj.shape[1], 1).expand(-1, -1, 3)
    ref_points = torch.gather(ref_line, 1, k)
    lane_error = torch.cat([plan_traj[:, 1::2, 0]-ref_points[:, 1::2, 0], plan_traj[:, 1::2, 1]-ref_points[:, 1::2, 1]], dim=1)
    
    return  torch.mean(lane_error, dim=1)

def convert2np(*data:torch.Tensor):
    res = []
    for elem in data:
        res.append(elem.cpu().numpy())
    return tuple(res)

# 行为类型划分有问题：要改。 10类(start/stop?)
action_type_dict =  {'stationary':0, 'straight':1, 'straight_right':1, 
                     'straight_left':1, 'slight_right_turn':2, 'sharp_right_turn':2, 
                     'right_u_turn':2, 'slight_left_turn':2, 'sharp_left_turn':2, 
                     'left_u_turn':2, 'None':0} # left big turn
behavior_list = [1,2] # 有效的多风格行为
# 不同行为聚类个数不同
cluster_N = [0,3,3]
def judge_action_type(ego, ground_truth):
    # traj = np.concatenate([ego[:, :5], ground_truth[0, :, :5]], axis=0)
    traj = ground_truth[0, :, :5]
    valid = np.where(traj[:, -1] == 0, False, True)
    future_xy = traj[:, :2]
    future_yaw = traj[:, 2]
    future_speed = traj[:, 3:]
    future_speed = np.linalg.norm(future_speed, axis=-1)

    kMaxSpeedForStationary = 2.0                 # (m/s) 2.0
    kMaxDisplacementForStationary = 2.0          # (m) 5.0
    kMaxLateralDisplacementForStraight = 2.0     # (m) 5.0
    kMinLongitudinalDisplacementForUTurn = -5.0  # (m) -5.0
    kMaxAbsHeadingDiffForStraight = np.pi / 6.0   # (rad) np.pi / 6.0
    kMaxAbsHeadingDiffForSharpTurn = np.pi * 1.0 / 3.0  # np.pi * 4.0 / 9.0 80°  # (rad) np.pi / 3.0
    first_valid_index, last_valid_index = 0, None
    for i in range(1, len(valid)):
        if valid[i] == 1:
            last_valid_index = i
    if valid[first_valid_index] == 0 or last_valid_index is None:
        return None

    xy_delta = future_xy[last_valid_index] - future_xy[first_valid_index] # xy_delta 分别为[Longitudinal, Lateral][纵向，横向]
    final_displacement = np.linalg.norm(xy_delta)
    heading_delta = future_yaw[last_valid_index] - future_yaw[first_valid_index]
    # max_speed = max(future_speed[last_valid_index], future_speed[first_valid_index])
    max_speed = max(future_speed[first_valid_index:last_valid_index])

    if max_speed < kMaxSpeedForStationary and final_displacement < kMaxDisplacementForStationary: # 静止
        return action_type_dict["stationary"]
    if np.abs(heading_delta) < kMaxAbsHeadingDiffForStraight: # 直行
        if np.abs(xy_delta[1]) < kMaxLateralDisplacementForStraight:
            return action_type_dict["straight"]
        elif xy_delta[1] < 0:
            return action_type_dict["straight_right"] 
        else:
            return action_type_dict["straight_left"]
    if heading_delta < -kMaxAbsHeadingDiffForStraight and xy_delta[1]: # 右转
        if xy_delta[0] < kMinLongitudinalDisplacementForUTurn:
            return action_type_dict["right_u_turn"]
        elif heading_delta < -kMaxAbsHeadingDiffForStraight and heading_delta > -kMaxAbsHeadingDiffForSharpTurn:
            return action_type_dict["slight_right_turn"]
        elif heading_delta < -kMaxAbsHeadingDiffForSharpTurn:
            return action_type_dict["sharp_right_turn"]
        # else: 不存在这种情况，前面情况区间全覆盖了
        #     return action_type_dict["None"]
    if heading_delta > kMaxAbsHeadingDiffForStraight and (-xy_delta[1]): # 左转
        if xy_delta[0] < kMinLongitudinalDisplacementForUTurn:
            return action_type_dict["left_u_turn"]
        elif heading_delta > kMaxAbsHeadingDiffForStraight and heading_delta < kMaxAbsHeadingDiffForSharpTurn:
            return action_type_dict["slight_left_turn"]
        elif heading_delta > kMaxAbsHeadingDiffForSharpTurn:
            return action_type_dict["sharp_left_turn"]
        # else:
        #     return action_type_dict["None"]
    # 不存在None这种情况，前面情况区间全覆盖了
    return action_type_dict["None"]

def get_action_np(ego_batch, ground_truth_batch):
    action_type = []
    for ego,ground_truth in zip(ego_batch, ground_truth_batch):
        # action_type.append(torch.nn.functional.one_hot(judge_action_type(ego,ground_truth), num_classes=11))
        action_type.append(judge_action_type(ego,ground_truth))
    return np.hstack(action_type)

def data_generation():
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() and args.gpu_id != "cpu" else "cpu")
    # logging
    log_path = f"./task_log/{args.name}/"
    os.makedirs(log_path, exist_ok=True)
    initLogging(log_file=log_path+f'task_processing_{args.seed}.log')

    logging.info("------------- {} -------------".format(args.name))
    logging.info("Use seed: {}".format(args.seed))
    logging.info("Use gpu_id: {}".format(args.gpu_id))
    logging.info("dataset size: {}".format(args.dataset_size))

    # process file
    task_manager = DataManager_Style(args.process_path + "raw_" +args.name, args.save_path + "style_" +args.name)
    data_loader = DataLoader(task_manager, batch_size=args.dataset_size, shuffle=False, num_workers=args.num_workers, drop_last=True)
    logging.info("Dataset Prepared: {} train data\n".format(len(task_manager)))

    # set seed
    fixed_seed(args.seed)
    check_find = True
    dataset_length = 0
    for i,batch in enumerate(data_loader):
        ego = batch[0].to(device)
        neighbors = batch[1].to(device)
        lanes = batch[2].to(device)
        crosswalks = batch[3].to(device)
        ref_line = batch[4].to(device)
        ground_truth = batch[5].to(device)
        
        scene_id = list(batch[6])
        time_step = batch[7].to(device)
        current_state = torch.cat([ego.unsqueeze(1), neighbors[..., :-1]], dim=1)[:, :, -1]

        # print("prediction",prediction.shape)
        # print("plan",plan.shape)
        # print("ref_line",ref_line.shape)
        # print("current_state",current_state.shape)
        # print("ground_truth",ground_truth.shape)
        # print("ego",ego.shape)
        """
        prediction torch.Size([8500, 10, 50, 3])
        plan torch.Size([8500, 50, 2])
        ref_line torch.Size([8500, 1200, 5])
        current_state torch.Size([8500, 11, 8])
        ground_truth torch.Size([8500, 11, 50, 5])
        ego torch.Size([8500, 20, 8])
        """

        """compute metrics"""
        features = cal_traj_features(ground_truth[:, 0], ref_line, ground_truth[:, 1:], current_state)
        logging.info(f"All data check: ")
        check_feature_min_max_info(features)
        features_np = features.cpu().numpy()
        # data save
        ego, neighbors, lanes, crosswalks = convert2np(ego, neighbors, lanes, crosswalks)
        ref_line, current_state, ground_truth = convert2np(ref_line, current_state, ground_truth)

        action_type = get_action_np(ego, ground_truth)
        
        # 取task计算k-mean
        """以下全是np计算"""
        behavior_N = 2
        for behavior_i in range(1,behavior_N+1):
            behavior_idx = action_type==behavior_i

            if sum(behavior_idx) <= 5:
                logging.info(f"error!!! --> behavior_{behavior_i} length: {np.sum(behavior_idx)}  percent: {np.sum(behavior_idx)*100/args.dataset_size:.1f}%")
                continue
                # raise Exception("TODO: solve the case")
            # traj data
            ego_behavior = ego[behavior_idx]
            neighbors_behavior = neighbors[behavior_idx]
            lanes_behavior = lanes[behavior_idx]
            crosswalks_behavior = crosswalks[behavior_idx]
            ref_line_behavior = ref_line[behavior_idx]
            ground_truth_behavior = ground_truth[behavior_idx]
            current_state_behavior = current_state[behavior_idx]

            # cost feature
            features_behavior = features_np[behavior_idx]
            logging.info(f"behavior_{behavior_i} length: {np.sum(behavior_idx)}  percent: {np.sum(behavior_idx)*100/args.dataset_size:.1f}%")
        
            # k_means
            k_means = K_Means(k=cluster_N[behavior_i], tolerance=0.0001, max_iter=300)
            k_means.fit(features_behavior)
            kmean_find_Max_N = 64
            select_task_label, all_task_label, center_value_idx = k_means.classify_dataset_find_N_Max(features_behavior,find_Max_N=kmean_find_Max_N) # 3600 为1h

            for style_i in center_value_idx:
                # find N
                style_idx = select_task_label[style_i]
                # style data
                ego_style = ego_behavior[style_idx]
                neighbors_style = neighbors_behavior[style_idx]
                lanes_style = lanes_behavior[style_idx]
                crosswalks_style = crosswalks_behavior[style_idx]
                ref_line_style = ref_line_behavior[style_idx]
                ground_truth_style = ground_truth_behavior[style_idx]
                current_state_style = current_state_behavior[style_idx]
                style_length = ref_line_style.shape[0]

                features_style = features_behavior[style_idx]
                logging.info(f"behavior-{behavior_i}th style-{style_i}th check: ")   
                check_feature_min_max_info(features_style)

                # find N/All
                all_style_idx = all_task_label==style_i
                logging.info(f"--> style_{style_i}  idx: {np.sum(all_style_idx)}  length: {style_length}  percent: {np.sum(all_style_idx)*100/np.sum(behavior_idx):.1f}%")
                
                dataset_length += style_length
                if style_length<kmean_find_Max_N:
                    check_find = False
                if behavior_i in behavior_list:
                    task_manager.save_data(behavior_i-1, style_i, ego_style, neighbors_style, lanes_style, crosswalks_style, ref_line_style, current_state_style, ground_truth_style, features_style)

        # print()
        # print("prediction: ",prediction.shape)
        # print("plan: ",plan.shape)
        # print("ref_line: ",ref_line.shape)
        # print("current_state: ",current_state.shape)
        # print("ground_truth: ",ground_truth.shape)
        # print("time_step: ",time_step.shape)
        # print("comfort_cost: ",comfort_cost.shape)
        # print("efficiency_cost: ",efficiency_cost.shape)
        # print("action_type: ",action_type.shape)
        """
        prediction:  (128, 10, 50, 3)
        plan:  (128, 50, 2)
        ref_line:  (128, 1200, 5)
        current_state:  (128, 11, 8)
        ground_truth:  (128, 11, 50, 5)
        time_step:  (128,)
        comfort_cost:  (128,)
        efficiency_cost:  (128,)
        action_type:  (128,)
        """
        break  # 退出循环，保证只取一个batch
    logging.info(f"check_find: {check_find}")
    logging.info(f"dataset_length: {dataset_length}")
    logging.info("task datasets generate: finished !!!")

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Data Generation')
    parser.add_argument('--name', type=str, help='log name (options: ["train","test"])', default="test")
    parser.add_argument('--process_path', type=str, help='path to processing datasets', default='/data/lin_funster/waymo_dataset/')
    parser.add_argument('--seed', type=int, help='fix random seed', default=25)
    parser.add_argument("--num_workers", type=int, help="number of workers used for dataloader", default=0)
    parser.add_argument('--save_path', type=str, help='path to mid result', default='/data/lin_funster/waymo_dataset/')
    parser.add_argument('--dataset_size', type=int, help='dataset number (default: 1000)', default=10000)
    parser.add_argument('--gpu_id', type=str, help='run on which gpu (default: cpu)', default='0')
    args = parser.parse_args()

    # Run
    data_generation()
