import torch
import sys
import csv
import time
import argparse
import logging
import os
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
# from utils.train_utils import *
from model.Pipeline import Predictor
from torch.utils.data import DataLoader
from model.LfD_IRL import *
from utils.test_utils import *
from utils.train_utils import fixed_seed, DrivingData
from dataset_style import convert2np,get_action_np
from model.DataManager import DataManager_Train

def list2tensor(*data:list):
    res = []
    for elem in data:
        res.append(torch.Tensor(elem).mean())
    return tuple(res)

def test_metrics(plan_trajectory, prediction_trajectories, ground_truth_trajectories, weights):
    prediction_trajectories = prediction_trajectories * weights
    plan_distance = torch.norm(plan_trajectory[:, :, :2] - ground_truth_trajectories[:, 0, :, :2], dim=-1)
    prediction_distance = torch.norm(prediction_trajectories[:, :, :, :2] - ground_truth_trajectories[:, 1:, :, :2], dim=-1)
    # planning
    plannerADE1 = torch.mean(plan_distance[:, :10])
    plannerADE2 = torch.mean(plan_distance[:,:30])
    plannerADE3 = torch.mean(plan_distance[:,:50])
    plannerFDE1 = torch.mean(plan_distance[:, 9])
    plannerFDE2 = torch.mean(plan_distance[:, 29])
    plannerFDE3 = torch.mean(plan_distance[:, 49])    
    # prediction
    predictorADE = torch.mean(prediction_distance, dim=-1)
    predictorADE = torch.masked_select(predictorADE, weights[:, :, 0, 0])
    predictorADE = torch.mean(predictorADE)
    predictorFDE = prediction_distance[:, :, -1]
    predictorFDE = torch.masked_select(predictorFDE, weights[:, :, 0, 0])
    predictorFDE = torch.mean(predictorFDE)
    
    return [plannerADE1.item(),plannerADE2.item(),plannerADE3.item(),plannerFDE1.item(),plannerFDE2.item(),plannerFDE3.item()],\
            [predictorADE.item(), predictorFDE.item()]

def batch_check_plan_result(traj, ref_line, grouth_truth, current_state): # 先横再纵：十字顺序
    """base element"""
    # lat_speed, lon_speed = torch.diff(traj[:, :, 0]) / 0.1, torch.diff(traj[:, :, 1]) / 0.1 # dim 49
    # lat_speed, lon_speed = torch.clamp(lat_speed, min=-10., max=50.), torch.clamp(lon_speed, min=-20., max=40.)
    speed = traj[:, :, 3] / torch.cos(traj[:, :, 2])
    speed = torch.clamp(speed, min=0., max=None)
    speed_mean = torch.mean(speed)
    """acc""" 
    acc = torch.diff(speed) / 0.1 # dim 49
    acc_mean = torch.mean(acc)
    """jerk"""
    jerk = torch.diff(speed, n=2) / 0.01 # dim 48
    jerk_mean = torch.mean(jerk)
    """speed efficiency"""
    # 超速
    speed_limit = torch.max(ref_line[:, :, -1], dim=-1, keepdim=True)[0]
    speed_limit_error = speed - speed_limit
    out_of_speed_limit = torch.sum(speed_limit_error>0, dim=1) # param 1
    out_of_speed_limit_rate = torch.sum(out_of_speed_limit>0)/speed.shape[0]
    # 效率
    f_efficiency = torch.mean(speed_limit_error, dim=1) # dim 1
    efficiency_mean = torch.mean(f_efficiency)
    # f_efficiency2 = speed[:,:47] - speed_limit
    """red light"""
    dt = 0.1
    red_light = ref_line[..., -1]
    s = torch.cumsum(speed * dt, dim=-1)
    stop_point = torch.max(red_light[:, 200:]==0, dim=-1)[1] * 0.1
    stop_distance = stop_point.view(-1, 1)  # param 2
    # print(((s > stop_distance) * (stop_point.unsqueeze(-1) != 0)))
    # print(torch.sum((s > stop_distance) * (stop_point.unsqueeze(-1) != 0),dim=1)!=0)
    red_light_rate = torch.sum(torch.sum((s > stop_distance) * (stop_point.unsqueeze(-1) != 0),dim=1)!=0)/speed.shape[0]
    """lane: off route"""
    distance_to_ref = torch.cdist(traj[:, :, :2], ref_line[:, :, :2])  # L2范数
    k = torch.argmin(distance_to_ref, dim=-1).view(-1, traj.shape[1], 1).expand(-1, -1, 3)
    ref_points = torch.gather(ref_line, 1, k)
    # f_lane_error = torch.cat([traj[:, 1::2, 0]-ref_points[:, 1::2, 0], traj[:, 1::2, 1]-ref_points[:, 1::2, 1]], dim=1)
    f_lane_error = torch.hypot(traj[:, :, 0]-ref_points[:, :, 0], traj[:, :, 1]-ref_points[:, :, 1]) # dim47
    min_lane_error = torch.min(f_lane_error, dim=1)[0]
    off_route_rate = torch.sum(min_lane_error>5)/speed.shape[0] # 单位：m
    """collision""" 
    collisions = []
    T = 50
    traj_batch,grouth_truth_batch,current_state_batch = traj.cpu().numpy(), grouth_truth.cpu().numpy(), current_state.cpu().numpy()
    for traj_np,grouth_truth_np,current_state_np in zip(traj_batch,grouth_truth_batch,current_state_batch):
        # print(traj_np.shape) (50, 4)
        # print(grouth_truth_np.shape) (11, 50, 5)
        # print(current_state_np.shape) (11, 8)
        collision = check_collision(traj_np[:T],grouth_truth_np[1:,:T],current_state_np[:, 5:])
        collisions.append(collision)
    collision_rate = torch.tensor(np.stack(collisions).sum()).to(speed.device)/speed.shape[0]

    # features
    dynamic_meatures = [speed_mean,acc_mean,jerk_mean,efficiency_mean]
    behavior_meatures = [out_of_speed_limit_rate, red_light_rate, off_route_rate, collision_rate]
    
    return dynamic_meatures, behavior_meatures

def plan_optimizer(planner, cost_function_weights, plan, prediction, ref_line_info, ground_truth, current_state):
    # plan 
    planner_inputs = {
        "control_variables": plan.view(-1, 100), # generate initial control sequence
        "predictions": prediction, # generate predictions for surrounding vehicles 
        "ref_line_info": ref_line_info,
        "current_state": current_state
    }
    
    for i in range(cost_function_weights.shape[1]):
        planner_inputs[f'cost_function_weight_{i+1}'] = cost_function_weights[:, i].unsqueeze(1)

    # best solution mode
    with torch.no_grad():
        final_values, info = planner.layer.forward(planner_inputs, optimizer_kwargs={'track_best_solution': True})
        plan = info.best_solution['control_variables'].view(-1, 50, 2).to(prediction.device)
    plan = bicycle_model_test(plan, current_state[:, 0])

    plan_cost = planner.objective.error_squared_norm().mean() / planner.objective.dim()
    plan_loss = F.smooth_l1_loss(plan[:, :, :3], ground_truth[:, 0, :, :3])
    plan_loss += F.smooth_l1_loss(plan[:, -1, :3], ground_truth[:, 0, -1, :3])
    loss = plan_loss + 1e-3 * plan_cost # planning loss
    return loss.detach().item(), plan

def model_training():
    #device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() and args.gpu_id is not None else "cpu")
    # Logging
    log_name = __file__.split('/')[-1].split('.')[0] + f"_{args.name}" + f"_{args.load_epoch}"
    log_path = f"./testing_log/{log_name}/"
    os.makedirs(log_path, exist_ok=True)
    initLogging(log_file=log_path+log_name+'_train.log')

    logging.info("------------- {} -------------".format(log_name))
    logging.info("test model name: {}".format(args.name))
    logging.info("load_model_path: {}".format(args.load_model_path))
    logging.info("load_epoch: {}".format(args.load_epoch))
    logging.info("Batch size: {}".format(args.batch_size))
    logging.info("Use integrated planning module: {}".format(args.use_planning))
    logging.info("Use device: {}".format(device))

    # set seed
    fixed_seed(args.seed)

    predictors = [None]*3
    cf_params = [None]*3
    for i in range(3): # style
        predictors[i] = Predictor(50).to(device)
        predictors[i].load_state_dict(torch.load(args.load_model_path+f'_{args.name}/save_model/'+f"Predictor_style_{i}_{args.load_epoch}.pth", map_location=device))
        predictors[i].eval()
        if args.use_planning:
            cost_function_weights = np.load(args.load_model_path+f'_{args.name}/save_model/'+f'CostFunction_style_{i}_{args.load_epoch}.npy')
            cf_params[i] = torch.Tensor(cost_function_weights).to(device)
    # set up planner
    if args.use_planning:
        trajectory_len, feature_len = 50, 9
        planner = MotionPlanner(trajectory_len, feature_len, device, test=True)
    else:
        planner = None
    # set up data loaders
    test_set = DrivingData(args.test_set+'/*')
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,drop_last=True)
    logging.info("Dataset Prepared: {} test data\n".format(len(test_set)))

    train_manager = DataManager_Train(args.expert_path)
    train_loader = DataLoader(train_manager, batch_size=1, shuffle=False, num_workers=args.num_workers)


    logging.info("In Domain Test: ")
    expert_features = {0:{},1:{},2:{}}
    for x_spt in train_loader:
        ego = x_spt[0][0].to(device)
        neighbors = x_spt[1][0].to(device)
        lanes = x_spt[2][0].to(device)
        crosswalks = x_spt[3][0].to(device)
        ref_line = x_spt[4][0].to(device)
        current_state = x_spt[5][0].to(device)
        ground_truth = x_spt[6][0].to(device)
        behavior_i = x_spt[7][0].item()
        style_i = x_spt[8][0].item()
        weights = torch.ne(ground_truth[:, 1:, :, :3], 0)
        E_expert_features = cal_traj_features_train(ground_truth[:, 0],ref_line,ground_truth[:, 1:],current_state).mean(dim=0)
        expert_features[style_i][behavior_i] = E_expert_features
        with torch.no_grad(): 
            plans, predictions, scores, raw_cost_function_weights = predictors[style_i](ego, neighbors, lanes, crosswalks) 
            plan, prediction = select_future_dg(plans, predictions, scores) 
        cost_function_weights = cf_params[i]
        ###################################################################################################
        """learn process""" 
        if args.use_planning:
            loss_item, plan = plan_optimizer(planner, cf_params[style_i][behavior_i][None,:],plan,prediction,ref_line,ground_truth,current_state)
        else:
            plan = bicycle_model_test(plan, current_state[:, 0])
            loss_item = torch.tensor(0.)
        logging.info(f"====================================style-{style_i} results====================================")
        # check prediction error
        plan_metrics, _ = test_metrics(plan, prediction, ground_truth, weights)
        logging.info(f'S{style_i}-plannerADE1: {plan_metrics[0]:.4f}, '  + f'S{style_i}-plannerFDE1: {plan_metrics[3]:.4f}')
        logging.info(f'S{style_i}-plannerADE2: {plan_metrics[1]:.4f}, '  + f'S{style_i}-plannerFDE2: {plan_metrics[4]:.4f}')
        logging.info(f'S{style_i}-plannerADE3: {plan_metrics[2]:.4f}, '  + f'S{style_i}-plannerFDE3: {plan_metrics[5]:.4f}')

        # check irl loss
        test_feature = cal_traj_features_train(plan,ref_line,ground_truth[:,1:],current_state).mean(dim=0)
        IRL_loss = F.l1_loss(test_feature, expert_features[style_i][behavior_i])
        logging.info(f"IRL Loss: {IRL_loss.item():.5f}")
        
    # print(expert_features)
    logging.info("Out of Domain Test: ")
    style_i = 0
    batch_count = 0
    running_time_list = []
    out_of_speed_limit_list,red_light_list, off_route_list,collisions_list = [], [], [], []
    Speeds_list, Accs_list, Jerks_list,efficiency_list = [], [], [], []
    prediction_ADE_list, prediction_FDE_list = [], []
    IRL_metrics_list = []
    epoch_loss_list = []
    
    # begin training
    current = 0
    size = len(test_loader.dataset)
    start_time = time.time()
    batch_count = 0
    while style_i<=2:
        for batch in test_loader:
            if batch_count>=1000: #  # TODO
                # show result
                running_time,out_of_speed_limit, red_light, off_route, collisions,\
                    Speeds, Accs, Jerks, efficiency_means,prediction_ADE, prediction_FDE,\
                        IRL_metrics, epoch_loss = list2tensor(running_time_list,out_of_speed_limit_list,red_light_list, off_route_list,collisions_list,Speeds_list, Accs_list, Jerks_list,efficiency_list,prediction_ADE_list, prediction_FDE_list,IRL_metrics_list,epoch_loss_list)
                running_time,out_of_speed_limit, red_light, off_route, collisions,\
                    Speeds, Accs, Jerks, efficiency_means,prediction_ADE, prediction_FDE,\
                        IRL_metrics, epoch_loss = convert2np(running_time,out_of_speed_limit,red_light, off_route,collisions,Speeds, Accs, Jerks,efficiency_means,prediction_ADE, prediction_FDE,IRL_metrics,epoch_loss)
                logging.info(f"\n================================style: {style_i}================================")
                logging.info(f"Style-{style_i}th Progress: [{current:>6d}/{size:>6d}]  Loss: {np.mean(epoch_loss_list):>.4f}  {(time.time()-start_time)/current:>.4f}s/sample")
                logging.info(f"running_time: {running_time} ")
                logging.info(f"out_of_speed_limit: {out_of_speed_limit*100:.2f}% red_light: {red_light*100:.2f}% off_route: {off_route*100:.2f}% collisions: {collisions*100:.2f}%")
                logging.info(f"efficiency: {efficiency_means:.4f} Speeds: {Speeds:.4f} Accs: {Accs:.4f} Jerks: {Jerks:.4f}")
                logging.info(f"prediction_ADE: {prediction_ADE:.4f} prediction_FDE: {prediction_FDE:.4f} IRL_metrics: {IRL_metrics:.5f} epoch_loss: {epoch_loss:.4f}")
                style_i+=1
                if style_i>2:
                    time.sleep(5)
                    break
                
                # init
                running_time_list = []
                out_of_speed_limit_list,red_light_list, off_route_list,collisions_list = [], [], [], []
                Speeds_list, Accs_list, Jerks_list,efficiency_list = [], [], [], []
                prediction_ADE_list, prediction_FDE_list = [], []
                IRL_metrics_list = []
                epoch_loss_list = []
                # begin training
                current = 0
                size = len(test_loader.dataset)
                start_time = time.time()
                batch_count = 0
            # prepare data
            ego = batch[0].to(device)
            neighbors = batch[1].to(device)
            map_lanes = batch[2].to(device)
            map_crosswalks = batch[3].to(device)
            ref_line_info = batch[4].to(device)
            ground_truth = batch[5].to(device)
            current_state = torch.cat([ego.unsqueeze(1), neighbors[..., :-1]], dim=1)[:, :, -1]
            weights = torch.ne(ground_truth[:, 1:, :, :3], 0)
            
            # predict
            with torch.no_grad():
                plans, predictions, scores, _ = predictors[style_i](ego, neighbors, map_lanes, map_crosswalks)
            plan, prediction = select_future_dg(plans, predictions, scores)
            
            ego_np, ground_truth_np = convert2np(ego, ground_truth)
            action_type = torch.tensor(get_action_np(ego_np,ground_truth_np))
            for behavior_i in range(2):
                behavior_idx = action_type==behavior_i
                t0 = time.time()
                plan_behavior = plan[behavior_idx]
                prediction_behavior = prediction[behavior_idx]
                ref_line_info_behavior = ref_line_info[behavior_idx]
                ground_truth_behavior = ground_truth[behavior_idx]
                current_state_behavior = current_state[behavior_idx]
                weights_behavior = weights[behavior_idx]
                if args.use_planning:
                    loss_item, plan_behavior = plan_optimizer(planner, cost_function_weights[behavior_i][None,:], plan_behavior, prediction_behavior, ref_line_info_behavior, ground_truth_behavior, current_state_behavior)
                else:
                    plan_behavior = bicycle_model_test(plan_behavior, current_state_behavior[:, 0])
                    loss_item = torch.tensor(0.)
                # compute metrics
                epoch_loss_list.append(loss_item)

                # cal time
                run_t = (time.time() - t0)
                running_time_list.append(int(run_t*1000))
                
                # check open loop metrics
                dynamic_meatures, behavior_meatures = batch_check_plan_result(plan_behavior,ref_line_info_behavior,ground_truth_behavior,current_state_behavior)

                Speeds_list.append(dynamic_meatures[0])
                Accs_list.append(dynamic_meatures[1])
                Jerks_list.append(dynamic_meatures[2])
                efficiency_list.append(dynamic_meatures[3])

                out_of_speed_limit_list.append(behavior_meatures[0])
                red_light_list.append(behavior_meatures[1])
                off_route_list.append(behavior_meatures[2])
                collisions_list.append(behavior_meatures[3])
                
                # check prediction error
                _, pred_metrics = test_metrics(plan_behavior, prediction_behavior, ground_truth_behavior, weights_behavior)
                prediction_ADE_list.append(pred_metrics[0])
                prediction_FDE_list.append(pred_metrics[1])

                # check irl loss
                test_feature = cal_traj_features_train(plan_behavior,ref_line_info_behavior,ground_truth_behavior[:,1:],current_state_behavior).mean(dim=0)
                IRL_loss = F.l1_loss(test_feature, expert_features[style_i][behavior_i])
                IRL_metrics_list.append(IRL_loss.item())
            batch_count+=1

            # show progress
            current += batch[0].shape[0]
            sys.stdout.write(f"\rStyle-{style_i}th Progress: [{current:>6d}/{size:>6d}]  Loss: {np.mean(epoch_loss_list):>.4f}  {(time.time()-start_time)/current:>.4f}s/sample")
            sys.stdout.flush()
    
    
if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Pipeline_Framework_Training')
    parser.add_argument('--name', type=str, help='name to test model', default='None') # 
    parser.add_argument('--load_model_path', type=str, help='path to saved model', 
                        default='/data/lin_funster/proj_paper1/Final_training_log/2_Style_FineTune_Run') # model_5_0.7052.pth/ model_20_0.6989.pth
    parser.add_argument('--load_epoch', type=int, help='batch size (default: 100)', default=100)
    parser.add_argument('--expert_path', type=str, help='path to train datasets', default='/data/lin_funster/waymo_dataset/style_test')
    parser.add_argument('--test_set', type=str, help='path to validation datasets', default='/data/lin_funster/waymo_dataset/raw_test')
    parser.add_argument('--seed', type=int, help='fix random seed', default=25)
    parser.add_argument("--num_workers", type=int, help="number of workers used for dataloader", default=8)
    parser.add_argument('--batch_size', type=int, help='batch size (default: 32)', default=64)
    parser.add_argument('--use_planning', action="store_true", help='if use integrated planning module (default: True)', default=False)
    parser.add_argument('--gpu_id', type=int, help='run on which device (default: cuda)', default=0)
    args = parser.parse_args()

    # Run
    model_training()
