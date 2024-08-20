import torch
import sys
import csv
import time
import argparse
import logging
import os
import numpy as np
from torch import nn, optim
from utils.train_utils import *
from torch.utils.data import DataLoader
from model.DataManager import *
import copy
from model.L_IRL import *
from model.Pipeline import Predictor,Pipeline # , bicycle_model_train
from dataset_style import convert2np,get_action_np

"""
nn+traj loss
"""
# 设置打印选项
np.set_printoptions(suppress=True, formatter={'float_kind': '{:.4f}'.format})


def LfD_Loss(E_expert_features, control_var, ground_truth,ref_line,current_state):
    traj = bicycle_model_train(control_var, current_state[:, 0])
    # IRL objective 
    E_learner_features = cal_traj_features_train(traj, ref_line, ground_truth[:,1:], current_state).mean(dim=0)
    # check 
    if torch.isnan(E_learner_features).any().item() or torch.isnan(E_expert_features).any().item():
        raise Exception("There are Nan in IRL_loss !!!")
    IRL_loss = F.smooth_l1_loss(E_learner_features, E_expert_features)
    traj_loss = F.smooth_l1_loss(traj[:, :, :3], ground_truth[:, 0, :, :3]) 
    end_loss = F.smooth_l1_loss(traj[:, -1, :3], ground_truth[:, 0, -1, :3])
    loss = 100*IRL_loss + traj_loss + end_loss # + 1e-3 * plan_cost# + traj_loss + end_loss # + 1e-3 * plan_cost
    # info_str = f"loss:{loss.item():.4f}  IRL-loss:{IRL_loss.item():.4f}  end-loss:{end_loss.item():.4f}  traj-loss:{traj_loss.item():.4f}  plan-cost:{plan_cost.item():.4f}"
    # print(info_str)
    loss_value_list = [loss.detach().item(),IRL_loss.detach().item(),traj_loss.detach().item()]
    return loss, traj, loss_value_list 

def train_task():
    # device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() and args.gpu_id is not None else "cpu")
    # Logging
    log_name = __file__.split('/')[-1].split('.')[0]
    log_path = f"./training_log/{log_name}/"
    os.makedirs(log_path, exist_ok=True)
    initLogging(log_file=log_path+log_name+'_train.log')

    # set seed
    fixed_seed(args.seed)
    logging.info("------------- {} -------------".format(log_name))
    logging.info("seed: {}".format(args.seed))
    logging.info("predictor_lr: {}".format(args.predictor_lr))
    logging.info("predictor_model_path: {}".format(args.predictor_model_path))
    logging.info("cost_function_lr: {}".format(args.cost_function_lr))
    logging.info("Use device: {}".format(device))
    logging.info("scheduler_step_size: {}".format(args.scheduler_step_size))

    train_set = DrivingData(args.train_path+'/*')
    train_loader = DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers)
    logging.info("Dataset Prepared: {} train data \n".format(len(train_set)))

    # set up data loaders
    expert_manager = DataManager_Train(args.expert_path)
    expert_loader = DataLoader(expert_manager, batch_size=1, shuffle=False, num_workers=args.num_workers)
    logging.info("calculate expert feature...")
    expert_features_set = {0:{},1:{},2:{}}
    expert_data_set = {0:{},1:{},2:{}}
    for x_spt in expert_loader:
        ref_line = x_spt[4][0].to(device)
        current_state = x_spt[5][0].to(device)
        ground_truth = x_spt[6][0].to(device)
        behavior_i = x_spt[7][0].item()
        style_i = x_spt[8][0].item()
        E_expert_features = cal_traj_features_train(ground_truth[:, 0],ref_line,ground_truth[:,1:],current_state).mean(dim=0)
        expert_features_set[style_i][behavior_i] = E_expert_features
        expert_data_set[style_i][behavior_i] = x_spt
    
    # train init
    optimizers = [None]*3
    predictors = [None]*3
    schedulers = [None]*3
    
    for i in range(3): # style
        predictors[i] = Predictor(50).to(device)
        predictors[i].load_state_dict(copy.deepcopy(torch.load(args.predictor_model_path, map_location=device)))
        predictors[i].train()
        optimizers[i] = optim.Adam([
                                    {"params":predictors[i].parameters(),'lr':args.predictor_lr}# ,
                                    # {"params":paramModels[i][0].parameters(),"lr":args.cost_function_lr},
                                    # {"params":paramModels[i][1].parameters(),"lr":args.cost_function_lr}
                                    ]) 
        schedulers[i] = optim.lr_scheduler.StepLR(optimizers[i], step_size = args.scheduler_step_size, gamma=0.5) 
    
    show_epoch_N = 10 
    start_time = time.time()
    for style_i in range(3):
        for epoch in range(1, args.train_epochs+1):
            epoch_loss_list = []
            epoch_irl_loss_list = []
            epoch_traj_loss_list = []
            epoch_metrics_list = []
            epoch_pred_loss_list = []
            """Training"""
            if epoch%show_epoch_N == 0:
                sys.stdout.write("\r                                                                                                                                              ")
                logging.info(f"\r=============================style_i-{style_i} epoch: {epoch}===============================") 
            batch = next(iter(train_loader))
            ego = batch[0].to(device)
            ground_truth = batch[5].to(device)
            try:
                ego_np, ground_truth_np = convert2np(ego, ground_truth)
                action_type = torch.tensor(get_action_np(ego_np, ground_truth_np))
            except:
                logging.info(f"\nepoch: {epoch} ==> running error and skip current epoch")
                continue
            for behavior_i in range(2):
                x_spt = expert_data_set[style_i][behavior_i]
                behavior_idx = action_type==behavior_i
                x_index = torch.randperm(64)
                x_behavior_idx = x_index[:args.batch_size-torch.sum(behavior_idx)]
                # prepare data
                ego = torch.concatenate((batch[0].to(device)[behavior_idx], x_spt[0][0].to(device)[x_behavior_idx]),dim=0)
                neighbors = torch.concatenate((batch[1].to(device)[behavior_idx], x_spt[1][0].to(device)[x_behavior_idx]),dim=0)
                lanes = torch.concatenate((batch[2].to(device)[behavior_idx], x_spt[2][0].to(device)[x_behavior_idx]),dim=0)
                crosswalks = torch.concatenate((batch[3].to(device)[behavior_idx], x_spt[3][0].to(device)[x_behavior_idx]),dim=0)
                ref_line = torch.concatenate((batch[4].to(device)[behavior_idx], x_spt[4][0].to(device)[x_behavior_idx]),dim=0)
                ground_truth = torch.concatenate((batch[5].to(device)[behavior_idx], x_spt[6][0].to(device)[x_behavior_idx]),dim=0)
                current_state = torch.cat([ego.unsqueeze(1), neighbors[..., :-1]], dim=1)[:, :, -1]
                weights = torch.ne(ground_truth[:, 1:, :, :3], 0)
                E_expert_features = expert_features_set[style_i][behavior_i]

                if ego.shape[0]!=args.batch_size:
                    logging.info(f"error!!! data handler wrong")
                # print("ego",ego.shape)
                # print("neighbors",neighbors.shape)
                # print("lanes",lanes.shape)
                # print("crosswalks",crosswalks.shape)
                # print("ref_line",ref_line.shape)
                # print("current_state",current_state.shape)
                # print("ground_truth",ground_truth.shape)
                """
                ego torch.Size([64, 20, 8])
                neighbors torch.Size([64, 10, 20, 9])
                lanes torch.Size([64, 11, 6, 100, 17])
                crosswalks torch.Size([64, 11, 4, 100, 3])
                ref_line torch.Size([64, 1200, 5])
                current_state torch.Size([64, 11, 8])
                ground_truth torch.Size([64, 11, 50, 5])
                """
                ################################################################################################### 
                """initial traj generation: 生成init轨迹""" 
                # with torch.no_grad(): 
                plans, predictions, scores, raw_cost_function_weights = predictors[style_i](ego, neighbors, lanes, crosswalks) 
                plan_trajs = torch.stack([bicycle_model(plans[:, i], ego[:, -1])[:, :, :3] for i in range(3)], dim=1)
                prediction_loss = Pipeline.MFMA_loss(plan_trajs, predictions, scores, ground_truth, weights) # multi-future multi-agent loss
                control_var, prediction = select_future_dg(plans, predictions, scores) 
                ###################################################################################################
                """learn process""" 
                learn_loss, learn_traj, loss_value_list = LfD_Loss(E_expert_features,control_var,ground_truth,ref_line,current_state)
                loss = prediction_loss + learn_loss
                # loss backward 
                optimizers[style_i].zero_grad() 
                loss.backward() # retain_graph=True 
                optimizers[style_i].step() 
                # save data 
                learn_metrics = msirl_metrics(learn_traj, ground_truth) 
                epoch_loss_list.append(loss.item()) 
                epoch_pred_loss_list.append(prediction_loss.item()) 
                epoch_irl_loss_list.append(loss_value_list[1]) 
                epoch_traj_loss_list.append(loss_value_list[2]) 
                epoch_metrics_list.append(learn_metrics) 
            sys.stdout.write(f"\rProgress:[{epoch:>4d}/{args.train_epochs:>4d}]  Loss:{np.mean(epoch_loss_list):>.5f}  Pred-Loss:{np.mean(epoch_pred_loss_list):>.5f}  IRL-Loss:{np.mean(epoch_irl_loss_list):>.5f}  Traj-Loss:{np.mean(epoch_traj_loss_list):>.5f}  {(time.time()-start_time)/epoch:>.5f}s/sample") 
            sys.stdout.flush()
            schedulers[style_i].step()
            """epoch""" 
            if epoch%show_epoch_N == 0: 
                # show train metrics 
                logging.info(f"\nstyle-{style_i} ==> train metrics")
                logging.info(f"Progress:[{epoch:>4d}/{args.train_epochs:>4d}]  Loss:{np.mean(epoch_loss_list):>.5f}  Pred-Loss:{np.mean(epoch_pred_loss_list):>.5f}  IRL-Loss:{np.mean(epoch_irl_loss_list):>.5f}  Traj-Loss:{np.mean(epoch_traj_loss_list):>.5f}  {(time.time()-start_time)/epoch:>.5f}s/sample")
                learn_epoch_metrics = show_metrics(epoch_metrics_list,str="learn") 

                # save to training log
                log = {
                        # time
                        'style':style_i, 'epoch': epoch, 
                        # loss
                        'epoch_loss_mean':np.mean(epoch_loss_list), 
                        'epoch_pred_loss_list':np.mean(epoch_pred_loss_list), 
                        'epoch_irl_loss_mean':np.mean(epoch_irl_loss_list), 
                        'epoch_traj_loss_mean':np.mean(epoch_traj_loss_list), 
                        # metrics
                        'learn-plannerADE1': learn_epoch_metrics[0], 'learn-plannerFDE1': learn_epoch_metrics[3], 
                        'learn-plannerADE2': learn_epoch_metrics[1], 'learn-plannerFDE2': learn_epoch_metrics[4], 
                        'learn-plannerADE3': learn_epoch_metrics[2], 'learn-plannerFDE3': learn_epoch_metrics[5], 
                        }

                if epoch == show_epoch_N and style_i==0: 
                    with open(f'./training_log/{log_name}/train_log.csv', 'w') as csv_file: 
                        writer = csv.writer(csv_file)
                        writer.writerow(log.keys())
                        writer.writerow(log.values())
                else:
                    with open(f'./training_log/{log_name}/train_log.csv', 'a') as csv_file: 
                        writer = csv.writer(csv_file)
                        writer.writerow(log.values())

            if epoch%100 == 0:
                """save learned model weights"""
                logging.info(f"\n==========================save learned model: epoch {epoch}th============================")
                save_path = log_path + f"save_model/"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                # save predictor
                torch.save(predictors[style_i].state_dict(), save_path+f"Predictor_style_{style_i}_{epoch}.pth")


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='LfD_Training')
    parser.add_argument('--train_path', type=str, help='path to train datasets', default='/data/lin_funster/waymo_dataset/raw_train')
    parser.add_argument('--expert_path', type=str, help='path to train datasets', default='/data/lin_funster/waymo_dataset/style_test')
    parser.add_argument('--predictor_model_path', type=str, help='path to saved model', default='/data/lin_funster/proj_paper1/model_param/model_5_0.7052.pth') # model_5_0.7052.pth/ model_20_0.6989.pth
    parser.add_argument('--seed', type=int, help='fix random seed', default=25)
    parser.add_argument('--num_workers', type=int, help="number of workers used for dataloader", default=8)
    parser.add_argument('--scheduler_step_size', type=int, help='epochs of learning rate ', default=100)
    parser.add_argument('--train_epochs', type=int, help='epochs of task training', default=100)
    parser.add_argument('--batch_size', type=int, help='batch size (default: 32)', default=64)
    parser.add_argument('--train_batch_size', type=int, help='batch size (default: 32)', default=64)
    parser.add_argument('--predictor_lr', type=float, help='learning rate (default: 2e-4)', default=2e-5)
    parser.add_argument('--cost_function_lr', type=float, help='learning rate (default: 1e-2)', default=1e-3)
    parser.add_argument('--gpu_id', type=int, help='run on which device (default: cuda)', default=0)
    args = parser.parse_args()

    # Run
    train_task()
