import torch
import argparse
import os
import logging
import pandas as pd
import tensorflow as tf
from utils.simulator import *
from utils.test_utils import *
from model.PPTIL import *
from model.Pipeline import Predictor
from waymo_open_dataset.protos import scenario_pb2
import time
from dataset_style import judge_action_type
from utils.train_utils import fixed_seed

def closed_loop_test(model_name):
    # test file
    files = glob.glob(args.test_set+'/*')

    # cache results
    running_time = []
    collisions, off_routes, progress = [], [], []
    Accs, Jerks, Lat_Accs = [], [], []
    Human_Accs, Human_Jerks, Human_Lat_Accs = [], [], []
    similarity_3s, similarity_5s, similarity_10s = [], [], []
    action_type_list = []
    style_error_list = []

    # set up simulator
    simulator = Simulator(150) # temporal horizon 15s
    scene_count = 0
    render_count = 0
    strat_idx = 0

    for file in files:
        scenarios = tf.data.TFRecordDataset(file)
        if scene_count > scene_N:
            break
        # iterate scenarios in the test file
        for scenario in scenarios:
            if strat_idx < args.idx:
                strat_idx += 1
                continue
            parsed_data = scenario_pb2.Scenario()
            parsed_data.ParseFromString(scenario.numpy())
            simulator.load_scenario(parsed_data)
            print(f" model_name: {model_name} ".center(30,"="))
            print(f'Scenario: {simulator.scenario_id}')
            
            obs = simulator.reset()
            done = False
            error = False

            scene_count += 1
            if scene_count > scene_N:
                break
            
            run_t_list = []
            while not done:
                print(f" model_name: {model_name} ".center(30,"="))
                print(f'Time: {simulator.timestep-19}')
                print(f'scene_count: {scene_count}')
                ego = torch.from_numpy(obs[0]).to(device)
                neighbors = torch.from_numpy(obs[1]).to(device)
                lanes = torch.from_numpy(obs[2]).to(device)
                crosswalks = torch.from_numpy(obs[3]).to(device)
                ref_line = torch.from_numpy(obs[4]).to(device)
                ground_truth = obs[5]
                current_state = torch.cat([ego.unsqueeze(1), neighbors[..., :-1]], dim=1)[:, :, -1]
                
                # init t
                t0 = time.time()
                run_t = 0.
                try:
                    # predict
                    with torch.no_grad():
                        plans, predictions, scores, _ = predictor(ego, neighbors, lanes, crosswalks)
                        plan, prediction = select_future(plans, predictions, scores)
                    if model_name in ['NNcf_cf_Human', 'NNcf_None', 'NNcf_DIPP', 'NNcf_cf_LfD','NNcf','NNcf_cf','NNcf_NN']:
                        # plan
                        planner_inputs = {
                            "control_variables": plan.view(-1, 100),
                            "predictions": prediction, # torch.tensor(prediction[None,:,:,:3]).to(device),
                            "ref_line_info": ref_line,
                            "current_state": current_state
                        }

                        plan_traj = bicycle_model_test(plan, ego[:, -1])
                        action_type = judge_action_type(ego.cpu().numpy()[0], plan_traj.cpu().numpy()) 
                        action_type_list.append(action_type)
                        if action_type==1:
                            cost_i = 0
                        elif action_type==2:
                            cost_i = 1
                        for i in range(feature_len):
                            planner_inputs[f'cost_function_weight_{i+1}'] = cost_function_weights[cost_i, i].view(1, 1)

                        with torch.no_grad():
                            final_values, info = planner.layer.forward(planner_inputs, optimizer_kwargs={'track_best_solution': True})
                            plan = info.best_solution['control_variables'].view(-1, 50, 2).to(device)

                        plan_irl = bicycle_model_test(plan, ego[:, -1])
                        plan_traj = bicycle_model(plan, ego[:, -1])[:, :, :3]
                        plan_traj = plan_traj.cpu().numpy()[0]
                    elif model_name in ['PreTrain','NN','NN_L1','NN_L2']:
                        plan_irl = bicycle_model_test(plan, ego[:, -1])
                        plan_traj = bicycle_model(plan, ego[:, -1])[:, :, :3]
                        plan_traj = plan_traj.cpu().numpy()[0]
                    elif model_name == 'User':
                        plan_irl = torch.tensor(ground_truth[0][None,:]).to(device)
                        plan_traj = ground_truth[0]
                    prediction = prediction.cpu().numpy()[0]
                    
                    # take one step
                    obs, done, info = simulator.step(plan_traj, prediction)
                except Exception as e:
                    logging.info(e)
                    error = True
                    break
                print(f'Collision: {info[0]}, Off-route: {info[1]}')

                # cal time
                run_t = (time.time() - t0)
                run_t_list.append(int(run_t*1000))
                print(f"Running Time: {int(run_t*1000):d}ms")
                # render
                if args.render:
                    simulator.render()
                error = False
            if error == False:
                # cal style
                action_type = judge_action_type(ego.cpu().numpy()[0], plan_irl.cpu().numpy()) 
                action_type_list.append(action_type)
                if action_type==2:
                    feture_expert =  np.array([0.04955,0.16039,0.05637,0.01409,0.03583,0.02098,0.00207])
                else: # 0/1 
                    feture_expert =  np.array([0.03089,0.00030,0.00620,0.00417,0.03358,0.00671,0.00127])
                # print(plan_irl.shape,ref_line.shape,torch.tensor(ground_truth[None,1:]).shape,current_state.shape)
                # torch.Size([1, 50, 5]) torch.Size([1, 1200, 5]) torch.Size([1, 10, 50, 5]) torch.Size([1, 11, 8])
                feature = cal_traj_features_train(plan_irl,ref_line,torch.tensor(ground_truth[None,1:]).to(device),current_state).cpu().numpy()[0]
                style_error = np.mean(np.abs(feature-feture_expert))
                style_error_list.append(style_error)
                print(f'style_error: {style_error:.4f}')
            
                # calculate metrics
                running_time.append(int(np.mean(run_t_list)))
                
                collisions.append(info[0])
                off_routes.append(info[1])
                progress.append(simulator.calculate_progress())

                dynamics = simulator.calculate_dynamics()
                acc = np.mean(np.abs(dynamics[0]))
                jerk = np.mean(np.abs(dynamics[1]))
                lat_acc = np.mean(np.abs(dynamics[2]))
                Accs.append(acc)
                Jerks.append(jerk)
                Lat_Accs.append(lat_acc)

                error, human_dynamics = simulator.calculate_human_likeness()
                h_acc = np.mean(np.abs(human_dynamics[0]))
                h_jerk = np.mean(np.abs(human_dynamics[1]))
                h_lat_acc = np.mean(np.abs(human_dynamics[2]))
                Human_Accs.append(h_acc)
                Human_Jerks.append(h_jerk)
                Human_Lat_Accs.append(h_lat_acc)

                similarity_3s.append(error[29])
                similarity_5s.append(error[49])
                similarity_10s.append(error[99])

                # save animation
                if args.save and render_N >= render_count:
                    save_path = log_path + f"vedio_{model_name}/"
                    os.makedirs(save_path, exist_ok=True)
                    simulator.save_animation(save_path)
                    render_count += 1
            else:
                scene_count -= 1
        
            # save metircs
            # if action_type_list == []:
            if scene_count>0 and scene_count%50==0:
                df = pd.DataFrame(data={
                                    'collision':collisions, 'off_route':off_routes, 'progress': progress, # "Action_type":action_type_list,
                                    'Acc':Accs, 'Jerk':Jerks, 'Lat_Acc':Lat_Accs, "Running_Time":running_time, 'style_error': style_error_list,
                                    'Human_Acc':Human_Accs, 'Human_Jerk':Human_Jerks, 'Human_Lat_Acc':Human_Lat_Accs,
                                    'Human_L2_3s':similarity_3s, 'Human_L2_5s':similarity_5s, 'Human_L2_10s':similarity_10s})
                df.to_csv(log_path + f'testing_{model_name}_{scene_count}.csv')
                logging.info(f"save results -- scene_count:{scene_count}!!!")
    logging.info(f"========================Finised Closed-Loop Test========================")
    time.sleep(5)
    

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Closed-loop Test')
    parser.add_argument('--name', type=str, help='log name (default: "Test1")', default="User")
    parser.add_argument('--load_epoch', type=int, help='load weights epoch', default=1000)
    parser.add_argument('--idx', type=int, help='load weights epoch', default=0)
    parser.add_argument('--seed', type=int, help='fix random seed', default=25)
    parser.add_argument('--test_set', type=str, help='path to the test file', default='/data/lin_funster/waymo_dataset/test_20s')
    parser.add_argument('--model_path', type=str, help='path to saved model', default='/data/lin_funster/proj_paper1/model_param/')
    parser.add_argument('--render', action="store_true", help='if render the scene (default: False)', default=False)
    parser.add_argument('--save', action="store_true", help='if save animation (default: False)', default=False)
    parser.add_argument('--gpu_id', type=int, help='run on which device (default: 0)', default=0)
    args = parser.parse_args()

    # fixed Param
    fixed_seed(args.seed)
    model_name_list = [f'{args.name}'] # 'NNcf_cf_Human', 'User', 'PreTrain', 'NN','NN_L1','NN_L2', 'NNcf_cf','NNcf'
    # changeable Param
    render_N = 0
    scene_N = 200
    # logging
    log_path = f"./testing_log/{args.name}_Closed_TestStyle_{args.load_epoch}_{args.idx}/"
    os.makedirs(log_path, exist_ok=True)
    initLogging(log_file=log_path+'test.log')

    logging.info("------------- {} -------------".format(args.name))
    logging.info("Use gpu_id: {}".format(args.gpu_id))
    # logging.info("Style_N: {}".format(Style_N))
    logging.info("render_N: {}".format(render_N))
    logging.info("scene_N: {}".format(scene_N))
    logging.info("load_epoch: {}".format(args.load_epoch))
    logging.info("idx: {}".format(args.idx))
    logging.info("model_path: {}".format(args.model_path))
    # device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() and args.gpu_id is not None else "cpu")
    trajectory_len, feature_len = 50, 9
    planner = MotionPlanner(trajectory_len, feature_len, device=device, test=True)
    # run test
    for model_i, model_name in enumerate(model_name_list):
        if model_name in ['NN','NN_L1','NN_L2','NNcf','NNcf_DIPP']:
            # load model
            predictor = Predictor(50).to(device)
            predictor.load_state_dict(torch.load(args.model_path+f'res_modelF/Predictor_{model_name}_0_{args.load_epoch}.pth', map_location=device))
            predictor.eval()
        elif model_name in ['User','PreTrain','NNcf_cf_Human','NNcf_None','NNcf_cf_LfD','NNcf_cf']: # model_name == 'Raw'
            predictor = Predictor(50).to(device)
            predictor.load_state_dict(torch.load(args.model_path+'model_5_0.7052.pth', map_location=device))
            predictor.eval()
        if model_name in ['NNcf','NNcf_DIPP','NNcf_cf','NNcf_cf_LfD']:
            cost_function_weights = np.load(args.model_path+f'res_modelF/CostFunction_{model_name}_0_{args.load_epoch}.npy')
            cost_function_weights = torch.Tensor(cost_function_weights).to(device)
        elif model_name == 'NNcf_cf_Human':
            cost_function_weights = torch.Tensor([[0.1,0.5,0.1,0.01,0.5,0.5,5,10,10],
                                                  [0.1,0.5,0.1,0.01,0.5,0.5,5,10,10]]).to(device).view(2,-1)
        elif model_name == 'NNcf_None':
            cost_function_weights = torch.Tensor([[0.5,0.5,0.5,0.5,0.5,0.5,5,10,10],
                                                  [0.5,0.5,0.5,0.5,0.5,0.5,5,10,10]]).to(device).view(2,-1)
        closed_loop_test(model_name)

