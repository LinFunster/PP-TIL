#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import numpy as np
import os
from torch.utils.data import Dataset
import random

import glob

class DataManager_Train(Dataset):
    def __init__(self, running_dir, save_dir = None, is_rand = False):
        self.is_rand = is_rand
        self.running_dir = running_dir
        self.save_dir = save_dir
        if not os.path.exists(self.running_dir):
            os.mkdir(self.running_dir)
        if save_dir is not None:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir) #递归创建目录
        self.data_list = glob.glob(self.running_dir+'/*')

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx): 
        path_list = self.data_list[idx].split('/')[-1].split('_')
        behavior_i, style_i = int(path_list[0].split('behavior')[-1]), int(path_list[1].split('style')[-1].split('.')[0])

        if self.is_rand:
            style_i = random.randint(0, 4)
        return self.load_data(behavior_i, style_i)
    
    def _check_nan_(self, *data):
        is_nan = False
        for elem in data:
            if np.isnan(elem).any():
                is_nan = True
        return is_nan
    
    def load_data(self, behavior_i, style_i): # load traj
        filename = f"{self.running_dir}/behavior{behavior_i}_style{style_i}.npz"
        # load data
        data = np.load(filename)
        # prediction = data['prediction']
        ego = data['ego']
        neighbors = data['neighbors']
        lanes = data['lanes']
        crosswalks = data['crosswalks']
        ref_line = data['ref_line']
        current_state = data['current_state']
        ground_truth = data['ground_truth']
        return ego, neighbors, lanes, crosswalks, ref_line, current_state, ground_truth, behavior_i, style_i

class DataManager_Test(Dataset):
    def __init__(self, running_dir,save_dir):
        self.running_dir = running_dir
        self.save_dir = save_dir
        if not os.path.exists(self.running_dir):
            os.mkdir(self.running_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir) #递归创建目录
        self.data_list = glob.glob(self.running_dir+'/*')

    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, idx): # load raw
        data = np.load(self.data_list[idx])
        ego = data['ego']
        neighbors = data['neighbors']
        ref_line = data['ref_line']
        map_lanes = data['map_lanes']
        map_crosswalks = data['map_crosswalks']
        gt_future_states = data['gt_future_states']
        
        return ego, neighbors, map_lanes, map_crosswalks, ref_line, gt_future_states
    
class DataManager_Style(Dataset):
    def __init__(self, running_dir,save_dir):
        self.running_dir = running_dir
        self.save_dir = save_dir
        if not os.path.exists(self.running_dir):
            os.mkdir(self.running_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir) #递归创建目录
        self.data_list = glob.glob(self.running_dir+'/*')

    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, idx): # load raw
        path_list = self.data_list[idx].split('/')[-1].split('_')
        scene_id, time_step = path_list[0], int(path_list[1].split('.')[0])
        
        data = np.load(self.data_list[idx])
        ego = data['ego']
        neighbors = data['neighbors']
        ref_line = data['ref_line']
        map_lanes = data['map_lanes']
        map_crosswalks = data['map_crosswalks']
        gt_future_states = data['gt_future_states']
        
        return ego, neighbors, map_lanes, map_crosswalks, ref_line, gt_future_states, scene_id, time_step
    
    def _check_nan_(self, *data):
        is_nan = False
        for elem in data:
            if np.isnan(elem).any():
                is_nan = True
        return is_nan

    def save_data(self, behavior_i, style_i, ego, neighbors, lanes, crosswalks, ref_line, current_state, ground_truth, features): # save task
        if self._check_nan_(ego, neighbors, lanes, crosswalks, ref_line, current_state, ground_truth,features):
            return 
        filename = f"{self.save_dir}/behavior{behavior_i}_style{style_i}.npz"
        # save data
        np.savez(filename, ego=ego, neighbors=neighbors, lanes=lanes, crosswalks=crosswalks, ref_line=ref_line, current_state=current_state, ground_truth=ground_truth,features=features)


if __name__ == "__main__":
    pass
