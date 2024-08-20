# -*- coding:utf-8 -*-
import numpy as np
from matplotlib import pyplot


class K_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, k=2, tolerance=0.0001, max_iter=300):
        self.k_ = k
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter

    def fit(self, data):
        self.centers_ = {}
        for i in range(self.k_):
            self.centers_[i] = data[i] + 1e-6

        for i in range(self.max_iter_):
            self.clf_ = {}
            for i in range(self.k_):
                self.clf_[i] = []
            # print("质点:",self.centers_)
            for feature in data:
                distances = []
                for center in self.centers_:
                    # 欧拉距离
                    distances.append(np.linalg.norm(feature - self.centers_[center]))
                classification = distances.index(min(distances))
                self.clf_[classification].append(feature)

            # print("分组情况:",self.clf_)
            prev_centers = dict(self.centers_)
            for c in self.clf_:
                self.centers_[c] = np.average(self.clf_[c], axis=0)

            # '中心点'是否在误差范围
            optimized = True
            for center in self.centers_:
                org_centers = prev_centers[center]
                cur_centers = self.centers_[center]
                if np.sum((cur_centers - org_centers) / org_centers * 100.0) > self.tolerance_:
                    optimized = False
            if optimized:
                break

    def predict(self, p_data):
        distances = [np.linalg.norm(p_data - self.centers_[center]) for center in self.centers_]
        index = distances.index(min(distances))
        return index
    
    def classify_dataset(self,raw_data):
        N = len(raw_data)
        res = []
        for i in range(N):
            res.append(self.predict(raw_data[i]))
        return np.hstack(res)
    
    def classify_dataset_find_N(self,raw_data,find_N=10): # find_N 找到对应最近的N个样本
        label_res = np.zeros((self.k_,find_N), dtype=np.int) -1
        distance_res = np.zeros((self.k_,find_N)) + 100000.
        all_label_res = []
        for data_i, p_data in enumerate(raw_data):
            distances = [np.linalg.norm(p_data - self.centers_[center]) for center in self.centers_]
            min_distance = min(distances)
            label = distances.index(min_distance)
            all_label_res.append(label)
            
            max_idx = np.argmax(distance_res[label,:])
            if min_distance<distance_res[label, max_idx]: # update
                distance_res[label,max_idx] = min_distance
                label_res[label,max_idx] = data_i
                
        if np.sum(label_res==-1):
            print(label_res)
            raise Exception("No enough sample for this classification")
        return label_res, np.hstack(all_label_res)
    
    def classify_dataset_find_N_Max(self,raw_data,find_Max_N=10): # find_N 找到对应最近的N个样本,可以小于N
        label_res = [None]*self.k_
        distance_res = [None]*self.k_
        for i in range(self.k_):
            label_res[i] = []
            distance_res[i] = [100000.]
        all_label_res = []
        for data_i, p_data in enumerate(raw_data):
            distances = [np.linalg.norm(p_data - self.centers_[center]) for center in self.centers_]
            min_distance = min(distances)
            label = distances.index(min_distance)
            all_label_res.append(label)
            
            arr = np.hstack(distance_res[label])
            max_idx = np.argmax(arr)
            if min_distance<arr[max_idx]: # update
                distance_res[label].append(min_distance)
                label_res[label].append(data_i)
                if len(label_res[label])>find_Max_N:
                    after_max_idx = np.argmax(np.hstack(distance_res[label]))
                    label_res[label].pop(after_max_idx)
                    distance_res[label].pop(after_max_idx)
        
        center_value_np = np.array([self.centers_[center][4] for center in self.centers_])
        center_value_idx_np = np.argsort(center_value_np) # 升序索引 保守>激进
        find_label = [np.hstack(label_res[i]) for i in range(self.k_)]
        return find_label, np.hstack(all_label_res), center_value_idx_np


if __name__ == '__main__':
    x = np.array([[1, 2, 4], [1.5, 1.8, 1.6], [5, 8, 6], [8, 8, 7], [1, 0.6, 0.8], [9, 11, 10]])
    k_means = K_Means(k=3)
    k_means.fit(x)
    # cat = k_means.predict(predict)
    print(k_means.centers_)
    for center in k_means.centers_:
        pyplot.scatter(k_means.centers_[center][0], k_means.centers_[center][1], marker='*', s=150)

    for cat in k_means.clf_:
        for point in k_means.clf_[cat]:
            pyplot.scatter(point[0], point[1], c=('r' if cat == 0 else 'b'))

    predict = [[2, 1, 2], [6, 9, 7]]
    for feature in predict:
        cat = k_means.predict(predict)
        pyplot.scatter(feature[0], feature[1], c=('r' if cat == 0 else 'b'), marker='x')

    pyplot.show()
