#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   lin_utils.py
@Author  :   Fangze Lin
@System  :   Ubuntu20.04
@Time    :   2023/06/20 15:53:00
@Brief   :   Lin utils
'''

import torch as t
import torch.nn as nn
import numpy as np

def torch_to_gpu(data: nn.Module, gpu_id):
    device = t.device(f"cuda:{gpu_id}" if t.cuda.is_available() and gpu_id is not None else "cpu")
    data.to(device)
    # print(f"model move to {device.__str__()}")
    # model.eval()


# 将其他类型的数据转换成tensor类型的数据，支持多变量传入，支持移动gpu参数传入
def convert2tensor(*datas, convert2type=t.float32, gpu_device=None, check_data=False) -> tuple(
        t.float32, ...):
    """
    convert one or more than one data to tensor
    """

    return_data = []

    # 如果没有数据传入，则返回
    if datas is ():
        raise Exception("data is None")

    for data in datas:
        # 如果类型已经满足，则报错
        if type(data) is t.Tensor:
            if check_data:
                raise Exception("data convert is not valid")
        # 处理numpy类型
        elif type(data) is np.ndarray:
            try:
                data = t.from_numpy(data).type(convert2type)  # 转换成torch.float64 -> torch.float32
            except:
                raise Exception("data can not convert to tensor from np.ndarray")

        # 处理list数据
        elif type(data) is list:
            try:
                data = list2tensor(data, convert2type=convert2type)
            except:
                raise Exception("data can not convert to tensor from list")

        # 处理非tensor类型
        elif not type(data) is t.Tensor:
            try:
                data = t.FloatTensor(np.array(data)).type(convert2type)
            except:
                print(type(data))
                raise Exception("data can not convert to tensor")

        # 处理非tensor.float32类型
        elif data != convert2type:
            try:
                data = data.type(convert2type)  # 转换成float32
            except:
                raise Exception("data can not convert to float")

        # 将其移动到gpu上
        if gpu_device is not None:
            data = data.to(gpu_device)
        # 添加返回
        return_data.append(data)

    # 因为元组如果只有一个元素的话会多输出一个逗号
    if return_data.__len__() == 1:
        return return_data[0]
    else:
        return tuple(return_data)


def list2tensor(*datas, convert2type=t.float32):
    """
        convert one or more than one list to numpy
        """
    return_data = []

    # 如果没有数据传入，则返回
    if datas is ():
        return print("data is None")

    for data in datas:
        if type(data) is list:
            # 如果元素type都为t.Tensor
            if np.array([type(e) == t.Tensor for e in data]).all():
                data = t.cat(data, dim=0).view(len(data), -1).type(convert2type)
            # 其余元素类型太多，直接强转，转不了则报错。
            else:
                try:
                    data = np.vstack(data)
                    data = t.tensor(data).type(convert2type)
                except:
                    raise Exception("This feature is not currently supported!!!")
        return_data.append(data)

    # 因为元组如果只有一个元素的话会多输出一个逗号
    if return_data.__len__() == 1:
        return return_data[0]
    else:
        return tuple(return_data)