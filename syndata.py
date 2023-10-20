import time

import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Any, Callable, List, Optional, Tuple
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
import plotly.graph_objects as go


def time_scaling(pat_temp_length, stride: int = 5) -> Tuple[int, int]:
    # pt_dim = pattern_template.shape[-1]
    pattern_length = (pat_temp_length - 1) * stride + 1
    # aux1_pattern = pattern_template.reshape(1, pt_dim, 1)
    # aux2_pattern = F.pad(aux1_pattern, (0, stride - 1))
    # aux3_pattern = aux2_pattern.reshape(1, 1, pt_dim * stride)
    # pattern = aux3_pattern[:, :, :pattern_length]
    return pattern_length, stride


def time_warping(pattern_length, stride: int) -> List:
    """Documentacion
    examples
    fff"""
    indices = [0]
    aux_index = 0
    # n = pattern.shape[-1]
    for i in range(stride, pattern_length - 1, stride):
        # index = i
        # aux_index = np.random.choice(range(aux_index + 1, i + int(stride/2)))
        aux_index = np.random.choice(range(i - int(stride/2), i + int(stride/2)))
        # if aux_index < pattern_length:
        # pattern[:, :, aux_index] = pattern[:, :, i]
        # index = aux_index
        # if i != 0 and i != n - 1:
        # pattern[:, :, i] = 0
        indices.append(aux_index)
        # if 0 not in indices:
        #     indices = [0] + indices
        # if pattern_length-1 not in indices:
        #     indices = indices + [pattern_length-1]
    return indices + [pattern_length - 1]


# def linear_interpolation(pattern: Tensor, indices: List):
#     aux1_pattern = torch.diff(pattern, dim=-1)


def linear_interpolation(start, end, steps):
    f = torch.vmap(torch.linspace)
    return f(start, end, steps=steps)


def interpolation(pattern_template: Tensor, indices: List) -> Tensor:
    # index = torch.where(pattern[:, :, :] > 0)[2]
    # aux_pattern = pattern_template
    # aux_sizes = indices
    # pt_length
    # print(pattern_template)
    # print(aux_pattern)
    # print(f'vvvvvvvvvvvvvvvvv {indices}')
    # aux_pattern = pattern_template.reshape(-1)
    interp_list = [pattern_template[:, 0:1]]
    for i in range(1, pattern_template.shape[-1]):
        # start, end = aux_pattern[:,i-1], aux_pattern[:,i]
        # interp = F.interpolate(pattern_template.reshape(1, 1, -1)[:, :, [i - 1, i]],
        #                        size=indices[i] - indices[i - 1] + 1, mode='linear')
        # interp = np.linspace()
        interp = torch.tensor(np.linspace(pattern_template[:, i-1].item(), pattern_template[:, i].item(), indices[i] - indices[i-1] + 1)) #.reshape(1, -1)
        # interp1 = linear_interpolation(start, end, steps=indices[i] - indices[i - 1] + 1).reshape(1, 1, -1)
        interp_list.append(interp.reshape(1, -1)[:, 1:])
    return torch.cat(interp_list, dim=-1)


def noise_adding(pattern: Tensor) -> Tensor:
    for i in range(pattern.shape[-1]):
        if np.random.choice(np.arange(0, 1.1, 0.1)) < 1.0:
            diff = 0 #pattern[:, i]
            if i < pattern.shape[-1] - 1:
                diff = pattern[:, i + 1] - pattern[:, i]
            pattern[:, i:i + 1] += np.random.choice(np.arange(-0.7, 0.75, 0.1)) * diff
    return pattern

def noise_adding_min(pattern: Tensor) -> Tensor:
    for i in range(pattern.shape[-1]):
        if np.random.choice(np.arange(0, 1.1, 0.1)) < 1.0:
            diff = 0 #pattern[:, i]
            if i < pattern.shape[-1] - 1:
                diff = pattern[:, i + 1] - pattern[:, i]
            pattern[:, i:i + 1] += np.random.choice(np.arange(-1.0, -0.4, 0.5)) * abs(diff)
    return pattern

def noise_adding_max(pattern: Tensor) -> Tensor:
    for i in range(pattern.shape[-1]):
        if np.random.choice(np.arange(0, 1.1, 0.1)) < 1.0:
            diff = 0 #pattern[:, i]
            if i < pattern.shape[-1] - 1:
                diff = pattern[:, i + 1] - pattern[:, i]
            pattern[:, i:i + 1] += np.random.choice(np.arange(0.4, 1.0, 0.5)) * abs(diff)
    return pattern


def synthetic_data(pattern_template: Tensor, random_stride: int = 2) -> Tensor:
    pt_length = pattern_template.shape[-1]
    out1, out2 = time_scaling(pt_length, stride=random_stride)
    out3 = time_warping(out1, out2)
    # print(out3)
    output3 = interpolation(pattern_template, out3)
    output4 = noise_adding(output3)
    return output4
    # return torch.cat([torch.full((1,63-random_stride*(pt_length-1)), torch.mean(pattern_template)), output4], dim=-1)


def random_tensor(dim0: int, dim1: int):
    f = torch.vmap(np.random.rand)
    return f(dim0, dim1)


def normalization(data: Tensor) -> Tensor:
    std, mean = torch.std_mean(data, dim=-1)
    data = torch.cat([torch.tensor(np.random.uniform(low= torch.min(data), high=torch.max(data), size=(32 - data.shape[-1]))).reshape(1,-1), data], dim=-1)
    output = (data - mean) / std
    return output

def normalization_real(data: Tensor) -> Tensor:
    std, mean = torch.std_mean(data, dim=-1)
    data = torch.cat([torch.tensor(np.random.uniform(low= torch.min(data), high=torch.max(data), size=(31 - data.shape[-1]))), data], dim=-1)
    output = (data - mean) / std
    return output

def normalization64(data: Tensor) -> Tensor:
    std, mean = torch.std_mean(data)
    data = torch.cat([torch.tensor(np.random.uniform(low= torch.min(data), high=torch.max(data), size=(62 - data.shape[-1]))), data], dim=-1)
    output = (data - mean) / std
    return output

def normalization62(data: Tensor) -> Tensor:
    std, mean = torch.std_mean(data)
    data = torch.cat([torch.Tensor(np.random.uniform(low= torch.min(data), high=torch.max(data), size=(62 - data.shape[-1]))).reshape(1,-1), data], dim=-1)
    output = (data - mean) / std
    return output


def vec_normalization(data: Tensor) -> Tensor:
    f = torch.vmap(normalization)
    return f(data)


def data_generator(pattern_templates: Tensor, data_size: int, filename: str) -> None:
    # wedge_rising = torch.Tensor([[[5, 1, 5.5, 2.5, 6, 4, 6.5]]])
    # head_and_shoulders = torch.Tensor([[[1, 4, 2, 6, 2, 4, 1]]])
    # cup_with_handle = torch.Tensor([[[5, 3, 2, 3, 5, 4, 5]]])
    # triangle_ascending = torch.Tensor([[[6, 2, 6, 3, 6, 4, 6]]])
    # triple_tops = torch.Tensor([[[2, 6, 2, 6, 2, 6, 2]]])
    # double_bottoms = torch.Tensor([[[6, 3, 2, 3, 6, 2, 6]]])
    # list_patterns = [wedge_rising, head_and_shoulders, cup_with_handle, triangle_ascending, triple_tops, double_bottoms]
    # patterns = torch.cat(list_patterns, dim=0)
    # print(patterns)
    # print(patterns[0].shape)
    # vec_generator = torch.vmap(synthetic_data, in_dims=0, chunk_size=1)
    # out = vec_generator(pattern_templates)
    # print('Hello')
    # print(f'eeeeeeeeeeeeeeeee {out}')
    #
    # pattern_templates = ['wedge_rising', 'head_and_shoulders', 'cup_with_handle', 'triangle_ascending', 'triple_tops',
    #                      'double_bottoms']
    list_data = []
    list_labels = []

    # labels = torch.tensor([[[0]], [[1]], [[2]], [[3]], [[4]], [[5]]])
    for _ in range(data_size):
        #     list_tensors = []
        list_candle = []
        random_pattern = np.random.choice(np.arange(0, 7, 1))
        random_stride = np.random.choice(np.arange(5, 6, 1))
        # for _ in range(4):

        if random_pattern == 6:
            out = torch.rand(1,32, dtype=torch.float32)
        else:
                # random_stride = 5
            out = synthetic_data(pattern_templates[random_pattern], random_stride=random_stride)
        out = normalization(out)
        # out0 = out.detach().clone()
        # # np.random.seed(23432)
        # out1 = noise_adding(out0.detach().clone())
        # # np.random.seed(8762)
        # out_min = noise_adding_min(out0.detach().clone())
        # # out_min2 = noise_adding_min(out1.detach().clone())
        # # np.random.seed(6348)
        # out_max = noise_adding(out0.detach().clone())
        # # out_max2 = noise_adding_min(out1.detach().clone())
        #     # out = vec_generator(pattern_templates, random_stride=random_stride)
        #     # print(out)
        #     # std, mean = torch.std_mean(out, dim=-1)
        #     # out1 = (out - mean) / std
        #     # out1 = vec_normalization(out)
        # out0 = normalization(out0)
        # out1 = normalization(out_max)
        # out2 = normalization(out_min)
        # out3 = normalization(out1)
        #     # n = out1.shape[-1]
        #     # x = torch.linspace(0, n - 1, steps=n)
        #     # plt.plot(x, out1[0,:])
        #     # plt.show()
        #     # print(out1)
        #     # time.sleep(60)
        # out0 = out0.type(torch.float32).reshape(-1)
        # out1 = out1.type(torch.float32).reshape(-1)
        # out2 = out2.type(torch.float32).reshape(-1)
        # out3 = out3.type(torch.float32).reshape(-1)
        # # list_candle.append(out1)
        # list_candle = [out0, out1, out2, out3]
        # out1 = torch.stack(list_candle)
        # min = torch.min(out1, dim=0).values
        # max = torch.max(out1, dim=0).values
        # out1 = torch.stack([out1[0], max, min, out1[-1]])
        # # out1 = torch.stack([max, min])
        list_data.append(out.reshape(1,-1))
        label = torch.tensor(random_pattern).reshape(1,-1)
        list_labels.append(label)
    syn_data = torch.stack(list_data, dim=0)
    syn_data = torch.Tensor(syn_data)
    print(syn_data.shape)

    syn_label = torch.stack(list_labels, dim=0)
    syn_label = torch.Tensor(syn_label)
    print(syn_label.shape)
    with open(filename + '.pt', 'wb') as f:
        torch.save((syn_data, syn_label), f)
    #         for i in range(data_size):
    #             output1 = time_scaling(eval(template))
    #             output2, indices = time_warping(output1)
    #             output3 = interpolation(output2, indices)
    #             output4 = noise_adding(output3)
    #             list_tensors.append(output4)
    #             # data = torch.cat(output4, dim=0)
    #         torch.save(torch.cat(list_tensors, dim=0), f)


def pattern_templates(filename: str):
    # triangle_symmetrical = torch.Tensor([[[7,1,6,2,5,3,4]]])
    # double_top_adam_adam = torch.Tensor([[[1,3,5,3,5,3,1]]])
    # head_and_shoulders_bottom = torch.Tensor([[[5.5,2.5,4,1,4,2.5,5.5]]])
    # triple_top = torch.Tensor([[[1,5,2,5,2,5,1]]])
    wedge_rising = torch.Tensor([[[8, 1, 9, 5, 10, 9, 11]]])
    head_and_shoulders = torch.Tensor([[[1, 7, 4, 10, 4, 7, 1]]])
    cup_with_handle = torch.Tensor([[[6, 2.5, 2, 2.5, 6, 4, 6]]])
    triangle_ascending = torch.Tensor([[[10, 1, 10, 5, 10, 9, 10]]])
    triple_tops = torch.Tensor([[[1, 9, 5, 9, 5, 9, 1]]])
    double_bottoms = torch.Tensor([[[7, 1, 3, 5, 3, 1, 7]]])
    list_patterns = [wedge_rising, head_and_shoulders, cup_with_handle, triangle_ascending, triple_tops, double_bottoms]
    # list_patterns = [double_top_adam_adam, head_and_shoulders_bottom, triple_top]
    patterns = torch.cat(list_patterns, dim=0)
    with open(filename + '.pt', 'wb') as f:
        torch.save(patterns, f)


def load_data(filename: str) -> Tensor:
    # pattern_templates = ['wedge_rising', 'head_and_shoulders', 'cup_with_handle', 'triangle_ascending', 'triple_tops',
    #                      'double_bottoms']
    # inputs = []
    # for template in pattern_templates:
    with open(filename + '.pt', 'rb') as f:
        # data = torch.cat(torch.load(f), dim=0)
        inputs = torch.load(f)
    return inputs

def transform_data(candle_data: Tensor):
    n = candle_data.shape[-1]
    new_timeseries = []
    for i in range(n):
        value1 = candle_data[2,i]
        value2 = candle_data[1,i]
        if (candle_data[1,i] - candle_data[0,i]) < (candle_data[0,i] - candle_data[2,i]):
            value1 = candle_data[1,i]
            value2 = candle_data[2,i]
        new_timeseries.append(value1)
        new_timeseries.append(value2)
    return torch.Tensor(new_timeseries).reshape(1,1,1,-1)

def transform_database(data: Tensor):
    n = data.shape[0]
    list_transform_data = []
    for i in range(n):
        t_data = transform_data(data[i,0])
        list_transform_data.append(t_data)
    return torch.cat(list_transform_data, dim=0)



if __name__ == '__main__':
    filename = 'pattern_templates_v3'
    pattern_templates(filename)
    templates = load_data(filename)
    # print(templates)
    # n = templates.shape[0]
    # x = torch.linspace(0, 6, steps=7)
    # for i in range(n):
    #     plt.plot(x, templates[i].reshape(-1))
    #     plt.show()
    # pt = torch.Tensor([[[1, 4, 2, 6, 2, 4, 1]]])
    # output1 = time_scaling(pt)
    # print(output1)
    # output2 = time_warping(output1)
    # print(output2)
    # output3 = interpolation(output2)
    # print(output3)
    # output4 = noise_adding(output3)
    # print(output4)

    filename2 = 'synthetic_data_v25'
    data_generator(templates, 30000, filename2)
    X_data, Y_data = load_data(filename2)
    print(X_data[25], Y_data[25])
    # X_test = X_data

    # one_data = X_data[0,0]
    # h = transform_data(one_data)
    # print(h.shape)
    # print(h)
    x = torch.linspace(0,31,steps=32)
    plt.plot(x, X_data[2,0].reshape(-1))
    # plt.plot(x, X_data[25,0,1].reshape(-1))
    # plt.plot(x, X_data[25,0,2].reshape(-1))
    # plt.plot(x, X_data[25,0,3].reshape(-1))
    plt.show()

    # for n in range(0,10):
    #     X_data = X_test[n, 0]
    # #     b = Y_test[n]
    # #
    # # #     # x_date = pd.DataFrame(x.numpy())
    #     dates = pd.date_range(start='1/1/2023', end='1/31/2023')
    #     fig = go.Figure(data=[go.Candlestick(x=dates,
    #                                          open=X_data[0],
    #                                          high=X_data[1],
    #                                          low=X_data[2],
    #                                          close=X_data[3])])
    #     fig.show()

    # inputs = load_data()
    # print(inputs)
    # print(inputs[5].shape)
    # for i in range(6):
    #     y = inputs[i][0]
    #     n = y.shape[2]
    #     x = torch.linspace(0, n - 1, steps=n)
    #     # print(x)
    #     plt.plot(x, y[0, 0, :])
    #     plt.show()
