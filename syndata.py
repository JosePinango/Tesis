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
from scipy import linalg


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
        aux_index = np.random.choice(range(i - int(stride / 2), i + int(stride / 2)))
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
        interp = torch.tensor(np.linspace(pattern_template[:, i - 1].item(), pattern_template[:, i].item(),
                                          indices[i] - indices[i - 1] + 1))  # .reshape(1, -1)
        # interp1 = linear_interpolation(start, end, steps=indices[i] - indices[i - 1] + 1).reshape(1, 1, -1)
        interp_list.append(interp.reshape(1, -1)[:, 1:])
    return torch.cat(interp_list, dim=-1)


def noise_adding(pattern: Tensor) -> Tensor:
    for i in range(pattern.shape[-1]):
        if np.random.choice(np.arange(0, 1.1, 0.1)) < 0.7:
            diff = 0  # pattern[:, i]
            if i < pattern.shape[-1] - 1:
                diff = pattern[:, i + 1] - pattern[:, i]
            pattern[:, i:i + 1] += np.random.choice(np.arange(-0.8, 0.85, 0.1)) * diff
    return pattern


# def noise_adding(pattern: Tensor) -> Tensor:
#     noise = torch.normal(0, 0.25, size=pattern.shape)
#     return pattern + noise


def noise_adding_min(pattern: Tensor) -> Tensor:
    for i in range(pattern.shape[-1]):
        if np.random.choice(np.arange(0, 1.1, 0.1)) < 1.0:
            diff = 0  # pattern[:, i]
            if i < pattern.shape[-1] - 1:
                diff = pattern[:, i + 1] - pattern[:, i]
            pattern[:, i:i + 1] += np.random.choice(np.arange(-1.0, -0.4, 0.5)) * abs(diff)
    return pattern


def noise_adding_max(pattern: Tensor) -> Tensor:
    for i in range(pattern.shape[-1]):
        if np.random.choice(np.arange(0, 1.1, 0.1)) < 1.0:
            diff = 0  # pattern[:, i]
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
    # data = torch.cat([torch.tensor(np.random.uniform(low= torch.min(data), high=torch.max(data), size=(31 - data.shape[-1]))).reshape(1,-1), data], dim=-1)
    noise = torch.normal(3, 1, size=(1, 31 - data.shape[-1]))
    data = torch.cat([noise, data], dim=-1)
    output = (data - mean) / std
    return output


def normalization_realdata(data: Tensor) -> Tensor:
    std, mean = torch.std_mean(data, dim=-1)
    # data = torch.cat([torch.tensor(np.random.uniform(low= torch.min(data), high=torch.max(data), size=(31 - data.shape[-1]))).reshape(1,-1), data], dim=-1)
    # noise = torch.normal(3, 1, size=(1, 31 - data.shape[-1]))
    # data = torch.cat([noise, data], dim=-1)
    output = (data - mean) / std
    return output


def normalization_real(data: Tensor) -> Tensor:
    std, mean = torch.std_mean(data, dim=-1)
    data = torch.cat(
        [torch.tensor(np.random.uniform(low=torch.min(data), high=torch.max(data), size=(31 - data.shape[-1]))), data],
        dim=-1)
    output = (data - mean) / std
    return output


def normalization64(data: Tensor) -> Tensor:
    std, mean = torch.std_mean(data)
    data = torch.cat(
        [torch.tensor(np.random.uniform(low=torch.min(data), high=torch.max(data), size=(62 - data.shape[-1]))), data],
        dim=-1)
    output = (data - mean) / std
    return output


def normalization62(data: Tensor) -> Tensor:
    std, mean = torch.std_mean(data)
    data = torch.cat([torch.Tensor(
        np.random.uniform(low=torch.min(data), high=torch.max(data), size=(62 - data.shape[-1]))).reshape(1, -1), data],
                     dim=-1)
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
            # out = torch.normal(1,31, dtype=torch.float32)
            out = torch.normal(0, 1, size=(1, 31))
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
        list_data.append(out.reshape(1, -1))
        label = torch.tensor(random_pattern).reshape(1, -1)
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
    wedge_rising = torch.Tensor([[[2, 5.5, 3.5, 5.75, 5, 6, 6.5]]])
    head_and_shoulders = torch.Tensor([[[2, 5, 3, 7, 3, 5, 2]]])
    cup_with_handle = torch.Tensor([[[6, 2.5, 2, 2.5, 6, 4, 6]]])
    triangle_ascending = torch.Tensor([[[5.5, 1, 5.5, 3, 5.5, 5, 5.5]]])
    triple_tops = torch.Tensor([[[1, 5, 2, 5, 2, 5, 1]]])
    double_bottoms_eve_adam = torch.Tensor([[[6, 2.5, 2, 2.5, 6, 2, 6]]])
    list_patterns = [wedge_rising, head_and_shoulders, cup_with_handle, triangle_ascending, triple_tops,
                     double_bottoms_eve_adam]
    # list_patterns = [head_and_shoulders, triangle_ascending, triple_tops,
    #                  double_bottoms_eve_adam]
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
        value1 = candle_data[2, i]
        value2 = candle_data[1, i]
        if (candle_data[1, i] - candle_data[0, i]) < (candle_data[0, i] - candle_data[2, i]):
            value1 = candle_data[1, i]
            value2 = candle_data[2, i]
        new_timeseries.append(value1)
        new_timeseries.append(value2)
    return torch.Tensor(new_timeseries).reshape(1, 1, 1, -1)


def transform_database(data: Tensor):
    n = data.shape[0]
    list_transform_data = []
    for i in range(n):
        t_data = transform_data(data[i, 0])
        list_transform_data.append(t_data)
    return torch.cat(list_transform_data, dim=0)


def homotopy(Tensor, filename: str):
    n = Tensor.shape[-1]
    N = Tensor.shape[0]
    list_h_tensor = []
    for j in range(N):
        for k in range(N):
            if j == k:
                pass
            else:
                for t in np.arange(0, 1, 0.01):
                    h_tensor = []
                    for i in range(n):
                        Tensor1 = normalization_realdata(Tensor[j,0])
                        Tensor2 = normalization_realdata(Tensor[k,0])
                        value = (1 - t) * Tensor1[i] + t * Tensor2[i]
                        h_tensor.append(value)
                    list_h_tensor.append(torch.Tensor([h_tensor]))
    homotopy_tensor = torch.stack(list_h_tensor)
    with open(filename + '.pt', 'wb') as f:
        torch.save(homotopy_tensor, f)


def barycentric_coordinates(pattern_name: str):
    X_data, Y_data, length_data = load_data(pattern_name + '_v3')
    n = X_data.shape[0]
    coordinates = np.random.randint(1,high=100,size=n)
    # coordinates = np.ones(n)*1/n
    # print(f'Random coordinates: {coordinates}')
    aux = np.sum(coordinates)
    norm_coordinates = (1/aux) * coordinates
    # print(f'Normalized coordinates: {norm_coordinates}')
    proof = np.sum(norm_coordinates)
    # print(f'Sum coordinates: {proof}')

    X_data = X_data.numpy()
    # print(X_data)
    new_tensor = np.zeros(31)
    for i in range(n):
        data = X_data[i,0]
        mean = np.mean(data)
        std = np.std(data)
        norm_pattern = (data-mean)/std
        new_tensor += norm_coordinates[i] * norm_pattern
        # print(new_tensor)
        plt.plot(np.arange(0,31,1, dtype=int), norm_pattern)
        plt.title('Pattern ' + str(i))
        plt.show()
        time.sleep(5)
    # plt.plot(np.arange(0, 31, 1, dtype=int), new_tensor)
    # plt.title('Synthetic pattern')
    # plt.show()
    # time.sleep(2)



    # t=1
    # data = X_data[0, 0]
    # mean = np.mean(data)
    # std = np.std(data)
    # norm_pattern = (data - mean) / std
    # plt.plot(np.arange(0, 31, 1, dtype=int), norm_pattern)
    # plt.title('Real pattern')
    # plt.show()
    # time.sleep(5)
    # aux = t*norm_pattern + (1-t)*new_tensor
    # plt.plot(np.arange(0, 31, 1, dtype=int), aux)
    # plt.title('Pattern out convex hull, t = ' + str(t))
    # plt.show()
    # time.sleep(5)
    return torch.from_numpy(new_tensor).reshape(1,-1), Y_data[0]


def recognition_algorithm(pattern_name: str, indexes: List):
    X_data, _ = load_data(pattern_name + '_v1')
    X_data = X_data[indexes].numpy()
    matrix = np.zeros((len(indexes),31))
    for i in range(len(indexes)):
        data = X_data[i,0]
        mean = np.mean(data)
        std = np.std(data)
        norm_pattern = (data - mean) / std
        matrix[i] = norm_pattern
    # print(matrix)
    # print(matrix.transpose())
    data = X_data[1, 0]
    mean = np.mean(data)
    std = np.std(data)
    norm_pattern = (data - mean) / std
    data1 = X_data[4, 0]
    mean1 = np.mean(data1)
    std1 = np.std(data1)
    norm_pattern1 = (data1 - mean1) / std1
    rng = np.random.default_rng()
    serie = rng.normal(0, 0.03, size=31)
    # np.set_printoptions(precision=4)
    # print(f'Gaussian Noise: {serie.round(decimals = 4)}')
    # print(f'Gaussian Noise: {serie}')

    bar_coordinates = linalg.lstsq(matrix.transpose(), 1/2*norm_pattern + 1/2*norm_pattern1)[0]
    print(f'Barycentric coordinates: {bar_coordinates.round(decimals = 4)}')
    print(bar_coordinates.sum())
    bar_coordinates1 = linalg.lstsq(matrix.transpose(), 1 / 2 * norm_pattern + 1 / 2 * norm_pattern1 + serie)[0]
    print(f'Barycentric coordinates with noise: {bar_coordinates1.round(decimals = 5)}')
    print(bar_coordinates1.sum())
    X_syn, Y_syn = load_data('synthetic_data_v32')
    p1 = Y_syn == 1
    p1 = p1.reshape(-1)
    # print(p1)
    # print(X_syn)
    # print(Y_syn)
    X_new = X_syn[p1]
    X_new = X_new[0].reshape(-1)
    X_new = X_new.numpy()

    bar_coordinates = linalg.lstsq(matrix.transpose(), X_new)[0]
    print(bar_coordinates)
    print(bar_coordinates.sum())
    # print(X_new)
    # plt.plot(np.arange(0, 31, 1, dtype=int), X_new[100].reshape(-1))
    # plt.title('Pattern')
    # plt.show()
    # time.sleep(5)




if __name__ == '__main__':
    list_filename = ['head_shoulders', 'cup_handle', 'triangle_ascending']
    # list_index = [[0, 1, 4, 6, 7, 8, 9, 10, 12], [0, 1, 2, 3, 4, 5, 6, 7, 8, 10],
    #               [0, 4, 5, 6, 9]]
    data_list = []
    label_list =[]
    for pattern_name in list_filename:
        for i in range(5):
    #     print(str(i))
    #     barycentric_coordinates('head_shoulders', list_index[0])
    # recognition_algorithm('head_shoulders', list_index[0])
            pattern, label = barycentric_coordinates(pattern_name)
            data_list.append(pattern)
            label_list.append(label)
    # print(data_list)
    # print(label_list)
    syn_data = torch.stack(data_list)
    syn_label = torch.stack(label_list)
    print(syn_data.shape)
    print(syn_label.shape)
    with open('data3patterns_v2.pt', 'wb') as f:
        torch.save((torch.Tensor(syn_data), torch.Tensor(syn_label)), f)

    # list_filename = ['wedge_rising', 'head_shoulders', 'cup_handle', 'triangle_ascending', 'eve_adam']
    # list_index = [[1,2,3,5,6,8,9,11,12], [0,1,4,6,7,8,9,10,12],[0,1,2,3,4,5,6,7,8,10],[0,4,5,6,9],[2,4]]
    # list_homotopy_Xdata = []
    # list_Ydata = []
    # for index, filename in zip(list_index, list_filename):
    #     # filename = 'head_shoulders'
    #     X_data, Y_data = load_data(filename + '_v1')
    #     # x = torch.linspace(0, 30, steps=31)
    #     # num = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, -1]
    #     # for i in range(0, X_data.shape[0], 1):
    #     #     plt.plot(x, X_data[i].reshape(-1))
    #     #     plt.title(filename + ' Pattern ' + str(i))
    #     #     plt.show()
    #     #     time.sleep(15)
    # #
    #     print(X_data)
    #     filename2 = 'homotopy' + filename + '_v3'
    #     homotopy(X_data[index], filename2)
    #     data = load_data(filename2)
    #     valueY = Y_data[0, 0, 0]
    #     Y_label = torch.full((data.shape[0], 1, 1), valueY)
    #     list_Ydata.append(Y_label)
    #     print(Y_label)
    #     print(data)
    #     list_homotopy_Xdata.append(data)
    #
    # X_all_data = torch.cat(list_homotopy_Xdata)
    # Y_all_data = torch.cat(list_Ydata)
    # with open('all_data_v3.pt', 'wb') as f:
    #     torch.save((X_all_data, Y_all_data), f)
    #
    #
    # X1, Y1 = load_data('all_data_v3')
    # print(X1.shape)
    # print(Y1.shape)
    #
    # x = torch.linspace(0, 30, steps=31)
    # # num = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, -1]
    # for i in range(0,X1.shape[0],250):
    #     plt.plot(x, X1[i].reshape(-1))
    #     plt.title('Pattern ' + str(i))
    #     plt.show()
    #     time.sleep(3)
    #

    # filename = 'pattern_templates_v3'
    # pattern_templates(filename)
    # templates = load_data(filename)
    # # print(templates)
    # # n = templates.shape[0]
    # # x = torch.linspace(0, 6, steps=7)
    # # for i in range(n):
    # #     plt.plot(x, templates[i].reshape(-1))
    # #     plt.show()
    # # pt = torch.Tensor([[[1, 4, 2, 6, 2, 4, 1]]])
    # # output1 = time_scaling(pt)
    # # print(output1)
    # # output2 = time_warping(output1)
    # # print(output2)
    # # output3 = interpolation(output2)
    # # print(output3)
    # # output4 = noise_adding(output3)
    # # print(output4)
    #
    # filename2 = 'synthetic_data_v32'
    # data_generator(templates, 50000, filename2)
    # X_data, Y_data = load_data(filename2)
    # # print(X_data[25], Y_data[25])
    # # X_test = X_data
    #
    # # one_data = X_data[0,0]
    # # h = transform_data(one_data)
    # # print(h.shape)
    # # print(h)
    # n = 5
    # for i in range(n, n + 50):
    #     x = torch.linspace(0, 30, steps=31)
    #     plt.plot(x, X_data[i, 0].reshape(-1))
    #     # plt.plot(x, X_data[25,0,1].reshape(-1))
    #     # plt.plot(x, X_data[25,0,2].reshape(-1))
    #     # plt.plot(x, X_data[25,0,3].reshape(-1))
    #     plt.show()
    #     time.sleep(5)
    # # for n in range(0,10):
    # #     X_data = X_test[n, 0]
    # # #     b = Y_test[n]
    # # #
    # # # #     # x_date = pd.DataFrame(x.numpy())
    # #     dates = pd.date_range(start='1/1/2023', end='1/31/2023')
    # #     fig = go.Figure(data=[go.Candlestick(x=dates,
    # #                                          open=X_data[0],
    # #                                          high=X_data[1],
    # #                                          low=X_data[2],
    # #                                          close=X_data[3])])
    # #     fig.show()
    #
    # # inputs = load_data()
    # # print(inputs)
    # # print(inputs[5].shape)
    # # for i in range(6):
    # #     y = inputs[i][0]
    # #     n = y.shape[2]
    # #     x = torch.linspace(0, n - 1, steps=n)
    # #     # print(x)
    # #     plt.plot(x, y[0, 0, :])
    # #     plt.show()
    #
    # # templates = load_data('pattern_templates_v3')
    # # x = torch.linspace(0, 30, steps=31)
    # # list_patterns = ['wedge_rising', 'head_and_shoulders', 'cup_with_handle', 'triangle_ascending', 'triple_tops', 'double_bottoms_eve_adam']
    # # for i in range(templates.shape[0]):
    # #     pt_length = templates.shape[-1]
    # #     out1, out2 = time_scaling(pt_length, stride=5)
    # #     out3 = time_warping(out1, out2)
    # #     pattern_template = templates[i]
    # #     output3 = interpolation(pattern_template, out3)
    # #     output3 = normalization(output3)
    # #     plt.plot(x, output3[0].reshape(-1))
    # #     plt.title(list_patterns[i])
    # #     plt.show()
    # #     time.sleep(1)
    # #     output4_1 = noise_adding(output3)
    # #     output4_1 = normalization(output4_1)
    # #     output4_2 = noise_adding(output3)
    # #     output4_2 = normalization(output4_2)
    # #     output4_3 = noise_adding(output3)
    # #     output4_3 = normalization(output4_3)
    # #     output4_4 = noise_adding(output3)
    # #     output4_4 = normalization(output4_4)
    # #     output4_5 = noise_adding(output3)
    # #     output4_5 = normalization(output4_5)
    # #     for j in range(1,6):
    # #         plt.plot(x, eval('output4_' + str(j))[0].reshape(-1))
    # #         plt.title(list_patterns[i] + ' Gaussian noise ' + str(j))
    # #         plt.show()
    # #         time.sleep(3)
