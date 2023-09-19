import time

import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Any, Callable, List, Optional, Tuple
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch.nn.functional as F


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
        aux_index = np.random.choice(range(aux_index + 1, i + stride))
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
        if np.random.choice(np.arange(0, 1.1, 0.1)) < 0.6:
            diff = pattern[:, i]
            if i < pattern.shape[-1] - 1:
                diff = pattern[:, i + 1] - pattern[:, i]
            pattern[:, i:i + 1] += np.random.choice(np.arange(-0.3, 0.35, 0.5)) * diff
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
    std, mean = torch.std_mean(data)
    data = torch.cat([torch.tensor(np.random.uniform(low= torch.min(data), high=torch.max(data), size=(1, 31 - data.shape[-1]))), data], dim=-1)
    output = (data - mean) / std
    return output

def normalization64(data: Tensor) -> Tensor:
    std, mean = torch.std_mean(data)
    data = torch.cat([torch.tensor(np.random.uniform(low= torch.min(data), high=torch.max(data), size=(1, 64 - data.shape[-1]))), data], dim=-1)
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
        random_pattern = np.random.choice(np.arange(0,7,1))
        # random_stride = np.random.choice(np.arange(2, 6, 1))
        if random_pattern == 6:
            out = torch.randn(1,31, dtype=torch.float32)
        else:
            random_stride = 5
            out = synthetic_data(pattern_templates[random_pattern], random_stride=random_stride)
        # out = vec_generator(pattern_templates, random_stride=random_stride)
        # print(out)
        # std, mean = torch.std_mean(out, dim=-1)
        # out1 = (out - mean) / std
        # out1 = vec_normalization(out)
        out1 = normalization(out)
        # n = out1.shape[-1]
        # x = torch.linspace(0, n - 1, steps=n)
        # plt.plot(x, out1[0,:])
        # plt.show()
        # print(out1)
        # time.sleep(60)
        out1 = out1.type(torch.float32)
        list_data.append(out1)
        label = torch.tensor(random_pattern).reshape(1,-1)
        list_labels.append(label)
    syn_data = torch.stack(list_data, dim=0)
    print(syn_data.shape)

    syn_label = torch.stack(list_labels, dim=0)
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
    wedge_rising = torch.Tensor([[[5, 1, 5.5, 2.5, 6, 4, 6.5]]])
    head_and_shoulders = torch.Tensor([[[1.5, 5.5, 3.5, 7, 3.5, 5.5, 1.5]]])
    cup_with_handle = torch.Tensor([[[5, 2.5, 2, 2.5, 5, 4, 5]]])
    triangle_ascending = torch.Tensor([[[6, 2, 6, 3, 6, 4, 6]]])
    triple_tops = torch.Tensor([[[1.5, 6.0, 3.5, 6.0, 3.5, 6.0, 1.5]]])
    double_bottoms = torch.Tensor([[[5, 2.5, 2, 2.5, 5, 2, 5]]])
    list_patterns = [wedge_rising, head_and_shoulders, cup_with_handle, triangle_ascending, triple_tops, double_bottoms]
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


if __name__ == '__main__':
    filename = 'pattern_templates'
    pattern_templates(filename)
    templates = load_data(filename)
    # print(templates)
    # n = templates.shape[0]
    # x = torch.linspace(0, 6, steps=7)
    # for i in range(n):
    #     plt.plot(x, templates[i,0,:])
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
    filename2 = 'synthetic_data_v6'
    data_generator(templates, 10000, filename2)
    X_data, Y_data = load_data(filename2)
    print(X_data[25])
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
