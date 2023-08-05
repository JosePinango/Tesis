import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Any, Callable, List, Optional, Tuple
import numpy as np
import pickle
import matplotlib.pyplot as plt


def time_scaling(pattern_template: Tensor, stride: int = 4) -> Tensor:
    pattern_length = (pattern_template.shape[2] - 1) * stride + 1
    pattern = torch.zeros(1, 1, pattern_length)
    pattern[:, :, ::stride] = pattern_template
    return pattern


def time_warping(pattern: Tensor, stride: int = 4) -> Tensor:
    """Documentacion
    examples
    fff"""
    aux_index = 0
    n = pattern.shape[2]
    for i in range(0, n, stride):
        aux_index = np.random.choice(range(aux_index + 1, i + stride))
        if aux_index != i and aux_index < n:
            pattern[:, :, aux_index] = pattern[:, :, i]
            if i != 0 and i != n - 1:
                pattern[:, :, i] = 0
    return pattern


def interpolation(pattern: Tensor) -> Tensor:
    index = torch.where(pattern[:, :, :] > 0)[2]
    interp_list = [pattern[:, :, 0:1]]
    for i, j in zip(index, index[1:]):
        # interp = F.interpolate(pattern[:, :, [i, j]], size=j - i + 1, mode='linear')
        # interp = torch.tensor(np.linspace(pattern[:, :, i].item(), pattern[:, :, j].item(), j - i + 1)).reshape(1, 1, -1)
        interp = torch.linspace(pattern[:, :, i].item(), pattern[:, :, j].item(), steps=j - i + 1).reshape(1, 1, -1)
        interp_list.append(interp[:, :, 1:])
    return torch.cat(interp_list, dim=-1)


def noise_adding(pattern: Tensor) -> Tensor:
    for i in range(pattern.shape[2]):
        if np.random.choice(np.arange(0, 1.1, 0.1)) < 0.6:
            diff = pattern[:, :, i]
            if i < pattern.shape[2] - 1:
                diff = pattern[:, :, i + 1] - pattern[:, :, i]
            pattern[:, :, i:i + 1] += np.random.choice(np.arange(-0.3, 0.35, 0.5)) * diff
    return pattern


def data_generator(n: int) -> None:
    wedge_rising = torch.Tensor([[[5, 1, 5.5, 2.5, 6, 4, 6.5]]])
    head_and_shoulders = torch.Tensor([[[1, 4, 2, 6, 2, 4, 1]]])
    cup_with_handle = torch.Tensor([[[5, 3, 2, 3, 5, 4, 5]]])
    triangle_ascending = torch.Tensor([[[6, 2, 6, 3, 6, 4, 6]]])
    triple_tops = torch.Tensor([[[2, 6, 2, 6, 2, 6, 2]]])
    double_bottoms = torch.Tensor([[[6, 3, 2, 3, 6, 2, 6]]])
    pattern_templates = ['wedge_rising', 'head_and_shoulders', 'cup_with_handle', 'triangle_ascending', 'triple_tops', 'double_bottoms']

    for template in pattern_templates:
        list_tensors = []
        with open(template + '.pt', 'wb') as f:
            for i in range(n):
                output1 = time_scaling(eval(template))
                output2 = time_warping(output1)
                output3 = interpolation(output2)
                output4 = noise_adding(output3)
                list_tensors.append(output4)
            torch.save(list_tensors, f)


def load_data() -> List[Tensor]:
    pattern_templates = ['wedge_rising', 'head_and_shoulders', 'cup_with_handle', 'triangle_ascending', 'triple_tops',
                         'double_bottoms']
    inputs = []
    for template in pattern_templates:
        with open(template + '.pt', 'rb') as f:
            inputs.append(torch.load(f))
    return inputs


if __name__ == '__main__':
    # pt = torch.Tensor([[[1, 4, 2, 6, 2, 4, 1]]])
    # output1 = time_scaling(pt)
    # print(output1)
    # output2 = time_warping(output1)
    # print(output2)
    # output3 = interpolation(output2)
    # print(output3)
    # output4 = noise_adding(output3)
    # print(output4)
    # data_generator(100)
    inputs = load_data()
    # print(inputs)
    for i in range(6):
        y = inputs[i][0]
        n = y.shape[2]
        x = torch.linspace(0, n - 1, steps=n)
        # print(x)
        plt.plot(x, y[0, 0, :])
        plt.show()
