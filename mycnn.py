import torch
from torch import nn, Tensor
import numpy as np
from typing import Any, Callable, List, Optional, Tuple
import torch.nn.functional as F

class CauDilConv1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_width: int, dilation: int,
                 causal_padding: int = None, stride: int = 1) -> None:
        super(CauDilConv1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_width = kernel_width
        self.kernel_width_aux = (kernel_width - 1) * dilation + 1
        self.dilation = dilation
        self.causal_padding = causal_padding
        self.stride = stride
        self.kernel = nn.Linear(kernel_width * in_channels, out_channels, bias=False)
        # weights = torch.ones(out_channels, kernel_width * in_channels)
        # weights[0] = -2
        # weights[5] = -3
        # self.kernel.weight = nn.Parameter(weights, requires_grad=False)

    def convolution(self, x: Tensor) -> Tensor:
        if self.causal_padding is not None:
            x = F.pad(x, (self.causal_padding, 0))
            # x2
            # print(f'eeeeeeeeeeee {x2}')
            # l_padding = torch.zeros(x.shape[0], self.causal_padding)
            # x = torch.cat([l_padding, x], dim=-1)
            # x1
            # print(f'sssssssssssssss {x}')
        conv1d_output = [self.kernel(x[:, i - self.kernel_width_aux:i:self.dilation].reshape(1, self.kernel_width * self.in_channels)) for i in range(self.kernel_width_aux, x.shape[-1] + 1, self.stride)]
        # return torch.stack(conv1d_output, dim=1).transpose(1, 2)
        return torch.cat(conv1d_output,dim=0).transpose(0,1)

    def forward(self, x: Tensor) -> Tensor:
        if len(x.shape)==3:
            f = torch.vmap(self.convolution)
            return f(x)
        return self.convolution(x).reshape(1,1,-1)



class Inception(nn.Module):
    def __init__(self, in_channels: int, pool_features: int, conv1D: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__()
        if conv1D is None:
            conv1D = CauDilConv1D
        self.mod7_1 = conv1D(in_channels, 6*2, 3, 1)
        self.mod7_2 = nn.ReLU()
        self.mod7_3 = conv1D(6*2, pool_features, 3, 2)
        self.mod7_4 = nn.ReLU()

        self.mod10_1 = conv1D(in_channels, 6*2, 3, 1)
        self.mod10_2 = nn.ReLU()
        self.mod10_3 = conv1D(6*2, 6*3, 3, 2)
        self.mod10_4 = nn.ReLU()
        self.mod10_5 = conv1D(6*3, pool_features, 2, 4, causal_padding=4)
        self.mod10_6 = nn.ReLU()

        self.mod15_1 = conv1D(in_channels, 6*2, 3, 1)
        self.mod15_2 = nn.ReLU()
        self.mod15_3 = conv1D(6*2, 6*3, 3, 2)
        self.mod15_4 = nn.ReLU()
        self.mod15_5 = conv1D(6*3, pool_features, 3, 4, causal_padding=8)
        self.mod15_6 = nn.ReLU()

    def _forward(self, x: Tensor) -> List[Tensor]:
        mod7 = self.mod7_1(x)

        mod7 = self.mod7_2(mod7)
        # print(mod7)
        mod7 = self.mod7_3(mod7)
        mod7 = self.mod7_4(mod7)
        # print(mod7)

        mod10 = self.mod10_1(x)
        mod10 = self.mod10_2(mod10)
        mod10 = self.mod10_3(mod10)
        mod10 = self.mod10_4(mod10)
        mod10 = self.mod10_5(mod10)
        mod10 = self.mod10_6(mod10)
        # print(mod10)

        mod15 = self.mod15_1(x)
        mod15 = self.mod15_2(mod15)
        mod15 = self.mod15_3(mod15)
        mod15 = self.mod15_4(mod15)
        mod15 = self.mod15_5(mod15)
        mod15 = self.mod15_6(mod15)
        # print(mod15)

        outputs = [mod7, mod10, mod15]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        # print(torch.cat(outputs, 1).shape)
        return torch.cat(outputs, 1)


class CNN(nn.Module):
    def __init__(self, in_channels: int, pool_features: int) -> None:
        super().__init__()
        # Inception_layer = torch.vmap(Inception(in_channels, pool_features))

        self.model = nn.Sequential(
            # Inception_layer,
            # N x 1 X 25
            Inception(in_channels, pool_features),
            # nn.Conv1d(in_channels, pool_features, 7),   #####solo para probar
            # N x 3*pool_features x 19
            nn.MaxPool1d(3),
            # N x 3*pool_features x 6
            nn.Conv1d(3 * pool_features, pool_features, 1),
            # N x pool_features x 6
            # nn.Flatten(start_dim=0, end_dim=-1),
            nn.Flatten(),
            # N x 3*pool_features*7
            nn.Linear(1 * pool_features * 6, pool_features * 2),
            nn.Dropout(p=0.5),
            nn.Linear(pool_features * 2, 6),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.model(x)


def main():
    # input = torch.tensor([[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.]])
    input =  torch.randn(4, 1, 15)
    model = CNN(1, 6)
    # for name, param in model.named_parameters():
    #     print(name, param)

    # model_vectorized = torch.vmap(model)
    # output = model_vectorized(input)
    output = model(input)
    # print(output)
    return output


if __name__ == '__main__':
    print(main())
    # conv =CauDilConv1D(1, 6, 3, 1, causal_padding=0)
    # input = torch.tensor([[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.]])
    # print(conv(input))
