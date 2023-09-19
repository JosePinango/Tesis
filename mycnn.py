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
            # x = x.type(torch.float64)
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
        self.mod7_1 = conv1D(in_channels, 7*2, 3, 1)
        self.mod7_2 = nn.ReLU()
        self.mod7_3 = conv1D(7*2, pool_features, 3, 2)
        self.mod7_4 = nn.ReLU()

        self.mod10_1 = conv1D(in_channels, 7*2, 3, 1)
        self.mod10_2 = nn.ReLU()
        self.mod10_3 = conv1D(7*2, 7*3, 3, 2)
        self.mod10_4 = nn.ReLU()
        self.mod10_5 = conv1D(7*3, pool_features, 2, 4, causal_padding=4)
        self.mod10_6 = nn.ReLU()

        self.mod15_1 = conv1D(in_channels, 7*2, 3, 1)
        self.mod15_2 = nn.ReLU()
        self.mod15_3 = conv1D(7*2, 7*3, 3, 2)
        self.mod15_4 = nn.ReLU()
        self.mod15_5 = conv1D(7*3, pool_features, 3, 4, causal_padding=8)
        self.mod15_6 = nn.ReLU()

        # self.mod19_1 = conv1D(in_channels, 6 * 2, 3, 1)
        # self.mod19_2 = nn.ReLU()
        # self.mod19_3 = conv1D(6 * 2, 6 * 3, 3, 2)
        # self.mod19_4 = nn.ReLU()
        # self.mod19_5 = conv1D(6 * 3, pool_features, 4, 4, causal_padding=12)
        # self.mod19_6 = nn.ReLU()

    def _forward(self, x: Tensor) -> List[Tensor]:
        mod7 = self.mod7_1(x)

        mod7 = self.mod7_2(mod7)
        # print(mod7)
        mod7 = self.mod7_3(mod7)
        mod7 = self.mod7_4(mod7)
        # print(mod7.shape)

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

        # mod19 = self.mod19_1(x)
        # mod19 = self.mod19_2(mod19)
        # mod19 = self.mod19_3(mod19)
        # mod19 = self.mod19_4(mod19)
        # mod19 = self.mod19_5(mod19)
        # mod19 = self.mod19_6(mod19)

        # outputs = [mod7, mod10, mod15, mod19]
        outputs = [mod7, mod10, mod15]
        # print(outputs[2].shape)
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
            nn.Linear(pool_features * 8, pool_features * 2),
            nn.Dropout(p=0.5),
            nn.Linear(pool_features * 2, 7),
            nn.Softmax(dim=-1)
        )

    def _sliding_window(self, x):
        position = 0
        list_out = []
        while position < 33:
            output = self.model(x[:, position:position + 31])
            if torch.argmax(output, dim=1) == 6:
                position += 1
            else:
                position += 31
            list_out.append(output)
        out_aux = torch.stack(list_out)
        return torch.max(out_aux, dim=0).values

    def _forward(self,x):
        f = torch.vmap(self._sliding_window, in_dims=1)
        return f(x)

    def forward(self, x):
        return self.model(x)
        # position = 0
        # list_out = []
        # if x.shape[-1] == 31:
        #     print('size 31')
        #     return self.model(x)
        # else:
        #     print('size > 31')
        #     print(x)
        #     # print(self._forward(x))
        #     # return self._forward(x)
        #     while position < 33:
        #         output = self.model(x[:,:,position:position+31])
        #         # if torch.argmax(output, dim=1) == 6:
        #         position +=1
        #         # else:
        #         #     return output
        #     # return self.model(x[:,position:position+31])
        #             # position +=31
        #         list_out.append(output)
        #     out_aux = torch.stack(list_out)
        #     return torch.max(out_aux, dim=0).values

class CNN_sliding_window(nn.Module):
    def __init__(self, in_channels: int, pool_features: int) -> None:
        super().__init__()
        # self.in_channels = in_channels
        # self.pool_features = pool_features
        self.model = CNN(in_channels, pool_features)

    def forward(self, x):
        # model_aux = CNN(self.in_channels, self.pool_features)
        position = 0
        list_out = []
        if x.shape[-1] == 31:
            print('size 31')
            return self.model(x)
        else:
            print('size > 31')
            print(x)
            # print(self._forward(x))
            # return self._forward(x)
            while position < 33:
                output = self.model(x[:,position:position+31])
                # if torch.argmax(output, dim=1) == 6:
                position +=1
                # else:
                    # return output
            # return self.model(x[:,position:position+31])
            #         position +=31
                list_out.append(output)
            out_aux = torch.stack(list_out)
            return torch.max(out_aux, dim=0).values



def main():
    # input = torch.tensor([[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.]])
    input =  torch.randn(4, 1, 32)
    print(input.dtype)
    model = CNN(1, 7)
    # for name, param in model.named_parameters():
    #     print(name, param)

    # model_vectorized = torch.vmap(model)
    # output = model_vectorized(input)
    # for i in range(25,65):
    #     print(i)
    output = model(input)
    # print(output)
    # position = 0
    # list_out = []
    # if input.shape[-1] == 31:
    #     print('size 31')
    #     return model(input)
    # else:
    #     print('size > 31')
    #     print(input)
    #     # print(self._forward(x))
    #     # return self._forward(x)
    #     while position < 33:
    #         output = model(input[:, position:position + 31])
    #         if torch.argmax(output, dim=1) == 6:
    #             position += 1
    #         else:
    #             # return output
    #             # return self.model(x[:,position:position+31])
    #             position += 31
    #         list_out.append(output)
    #     out_aux = torch.stack(list_out)
    #     return torch.max(out_aux, dim=0).values
    return output


if __name__ == '__main__':
    print(main())
    # conv =CauDilConv1D(1, 6, 3, 1, causal_padding=0)
    # input = torch.tensor([[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.]])
    # print(conv(input))
