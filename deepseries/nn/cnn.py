# encoding: utf-8
# Author: 周知瑞
# Mail: evilpsycho42@gmail.com

from torch import nn
from torch.nn import functional as F
import torch


class CausalConv1d(nn.Conv1d):
    """1D Causal Convolution Layer

    Args:
        inputs, Tensor(batch, input_unit(kernel_size), sequence)

    Returns:
        Tensor(batch, output_unit(kernel_size), sequence)
    """

    def __init__(self, in_channels, out_channels, dilation=1, bias=True,
                 padding_mode='zeros'):
        kernel_size = 2
        self.shift = (kernel_size - 1) * dilation
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=self.shift,
            dilation=dilation,
            groups=1,
            bias=bias,
            padding_mode=padding_mode)

    def forward(self, inputs):
        return super(CausalConv1d, self).forward(inputs)[:, :, :-self.shift]


class WaveLayer(nn.Module):

    def __init__(self, residual_channels, skip_channels, dilation):
        super(WaveLayer, self).__init__()
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.dilation = dilation
        self.conv_dilation = CausalConv1d


class WaveNetBlockV1(nn.Module):
    """
        References:
            5th solution for KAGGLE web traffic competition
            https://github.com/vincentherrmann/pytorch-wavenet/blob/master/wavenet_model.py
        Notes:
                #            |----------------------------------------|     *residual*
                #            |                                        |
                #            |    |-- conv -- tanh --|                |
                # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
                #                 |-- conv -- sigm --|     |
                #                                         1x1
                #                                          |
                # ---------------------------------------> + ------------->	*skip*
        """
    def __init__(self, residual_channels, skip_channels, dilation):
        super().__init__()
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.dilation = dilation
        self.dilation_conv = CausalConv1d(residual_channels, 2*residual_channels, dilation=dilation, kernel_size=2)
        self.conv1x1 = nn.Conv1d(residual_channels, residual_channels+skip_channels, kernel_size=1)

    def forward(self, x):
        conv_dilation = self.dilation_conv(x)
        conv_filter, conv_gate = torch.split(conv_dilation, self.residual_channels, dim=1)
        output = torch.tanh(conv_filter) * torch.sigmoid(conv_gate)
        output = self.conv1x1(output)
        skip, residual = torch.split(output, [self.skip_channels, self.residual_channels], dim=1)
        next_input = x + residual
        return next_input, skip

    def fast_forward(self, input):
        input_len = input.shape[2]
        if input_len >= self.dilation+1:
            input_short = input[:, :, [input_len-(self.dilation+1), -1]]
        else:
            last = input[:, :, [-1]]
            input_short = torch.cat([torch.zeros_like(last), last], dim=2)
        dilation = torch.conv1d(input_short, self.dilation_conv.weight, self.dilation_conv.bias)
        conv_filter, conv_gate = torch.split(dilation, self.residual_channels, dim=1)
        output = torch.tanh(conv_filter) * torch.sigmoid(conv_gate)
        output = self.conv1x1(output)
        skip, residual = torch.split(output, [self.skip_channels, self.residual_channels], dim=1)
        next_input = input + residual
        return next_input, skip

