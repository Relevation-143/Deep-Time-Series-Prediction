# encoding: utf-8
# Author: 周知瑞
# Mail: evilpsycho42@gmail.com

import torch
from torch import nn
from torch.nn import functional as F


class TimeDistributedDense1D(nn.Module):
    """
    input shape (batch, in_channels, seqs), returns (batch, out_channels, seqs)

    Args:
        in_features: in_channels.
        out_features: out_channels.
        bias: (bool), default True.
        activation: (callable)
        batch_norm: (bool), default false, if true then add BatchNorm1d layer
        dropout: (float)
    """

    def __init__(self, in_features, out_features, bias=True, activation=None,
                 batch_norm=False, dropout=None):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias)
        self.activation = activation
        self.batch_norm = nn.BatchNorm1d(out_features) if batch_norm else None
        self.dropout = dropout

    def forward(self, inputs):
        x = self.fc(inputs.transpose(1, 2)).transpose(1, 2)
        x = self.batch_norm(x) if self.batch_norm else x
        x = F.dropout(x, self.dropout) if self.dropout else x
        return x


class CausalConv1d(nn.Conv1d):
    """1D Causal Convolution Layer

    Args:
        inputs, Tensor(batch, input_unit(kernel_size), sequence)

    Returns:
        Tensor(batch, output_unit(kernel_size), sequence)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 dilation=1, groups=1, bias=True, padding_mode='zeros', batch_norm=False):
        self.shift = (kernel_size - 1) * dilation
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.shift,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode)
        self.batch_norm = nn.BatchNorm1d(out_channels) if batch_norm else None

    def forward(self, inputs):
        result = super(CausalConv1d, self).forward(inputs)
        if self.padding != 0:
            return result[:, :, :-self.shift]
        if self.batch_norm is not None:
            return self.batch_norm(result)
        else:
            return result
