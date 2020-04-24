# encoding: utf-8
# Author: 周知瑞
# Mail: evilpsycho42@gmail.com

from torch import nn
from torch.nn import functional as F
from torch.nn import LayerNorm

class CausalConv1d(nn.Conv1d):
    """1D Causal Convolution Layer

    Args:
        inputs, Tensor(batch, input_unit(kernel_size), sequence)

    Returns:
        Tensor(batch, output_unit(kernel_size), sequence)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', batch_norm=False, dropout=0., activation=None):
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
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, inputs):
        result = super(CausalConv1d, self).forward(inputs)[:, :, :-self.shift]
        if self.batch_norm is not None:
            result = self.batch_norm(result)
        if self.activation is not None:
            result = self.activation(result)
        result = F.dropout(result)
        return result
