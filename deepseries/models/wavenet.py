# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/4/22 16:13
"""
import torch
import torch.nn as nn
from deepseries.nn.cnn import WaveNetBlockV1
from deepseries.nn.comm import Inputs, Concat


class WaveNet(nn.Module):

    def __init__(self, target_size, num_size=None, cat_size=None,
                 residual_channels=32, num_blocks=8, num_layers=3, dropout=0.0):
        super().__init__()
        self.input = Inputs(num_size, cat_size, seq_last=True, dropout=dropout)
        self.concat = Concat(dim=1)
        self.conv_h = nn.Conv1d(self.input.output_size + target_size, residual_channels, kernel_size=1)
        self.conv_c = nn.Conv1d(self.input.output_size + target_size, residual_channels, kernel_size=1)
        self.wave_blocks = nn.ModuleList(
            [WaveNetBlockV1(residual_channels, 2**block) for layer in range(num_layers) for block in range(num_blocks)]
        )
