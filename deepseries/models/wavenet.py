# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/4/22 16:13
"""
import torch
import torch.nn as nn
from deepseries.nn.cnn import CausalConv1d
from deepseries.nn.comm import Inputs


class WaveEncoderV2(nn.Module):

    def __init__(self, series_dim, enc_num=None, enc_cat=None, residual_channels=32, n_blocks=3, n_layers=8, dropout=0.):
        super().__init__()
        self.enc_input = Inputs(enc_num, enc_cat, dropout=dropout, seq_last=True)
        self.conv_input = nn.Conv1d(in_channels=self.enc_input.output_dim+series_dim,
                                    out_channels=residual_channels, kernel_size=1)


class WaveNetV2(nn.Module):
    """
            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*
    """

    def __init__(self, series_dim, enc_num=None, enc_cat=None, dec_num=None, dec_cat=None, n_blocks=3, n_layers=8, dropout=0.):
        super().__init__()
        self.enc_input = Inputs(enc_num, enc_cat, dropout=dropout, seq_last=True)
        self.dec_input = Inputs(dec_num, dec_cat, dropout=dropout, seq_last=True)
