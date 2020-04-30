# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/4/22 16:12
"""
import torch
import torch.nn as nn
from deepseries.nn.comm import Inputs


class RNN2RNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.enc_inputs =