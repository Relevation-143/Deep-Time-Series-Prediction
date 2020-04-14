# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/4/14 16:26
"""

import torch
from torch.nn import BatchNorm1d, BatchNorm2d, Dropout, Conv1d

@torch.no_grad()
def wavenet_init(m):
    if not isinstance(m, (BatchNorm1d, BatchNorm2d, Dropout)):
        torch.nn.init.xavier_normal_(m)
