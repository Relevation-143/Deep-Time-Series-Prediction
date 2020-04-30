# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/4/30 14:58
"""
import torch
import torch.nn as nn


@torch.no_grad
def init_weight(x):
    if isinstance(x, nn.Embedding)