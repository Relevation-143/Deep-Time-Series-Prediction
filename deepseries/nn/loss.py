# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/4/16 10:58
"""
import torch
from torch.nn import functional as F


class MSELoss:

    def __call__(self, input, target, weight=None):
        if weight is None:
            return F.mse_loss(input, target, reduction='mean')
        else:
            ret = F.mse_loss(input, target, reduction='none')
            loss = torch.mean(ret * weight.expand_as(ret))
            return loss


class RMSELoss:

    def __call__(self, input, target, weight=None):
        if weight is None:
            return torch.sqrt(F.mse_loss(input, target, reduction='mean'))
        else:
            ret = F.mse_loss(input, target, reduction='none')
            loss = torch.sqrt(torch.mean(ret * weight.expand_as(ret)) * 1e-6)
            return loss
