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
            loss = torch.mean(ret * weight)
            return loss


class RMSELoss:

    def __call__(self, input, target, weight=None):
        if weight is None:
            return torch.sqrt(F.mse_loss(input, target, reduction='mean'))
        else:
            ret = F.mse_loss(input, target, reduction='none')
            loss = torch.sqrt(torch.mean(ret * weight))
            return loss


class RNNStabilityLoss:
    """

    RNN outputs -> loss

    References:
        https://arxiv.org/pdf/1511.08400.pdf
    """

    def __init__(self, beta=1e-5):
        self.beta = beta

    def __call__(self, rnn_output):
        if self.beta == .0:
            return .0
        l2 = torch.sqrt(torch.sum(torch.pow(rnn_output, 2), dim=-1))
        return self.beta * torch.mean(torch.pow(l2[1:] - l2[:-1], 2))


class RNNActivationLoss:

    """
    RNN outputs -> loss
    """

    def __init__(self, beta=1e-5):
        self.beta = beta

    def __call__(self, rnn_output):
        if self.beta == .0:
            return .0
        return torch.sum(torch.norm(rnn_output)) * self.beta
