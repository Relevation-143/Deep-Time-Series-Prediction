# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/4/13 9:53
"""
import torch
from torch import nn


class Dense(nn.Module):

    def __init__(self, in_features, out_features, bias=True, dropout=0.1, nonlinearity=None):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.nonlinearity = getattr(nn, nonlinearity)() if nonlinearity else None
        self.reset_parameters()

    def forward(self, x):
        x = self.dropout(self.fc(x))
        if self.nonlinearity is not None:
            x = self.nonlinearity(x)
        return x

    def reset_parameters(self):
        nn.init.xavier_normal_(self.fc.weight)


class Embeddings(nn.Module):

    """
    References:
      embedding weight initialize https://arxiv.org/pdf/1711.09160.pdf
    """

    def __init__(self, embeds_size=None, seq_last=False):
        super().__init__()
        self.embeds_size = embeds_size
        self.embeddings = nn.ModuleList([nn.Embedding(i, o) for i, o in embeds_size]) if embeds_size else None
        self.seq_last = seq_last
        self.output_size = sum([i for _, i in embeds_size]) if embeds_size else 0

    def forward(self, inputs):
        if inputs is None:
            return None
        if self.seq_last:
            embed = torch.cat(
                [self.embeddings[d](inputs[:, d]).transpose(1, 2) for d in range(inputs.shape[1])], dim=1)
        else:
            embed = torch.cat(
                [self.embeddings[d](inputs[:, :, d]) for d in range(inputs.shape[2])], dim=2)
        return embed

    def reset_parameters(self):
        if self.embeds_size:
            for layer in self.embeddings:
                nn.init.xavier_normal_(layer.weight)


class Concat(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, *x):
        return torch.cat([i for i in x if i is not None], self.dim)


class Inputs(nn.Module):

    def __init__(self, num_size=None, cat_size=None, seq_last=False, dropout=0.):
        super().__init__()
        self.num_size = num_size
        self.cat_size = cat_size
        self.embeddings = Embeddings(cat_size, seq_last) if cat_size else None
        self.output_size = ((0 if num_size is None else num_size)
                            + 0 if cat_size is None else self.embeddings.output_size)
        self.dropout = nn.Dropout(dropout)
        self.seq_last = seq_last

    def forward(self, x, num=None, cat=None):
        if self.num_size is None and self.cat_size is None:
            return x
        concat = [x]
        if self.num_size is not None:
            concat.append(num)
        if self.cat_size is not None:
            concat.append(self.embeddings(cat))
        concat = torch.cat(concat, dim=1 if self.seq_last else 2)
        return self.dropout(concat)


class TimeDistributedDense1d(nn.Module):
    """
    input shape (batch, in_channels, seqs), returns (batch, out_channels, seqs)

    Args:
        in_dim: in_channels.
        out_dim: out_channels.
        bias: (bool), default True.
        activation: (callable)
        batch_norm: (bool), default false
        dropout: (float), default zero
    """

    def __init__(self, in_dim, out_dim, bias=True, activation=None,
                 batch_norm=False, dropout=0., seq_last=True):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias)
        self.activation = activation
        self.batch_norm = nn.BatchNorm1d(out_dim) if batch_norm else None
        self.dropout = nn.Dropout(dropout)
        self.seq_last = seq_last

    def forward(self, inputs):
        if self.seq_last:
            x = self.fc(inputs.transpose(1, 2)).transpose(1, 2)
        else:
            x = self.fc(inputs)
        x = self.batch_norm(x) if self.batch_norm is not None else x
        x = self.activation(x) if self.activation is not None else x
        x = self.dropout(x)
        return x
