# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/4/13 9:53
"""
import torch
from torch import nn


class Embeddings(nn.Module):

    def __init__(self, embeds_dim, seq_last=True):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(i, o) for i, o in embeds_dim]) if embeds_dim else None
        self.seq_last = seq_last

    def forward(self, inputs):
        if self.seq_last:
            embed = torch.cat(
                [self.embeddings[d](inputs[:, d]).transpose(1, 2) for d in range(inputs.shape[1])], dim=1)
        else:
            embed = torch.cat(
                [self.embeddings[d](inputs[:, :, d]) for d in range(inputs.shape[2])], dim=2)
        return embed


class Inputs(nn.Module):

    def __init__(self, num_features=None, cat_features=None, batch_norm=False,
                 activation=None, dropout=0., seq_last=True):
        super().__init__()
        self.num_features = num_features
        self.cat_features = cat_features
        self.num_dim = 0 if num_features is None else num_features
        self.cat_dim = 0 if cat_features is None else sum([i for _, i in cat_features])
        self.output_dim = self.num_dim + self.cat_dim

        self.embeddings = Embeddings(cat_features, seq_last) if cat_features else None
        self.batch_norm = nn.BatchNorm1d(self.output_dim) if batch_norm else None
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.seq_last = seq_last

    def forward(self, num=None, cat=None):
        if self.num_features is None and self.cat_features is None:
            return None
        concat = []
        if self.num_features is not None:
            concat.append(num)
        if self.cat_features is not None:
            concat.append(self.embeddings(cat))
        concat = torch.cat(concat, dim=1 if self.seq_last else 2)
        return concat


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
