# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/5/7 9:22
"""
import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class Align(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class Attention(nn.Module):
    """
    Take in model size and number of heads.
    general attention
    """

    def __init__(self, heads, attn_size, query_size, key_size, value_size, dropout=0.1):
        super().__init__()
        assert attn_size % heads == 0

        # We assume d_v always equals d_k
        self.d_k = attn_size // heads
        self.h = heads

        self.linear_layers = nn.ModuleList([nn.Linear(s, attn_size) for s in [query_size, key_size, value_size]])
        self.output_linear = nn.Linear(attn_size, attn_size)
        self.align = Align()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """inputs shape (B, S, N)"""

        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.align(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x), attn


# class Align(nn.Module):
#
#     def __init__(self, score='dot', size=None):
#         super().__init__()
#         self.score = score
#         self.size = size
#         if score == 'concat':
#             self.W = nn.Parameter(torch.zeros(size * 2, size), requires_grad=True)
#             self.V = nn.Parameter(torch.zeros(size, 1), requires_grad=True)
#             nn.init.xavier_normal_(self.W)
#             nn.init.xavier_normal_(self.V)
#         elif score == 'general':
#             self.W = nn.Parameter(torch.zeros(size, size), requires_grad=True)
#             nn.init.xavier_normal_(self.W)
#         elif score == 'dot':
#             pass
#         else:
#             raise ValueError
#
#     def forward(self, q, k, v):
#         """
#
#         Args:
#             q: B x S x N
#             k: B x S x N
#             v: B x S x N
#
#         Returns:
#
#         """
#         lens = k.shape[1]
#         if self.score == 'concat':
#             align = torch.tanh(torch.cat([q.repeat(1, lens, 1), k], dim=2) @ self.W) @ self.V
#         elif self.score == 'general':
#             align = k @ (q @ self.W).transpose(1, 2)
#         elif self.score == 'dot':
#             align = k @ q.transpose(1, 2)
#         else:
#             raise ValueError
#         weight = torch.softmax(align / math.sqrt(lens), 1)  # B x S x 1
#         values = (v.transpose(1, 2) @ weight).transpose(1, 2)  # B x 1 x N
#         return values, weight


if __name__ == "__main__":

    q = torch.rand(4, 1, 30)
    k = torch.rand(4, 12, 9)

    attn = RNNAttention(3, 12, 30, 9, 9, dropout=0.)
    values, weights = attn(q, k, k)
