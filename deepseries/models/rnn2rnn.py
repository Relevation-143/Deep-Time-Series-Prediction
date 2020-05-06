# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/5/6 10:40
"""
import torch.nn as nn
from deepseries.nn.comm import InputsEXP


class RNNEncoder(nn.Module):

    def __init__(self, series_dim, hidden_dim, compress_dim, activation='SELU',
                 dropout=0.5, num_feat=None, cat_feat=None, rnn_type='gru', n_layers=1):
        super().__init__()
        self.input = InputsEXP(num_feat, cat_feat, seq_last=False)
        self.rnn_type = rnn_type
        self.rnn = getattr(nn, rnn_type)(self.encoder_input.output_dim + series_dim, hidden_dim,
                                         num_layers=n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.compress = nn.Linear(hidden_dim, compress_dim)
        self.activation = getattr(nn, activation)()

    def forward(self, x, num=None, cat=None):
        input_concat = self.input(x, num, cat)
        input_concat = self.dropout(input_concat)
        rnn_outputs, rnn_states = self.rnn(input_concat)
        rnn_compress = self.activation(self.compress(rnn_outputs))
        return rnn_compress, rnn_states


class LuongDecoder(nn.Module):
    """
    References:
    https://stackoverflow.com/questions/44238154/what-is-the-difference-between-luong-attention-and-bahdanau-attention
    https://distill.pub/2016/augmented-rnns/
    https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html#global-vs-local-attention
    https://towardsdatascience.com/sequence-2-sequence-model-with-attention-mechanism-9e9ca2a613a
    """
    pass