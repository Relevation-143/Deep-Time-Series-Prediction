# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/5/6 10:40
"""
import torch.nn as nn
import torch
from deepseries.nn.comm import InputsEXP, Dense
from deepseries.nn.init import rnn_init
from deepseries.nn.attention import RNNAttention


class RNNEncoder(nn.Module):

    def __init__(self, series_dim, hidden_dim, compress_dim, activation='SELU',
                 dropout=0.5, num_feat=None, cat_feat=None, rnn_type='gru', n_layers=1, initializer='xavier'):
        super().__init__()
        self.input = InputsEXP(num_feat, cat_feat, seq_last=False, dropout=dropout)
        self.rnn = getattr(nn, rnn_type)(self.input.output_dim + series_dim, hidden_dim,
                                         num_layers=n_layers, dropout=dropout, batch_first=True)
        self.compress = Dense(hidden_dim, compress_dim, dropout=dropout, activation=activation)
        rnn_init(self.rnn, initializer)

    def forward(self, x, num=None, cat=None):
        """

        Args:
            x: B x S x N
            num:
            cat:

        Returns:
            rnn_compress: B x S x C
            rnn_states: ...

        """
        concat = self.input(x, num, cat)
        outputs, hidden = self.rnn(concat)
        compress = self.compress(outputs)
        return compress, outputs, hidden


class RNNDecoder(nn.Module):

    def __init__(self, series_dim, hidden_dim, enc_output_dim, activation='SELU', residual=True, attn_heads=None, attn_size=None,
                 dropout=0.1, num_feat=None, cat_feat=None, rnn_type='gru', n_layers=1, initializer='xavier'):
        """
            https://stackoverflow.com/questions/44238154/what-is-the-difference-between-luong-attention-and-bahdanau-attention
            https://distill.pub/2016/augmented-rnns/
            https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html#global-vs-local-attention
            https://towardsdatascience.com/sequence-2-sequence-model-with-attention-mechanism-9e9ca2a613a
        Args:
            series_dim:
            hidden_dim:
            enc_output_dim:
            activation:
            residual:
            attn_heads:
            attn_size:
            dropout:
            num_feat:
            cat_feat:
            rnn_type:
            n_layers:
            initializer:
        """
        super().__init__()
        self.input = InputsEXP(num_feat, cat_feat, seq_last=False, dropout=dropout)
        self.rnn = getattr(nn, rnn_type)(self.input.output_dim + series_dim, hidden_dim,
                                         num_layers=n_layers, dropout=dropout, batch_first=True)
        self.residual = residual
        concat_dim = hidden_dim
        if attn_heads is not None:
            self.attn = RNNAttention(attn_heads, attn_size, hidden_dim, enc_output_dim, enc_output_dim, dropout=dropout)
            hidden_dim += enc_output_dim
        if residual:
            hidden_dim += self.input.output_dim
        self.output = Dense(concat_dim, series_dim, dropout=dropout, activation=activation)

        rnn_init(self.rnn, initializer)

    def forward(self, x, num_feat=None, cat_feat=None, enc_outputs=None, hidden=None):
        inputs = self.input(x, num_feat, cat_feat)
        outputs, _, hidden = self.rnn(inputs, hidden)
        if hasattr(self, 'attn'):
            attn_outputs, p_attn = self.attn(outputs, enc_outputs, enc_outputs)
            if self.residual:
                concat = torch.cat([outputs, attn_outputs, x], dim=2)
                y = self.output(concat)
            else:
                concat = torch.cat([outputs, attn_outputs], dim=2)
                y = self.output(concat)
            return y, hidden, p_attn
        else:
            if self.residual:
                concat = torch.cat([outputs, x], dim=2)
                y = self.output(concat)
            else:
                y = self.output(outputs)
            return y, outputs, hidden, None


class RNN2RNN(nn.Module):

    def __init__(self, series_dim, hidden_dim, compress_dim, activation='SELU', residual=True, attn_heads=None, attn_size=None,
                 dropout=0.1, enc_num_feat=None, enc_cat_feat=None, dec_num_feat=None, dec_cat_feat=None, rnn_type='gru', n_layers=1, initializer='xavier'):
        super().__init__()
        self.encoder = RNNEncoder(series_dim, hidden_dim, compress_dim, activation,
                                  dropout, enc_num_feat, enc_cat_feat, rnn_type, n_layers, initializer)
        self.decoder = RNNEncoder(series_dim, hidden_dim, compress_dim, activation, dropout, dec_num_feat,
                                  dec_cat_feat, rnn_type, n_layers, initializer)

    def forward(self, enc_x, dec_x):
        enc_outputs, enc_hidden = self.encoder()