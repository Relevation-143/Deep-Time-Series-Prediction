# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/4/15 16:18
"""
import torch
import torch.nn as nn
from deepseries.nn.comm import Inputs


class BasicSeq2Seq(nn.Module):

    def __init__(self, xy_dim, hidden_dim, dropout=0.5, enc_num=None,
                 enc_cat=None, dec_num=None, dec_cat=None, rnn_type='gru', n_layers=1):
        super().__init__()
        self.encoder_input = Inputs(enc_num, enc_cat, dropout=dropout, seq_last=False)
        self.decoder_input = Inputs(dec_num, dec_cat, dropout=dropout, seq_last=False)
        if rnn_type == "gru":
            rnn = nn.GRU
        elif rnn_type == "lstm":
            rnn = nn.LSTM
        elif rnn_type == "rnn":
            rnn = nn.RNN
        self.encoder = rnn(self.encoder_input.output_dim + xy_dim, hidden_dim, num_layers=n_layers,
                           dropout=dropout, batch_first=True)
        self.decoder = rnn(self.encoder_input.output_dim + xy_dim, hidden_dim, num_layers=n_layers,
                           dropout=dropout, batch_first=True)

    def encode(self, enc_x, enc_num=None, enc_cat=None):
        enc_features = self.encoder_input(enc_num, enc_cat)
        if enc_features is not None:
            enc_inputs = torch.cat([enc_x, enc_features], dim=2)
        else:
            enc_inputs = enc_x
