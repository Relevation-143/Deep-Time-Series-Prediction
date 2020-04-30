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

    def __init__(self, series_dim, hidden_dim, dropout=0.5, enc_num=None,
                 enc_cat=None, dec_num=None, dec_cat=None, rnn_type='gru', n_layers=1):
        super().__init__()
        self.encoder_input = Inputs(enc_num, enc_cat, dropout=dropout, seq_last=False)
        self.decoder_input = Inputs(dec_num, dec_cat, dropout=dropout, seq_last=False)
        self.rnn_type = rnn_type
        if rnn_type == "gru":
            rnn = nn.GRU
        elif rnn_type == "lstm":
            rnn = nn.LSTM
        elif rnn_type == "rnn":
            rnn = nn.RNN

        self.encoder = rnn(self.encoder_input.output_dim + series_dim, hidden_dim, num_layers=n_layers,
                           dropout=dropout, batch_first=True)
        self.decoder = rnn(self.decoder_input.output_dim + series_dim, hidden_dim, num_layers=n_layers,
                           dropout=dropout, batch_first=True)
        self.fc_out_1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out_2 = nn.Linear(hidden_dim, series_dim)

    def encode(self, enc_x, enc_num=None, enc_cat=None):
        enc_features = self.encoder_input(enc_num, enc_cat)
        if enc_features is not None:
            enc_inputs = torch.cat([enc_x, enc_features], dim=2)
        else:
            enc_inputs = enc_x
        # if self.bn is not None:
        #     enc_inputs = self.bn(enc_inputs.transpose(1, 2)).transpose(2, 1)
        outputs, hidden = self.encoder(enc_inputs)
        return outputs, hidden

    def decode(self, enc_outputs, hidden, dec_x, dec_num=None, dec_cat=None):
        dec_features = self.decoder_input(dec_num, dec_cat)
        if dec_features is not None:
            dec_inputs = torch.cat([dec_x, dec_features], dim=2)
        else:
            dec_inputs = dec_x
        outputs, hidden = self.decoder(dec_inputs, hidden)
        outputs = torch.relu(self.fc_out_1(outputs))
        outputs = self.fc_out_2(outputs)
        return outputs, hidden

    def forward(self, enc_x, dec_len, enc_num=None, enc_cat=None, dec_num=None, dec_cat=None):
        enc_outputs, hidden = self.encode(enc_x, enc_num, enc_cat)
        step_x = enc_x[:, [-1], :]
        result = []
        for step in range(dec_len):
            step_dec_num = None if dec_num is None else dec_num[:, [step]]
            step_dec_cat = None if dec_cat is None else dec_cat[:, [step]]
            step_x, hidden = self.decode(enc_outputs, hidden, step_x, step_dec_num, step_dec_cat)
            result.append(step_x)
        return torch.cat(result, dim=1)

    def predict(self, **kw):
        with torch.no_grad():
            y_hat = self(**kw)
        return y_hat


net = BasicSeq2Seq(1, 12, enc_cat=[(12, 2)])
net = net.apply(lambda x: print(x))


net2 = nn.Embedding(10, 10)
