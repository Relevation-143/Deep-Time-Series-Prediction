# encoding: utf-8
# Author: 周知瑞
# Mail: evilpsycho42@gmail.com

from deepseries.nn import CausalConv1d, TimeDistributedDense1d, Inputs
import torch
from torch import nn
from torch.nn import functional as F


class WaveEncoder(nn.Module):

    def __init__(self,
                 series_dim=1,
                 features_dim=None,
                 residual_channels=32,
                 skip_channels=32,
                 n_blocks=3,
                 n_layers=8,
                 dropout=0.,
                 debug=False,
                 ):
        super().__init__()
        self.features_dim = features_dim if features_dim is not None else 0
        self.inputs_dim = self.features_dim + 1
        self.source_dim = series_dim
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.n_blocks = n_blocks
        self.n_layers = n_layers

        self.fc_h = TimeDistributedDense1d(self.inputs_dim, residual_channels, torch.tanh, dropout=dropout)
        self.fc_c = TimeDistributedDense1d(self.inputs_dim, residual_channels, torch.tanh, dropout=dropout)
        self.cnns = nn.ModuleList(
            [CausalConv1d(residual_channels, residual_channels * 4, kernel_size=2, dilation=2**layer)
             for _ in range(n_blocks) for layer in range(n_layers)])  # TODO

    def forward(self, x, features=None):
        inputs = torch.cat([x, features], dim=1) if features is not None else x
        h = self.fc_h(inputs)
        c = self.fc_c(inputs)
        # TODO
        states = [h]
        for cnn in self.cnns[:-1]:
            dilation_inputs = cnn(h)
            input_gate, conv_filter, conv_gate, emit_gate = torch.split(dilation_inputs, self.residual_channels, dim=1)
            c = torch.sigmoid(input_gate) * c + torch.tanh(conv_filter) * torch.sigmoid(conv_gate)
            h = torch.sigmoid(emit_gate) * torch.tanh(c)
            states.append(h)
        return states


class WaveDecoder(nn.Module):

    def __init__(self,
                 series_dim=1,
                 features_dim=None,
                 residual_channels=32,
                 skip_channels=32,
                 n_blocks=3,
                 n_layers=8,
                 hidden_size=128,
                 dropout=0.,
                 debug=False,
                 ):
        super().__init__()
        self.features_dim = features_dim if features_dim is not None else 0
        self.series_dim = series_dim
        self.inputs_dim = self.features_dim + series_dim
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.debug = debug

        self.fc_h = TimeDistributedDense1d(self.inputs_dim, residual_channels, torch.tanh, dropout=dropout)
        self.fc_c = TimeDistributedDense1d(self.inputs_dim, residual_channels, torch.tanh, dropout=dropout)
        self.cnns = nn.ModuleList(
            [nn.Conv1d(residual_channels, residual_channels * 4, kernel_size=2)
             for _ in range(n_blocks) for layer in range(n_layers)])
        self.fc_out_1 = TimeDistributedDense1d(n_layers * n_blocks * skip_channels, hidden_size, F.relu, dropout=dropout)
        self.fc_out_2 = TimeDistributedDense1d(hidden_size, self.series_dim, dropout=dropout)

    def forward(self, x, features, queues):
        inputs = torch.cat([x, features], dim=1) if features is not None else x
        h = self.fc_h(inputs)
        c = self.fc_c(inputs)
        skips, update_queues = [], []  # TODO

        step_padding = 0
        dilations = [2 ** d for _ in range(self.n_blocks) for d in range(self.n_layers)]

        for state, cnn, dilation in zip(queues, self.cnns, dilations):
            update_queues.append(torch.cat([state, h], dim=2))
            state_len = state.shape[2]
            if state_len >= dilation:
                conv_inputs = torch.cat([state[:, :, [(state_len-1)-(dilation-1)]], h], dim=2)
            else:
                step_padding += 1
                conv_inputs = torch.cat([torch.zeros_like(h), h], dim=2)
            conv_outputs = cnn(conv_inputs)
            input_gate, conv_filter, conv_gate, emit_gate = torch.split(conv_outputs, self.residual_channels, dim=1)
            c = torch.sigmoid(input_gate) * c + torch.tanh(conv_filter) * torch.sigmoid(conv_gate)
            h = torch.sigmoid(emit_gate) * torch.tanh(c)
            skips.append(h)
        if self.debug:
            print(f"debug info: step padding {step_padding / len(queues) * 100:.0f}%")

        # TODO
        # update_queues.pop(-1)
        skips = torch.cat(skips, dim=1)
        y_hidden = self.fc_out_1(skips)
        h_hat = self.fc_out_2(y_hidden)
        return h_hat, update_queues


class Wave2WaveV1(nn.Module):

    def __init__(self,
                 series_dim=1,
                 enc_num=None,
                 enc_cat=None,
                 dec_num=None,
                 dec_cat=None,
                 residual_channels=32,
                 skip_channels=32,
                 dropout=0.,
                 n_blocks=3,
                 n_layers=8,
                 hidden_size=128,
                 debug=False
                 ):
        super().__init__()
        self.series_dim = series_dim
        self.enc_trans = Inputs(enc_num, enc_cat, dropout=dropout)
        self.enc = WaveEncoder(series_dim, self.enc_trans.output_dim, residual_channels,
                               skip_channels, n_blocks=n_blocks, n_layers=n_layers, dropout=dropout, debug=debug)
        self.dec_trans = Inputs(dec_num, dec_cat, dropout=dropout)
        self.dec = WaveDecoder(series_dim, self.dec_trans.output_dim, residual_channels,
                               skip_channels, n_blocks=n_blocks, n_layers=n_layers, dropout=dropout,
                               hidden_size=hidden_size, debug=debug)
        self.debug = debug

    def encode(self, x, numerical=None, categorical=None):
        enc_features = self.enc_trans(numerical, categorical)
        states = self.enc(x, enc_features)
        return states

    def decode(self, x, states, numerical=None, categorical=None):
        dec_features = self.dec_trans(numerical, categorical)
        y_hat, updated_queues = self.dec(x, dec_features, states)
        return y_hat, updated_queues

    def forward(self, enc_x, dec_len, enc_num=None, enc_cat=None,
                dec_num=None, dec_cat=None):
        results = []
        queues = self.encode(enc_x, enc_num, enc_cat)
        step_x = enc_x[:, :, -1].unsqueeze(2)
        for step in range(dec_len):
            step_numerical = dec_num[:, :, -1].unsqueeze(2) if dec_num is not None else None
            step_categorical = dec_cat[:, :, -1].unsqueeze(2) if dec_cat is not None else None
            step_x, queues = self.decode(step_x, queues, step_numerical, step_categorical)
            results.append(step_x)
        y_hat = torch.cat(results, dim=2)  # (B, N, S)
        return y_hat

    def predict(self, **kw):
        with torch.no_grad():
            y_hat = self(**kw)
        return y_hat
