# encoding: utf-8
# Author: 周知瑞
# Mail: evilpsycho42@gmail.com

from deepseries.layers import CausalConv1d, TimeDistributedDense1D
import torch
from torch import nn
from torch.nn import functional as F


class WaveEncoder(nn.Module):

    def __init__(self,
                 features_dim=None,
                 source_dim=1,
                 residual_channels=32,
                 skip_channels=32,
                 dilations=[2 ** i for i in range(8)] * 3,
                 kernels_size=[2 for i in range(8)] * 3):
        super().__init__()
        self.features_dim = features_dim if features_dim else 0
        self.inputs_dim = self.features_dim + 1
        self.source_dim = source_dim
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.dilations = dilations
        self.kernels_size = kernels_size

        self.fc_h = TimeDistributedDense1D(self.inputs_dim, residual_channels, torch.tanh)
        self.fc_c = TimeDistributedDense1D(self.inputs_dim, residual_channels, torch.tanh)
        self.cnn_layers = nn.ModuleList([CausalConv1d(residual_channels, residual_channels * 4, k, dilation=d)
                                         for k, d in zip(kernels_size[:-1], dilations[:-1])])

    def forward(self, x, features=None):
        inputs = torch.cat([x, features], dim=1) if features is not None else x
        h = self.fc_h(inputs)
        c = self.fc_c(inputs)
        states = [h]
        for cnn in self.cnn_layers:
            dilation_inputs = cnn(h)
            input_gate, conv_filter, conv_gate, emit_gate = torch.split(dilation_inputs, self.residual_channels, dim=1)
            c = torch.sigmoid(input_gate) * c + torch.tanh(conv_filter) * torch.sigmoid(conv_gate)
            h = torch.sigmoid(emit_gate) * torch.tanh(c)
            states.append(h)
        return states


class WaveDecoder(nn.Module):

    def __init__(self,
                 features_dim,
                 source_dim=1,
                 residual_channels=32,
                 skip_channels=32,
                 dilations=[2 ** i for i in range(8)] * 3,
                 kernels_size=[2 for i in range(8)] * 3):
        super().__init__()
        self.features_dim = features_dim if features_dim else 0
        self.source_dim = source_dim
        self.inputs_dim = self.features_dim + source_dim * 2
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.dilations = dilations
        self.kernels_size = kernels_size

        self.fc_h = TimeDistributedDense1D(self.inputs_dim, residual_channels, torch.tanh)
        self.fc_c = TimeDistributedDense1D(self.inputs_dim, residual_channels, torch.tanh)
        self.cnn_layers = nn.ModuleList([nn.Conv1d(residual_channels, residual_channels * 4, k)
                                         for k, d in zip(kernels_size, dilations)])
        self.fc_out_1 = TimeDistributedDense1D(len(dilations) * skip_channels, 128, F.relu)
        self.fc_out_2 = TimeDistributedDense1D(128, self.source_dim * 2)

    def forward(self, x, features, queues):
        inputs = torch.cat([x, features], dim=1) if features is not None else x
        h = self.fc_h(inputs)
        c = self.fc_c(inputs)
        skips, update_queues = [], []  # TODO

        for state, cnn, dilation in zip(queues, self.cnn_layers, self.dilations):
            state_len = state.shape[2]
            if state_len >= dilation:
                conv_inputs = torch.cat([state[:, :, (state_len-1)-(dilation-1)].unsqueeze(2), h], dim=2)
            else:
                conv_inputs = torch.cat([torch.zeros_like(h), h], dim=2)
            conv_outputs = cnn(conv_inputs)
            input_gate, conv_filter, conv_gate, emit_gate = torch.split(conv_outputs, self.residual_channels, dim=1)
            c = torch.sigmoid(input_gate) * c + torch.tanh(conv_filter) * torch.sigmoid(conv_gate)
            h = torch.sigmoid(emit_gate) * torch.tanh(c)
            skips.append(h)
            update_queues.append(torch.cat([state, h], dim=2))
        # TODO
        # update_queues.pop(-1)
        skips = torch.cat(skips, dim=1)
        y_hidden = self.fc_out_1(skips)
        h_hat = self.fc_out_2(y_hidden)
        return h_hat, update_queues


class WaveNet(nn.Module):

    def __init__(self,
                 source_dim=1,
                 enc_compress=None,
                 dec_compress=None,
                 enc_numerical=None,
                 enc_categorical=None,
                 dec_numerical=None,
                 dec_categorical=None,
                 residual_channels=32,
                 skip_channels=32,
                 dilations=[2 ** i for i in range(8)] * 3,
                 kernels_size=[2 for i in range(8)] * 3,
                 ):
        super().__init__()
        self.source_dim = source_dim
        self.enc_trans = SeriesFeatureTransformer(enc_compress, enc_numerical, enc_categorical)
        self.enc = WaveEncoder(self.enc_trans.output_dim, source_dim, residual_channels, skip_channels, dilations, kernels_size)
        self.dec_trans = SeriesFeatureTransformer(dec_compress, dec_numerical, dec_categorical)
        self.dec = WaveDecoder(self.dec_trans.output_dim, source_dim, residual_channels, skip_channels, dilations, kernels_size)

    def encode(self, x, numerical=None, categorical=None):
        enc_features = self.enc_trans(numerical, categorical)
        queues = self.enc(x, enc_features)
        return queues

    def decode(self, x, queues, numerical=None, categorical=None):
        dec_features = self.dec_trans(numerical, categorical)
        y_hat, updated_queues = self.dec(x, dec_features, queues)
        return y_hat, updated_queues

    def predict(self, enc_x, dec_len, enc_num=None, enc_cat=None,
                dec_num=None, dec_cat=None):
        if not isinstance(dec_len, int):
            dec_len = dec_len[0]
        results = []
        queues = self.encode(enc_x, enc_num, enc_cat)
        step_x = enc_x[:, :, -1].unsqueeze(2)
        for step in range(dec_len):
            step_numerical = dec_num[:, :, -1].unsqueeze(2) if dec_num is not None else None
            step_categorical = dec_cat[:, :, -1].unsqueeze(2) if dec_cat is not None else None
            step_x, queues = self.decode(step_x, queues, step_numerical, step_categorical)
            results.append(step_x)
        y_hat, p = torch.cat(results, dim=2).split(self.source_dim, dim=1)  # (B, 2*N, S)
        y_hat = torch.sigmoid(p) * y_hat
        return y_hat

    def forward(self, feed):
        y_hat = self.predict(**feed)
        return y_hat


class SeriesFeatureTransformer(nn.Module):

    """
    Args:
        compress_dim, int
        numerical, int
        categorical, list of tuple(n, embed_dim)
    """

    def __init__(self, compress_dim=None, numerical=None, categorical=None):
        super().__init__()
        self.compress_dim = compress_dim
        self.numerical = numerical
        self.numerical_dim = numerical if numerical else 0
        self.categorical = categorical
        self.categorical_dim = sum([o for _, o in categorical]) if categorical else 0

        self.embed_cat = nn.ModuleList([nn.Embedding(i, o) for i, o in categorical]) if categorical else None
        self.compress = TimeDistributedDense1D(
            self.categorical_dim + self.numerical_dim, compress_dim, activation=F.relu) if compress_dim else None
        self.output_dim = compress_dim if self.compress_dim else self.numerical_dim + self.categorical_dim

    def forward(self, numerical=None, categorical=None):
        if self.numerical is None and self.categorical is None:
            return None
        concat = []
        if self.numerical:
            concat.append(numerical)
        if self.categorical:
            cat = []
            for channel in range(categorical.shape[1]):
                cat.append(self.embed_cat[channel](categorical[:, channel]).transpose(1, 2))
            concat.append(torch.cat(cat, dim=1))
        concat = torch.cat(concat, dim=1)
        if self.compress:
            concat = self.compress(concat)
        return concat


if __name__ == "__main__":

    # x1 = torch.rand(4, 1, 10)
    # net1 = WaveNet()
    # net1.predict(x1, 10)

    # x2 = torch.rand(4, 1, 10)
    # f2 = torch.rand(4, 4, 10)
    # net2 = WaveNet(enc_compress=2, enc_numerical=4)
    # net2.predict(x2, 10, enc_numerical=f2).shape

    x3 = torch.rand(4, 1, 10)
    f3 = torch.randint(0, 10, (4, 3, 10))
    net3 = WaveNet(enc_compress=2, enc_categorical=[(10, 2), (10, 2), (10, 2)])
    net3.predict(x3, 10, enc_categorical=f3).shape
    from fastai.basic_train import BasicLearner, Learner, DataBunch
