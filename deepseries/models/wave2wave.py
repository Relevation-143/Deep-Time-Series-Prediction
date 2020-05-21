# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/5/20 10:51
"""
import torch.nn as nn
import torch
from deepseries.nn.cnn import WaveNetBlockV1
from deepseries.nn.comm import Embeddings, Concat
from deepseries.nn.loss import RMSELoss


class Wave2WaveEncoderV1(nn.Module):

    def __init__(self, target_size, cat_size=None, num_size=None, residual_channels=32,
                 skip_channels=32, num_blocks=8, num_layers=3, dropout=.0):
        super().__init__()
        self.embeds = Embeddings(cat_size)
        self.concat = Concat(dim=1)
        self.dropout = nn.Dropout(dropout)
        input_channels = self.embeds.output_size + target_size + num_size if isinstance(num_size, int) else 0
        self.input_conv = nn.Conv1d(input_channels, residual_channels, kernel_size=1)
        self.wave_blocks = nn.ModuleList([WaveNetBlockV1(residual_channels, skip_channels, 2**b)
                                          for l in range(num_layers) for b in range(num_blocks)])

    def forward(self, x, num=None, cat=None):
        input = self.concat(x, num, self.embeds(cat))
        input = self.dropout(input)
        input = self.input_conv(input)
        skips = []
        inputs = [input]
        for block in self.wave_blocks[:-1]:
            input, skip = block(input)
            skips.append(skip)
            inputs.append(inputs)
        return inputs, skips


class Wave2WaveDecoderV1(nn.Module):

    def __init__(self, target_size, cat_size=None, num_size=None, residual_channels=32, embeds=None,
                 skip_channels=32, num_blocks=8, num_layers=3, dropout=.0, output_size=128, loss_fn=RMSELoss()):
        super().__init__()
        self.embeds = embeds if isinstance(embeds, Embeddings) else Embeddings(cat_size)
        self.concat = Concat(dim=1)
        self.dropout = nn.Dropout(dropout)
        input_channels = self.embeds.output_size + target_size + num_size if isinstance(num_size, int) else 0
        self.input_conv = nn.Conv1d(input_channels, residual_channels, kernel_size=1)
        self.wave_blocks = nn.ModuleList([WaveNetBlockV1(residual_channels, skip_channels, 2 ** b)
                                          for l in range(num_layers) for b in range(num_blocks)])
        self.output_conv1 = nn.Conv1d(num_layers*num_blocks*skip_channels, output_size, kernel_size=1)
        self.output_conv2 = nn.Conv1d(output_size, target_size, kernel_size=1)
        self.loss_fn = loss_fn

    def forward(self, inputs_queue, x, num=None, cat=None):
        input = self.concat(x, num, self.embeds(cat))
        input = self.dropout(input)
        input = self.input_conv(input)
        inputs_queue[0] = torch.cat([inputs_queue[0], input], dim=2)
        skips = []
        for i, block in enumerate(self.wave_blocks):
            input, skip = block.fast_forward(inputs_queue[i])
            skips.append(skip)
            if i == len(self.wave_blocks)-1:
                break
            print(type(inputs_queue[i+1]
                       ), type(input))
            inputs_queue[i+1] = torch.cat([inputs_queue[i+1], input], dim=2)
        skips = torch.relu(torch.cat(skips, dim=1))
        skips = torch.relu(self.output_conv1(skips))
        output = self.output_conv2(skips)
        return output, inputs_queue


class Wave2WaveV1(nn.Module):

    def __init__(self, target_size, enc_cat_size=None, enc_num_size=None, dec_cat_size=None, dec_num_size=None,
                 residual_channels=32, share_embeds=None, skip_channels=32, num_blocks=8, num_layers=3,
                 dropout=.0, output_size=128):
        super().__init__()
        self.encoder = Wave2WaveEncoderV1(target_size, enc_cat_size, enc_num_size, residual_channels, skip_channels,
                                          num_blocks, num_layers, dropout)
        embeds = None
        if share_embeds:
            embeds = self.encoder.embeds
        self.decoder = Wave2WaveDecoderV1(target_size, dec_cat_size, dec_num_size, residual_channels, embeds,
                                          skip_channels, num_blocks, num_layers, dropout, output_size)

    def batch_loss(self, x, y, w=None):
        inputs, _ = self.encoder(x['enc_x'], x['enc_num'], x['enc_cat'])
        y_hats = []
        for step in range(x['dec_len']):
            y_hat, inputs = self.decoder(inputs, x['dec_x'], x['dec_num'], x['dec_cat'])
            y_hats.append(y_hat)
        y_hats = torch.cat(y_hats, dim=2)
        loss = self.loss_fn(y_hats, y, w)
        return loss

    @torch.no_grad()
    def predict(self, enc_x, dec_len, enc_num=None, enc_cat=None, dec_num=None, dec_cat=None):
        inputs, _ = self.encoder(enc_x, enc_num, enc_cat)
        preds = []
        last_y = enc_x[:, :, [-1]]
        for step in range(dec_len):
            last_y, inputs = self.decoder(inputs, last_y, dec_num, dec_cat)
            preds.append(last_y)
        return torch.cat(preds, dim=2)
