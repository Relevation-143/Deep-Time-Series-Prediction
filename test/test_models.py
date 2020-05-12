# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/5/12 16:33
"""
from deepseries.models.rnn2rnn import RNN2RNN
from deepseries.train import Learner
from deepseries.dataset import TimeSeries, FeatureStore, Seq2SeqDataLoader
import numpy as np
from torch.optim import Adam


batch_size = 16
enc_len = 24
dec_len = 12


series = np.sin(np.arange(0, 1000))
series -= series.mean()
series = series.reshape(1, -1, 1)
trn_dl = Seq2SeqDataLoader(TimeSeries(series[:, :800]), batch_size, enc_lens=enc_len, dec_lens=dec_len, seq_last=False)
val_dl = Seq2SeqDataLoader(TimeSeries(series[:, -200:]), batch_size, enc_lens=enc_len, dec_lens=dec_len,  seq_last=False)


model = RNN2RNN(1, 16, 8, num_layers=1, attn_heads=1, attn_size=9)
opt = Adam(model.parameters(), 0.001)
learner = Learner(model, opt, ".")
learner.fit(100, trn_dl, val_dl, early_stopping=False)
