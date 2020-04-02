# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/4/1 16:28
"""
from torch.utils.data import DataLoader
import random
import numpy as np
from collections import OrderedDict


class BaseFeature:

    def read(self, *args, **kwargs):
        raise NotImplemented


class SeriesFeature(BaseFeature):

    def __init__(self, name, ftype, series, idx_map=None, embedding=None, enc=True, dec=True):
        self.name = name
        self.ftype = ftype
        if ftype == "cat":
            assert embedding is not None
        self.embedding = embedding
        self.series = series
        self.idx_map = idx_map
        self.enc = enc
        self.dec = dec

    def read(self, series_idx, time_idx):
        if self.idx_map is not None:
            series_idx = [self.idx_map[i] for i in series_idx]
        return self.series[series_idx, time_idx]


class PropertyFeature(BaseFeature):

    def __init__(self, name, ftype, value_map, embedding=None):
        self.name = name
        self.ftype = ftype
        if ftype == "cat":
            assert embedding is not None
        self.embedding = embedding
        self.value_map = value_map

    def read(self, series_idx, time_idx):
        var = np.array([self.value_map[k] for k in series_idx])
        return np.repeat(np.expand_dims(var, axis=1), len(time_idx), axis=1)


class SeriesFrame:

    enc_num_feats = None
    enc_cat_feats = None
    dec_num_feats = None
    dec_cat_feats = None

    def __init__(self, series, features, batch_size, enc_lens, dec_lens, free_window=None, mode="train", shuffle=True):
        self.series_idx = np.arange(series.shape[0])
        self.times_idx = np.arange(series.shape[1])
        self.series_len = len(self.times_idx)
        self.series = series
        self.enc_num_feats = OrderedDict([(f.name, f) for f in features if f.ftype == "num" and f.enc])
        self.dec_num_feats = OrderedDict([(f.name, f) for f in features if f.ftype == "num" and f.dec])
        self.enc_cat_feats = OrderedDict([(f.name, f) for f in features if f.ftype == "cat" and f.enc])
        self.dec_cat_feats = OrderedDict([(f.name, f) for f in features if f.ftype == "cat" and f.dec])

        self.batch_size = batch_size
        self.enc_lens = enc_lens
        self.dec_lens = dec_lens
        assert free_window < enc_lens and free_window < dec_lens
        self.free_window = free_window
        self.is_single = len(self.series_idx) == 1
        self.mode = mode
        self.shuffle = shuffle

    def single_generate(self):
        start_idxes = np.arange(0, self.series - self.enc_lens - self.dec_lens - self.free_window * 2 + 1)
        if self.shuffle:
            random.shuffle(start_idxes)
        for batch in range(start_idxes // self.batch_size):
            batch_start_idxes = start_idxes[batch * self.batch_size: (batch + 1) * self.batch_size]
            free = random.randint(-self.free_window, self.free_window)
            batch_time_idxes = [
                (np.arange(start, start+self.enc_lens+free),
                 np.arange(start+self.enc_lens+free, start+self.enc_lens+self.dec_lens+free))
                for start in batch_start_idxes]

    def single_apply(self, batch_start_idxes):
        pass

    def multi_generate_batch_idx(self):
        if self.shuffle:
            random.shuffle(self.times_idx)


    def apply_single(self):
        pass

    def apply_multi(self):
        pass

    def __iter__(self):
        pass
