# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/4/3 14:33
"""
import torch
import random
import numpy as np


class TimeSeries:

    def __init__(self, values, idx_map=None):
        """

        Args:
            values: shape(N, dim, seq)
            idx_map: dict
        """
        assert isinstance(values, np.ndarray)
        assert len(values.shape) == 3
        self.n = values.shape[0]
        self.dim = values.shape[1]
        self.values = values
        self.idx_map = idx_map

    def read_batch(self, series_idx, time_idx):
        """
        Args:
            series_idx: shape(I)
            time_idx: shape(J, seq)

        Returns:
            shape(batch = J / I, dim, seq)
        """
        if self.n == 1:
            assert len(series_idx) == 1
        if self.idx_map is not None:
            series_idx = np.array([self.idx_map[i] for i in series_idx])
        batch_size = series_idx.shape[0] * time_idx.shape[0]
        seq_len = time_idx.shape[1]
        batch = self.values[series_idx][:, :, time_idx].transpose(0, 2, 1, 3).reshape(batch_size, self.dim, seq_len)
        return batch


class Property:

    def __init__(self, values, idx_map=None):
        """

        Args:
            values (np.ndarray): shape(N, dim)
            idx_map: dict
        """
        self.values = values
        self.idx_map = idx_map

    def read_batch(self, series_idx, time_idx):
        """

        Args:
            series_idx: shape(N)
            time_idx: (1, seq)

        Returns:
            shape(N, dim, seq)
        """
        if self.idx_map is not None:
            series_idx = np.array([self.idx_map[i] for i in series_idx])
        batch = np.repeat(np.expand_dims(self.values[series_idx], axis=2), time_idx.shape[1], axis=2)
        return batch


class FeatureStore:

    def __init__(self, features=None):
        self.features = [] if features is None else features

    def read_batch(self, series_idx, time_idx):
        if len(self.features) == 0:
            return None
        else:
            batch = np.concatenate([f.read_batch(series_idx, time_idx) for f in self.features], axis=1)
            return batch


class SeriesFrame:

    def __init__(self, xy, batch_size, enc_lens, dec_lens, shuffle=True, time_free_space=None, mode="train", drop_last=False,
                 enc_num_feats=None, dec_num_feats=None, enc_cat_feats=None, dec_cat_feats=None, use_cuda=False):
        self.xy = xy
        self.series_idx = np.arange(xy.values.shape[0])
        self.num_series = len(self.series_idx)
        self.times_idx = np.arange(xy.values.shape[2])
        self.series_len = len(self.times_idx)

        self.enc_num_feats = FeatureStore(enc_num_feats)
        self.dec_num_feats = FeatureStore(dec_num_feats)
        self.enc_cat_feats = FeatureStore(enc_cat_feats)
        self.dec_cat_feats = FeatureStore(dec_cat_feats)

        self.batch_size = batch_size
        self.enc_lens = enc_lens
        self.dec_lens = dec_lens

        self._time_free_space = time_free_space if time_free_space is not None else 0
        assert self._time_free_space < enc_lens and self._time_free_space < dec_lens

        self.is_single = self.num_series == 1
        assert mode in ["train", "eval"]
        self.mode = mode
        self.shuffle = shuffle
        self.use_cuda = use_cuda
        self.drop_last = drop_last

    @property
    def time_free_space(self):
        if self.mode == "eval":
            return 0
        else:
            return self._time_free_space

    def get_time_free(self):
        return random.randint(-self.time_free_space, self.time_free_space)

    def __len__(self):
        num_time = (self.series_len - self.enc_lens - self.dec_lens - self.time_free_space * 2 + 1)
        if self.is_single:
            num = num_time // self.batch_size
        else:
            num = self.num_series // self.batch_size
        if not self.drop_last:
            num += 1
        return num

    @property
    def num_batchs(self):
        return self.__len__()

    @property
    def num_samples(self):
        return self.__len__() * self.batch_size

    def _single_batch_generate(self):
        start_space = np.arange(0, self.series_len - self.enc_lens - self.dec_lens - self.time_free_space * 2 + 1)
        if self.shuffle and self.mode == "train":
            random.shuffle(start_space)
        for batch in range(len(self)):
            batch_start = start_space[batch * self.batch_size: (batch + 1) * self.batch_size]
            enc_len = self.enc_lens + self.get_time_free()
            dec_len = self.dec_lens + self.get_time_free()
            enc_time_idx, dec_time_idx = [], []
            for start in batch_start:
                enc_time_idx.append(np.arange(start, start+enc_len))
                dec_time_idx.append(np.arange(start+enc_len, start+enc_len+dec_len))
            enc_time_idx = np.array(enc_time_idx)
            dec_time_idx = np.array(dec_time_idx)
            enc_x = self.xy.read_batch(0, enc_time_idx)

            feed_dict = {
                "enc_x": self.check_to_tensor(enc_x),
                "enc_num": self.check_to_tensor(self.enc_num_feats.read_batch(0, enc_time_idx)),
                "enc_cat": self.check_to_tensor(self.enc_cat_feats.read_batch(0, enc_time_idx)),
                "dec_num": self.check_to_tensor(self.dec_num_feats.read_batch(0, dec_time_idx)),
                "dec_cat": self.check_to_tensor(self.dec_cat_feats.read_batch(0, dec_time_idx)),
                "dec_len": dec_len
            }
            if self.mode == "train":
                dec_x = self.xy.read_batch(0, dec_time_idx)
                yield feed_dict, self.check_to_tensor(dec_x).float()
            else:
                yield feed_dict

    def _multi_generate_batch(self):
        start_space = np.arange(0, self.series_len - self.enc_lens - self.dec_lens - self.time_free_space * 2 + 1)
        if self.shuffle and self.mode=="train":
            random.shuffle(self.series_idx)
        for batch in range(len(self)):
            batch_series_idx = self.series_idx[batch * self.batch_size: (batch + 1) * self.batch_size]
            start = random.choice(start_space)
            enc_len = self.enc_lens + self.get_time_free()
            dec_len = self.dec_lens + self.get_time_free()
            enc_time_idx = np.expand_dims(np.arange(start, start+enc_len), axis=0)
            dec_time_idx = np.expand_dims(np.arange(start+enc_len, start+enc_len+dec_len), axis=0)

            enc_x = self.xy.read_batch(batch_series_idx, enc_time_idx)

            feed_dict = {
                "enc_x": self.check_to_tensor(enc_x).float(),
                "enc_num": self.check_to_tensor(self.enc_num_feats.read_batch(batch_series_idx, enc_time_idx)).float(),
                "enc_cat": self.check_to_tensor(self.enc_cat_feats.read_batch(batch_series_idx, enc_time_idx)),
                "dec_num": self.check_to_tensor(self.dec_num_feats.read_batch(batch_series_idx, dec_time_idx)).float(),
                "dec_cat": self.check_to_tensor(self.dec_cat_feats.read_batch(batch_series_idx, dec_time_idx)),
                "dec_len": dec_len
            }

            if self.mode == "train":
                dec_x = self.xy.read_batch(batch_series_idx, dec_time_idx)
                yield feed_dict, self.check_to_tensor(dec_x).float()
            else:
                yield feed_dict

    def generate_batch(self):
        if self.is_single:
            return self._single_batch_generate()
        else:
            return self._multi_generate_batch()

    def __iter__(self):
        return self.generate_batch()

    def cuda(self):
        self.use_cuda = True

    def cpu(self):
        self.use_cuda = False

    def train(self):
        self.mode = 'train'

    def eval(self):
        self.mode = 'eval'

    def check_to_tensor(self, x):
        if x is None:
            return x
        if self.use_cuda:
            return torch.as_tensor(x).cuda()
        else:
            return torch.as_tensor(x)
