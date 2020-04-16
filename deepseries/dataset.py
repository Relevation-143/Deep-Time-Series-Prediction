# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/4/3 14:33
"""
import torch
import random
import copy
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
            shape(batch, dim, seq)
        """
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
            series_idx: shape(J)
            time_idx: (I, seq)

        Returns:
            shape(N, dim, seq)
        """
        if self.idx_map is not None:
            series_idx = np.array([self.idx_map[i] for i in series_idx])
        batch = np.repeat(np.expand_dims(self.values[series_idx], axis=2), time_idx.shape[1], axis=2)
        return batch


class FeatureStore:

    def __init__(self, features=None):
        self.features = features

    def read_batch(self, series_idx, time_idx):
        if self.features is None:
            return None
        else:
            batch = np.concatenate([f.read_batch(series_idx, time_idx) for f in self.features], axis=1)
            return batch


class Seq2SeqDataLoader:

    def __init__(self, xy, batch_size, enc_lens, dec_lens, use_cuda=False, weight=None,
                 time_free_space=0, time_interval=1, mode="train", drop_last=False,
                 enc_num_feats=None, dec_num_feats=None, enc_cat_feats=None, dec_cat_feats=None, random_seed=42):
        self.xy = xy
        self.series_size = xy.values.shape[0]
        self.series_idx = np.arange(xy.values.shape[0])
        self.time_size = xy.values.shape[2]
        self.time_idx = np.arange(xy.values.shape[2])
        self.batch_size = batch_size
        self.enc_lens = enc_lens
        self.dec_lens = dec_lens

        self.use_cuda = use_cuda
        self._time_free_space = time_free_space
        self.time_interval = time_interval
        self.mode = mode
        self.drop_last = drop_last

        self.weight = FeatureStore([weight]) if weight is not None else None
        self.enc_num_feats = FeatureStore(enc_num_feats)
        self.dec_num_feats = FeatureStore(dec_num_feats)
        self.enc_cat_feats = FeatureStore(enc_cat_feats)
        self.dec_cat_feats = FeatureStore(dec_cat_feats)

        self.random = np.random.RandomState(random_seed)

    @property
    def time_free_space(self):
        if self.mode == "train": return self._time_free_space
        else: return 0

    def get_time_free(self):
        if self.time_free_space == 0: return 0
        else: return self.random.randint(-self.time_free_space, self.time_free_space)

    @property
    def val_time_start_idx(self):
        return self.time_idx[:-self.dec_lens-self.enc_lens-self.time_free_space*2+1][::self.time_interval]

    def __len__(self):
        """ return num batch in one epoch"""
        if self.series_size == 1:
            return len(self.val_time_start_idx) // self.batch_size + 1 if not self.drop_last else 0
        else:
            return self.series_size // self.batch_size + 1 if not self.drop_last else 0

    def __iter__(self):
        if self.series_size == 1:
            series_idx = np.array([0])
            time_idx = copy.copy(self.val_time_start_idx)
            if self.mode == "train": self.random.shuffle(time_idx)
            for i in range(len(self)):
                batch_start = time_idx[i * self.batch_size: (i + 1) * self.batch_size]
                enc_len = self.enc_lens + self.get_time_free()
                dec_len = self.dec_lens + self.get_time_free()
                enc_time_idx, dec_time_idx = [], []
                for start in batch_start:
                    enc_time_idx.append(np.arange(start, start + enc_len))
                    dec_time_idx.append(np.arange(start + enc_len, start + enc_len + dec_len))
                enc_time_idx = np.stack(enc_time_idx, 0)
                dec_time_idx = np.stack(dec_time_idx, 0)
                yield self.read_batch(series_idx, enc_time_idx, dec_time_idx)
        else:
            series_idx = copy.copy(self.series_idx)
            time_idx = copy.copy(self.val_time_start_idx)
            if self.mode == "train": self.random.shuffle(series_idx)
            for i in range(len(self)):
                start = self.random.choice(time_idx)
                enc_len = self.enc_lens + self.get_time_free()
                dec_len = self.dec_lens + self.get_time_free()
                enc_time_idx = np.stack([np.arange(start, start + enc_len)], 0)
                dec_time_idx = np.stack([np.arange(start + enc_len, start + enc_len + dec_len)], 0)
                batch_series_idx = series_idx[i * self.batch_size: (i + 1) * self.batch_size]
                yield self.read_batch(batch_series_idx, enc_time_idx, dec_time_idx)

    def read_batch(self, batch_series_idx, enc_time_idx, dec_time_idx):

        dec_len = dec_time_idx.shape[1]
        enc_x = self.xy.read_batch(batch_series_idx, enc_time_idx)
        feed_dict = {
            "enc_x": self.check_to_tensor(enc_x).float(),
            "enc_num": self.check_to_tensor(self.enc_num_feats.read_batch(batch_series_idx, enc_time_idx)).float(),
            "enc_cat": self.check_to_tensor(self.enc_cat_feats.read_batch(batch_series_idx, enc_time_idx)),
            "dec_num": self.check_to_tensor(self.dec_num_feats.read_batch(batch_series_idx, dec_time_idx)).float(),
            "dec_cat": self.check_to_tensor(self.dec_cat_feats.read_batch(batch_series_idx, dec_time_idx)),
            "dec_len": dec_len
        }

        if self.mode != "test":
            dec_x = self.xy.read_batch(batch_series_idx, dec_time_idx)
            if self.weight is not None:
                weight = self.weight.read_batch(batch_series_idx, dec_time_idx)
                return feed_dict, self.check_to_tensor(dec_x).float(), self.check_to_tensor(weight).float()
            else:
                return feed_dict, self.check_to_tensor(dec_x).float(), None
        else:
            return feed_dict

    def cuda(self):
        self.use_cuda = True
        return self

    def cpu(self):
        self.use_cuda = False
        return self

    def train(self):
        self.mode = 'train'
        return self

    def eval(self):
        self.mode = 'eval'
        return self

    def test(self):
        self.mode = 'test'
        return self

    def check_to_tensor(self, x):
        if x is None:
            return x
        if self.use_cuda:
            return torch.as_tensor(x).cuda()
        else:
            return torch.as_tensor(x)


# class SeriesFrame:
#
#     def __init__(self, xy, batch_size, enc_lens, dec_lens,
#                  shuffle=True, time_free_space=None, mode="train", drop_last=False,
#                  enc_num_feats=None, dec_num_feats=None, enc_cat_feats=None, dec_cat_feats=None, use_cuda=False):
#         self.xy = xy
#         self.series_idx = np.arange(xy.values.shape[0])
#         self.num_series = len(self.series_idx)
#         self.times_idx = np.arange(xy.values.shape[2])
#         self.series_len = len(self.times_idx)
#
#         self.enc_num_feats = FeatureStore(enc_num_feats)
#         self.dec_num_feats = FeatureStore(dec_num_feats)
#         self.enc_cat_feats = FeatureStore(enc_cat_feats)
#         self.dec_cat_feats = FeatureStore(dec_cat_feats)
#
#         self.batch_size = batch_size
#         self.enc_lens = enc_lens
#         self.dec_lens = dec_lens
#
#         self._time_free_space = time_free_space if time_free_space is not None else 0
#         assert self._time_free_space < enc_lens and self._time_free_space < dec_lens
#
#         self.is_single = self.num_series == 1
#         assert mode in ["train", "eval", 'valid']
#         self.mode = mode
#         self.shuffle = shuffle
#         self.use_cuda = use_cuda
#         self.drop_last = drop_last
#
#     @property
#     def time_free_space(self):
#         if self.mode != "train":
#             return 0  # eval valid mode
#         else:
#             return self._time_free_space  # train mode
#
#     def get_time_free(self):
#         return random.randint(-self.time_free_space, self.time_free_space)
#
#     def __len__(self):
#         n_time = (self.series_len - self.enc_lens - self.dec_lens - self.time_free_space * 2 + 1)
#         n_series = self.num_series
#         if self.is_single: n = n_time // self.batch_size
#         else:
#             if self.mode == "train":
#                 n = n_series // self.batch_size
#             else:
#                 n = n_series * n_time // self.batch_size
#         if not self.drop_last:
#             n += 1
#         return n
#
#     def _single_batch_generate(self):
#         start_space = np.arange(0, self.series_len - self.enc_lens - self.dec_lens - self.time_free_space * 2 + 1)
#         if self.shuffle and self.mode == "train":
#             random.shuffle(start_space)
#         for batch in range(len(self)):
#             batch_start = start_space[batch * self.batch_size: (batch + 1) * self.batch_size]
#             enc_len = self.enc_lens + self.get_time_free()
#             dec_len = self.dec_lens + self.get_time_free()
#             enc_time_idx, dec_time_idx = [], []
#             for start in batch_start:
#                 enc_time_idx.append(np.arange(start, start + enc_len))
#                 dec_time_idx.append(np.arange(start + enc_len, start + enc_len + dec_len))
#             enc_time_idx = np.array(enc_time_idx)
#             dec_time_idx = np.array(dec_time_idx)
#             yield self.read_batch(0, enc_time_idx, dec_time_idx)
#
#     def _multi_generate_batch(self):
#         start_space = np.arange(0, self.series_len - self.enc_lens - self.dec_lens - self.time_free_space * 2 + 1)
#         if self.mode == "train":
#             if self.shuffle: random.shuffle(self.series_idx)
#             for batch in range(len(self)):
#                 batch_series_idx = self.series_idx[batch * self.batch_size: (batch + 1) * self.batch_size]
#                 start = random.choice(start_space)
#                 enc_len = self.enc_lens + self.get_time_free()
#                 dec_len = self.dec_lens + self.get_time_free()
#                 enc_time_idx = np.expand_dims(np.arange(start, start + enc_len), axis=0)
#                 dec_time_idx = np.expand_dims(np.arange(start + enc_len, start + enc_len + dec_len), axis=0)
#                 yield self.read_batch(batch_series_idx, enc_time_idx, dec_time_idx)
#         else:
#             for i in range(self.num_series // self.batch_size + 1):
#                 batch_series_idx = self.series_idx[i * self.batch_size: (i + 1) * self.batch_size]
#                 for j in range(self.series_len - self.enc_lens - self.dec_lens + 1):
#                     enc_time_idx = np.expand_dims(np.arange(j, j + self.enc_lens), axis=0)
#                     dec_time_idx = np.expand_dims(np.arange(j + self.enc_lens, j + self.enc_lens + self.dec_lens),
#                                                   axis=0)
#                     yield self.read_batch(batch_series_idx, enc_time_idx, dec_time_idx)
#
#     def read_batch(self, batch_series_idx, enc_time_idx, dec_time_idx):
#
#         dec_len = dec_time_idx.shape[1]
#         enc_x = self.xy.read_batch(batch_series_idx, enc_time_idx)
#         feed_dict = {
#             "enc_x": self.check_to_tensor(enc_x).float(),
#             "enc_num": self.check_to_tensor(self.enc_num_feats.read_batch(batch_series_idx, enc_time_idx)).float(),
#             "enc_cat": self.check_to_tensor(self.enc_cat_feats.read_batch(batch_series_idx, enc_time_idx)),
#             "dec_num": self.check_to_tensor(self.dec_num_feats.read_batch(batch_series_idx, dec_time_idx)).float(),
#             "dec_cat": self.check_to_tensor(self.dec_cat_feats.read_batch(batch_series_idx, dec_time_idx)),
#             "dec_len": dec_len
#         }
#
#         if self.mode != "eval":
#             dec_x = self.xy.read_batch(batch_series_idx, dec_time_idx)
#             return feed_dict, self.check_to_tensor(dec_x).float()
#         else:
#             return feed_dict
#
#     def generate_batch(self):
#         if self.is_single:
#             return self._single_batch_generate()
#         else:
#             return self._multi_generate_batch()
#
#     def __iter__(self):
#         return self.generate_batch()
#
#     def cuda(self):
#         self.use_cuda = True
#         return self
#
#     def cpu(self):
#         self.use_cuda = False
#         return self
#
#     def train(self):
#         self.mode = 'train'
#         return self
#
#     def valid(self):
#         self.mode = 'valid'
#         return self
#
#     def eval(self):
#         self.mode = 'eval'
#         return self
#
#     def check_to_tensor(self, x):
#         if x is None:
#             return x
#         if self.use_cuda:
#             return torch.as_tensor(x).cuda()
#         else:
#             return torch.as_tensor(x)
