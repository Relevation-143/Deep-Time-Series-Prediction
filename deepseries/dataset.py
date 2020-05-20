# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/4/3 14:33
"""
import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler, WeightedRandomSampler, Sampler
import copy
import numpy as np
from typing import List, Tuple
import numpy as np

DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class DeepSeriesDataSet(Dataset):

    def __init__(self, xy, enc_len, dec_len, weights=None, features=None, seq_last=True,
                 device=DEFAULT_DEVICE, mode='train'):
        self.xy = xy
        self.enc_len = enc_len
        self.dec_len = dec_len
        self.weights = weights
        self.features = features
        self.series_size = xy.values.shape[0]
        self.time_size = xy.values.shape[2]
        self.num_samples_per_series = self.time_size - self.enc_len - self.dec_len + 1

        self.enc_num = ValuesHouse(list(filter(lambda x: x.enc and not x.categorical, features)))
        self.enc_cat = ValuesHouse(list(filter(lambda x: x.enc and x.categorical, features)))
        self.dec_num = ValuesHouse(list(filter(lambda x: x.dec and not x.categorical, features)))
        self.dec_cat = ValuesHouse(list(filter(lambda x: x.dec and x.categorical, features)))

        self.seq_last = seq_last
        self.device = device
        self.mode = mode

    def __len__(self):
        return self.series_size * self.num_samples_per_series

    def __getitem__(self, items):
        series_idx = []
        enc_idx = []
        dec_idx = []
        for item in items:
            series_idx.append(item // self.num_samples_per_series)
            start_idx = item % self.num_samples_per_series
            mid_idx = start_idx + self.enc_len
            end_idx = mid_idx + self.dec_len
            enc_idx.append(np.arange(start_idx, mid_idx))
            dec_idx.append(np.arange(mid_idx, end_idx))
        series_idx = np.array(series_idx)
        enc_idx = np.stack(enc_idx, axis=0)
        dec_idx = np.stack(dec_idx, axis=0)

    def check_to_tensor(self, x, dtype):
        if x is None:
            return x
        x = torch.as_tensor(x)
        if dtype == 'float32':
            x = x.float()
        elif dtype == 'int64':
            x = x.long()
        return x.to(self.device)

    def read_batch(self, batch_series_idx, enc_time_idx, dec_time_idx):

        dec_len = dec_time_idx.shape[1]
        enc_x = self.xy.read_batch(batch_series_idx, enc_time_idx, self.seq_last)
        feed_dict = {
            "enc_x": self.check_to_tensor(enc_x, 'float32'),
            "enc_num": self.check_to_tensor(
                self.enc_num.read_batch(batch_series_idx, enc_time_idx, self.seq_last), 'float32'),
            "enc_cat": self.check_to_tensor(
                self.enc_cat.read_batch(batch_series_idx, enc_time_idx, self.seq_last), 'int64'),
            "dec_x": self.check_to_tensor(
                self.xy.read_batch(batch_series_idx, dec_time_idx-1, self.seq_last), 'float32'),
            "dec_num": self.check_to_tensor(
                self.dec_num.read_batch(batch_series_idx, dec_time_idx, self.seq_last), 'float32'),
            "dec_cat": self.check_to_tensor(
                self.dec_cat.read_batch(batch_series_idx, dec_time_idx, self.seq_last), 'int64'),
            "dec_len": dec_len
        }
        if self.mode != "test":
            dec_x = self.xy.read_batch(batch_series_idx, dec_time_idx, self.seq_last)
            if self.weights is not None:
                weights = self.weights.read_batch(batch_series_idx, dec_time_idx, self.seq_last)
                return feed_dict, self.check_to_tensor(dec_x, 'float32'), self.check_to_tensor(weights, 'float32')
            else:
                return feed_dict, self.check_to_tensor(dec_x, 'float32').float(), None
        else:
            return feed_dict


class ValuesHouse:

    def __init__(self, values=None):
        self.values = None if values is None or values == [] else values

    def read_batch(self, series_idx, time_idx, seq_last):
        if self.values is None:
            return None
        else:
            if seq_last:
                batch = np.concatenate([f.read_batch(series_idx, time_idx, seq_last) for f in self.values], axis=1)
            else:
                batch = np.concatenate([f.read_batch(series_idx, time_idx, seq_last) for f in self.values], axis=2)
            return batch


class Values:

    """
    default seq last
    """

    def __init__(self, values, name, enc=True, dec=True, categorical=False, category_embed=None, mapping=None):
        assert len(values.shape) in [2, 3]
        self.values = values
        self.name = name
        self.is_property = True if len(values.shape) == 2 else False
        self.enc = enc
        self.dec = dec
        self.categorical = categorical
        if self.categorical: assert isinstance(categorical, List) and isinstance(categorical[0], Tuple)
        self.category_embed = category_embed
        self.mapping = mapping

    @property
    def size(self):
        if self.categorical:
            return self.category_embed
        else:
            if self.is_property:
                return self.values.shape[1]
            else:
                return self.values.shape[2]

    def read_batch(self, series_idx, time_idx):
        if self.mapping is not None:
            series_idx = np.array([self.mapping.get(idx) for idx in series_idx])
        if self.is_property:
            batch = np.repeat(np.expand_dims(self.values[series_idx], axis=2), time_idx.shape[1], axis=2)
        else:
            batch_size = series_idx.shape[0] * time_idx.shape[0]
            seq_len = time_idx.shape[1]
            dim = self.values.shape[1]
            batch = self.values[series_idx][:, :, time_idx].transpose(0, 2, 1, 3).reshape(batch_size, dim, seq_len)
        return batch

    def split_by_time(self, *idxes):
        assert isinstance(idxes, tuple)
        if self.is_property:
            return [self for idx in idxes]
        else:
            return [Values(self.values[:, :, idx], self.name, self.enc, self.dec, self.categorical,
                           self.category_embed, self.mapping) for idx in idxes]


class DeepSeriesDataLoader:

    def __init__(self, xy, enc_len, dec_len, sample_rate, batch_size=32, weights=None, drop_last=False,
                 device=DEFAULT_DEVICE, random_seed=42, features=None, val_starts=None, val_ends=None):
        self.xy = xy
        self.enc_len = enc_len
        self.dec_len = dec_len
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.weights = weights
        self.drop_last = drop_last
        self.device = device
        self.random_seed = np.random.RandomState(random_seed)
        if features is not None:
            assert isinstance(features, list)
        self.features = features
        self.val_starts = val_starts if val_starts is not None else np.zeros(self.xy.values.shape[2])
        self.val_ends = val_ends if val_ends is not None else np.ones(self.xy.values.shape[2])*self.xy.values.shape[2]-1
        assert np.all((self.val_ends - self.val_starts - enc_len - dec_len + 1) > 0)

    def __len__(self):
        ns = np.floor(np.sum(self.val_ends - self.val_starts - self.enc_len - self.dec_len + 1) * self.sample_rate)
        nb = ns // self.batch_size + max(0, -int(self.drop_last))
        return nb

    def __iter__(self):
        pass

    @property
    def enc_num(self):
        if self.features is None: return None
        blocks = [block for block in self.features if block.enc and not block.categorical]
        if len(blocks) > 0:
            return blocks
        else:
            return None

    @property
    def enc_cat(self):
        if self.features is None: return None
        blocks = [block for block in self.features if block.enc and block.categorical]
        if len(blocks) > 0:
            return blocks
        else:
            return None

    @property
    def dec_num(self):
        if self.features is None: return None
        blocks = [block for block in self.features if block.dec and not block.categorical]
        if len(blocks) > 0:
            return blocks
        else:
            return None

    @property
    def dec_cat(self):
        if self.features is None: return None
        blocks = [block for block in self.features if block.dec and block.categorical]
        if len(blocks) > 0:
            return blocks
        else:
            return None



class TimeSeries:

    def __init__(self, values, idx_map=None):
        """
        Args:
            values: shape(N, dim, seq)
            idx_map: dict
        """
        assert isinstance(values, np.ndarray)
        assert len(values.shape) == 3
        self.values = values
        self.idx_map = idx_map

    def read_batch(self, series_idx, time_idx, seq_last):
        """
        Args:
            series_idx: shape(I)
            time_idx: shape(J, seq)
            seq_last(bool)

        Returns:
            shape(batch, dim, seq)
        """
        if self.idx_map is not None:
            series_idx = np.array([self.idx_map[i] for i in series_idx])
        batch_size = series_idx.shape[0] * time_idx.shape[0]
        seq_len = time_idx.shape[1]
        if seq_last:
            dim = self.values.shape[1]
            batch = self.values[series_idx][:, :, time_idx].transpose(0, 2, 1, 3).reshape(batch_size, dim, seq_len)
        else:
            dim = self.values.shape[2]
            batch = self.values[series_idx][:, time_idx].reshape(batch_size, seq_len, dim)
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

    def read_batch(self, series_idx, time_idx, seq_last):
        """

        Args:
            series_idx: shape(J)
            time_idx: (I, seq)
            seq_last(bool)

        Returns:
            shape(N, dim, seq)
        """
        if self.idx_map is not None:
            series_idx = np.array([self.idx_map[i] for i in series_idx])
        if seq_last:
            batch = np.repeat(np.expand_dims(self.values[series_idx], axis=2), time_idx.shape[1], axis=2)
        else:
            batch = np.repeat(np.expand_dims(self.values[series_idx], axis=1), time_idx.shape[1], axis=1)
        return batch


class FeatureStore:

    def __init__(self, features=None):
        self.features = features

    def read_batch(self, series_idx, time_idx, seq_last):
        if self.features is None:
            return None
        else:
            if seq_last:
                batch = np.concatenate([f.read_batch(series_idx, time_idx, seq_last) for f in self.features], axis=1)
            else:
                batch = np.concatenate([f.read_batch(series_idx, time_idx, seq_last) for f in self.features], axis=2)
            return batch


class Seq2SeqDataLoader:

    def __init__(self, xy, batch_size, enc_lens, dec_lens, use_cuda=False, weights=None,
                 time_free_space=0, time_interval=1, mode="train", drop_last=False, seq_last=True,
                 enc_num_feats=None, dec_num_feats=None, enc_cat_feats=None, dec_cat_feats=None, random_seed=42):
        self.xy = xy
        self.series_size = xy.values.shape[0]
        self.series_idx = np.arange(xy.values.shape[0])
        self.time_dim = 2 if seq_last else 1
        self.time_size = xy.values.shape[self.time_dim]
        self.time_idx = np.arange(xy.values.shape[self.time_dim])
        self.batch_size = batch_size
        self.enc_lens = enc_lens
        self.dec_lens = dec_lens
        self.seq_last = seq_last

        self.use_cuda = use_cuda
        self._time_free_space = time_free_space
        self.time_interval = time_interval
        self.mode = mode
        self.drop_last = drop_last

        self.weights = FeatureStore([weights]) if weights is not None else None
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
            size = len(self.val_time_start_idx) // self.batch_size
            if size % self.batch_size == 0 or self.drop_last:
                return size // self.batch_size
            else:
                return size // self.batch_size + 1
        else:
            if self.drop_last or self.series_size % self.batch_size == 0:
                return self.series_size // self.batch_size
            else:
                return self.series_size // self.batch_size + 1

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
        enc_x = self.xy.read_batch(batch_series_idx, enc_time_idx, self.seq_last)
        feed_dict = {
            "enc_x": self.check_to_tensor(enc_x, 'float32'),
            "enc_num": self.check_to_tensor(
                self.enc_num_feats.read_batch(batch_series_idx, enc_time_idx, self.seq_last), 'float32'),
            "enc_cat": self.check_to_tensor(
                self.enc_cat_feats.read_batch(batch_series_idx, enc_time_idx, self.seq_last), 'int64'),
            "dec_x": self.check_to_tensor(
                self.xy.read_batch(batch_series_idx, dec_time_idx-1, self.seq_last), 'float32'),
            "dec_num": self.check_to_tensor(
                self.dec_num_feats.read_batch(batch_series_idx, dec_time_idx, self.seq_last), 'float32'),
            "dec_cat": self.check_to_tensor(
                self.dec_cat_feats.read_batch(batch_series_idx, dec_time_idx, self.seq_last), 'int64'),
            "dec_len": dec_len
        }

        if self.mode != "test":
            dec_x = self.xy.read_batch(batch_series_idx, dec_time_idx, self.seq_last)
            if self.weights is not None:
                weights = self.weights.read_batch(batch_series_idx, dec_time_idx, self.seq_last)
                return feed_dict, self.check_to_tensor(dec_x, 'float32'), self.check_to_tensor(weights, 'float32')
            else:
                return feed_dict, self.check_to_tensor(dec_x, 'float32').float(), None
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

    def check_to_tensor(self, x, dtype):
        if x is None:
            return x
        x = torch.as_tensor(x)
        if dtype == 'float32':
            x = x.float()
        elif dtype == 'int64':
            x = x.long()
        if self.use_cuda:
            x = x.cuda()
        return x


def forward_split(time_idx, enc_len, dec_len, valid_size):
    if valid_size < 1:
        valid_size = int(len(time_idx) * valid_size)
    valid_idx = time_idx[-(valid_size + enc_len + dec_len):]
    train_idx = time_idx[:-valid_size]
    return train_idx, valid_idx
