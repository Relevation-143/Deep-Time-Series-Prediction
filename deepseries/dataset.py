# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/5/21 9:33
"""
import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler, WeightedRandomSampler, Sampler
import copy
from typing import List, Tuple
from deepseries.log import get_logger
import json
import numpy as np

DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
logger = get_logger(__name__)


class Values:

    """
    default seq last
    """

    def __init__(self, values, name, enc=True, dec=True, mapping=None):
        assert isinstance(values, np.ndarray)
        assert len(values.shape) in [2, 3]
        self.values = values
        self.name = name
        self.enc = enc
        self.dec = dec
        self.is_property = True if len(values.shape) == 2 else False
        self.is_cat = True if values.dtype.name in ["int8", "int16", "int32", "int64"] else False
        self.mapping = mapping

    def read_batch(self, series_idx, time_idx):
        if self.mapping is not None:
            series_idx = np.array([self.mapping.get(idx) for idx in series_idx])
        if self.is_property:
            batch = np.repeat(np.expand_dims(self.values[series_idx], axis=2), time_idx.shape[1], axis=2)
        else:
            batch = []
            for s, t in zip(series_idx, time_idx):
                batch.append(self.values[s][:, t])
            batch = np.stack(batch, axis=0)
        return batch

    def sub(self, time_idx):
        if self.is_property:
            return self
        else:
            return Values(self.values[:, :, time_idx], self.name, self.enc, self.dec, self.mapping)


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

        self.enc_num = self.filter(lambda x: x.enc and not x.is_cat)
        self.enc_cat = self.filter(lambda x: x.enc and x.is_cat)
        self.dec_num = self.filter(lambda x: x.dec and not x.is_cat)
        self.dec_cat = self.filter(lambda x: x.dec and x.is_cat)

        self.seq_last = seq_last
        self.device = device
        self.mode = mode

    @property
    def info(self):
        return {
            "series": {"shape": " x ".join(map(str, self.xy.values.shape))},
            "enc_num": None if self.enc_num is None else [{"name": f.name, "shape": " x ".join(map(str, f.values.shape))}
                                                          for f in self.enc_num],
            "dec_num": None if self.dec_num is None else [{"name": f.name, "shape": " x ".join(map(str, f.values.shape))}
                                                          for f in self.dec_num],
            "enc_cat": None if self.enc_cat is None else [{"name": f.name, "shape": " x ".join(map(str, f.values.shape))}
                                                          for f in self.enc_cat],
            "dec_cat": None if self.dec_cat is None else [{"name": f.name, "shape": " x ".join(map(str, f.values.shape))}
                                                          for f in self.dec_cat],
            "weights": None if self.weights is None else {"shape": " x ".join(map(str, self.weights.values.shape))}
        }

    def filter(self, func):
        if self.features is None:
            return None
        ret = list(filter(func, self.features))
        if len(ret) == 0: return None
        return ret

    def read_batch(self, features, series_idx, time_idx):
        if features is None:
            return None
        batch = np.concatenate([f.read_batch(series_idx, time_idx) for f in features], axis=1)
        if not self.seq_last:
            batch = batch.transpose([0, 2, 1])
        return torch.as_tensor(batch, dtype=torch.long if features[0].is_cat else torch.float, device=self.device)

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

        feed_x = {
            "enc_x": self.read_batch([self.xy], series_idx, enc_idx),
            "dec_x": self.read_batch([self.xy], series_idx, dec_idx-1),
            "enc_num": self.read_batch(self.enc_num, series_idx, enc_idx),
            "dec_num": self.read_batch(self.dec_num, series_idx, dec_idx),
            "enc_cat": self.read_batch(self.enc_cat, series_idx, enc_idx),
            "dec_cat": self.read_batch(self.dec_cat, series_idx, dec_idx),
            "dec_len": dec_idx.shape[1],
        }
        feed_y = self.read_batch([self.xy], series_idx, dec_idx)
        weight = self.read_batch([self.weights], series_idx, dec_idx) if self.weights is not None else None
        return feed_x, feed_y, weight


def seq2seq_collate_fn(batch):
    (x, y, weight) = batch[0]
    return x, y, weight


class DeepSeriesSampler(Sampler):

    def __init__(self, data_source, batch_size, num_iteration_per_epoch, seed=42):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_iteration_per_epoch = num_iteration_per_epoch
        self.seed = np.random.RandomState(seed)

    def __iter__(self):
        samples = np.arange(len(self.data_source))
        return iter([self.seed.choice(samples, self.batch_size, replace=False)
                     for i in range(self.num_iteration_per_epoch)])

    def __len__(self):
        return self.num_iteration_per_epoch


def forward_split(time_idx, enc_len, valid_size):
    if valid_size < 1:
        valid_size = int(np.floor(len(time_idx) * valid_size))
    valid_idx = time_idx[-(valid_size + enc_len):]
    train_idx = time_idx[:-valid_size]
    return train_idx, valid_idx


def create_seq2seq_data_loader(series, enc_len, dec_len, time_idx, batch_size, num_iteration_per_epoch, weights=None,
                               features=None, seq_last=False, device=DEFAULT_DEVICE, mode='train', seed=42):
    series = Values(series, 'series').sub(time_idx)
    weights = None if weights is None else Values(weights, 'weights').sub(time_idx)
    features = None if features is None else [f.sub(time_idx) for f in features]
    data_set = DeepSeriesDataSet(series, enc_len, dec_len, weights, features, seq_last, device, mode)
    sampler = DeepSeriesSampler(data_set, batch_size, num_iteration_per_epoch, seed)
    data_loader = DataLoader(data_set, sampler=sampler, collate_fn=seq2seq_collate_fn, num_workers=8)
    logger.info("---------- dataset information ----------")
    logger.info(json2str(data_loader.dataset.info))
    proportion = batch_size * num_iteration_per_epoch / len(data_set)
    logger.info(f"data loader sampling proportion of each epoch: {proportion*100:.1f}%")
    return data_loader


def json2str(data):
    return json.dumps(data, sort_keys=True, indent=4, separators=(', ', ': '), ensure_ascii=False)
