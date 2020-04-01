# coding: utf-8
"""
@Author: zhirui zhou
@Contact: evilpsycho42@gmail.com
@Time: 2020/3/31 下午10:14
"""
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import ConcatDataset
from fastai.train import DataBunch, Learner
import numpy as np


DEC_LEN = "dec_len"


def dict_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    x = batch[0][0]
    y = batch[0][1]
    xx = {k: torch.as_tensor(v) for k, v in x.items() if k != DEC_LEN}
    xx[DEC_LEN] = x.get(DEC_LEN)
    yy = torch.as_tensor(y)
    return xx, yy


class SeriesData(Dataset):
    """Single Series DataSet"""
    def __init__(self, x, enc_len, dec_len, mode="train", numerical=None, categorical=None, random_select_start=True):
        """

        Args:
            x: array like, shape(S, N)
            enc_len:
            dec_len:
            mode:
            numerical:
            categorical:
            random_select_start:
        """
        self.x = x.transpose(1, 0)
        self.seq_len = self.x.shape[1]
        self.enc_len = enc_len
        self.dec_len = dec_len
        assert mode in ["train", "eval"]
        self.mode = mode if mode else "train"
        self.numerical = numerical
        self.categorical = categorical
        self.random_select_start = random_select_start
        self.free_space = np.arange(self.seq_len - self.enc_len - self.dec_len + 1)

    def __len__(self):
        if self.random_select_start and self.mode == "train": return 1
        else: return self.seq_len - self.enc_len - self.dec_len + 1

    def __getitem__(self, item):
        if self.random_select_start and self.mode == "train":
            item = item + random.choice(self.free_space)
        enc_start = item
        enc_end = item + self.enc_len
        dec_start = enc_end
        dec_end = dec_start + self.dec_len

        feed = {
            "enc_x": self.x[:, enc_start: enc_end],
            "dec_len": self.dec_len,
        }

        if self.numerical:
            feed["enc_numerical"] = self.numerical[:, enc_start: enc_end]
            feed["dec_numerical"] = self.numerical[:, dec_start: dec_end]
        if self. categorical:
            feed["enc_categorical"] = self.categorical[:, enc_start: enc_end]
            feed["dec_categorical"] = self.categorical[:, dec_start: dec_end]

        return feed, self.x[:, dec_start: dec_end]

    def train(self):
        self.mode = "train"
        return self

    def eval(self):
        self.mode = "eval"
        return self

    def open_random(self):
        self.random_select_start = True

    def close_random(self):
        self.random_select_start = False


class ConcatDataSetEx(ConcatDataset):

    def train(self):
        for d in self.datasets:
            d.train()
        self.cumulative_sizes = self.cumsum(self.datasets)

    def eval(self):
        for d in self.datasets:
            d.eval()
        self.cumulative_sizes = self.cumsum(self.datasets)

    def open_random(self):
        for d in self.datasets:
            d.open_random()
        self.cumulative_sizes = self.cumsum(self.datasets)

    def close_random(self):
        for d in self.datasets:
            d.close_random()
        self.cumulative_sizes = self.cumsum(self.datasets)


class MultiSeriesData:

    def __new__(cls, x, enc_len, dec_len, series_id, mode="train",
                numerical=None, categorical=None, random_select_start=True):
        sid = np.unique(series_id)
        all_sets = []
        for i in sid:
            mask = series_id == i
            all_sets.append(
                SeriesData(x[mask], enc_len, dec_len, mode,
                           numerical=None if numerical is None else numerical[mask],
                           categorical=None if categorical is None else categorical[mask],
                           random_select_start=random_select_start
                           ))
        return ConcatDataSetEx(all_sets)
