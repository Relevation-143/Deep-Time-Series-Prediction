# coding: utf-8
"""
@Author: zhirui zhou
@Contact: evilpsycho42@gmail.com
@Time: 2020/4/9 上午1:35
"""
from deepseries.dataset import *
import numpy as np


class TestDataSet:

    def test_time_feature(self):
        v = np.random.rand(4, 4, 20)
        s_idx = np.array([0, 1])
        t_idx = np.array([[2, 3, 4], [3, 4, 5]])
        f = SeriesFeature(v)
        batch = f.read_batch(s_idx, t_idx)
        assert np.all(batch[0] == v[s_idx[0]][:, t_idx[0]])
        assert np.all(batch[2] == v[s_idx[1]][:, t_idx[0]])
