# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/5/20 17:17
"""
import pytest
from deepseries.dataset import DataBlock
import numpy as np


def test_data_block():
    x1 = np.random.rand(4, 8, 12)
    d1 = DataBlock(x1, 'x1', enc=True, dec=True, categorical=False, seq_last=True)
    d11, d12 = d1.split_by_time([1, 2, 3], [2, 3, 4])
    assert d11.size == 8
    assert d11.x.shape == (4, 8, 3)
