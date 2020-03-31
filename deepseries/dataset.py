# coding: utf-8
"""
@Author: zhirui zhou
@Contact: evilpsycho42@gmail.com
@Time: 2020/3/31 下午10:14
"""
import torch
from torch.utils.data import Dataset, DataLoader
from fastai.train import DataBunch


DEC_LEN = "dec_len"


def collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    x = batch[0][0]
    y = batch[0][1]
    xx = {k: torch.as_tensor(v) for k, v in x.items() if k != DEC_LEN}
    xx[DEC_LEN] = x.get(DEC_LEN)
    yy = torch.as_tensor(y)
    return xx, yy


class SingleSeriesData(Dataset):

    def __init__(self, values, encode_len, decode_len,):
        self.values = values
        self.encode_len = encode_len
        self.decode_len = decode_len
        self.source_len = values.shape[1]

    def __len__(self):
        return self.source_len - self.encode_len - self.decode_len + 1

    def __getitem__(self, item):
        return dict(enc_x=self.values[:, item: item+self.encode_len],
                    dec_len=self.decode_len), \
               self.values[:, item+self.encode_len: item+self.encode_len+self.decode_len]


if __name__ == "__main__":
    import numpy as np
    from fastai.train import DataBunch, Learner
    from fastai.basic_train import BasicLearner
    from torch.nn import MSELoss
    from deepseries.models import WaveNet
    values = (np.sin(np.arange(1, 200)) + np.log1p(np.arange(1, 200))).reshape(1, -1).astype("float32")
    mu, std = values.mean(), values.std()
    values = (values - mu) / std

    dset = SingleSeriesData(values, 100, 50)
    dl = DataLoader(dset, collate_fn=collate)

    db = DataBunch(dl, dl)
    net = WaveNet(
                residual_channels=4,
                 skip_channels=4,
                 dilations=[2 ** i for i in range(8)] * 2,
                 kernels_size=[2 for i in range(8)] * 2)
    # net.cuda()
    learner = Learner(db, net, loss_func=MSELoss())
    learner.fit(5)

    import matplotlib.pyplot as plt
    test_values = (np.sin(np.arange(100, 200)) + np.log1p(np.arange(100, 200))).reshape(1, -1).astype("float32")
    test_values = (test_values - mu) / std
    test_values_tensor = torch.as_tensor(np.expand_dims(test_values, 1)).cuda()
    plt.plot(net.predict(test_values_tensor, 50).detach().cpu().numpy().reshape(-1), label="pred")
    y = (((np.sin(np.arange(200, 250)) + np.log1p(np.arange(200, 250))).reshape(1, -1).astype("float32") - mu) / std).reshape(-1)
    plt.plot(y)
    plt.legend()

    import matplotlib.pyplot as plt
    offset = 100
    test_values = (np.sin(np.arange(100+offset, 200+offset)) + np.log1p(np.arange(100+offset, 200+offset))).reshape(1, -1).astype("float32")
    test_values = (test_values - mu) / std
    test_values_tensor = torch.as_tensor(np.expand_dims(test_values, 1)).cuda()
    plt.plot(net.predict(test_values_tensor, 50).detach().cpu().numpy().reshape(-1), label="pred")
    y = (((np.sin(np.arange(200+offset, 250+offset)) + np.log1p(np.arange(200+offset, 250+offset))).reshape(1, -1).astype("float32") - mu) / std).reshape(-1)
    plt.plot(y)
    plt.legend()
