# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/4/3 14:33
"""
import torch
import random
import numpy as np


VALID_FTYPE = ['num', 'cat']


class _BaseFeature:

    def read_batch(self, *args, **kwargs):
        raise NotImplemented


class SeriesFeature(_BaseFeature):

    def __init__(self, name, ftype, series, idx_map=None, embedding=None, enc=True, dec=True):
        """

        Args:
            name (string): feature name
            ftype (string): 'num' or 'cat'
            series (np.array): shape(N, seq)
            idx_map (dict): series_idx to custom_feature_idx
            embedding (tuple or list): (feature_num_unique, embedding_size)
            enc (bool): use for encoder
            dec (bool): use for decoder
        """
        self.name = name
        self.ftype = ftype
        assert self.ftype in VALID_FTYPE
        if ftype == "cat":
            assert embedding is not None
        self.embedding = embedding
        self.series = series
        self.idx_map = idx_map
        self.enc = enc
        self.dec = dec

    def read_batch(self, series_idx, time_idx):
        """

        Args:
            series_idx (int, 1D np.array):
            time_idx (1D or 2D np.array):

        Returns:

        """

        if isinstance(series_idx, int):
            series_idx = [series_idx]

        if len(series_idx) == 1:
            # single series
            batch = self.series[series_idx, :][:, time_idx].squeeze(0)
        else:
            # multi series_idx single time_idx
            if self.idx_map is not None:
                series_idx = np.array([self.idx_map[i] for i in series_idx])
            batch = self.series[series_idx, :][:, time_idx]
        return batch


class PropertyFeature(_BaseFeature):

    def __init__(self, name, ftype, value_map, embedding=None, enc=True, dec=True):
        """

        Args:
            name (string): feature name
            ftype (string): 'num' or 'cat'
            value_map (dict): series_idx to property mapping
            embedding (tuple or list): (feature_num_unique, embedding_size)
            enc (bool): use for encoder
            dec (bool): use for decoder
        """
        self.name = name
        self.ftype = ftype
        assert self.ftype in VALID_FTYPE
        if ftype == "cat":
            assert embedding is not None
        self.embedding = embedding
        self.value_map = value_map
        self.enc = enc
        self.dec = dec

    def read_batch(self, series_idx, time_idx):
        if isinstance(series_idx, int):
            series_idx = [series_idx]

        var = np.array([self.value_map[k] for k in series_idx])
        if len(time_idx.shape) == 1:
            # single time
            time_len = len(time_idx)
        else:
            # multi time
            time_len = time_idx.shape[2]
        return np.repeat(np.expand_dims(var, axis=1), time_len, axis=1)


class FeatureStore:

    def __init__(self, features=None):
        self.features = [] if features is None else features

    def read_batch(self, series_idx, time_idx):
        if len(self.features) == 0:
            return None
        else:
            batch = np.stack([f.read_batch(series_idx, time_idx) for f in self.features], axis=1)
            return batch

    def __str__(self):
        return f"{self.__class__.__name__}: {','.join([f.name for f in self.features])}"

    def __repr__(self):
        return self.__str__()


class SeriesFrame:

    def __init__(self, series, *features, batch_size, enc_lens, dec_lens,
                 time_free_space=None, mode="train", shuffle=True):
        self.series = series
        self.series_idx = np.arange(series.series.shape[0])
        self.num_series = len(self.series_idx)
        self.times_idx = np.arange(series.series.shape[1])
        self.series_len = len(self.times_idx)

        self.enc_num_feats = FeatureStore([f for f in features if f.ftype == "num" and f.enc])
        self.dec_num_feats = FeatureStore([f for f in features if f.ftype == "num" and f.dec])
        self.enc_cat_feats = FeatureStore([f for f in features if f.ftype == "cat" and f.enc])
        self.dec_cat_feats = FeatureStore([f for f in features if f.ftype == "cat" and f.dec])

        self.batch_size = batch_size
        self.enc_lens = enc_lens
        self.dec_lens = dec_lens

        self._time_free_space = time_free_space if time_free_space is not None else 0
        assert self._time_free_space < enc_lens and self._time_free_space < dec_lens

        self.is_single = self.num_series == 1
        self.mode = mode
        self._shuffle = shuffle

    def shuffle(self):
        if self.mode == 'train':
            random.shuffle(self.series_idx)

    @property
    def time_free_space(self):
        if self.mode == "eval":
            return 0
        else:
            return self._time_free_space

    def get_time_free(self):
        return random.randint(-self._time_free_space, self._time_free_space)

    def __len__(self):
        num_time = (self.series_len - self.enc_lens - self.dec_lens - self.time_free_space * 2 + 1) // self.batch_size
        return self.num_series * num_time

    @property
    def num_batchs(self):
        return self.__len__()

    @property
    def num_samples(self):
        return self.__len__() * self.batch_size

    def _single_batch_generate(self):
        start_space = np.arange(0, self.series_len - self.enc_lens - self.dec_lens - self.time_free_space * 2 + 1)
        if self.shuffle:
            random.shuffle(start_space)
        for batch in range(len(start_space) // self.batch_size):
            batch_start_idxes = start_space[batch * self.batch_size: (batch + 1) * self.batch_size]
            enc_len = self.enc_lens + self.get_time_free()
            dec_len = self.dec_lens + self.get_time_free()
            enc_time_idx, dec_time_idx = [], []
            for start in batch_start_idxes:
                enc_time_idx.append(np.arange(start, start+enc_len))
                dec_time_idx.append(np.arange(start+enc_len, start+enc_len+dec_len))
            enc_time_idx = np.array(enc_time_idx)
            dec_time_idx = np.array(dec_time_idx)
            enc_x = np.expand_dims(self.series.read_batch(0, enc_time_idx), 1)
            dec_x = self.series.read_batch(0, dec_time_idx)

            feed_dict = {
                "enc_x": check_to_tensor(enc_x),
                "enc_numerical": check_to_tensor(self.enc_num_feats.read_batch(0, enc_time_idx)),
                "enc_categorical": check_to_tensor(self.enc_cat_feats.read_batch(0, enc_time_idx)),
                "dec_numerical": check_to_tensor(self.dec_num_feats.read_batch(0, dec_time_idx)),
                "dec_categorical": check_to_tensor(self.dec_cat_feats.read_batch(0, dec_time_idx)),
                "dec_len": dec_len
            }
            yield feed_dict, check_to_tensor(dec_x)

    def _multi_generate_batch(self):
        start_space = np.arange(0, self.series_len - self.enc_lens - self.dec_lens - self.time_free_space * 2 + 1)
        if self.shuffle:
            random.shuffle(self.series_idx)
        for batch in range(len(self.series_idx) // self.batch_size):
            batch_series_idx = self.series_idx[batch * self.batch_size: (batch + 1) * self.batch_size]
            start = random.choice(start_space)
            enc_len = self.enc_lens + self.get_time_free()
            dec_len = self.dec_lens + self.get_time_free()
            enc_time_idx = np.arange(start, start+enc_len)
            dec_time_idx = np.arange(start+enc_len, start+enc_len+dec_len)

            enc_x = self.series.read_batch(batch_series_idx, enc_time_idx)
            dec_x = self.series.read_batch(batch_series_idx, dec_time_idx)

            feed_dict = {
                "enc_x": check_to_tensor(enc_x),
                "enc_numerical": check_to_tensor(self.enc_num_feats.read_batch(batch_series_idx, enc_time_idx)),
                "enc_categorical": check_to_tensor(self.enc_cat_feats.read_batch(batch_series_idx, enc_time_idx)),
                "dec_numerical": check_to_tensor(self.dec_num_feats.read_batch(batch_series_idx, dec_time_idx)),
                "dec_categorical": check_to_tensor(self.dec_cat_feats.read_batch(batch_series_idx, dec_time_idx))
            }
            yield feed_dict, check_to_tensor(dec_x)

    def generate_batch(self):
        if self.is_single:
            return self._single_batch_generate()
        else:
            return self._multi_generate_batch()

    def __iter__(self):
        return self.generate_batch()


def check_to_tensor(x):
    if x is None:
        return x
    return torch.as_tensor(x)


if __name__ == "__main__":
    # single test
    x = SeriesFeature("x", "num", np.random.rand(1, 200))
    f_series_num = SeriesFeature("SeriesFeature", "num", np.random.rand(1, 200))
    f_series_cat = SeriesFeature("SeriesCat", "cat", np.random.randint(0, 100, (1, 200)), embedding=(100, 2))
    frame = SeriesFrame(x, f_series_cat, f_series_num, batch_size=4, enc_lens=20, dec_lens=20, time_free_space=2)

    for x, y in frame:
        print(y.shape)

    # multi test
    x = SeriesFeature("x", "num", np.random.rand(20, 200))
    f_series_num = SeriesFeature("SeriesFeature", "num", np.random.rand(20, 200))
    f_series_cat = SeriesFeature("SeriesCat", "cat", np.random.randint(0, 100, (20, 200)), embedding=(100, 2))
    f_proper_num = PropertyFeature("proper", "num", np.random.randint(0,10,20))
    frame = SeriesFrame(x, f_series_cat, f_series_num, f_proper_num, batch_size=4, enc_lens=20, dec_lens=20, time_free_space=2)

    for x, y in frame:
        print(y.shape)
