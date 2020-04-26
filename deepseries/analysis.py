# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/4/26 14:10
"""
import numpy as np
import matplotlib.pyplot as plt
import typing


def single_autocorr(series, lag):
    """
    Autocorrelation for single data series
    :param series: traffic series
    :param lag: lag, days
    :return:
    """
    s1 = series[lag:]
    s2 = series[:-lag]
    ms1 = np.mean(s1)
    ms2 = np.mean(s2)
    ds1 = s1 - ms1
    ds2 = s2 - ms2
    divider = np.sqrt(np.sum(ds1 * ds1)) * np.sqrt(np.sum(ds2 * ds2))
    return np.sum(ds1 * ds2) / divider if divider != 0 else 0


def batch_autocorr(data, lag, starts, ends, threshold, backoffset=0, smooth=False):
    """
    Calculate autocorrelation for batch (many time series at once)
    :param data: Time series, shape [n_pages, n_days]
    :param lag: Autocorrelation lag
    :param starts: Start index for each series
    :param ends: End index for each series
    :param threshold: Minimum support (ratio of time series length to lag) to calculate meaningful autocorrelation.
    :param backoffset: Offset from the series end, days.
    :return: autocorrelation, shape [n_series]. If series is too short (support less than threshold),
    autocorrelation value is NaN
    """
    n_series = data.shape[0]
    n_days = data.shape[1]
    max_end = n_days - backoffset
    corr = np.empty(n_series, dtype=np.float64)
    support = np.empty(n_series, dtype=np.float64)
    for i in range(n_series):
        series = data[i]
        end = min(ends[i], max_end)
        real_len = end - starts[i]
        support[i] = real_len/lag
        if support[i] > threshold:
            series = series[starts[i]:end]
            if smooth:
                c_365 = single_autocorr(series, lag)
                c_364 = single_autocorr(series, lag-1)
                c_366 = single_autocorr(series, lag+1)
                # Average value between exact lag and two nearest neighborhoods for smoothness
                corr[i] = 0.5 * c_365 + 0.25 * c_364 + 0.25 * c_366
            else:
                corr[i] = single_autocorr(series, lag)
        else:
            corr[i] = np.NaN
    return corr  # support


class SeriesAnalysisModel:

    def __init__(self, series, valid_mask=None, zero_valid=False):
        self.series = series
        self.valid_mask = valid_mask
        self.zero_valid = zero_valid
        self.series_corr = None
        self.starts, self.ends = self.get_valid_start_end(self.series)
        self.valid_lens = self.ends - self.starts

    def get_autocorr(self, lag, threshold=1.5, backoffset=0, smooth=False):
        """

        Args:
            lag:
            threshold: Minimum support (ratio of time series length to lag) to calculate meaningful autocorrelation.
            backoffset:
            smooth:

        Returns:

        """
        if isinstance(lag, typing.Iterable):
            series_corr = np.concatenate([self.get_autocorr(l) for l in lag], axis=1)
        else:
            series_corr = np.expand_dims(
                batch_autocorr(self.series, lag, self.starts, self.ends, threshold, backoffset=backoffset,
                               smooth=smooth), 1)
        return series_corr

    def get_valid_start_end(self, data: np.ndarray):
        """
        Calculates start and end of real traffic data. Start is an index of first non-zero, non-NaN value,
         end is index of last non-zero, non-NaN value
        :param data: Time series, shape [n_pages, n_days]
        :return:
        """
        n_series = data.shape[0]
        n_time = data.shape[1]
        start_idx = np.full(n_series, -1, dtype=np.int32)
        end_idx = np.full(n_series, -1, dtype=np.int32)

        if self.valid_mask is None:
            if self.zero_valid:
                self.valid_mask = (~np.isnan(self.series)) & (self.series != 0)
            else:
                self.valid_mask = ~np.isnan(self.series)

        for s in range(n_series):
            # scan from start to the end
            for t in range(n_time):
                if self.valid_mask[s][t]:
                    start_idx[s] = t
                    break
            # reverse scan, from end to start
            for t in range(n_time - 1, -1, -1):
                if self.valid_mask[s][t]:
                    end_idx[s] = t
                    break
        return start_idx, end_idx

    def plot_valid(self, figsize=(8, 6)):
        f, ax = plt.subplots(1, 1, figsize=figsize)
        im = ax.imshow(self.valid_mask, aspect="auto", vmin=0, vmax=1)
        f.colorbar(im, ax=ax)
        ax.set_title("series valid value map (zero means invliad location)")
        return im

    def plot_autocorr(self, corr, figsize=(8, 6)):
        f, ax = plt.subplots(1, 1, figsize=figsize)
        im = ax.imshow(corr, aspect="auto", vmin=np.nan_to_num(corr).min(), vmax=np.nan_to_num(corr).max())
        f.colorbar(im, ax=ax)
        ax.set_title("series autocorr")
        return im
