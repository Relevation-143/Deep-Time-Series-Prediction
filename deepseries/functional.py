# encoding: utf-8
# Author: 周知瑞
# Mail: evilpsycho42@gmail.com
import typing
import numpy as np
from scipy.signal import convolve2d


def make_lags(series, n_lags, use_smooth=False):
    """

    Args:
        series (ndarray): shape N x S
        n_lags (int or list):
        use_smooth (bool):  default False

    Returns:
        series_lags, if n_lags == 1: shape N x S, else shape N x L x S
    """

    if isinstance(n_lags, typing.Iterable):
        return np.concatenate([make_lags(series, l, use_smooth) for l in n_lags], axis=1)
    else:
        assert series.shape[1] > n_lags
        if use_smooth and (series.shape[1] > 3):
            left = make_lags(series, n_lags-1, use_smooth=False)
            mid = make_lags(series, n_lags, use_smooth=False)
            right = make_lags(series, n_lags+1, use_smooth=False)
            return left * 0.25 + mid * 0.5 + right * 0.25
        else:
            lag = np.zeros_like(series)
            lag[:, n_lags:] = series[:, :-n_lags]
            lag[:, :n_lags] = np.nan
            return np.expand_dims(lag, axis=1)


# TODO numba jit speed
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


# TODO numba jit speed
def batch_autocorr(data, lags, starts, ends, threshold, backoffset=0, use_smooth=False, smooth_offset=1):
    """
    Args:
        data: Time series, shape [n_pages, n_days]
        lags: Autocorrelation lag
        starts: Start index for each series
        ends: End index for each series
        threshold: Minimum support (ratio of time series length to lag) to calculate meaningful autocorrelation.
        backoffset: Offset from the series end, days.
        use_smooth:

    Returns:
        autocorrelation, shape [n_series]. If series is too short (support less than threshold),
        autocorrelation value is NaN
    """
    if isinstance(lags, typing.Iterable):
        return np.concatenate(
            [batch_autocorr(data, l, starts, ends, threshold, backoffset, use_smooth, smooth_offset) for l in lags],
            axis=1)
    else:
        if lags < 3:
            use_smooth = False
        n_series = data.shape[0]
        n_days = data.shape[1]
        max_end = n_days - backoffset
        corr = np.empty(n_series, dtype=np.float64)
        support = np.empty(n_series, dtype=np.float64)
        for i in range(n_series):
            series = data[i]
            end = min(ends[i], max_end)
            real_len = end - starts[i]
            support[i] = real_len/lags
            if support[i] > threshold:
                series = series[starts[i]:end]
                if use_smooth:
                    c_365 = single_autocorr(series, lags)
                    c_364 = single_autocorr(series, lags-smooth_offset)
                    c_366 = single_autocorr(series, lags+smooth_offset)
                    # Average value between exact lag and two nearest neighborhoods for smoothness
                    corr[i] = 0.5 * c_365 + 0.25 * c_364 + 0.25 * c_366
                else:
                    corr[i] = single_autocorr(series, lags)
            else:
                corr[i] = np.NaN
        return np.expand_dims(corr, 1)  # support


def smooth(x, window=3, ratio=0.5, mode='center'):
    """

    Args:
        x (ndarray): input series with shape(N, S)
        window (int): smooth window
        ratio (float): (0, 1), the bigger the more smooth
        mode (string): center, causal

    Returns:
        smooth_x
    """
    assert window // 2 != 0
    weight = [ratio ** (window // 2 - i) if i <= window // 2 else ratio ** (i - window // 2) for i in
              range(window)]
    weight = np.array(weight) / np.sum(weight)
    weight = np.expand_dims(weight, 0)
    if mode == "center":
        pad_x = np.pad(x, [(0, 0), (int((window - 1) / 2), int((window - 1) / 2))], 'edge')
    elif mode == "causal":
        pad_x = np.pad(x, [(0, 0), (int((window - 1)), 0)], 'edge')
    else:
        raise ValueError("Only support center or causal mode.")
    return convolve2d(pad_x, weight, mode='valid')


def mask_zero_nan(x, mask_zero=True, mask_nan=True):
    mask = np.zeros_like(x).astype(bool)
    if mask_zero:
        mask = np.bitwise_or(mask, x == 0)
    if mask_nan:
        mask = np.bitwise_or(mask, np.isnan(x))
    return mask


def get_valid_start_end(data, mask=None):
    """

    Args:
        data (ndarray): shape N x S x D
        mask (ndarray of bool): invalid mask
    Returns:

    """
    ns = data.shape[0]
    nt = data.shape[1]
    start_idx = np.full(ns, -1, dtype=np.int32)
    end_idx = np.full(ns, -1, dtype=np.int32)

    for s in range(ns):
        # scan from start to the end
        for t in range(nt):
            if not mask[s][t]:
                start_idx[s] = t
                break
        # reverse scan, from end to start
        for t in range(nt - 1, -1, -1):
            if not mask[s][t]:
                end_idx[s] = t + 1
                break
    return start_idx, end_idx


def get_trend(x, max_T, use_smooth=True, smooth_windows=5, smooth_ration=0.5):
    if use_smooth:
        x = smooth(x, smooth_windows, smooth_ration)
    lag = make_lags(x, max_T, use_smooth).squeeze()
    return np.where(lag == 0, 0, x / lag)

