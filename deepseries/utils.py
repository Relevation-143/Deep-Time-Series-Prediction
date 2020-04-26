# encoding: utf-8
# Author: 周知瑞
# Mail: evilpsycho42@gmail.com
import typing
import numpy as np


class HyperParameters(dict):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getattr__(self, item):
        return self.get(item)

    def __setattr__(self, key, value):
        self[key] = value


class EMA:
    # TODO
    """Weights Exponential Moving Average.

    Args:
        model(torch.nn.module).
        decay(float).

    Examples:

        ema = EMA(model, 0.99)

        # train stage
        opt.step()
        ema.update()

        # eval stage
        ema.apply_shadow()
        model.predict(...)
        ema.restore()
    """

    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def make_lags(series, n_lags, smooth=False):
    if isinstance(n_lags, typing.Iterable):
        return np.stack([make_lags(series, l, smooth) for l in n_lags], axis=0)
    else:
        if n_lags < 2:
            smooth = False
        if smooth:
            left = make_lags(series, n_lags-1, smooth=False)
            mid = make_lags(series, n_lags, smooth=False)
            right = make_lags(series, n_lags+1, smooth=False)
            return left * 0.25 + mid * 0.5 + right * 0.25
        else:
            lag = np.zeros_like(series)
            lag[:, n_lags:] = series[:, :-n_lags]
            lag[:, :n_lags] = np.nan
            return lag
