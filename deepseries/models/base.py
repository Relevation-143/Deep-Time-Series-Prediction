# encoding: utf-8
# Author: 周知瑞
# Mail: evilpsycho42@gmail.com

from collections import deque
import os
import logging
from torch import nn, optim
from datetime import datetime
from deepseries.utils import HyperParameters
from fastai.train import Learner, ItemBase


class EMA:

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


class BaseTorchModel:

    """ Base Torch Model

    """

    hp = HyperParameters()

    def __init__(self, save_dir):
        self.hp.save_dir = save_dir
        self.hp.checkpoint_dir = os.path.join(self.hp.save_dir, "checkpoints")
        self.hp.log_dir = os.path.join(self.hp.save_dir, "logs")
        self.ema = None
        self.opt = None
        self.lr_scheduler = None
        self.init_logging()

    def set_opt(self, optimizer, lr, **kw):
        kw = dict() if kw is None else kw
        self.opt = getattr(nn, optimizer)(self.parameters(), lr, **kw)

    def get_lr_scheduler(self, opt, lr_scheduler, **kw):
        if lr_scheduler is not None:
            kw = dict() if kw is None else kw
            self.lr_scheduler = getattr(nn, lr_scheduler)(opt, lr_scheduler, **kw)

    def compile(self,
                max_training_step,
                min_steps_to_checkpoint,
                save_every_n_step,
                log_every_n_step,
                loss_averaging_window,
                lr=0.001,
                optimizer='adam',
                optimizer_kw=None,
                lr_scheduler=None,
                lr_scheduler_kw=None,
                grad_clip=5,
                patience=None):
        self.set_opt(optimizer, lr, **optimizer_kw)
        self.set_opt(self.opt, lr_scheduler, **lr_scheduler_kw)

    def restore(self):
        pass

    def save(self):
        pass

    def fit(self):
        assert self.opt is not None, "can't fit, need compile first or restore model."
        pass

    def calculate_loss(self):
        raise NotImplementedError("subclass must implement this")

    def predict(self):
        raise NotImplementedError("subclass must implement this")

    def init_logging(self):
        if not os.path.isdir(self.hp.log_dir):
            os.makedirs(self.hp.log_dir)

        date_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
        log_file = 'log_{}.txt'.format(date_str)
        logging.basicConfig(
            filename=os.path.join(self.hp.log_dir, log_file),
            level=logging.INFO,
            format='[[%(asctime)s]] %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p'
        )
        logging.getLogger().addHandler(logging.StreamHandler())


# 模型，超参数，训练记录，