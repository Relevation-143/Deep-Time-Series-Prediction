# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/4/1 17:03
"""
class Learner:

    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def fit(self, epochs, train_dl, valid_dl=None):
        pass

    def load(self):
        pass

    def predict(self):
        pass

    def save(self):
        pass
