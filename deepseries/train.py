# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/4/1 17:03
"""
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import logging


class Learner:

    def __init__(self, model, optimizer, loss_fn, root_dir, log_interval=4):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.root_dir = root_dir
        self.log_dir = os.path.join(root_dir, 'logs')
        self.model_dir = os.path.join(root_dir, 'checkpoints')
        for i in [self.root_dir, self.log_dir, self.model_dir]:
            if not os.path.exists(i):
                os.mkdir(i)
        self.epochs = 0
        self.losses = []
        self.init_logging()
        self.log_interval = log_interval

    def init_logging(self):

        date_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
        log_file = 'log_{}.txt'.format(date_str)
        logging.basicConfig(
            filename=os.path.join(self.log_dir, log_file),
            level=logging.INFO,
            format='[[%(asctime)s]] %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p'
        )
        logging.getLogger().addHandler(logging.StreamHandler())

    def fit(self, max_epochs, train_dl, valid_dl, early_stopping=True, patient=10, save_start_epoch=-1):
        with SummaryWriter(self.log_dir) as writer:
            # writer.add_graph(self.model)
            bad_epochs = 0
            global_steps = 0
            for epoch in range(max_epochs):
                self.model.train()
                train_loss = 0
                for i, (x, y) in enumerate(train_dl):
                    loss = self.loss_batch(x, y)
                    writer.add_scalar("Loss/train", loss, global_steps)
                    global_steps += 1
                    train_loss += loss
                    if global_steps % self.log_interval == 0:
                        logging.info(f"epoch: {epoch} / {max_epochs}, batch: {i/len(train_dl)*100:.0f}%, "
                                     f"train loss {train_loss / (i+1):.4f}")
                valid_loss = 0
                self.model.eval()
                for x, y in valid_dl:
                    loss = self.eval_batch(x, y)
                    valid_loss += loss / len(valid_dl)
                writer.add_scalar("Loss/valid", valid_loss, global_steps)
                logging.info(f"epoch: {epoch} / {max_epochs} finished, valid loss {valid_loss:.4f}")

                if epoch >= save_start_epoch:
                    self.save()

                self.losses.append(valid_loss)
                self.epochs += 1
                if early_stopping:
                    if self.epochs > 1:
                        if valid_loss >= min(self.losses):
                            bad_epochs += 1
                        else:
                            bad_epochs = 0
                        if bad_epochs > patient:
                            break

    def loss_batch(self, x, y):
        if isinstance(x, dict):
            y_hat = self.model.predict(**x)
        else:
            y_hat = self.model.predict(*x)
        loss = self.loss_fn(y_hat, y)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def eval_batch(self, x, y):
        with torch.no_grad():
            if isinstance(x, dict):
                y_hat = self.model.predict(**x)
            else:
                y_hat = self.model.predict(*x)
            loss = self.loss_fn(y, y_hat)
        return loss.item()

    def load(self, model_checkpoint_path):
        checkpoint = torch.load(model_checkpoint_path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epochs = checkpoint['epochs']

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def save(self):
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epochs": self.epochs,
        }

        name = f"model-epoch-{self.epochs}.pkl"
        torch.save(checkpoint, os.path.join(self.model_dir, name))
