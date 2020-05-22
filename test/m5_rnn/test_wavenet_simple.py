# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/5/22 16:18
"""
import os
import pandas as pd
import numpy as np
import scipy as sp
import gc
from sklearn.preprocessing import LabelEncoder
from deepseries.models import Wave2WaveV1
from deepseries.train import Learner
from deepseries.dataset import Values, create_seq2seq_data_loader, forward_split
import deepseries.functional as F
from deepseries.nn.loss import RMSELoss
from torch.optim import Adam
from deepseries.optim import ReduceCosineAnnealingLR


x = np.sin(np.arange(1000))


trn_ld = create_seq2seq_data_loader(x.reshape(1, 1, 1000), 24, 24, np.arange(800), 12, 4, seq_last=True)
val_ld = create_seq2seq_data_loader(x.reshape(1, 1, 1000), 24, 24, np.arange(800, 1000), 12, 4, seq_last=True)

model = Wave2WaveV1(1, num_layers=1)
opt = Adam(model.parameters(), 0.001)
loss_fn = RMSELoss()
lr_scheduler = ReduceCosineAnnealingLR(opt, 64, eta_min=1e-4, gamma=0.998)
model.cuda()
learner = Learner(model, opt, './m5_rnn', lr_scheduler=lr_scheduler, verbose=1)
learner.fit(50, trn_ld, val_ld, patient=64, start_save=-1, early_stopping=True)