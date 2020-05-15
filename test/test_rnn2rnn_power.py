# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/5/13 10:17
"""
from deepseries.models.rnn2rnn import RNN2RNN
from deepseries.train import Learner
from deepseries.dataset import TimeSeries, FeatureStore, Seq2SeqDataLoader
import numpy as np
from torch.optim import Adam
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from deepseries.models import BasicSeq2Seq
from deepseries.dataset import Property, TimeSeries, Seq2SeqDataLoader
from deepseries.nn.loss import MSELoss, RMSELoss
from deepseries.train import Learner
from deepseries.optim import ReduceCosineAnnealingLR
import deepseries.functional as F
from deepseries.analysis import SeriesAnalysisModel
from torch.optim import Adam
from torch import nn
import matplotlib as mpl
from sklearn.metrics import mean_absolute_error
import chinese_calendar as calendar
import datetime as dt


def normalize(x, axis, fill_zero=True):
    mu = np.nanmean(x, axis, keepdims=True)
    std = np.nanstd(x, axis, keepdims=True)
    x_norm = (x - mu) / std
    if fill_zero:
        x_norm = np.nan_to_num(x_norm)
    return x_norm, mu, std


power = pd.read_csv('./data/df.csv', parse_dates=['data_time'])[['data_time', 'cid', 'value']]
power = power.set_index("data_time").groupby("cid").resample("1H").sum().reset_index()
power = power.pivot(index='cid', columns='data_time', values='value')

N_TEST = 24 * 30
N_VALID = 24 * 2
DROP_ZERO = True
DEC_LEN = 24 * 2
ENC_LEN = 24 * 7
time_free_space = 0

drop_before = 15000


xy = power.values
xy_valid = np.bitwise_and(~np.isnan(xy), xy != 0)
starts, ends = F.get_valid_start_end(xy, ~xy_valid)
xy, xy_mean, xy_std = normalize(xy, axis=1)


corr_7 = F.batch_autocorr(xy, 7*24, starts, ends, 1.05, use_smooth=False, smooth_offset=None)
corr_14 = F.batch_autocorr(xy, 14*24, starts, ends, 1.05, use_smooth=False, smooth_offset=None)
corr_365 = F.batch_autocorr(xy, 365*24, starts, ends, 1.05, use_smooth=True, smooth_offset=24)
xy_auto_corr = np.concatenate([corr_7, corr_14, corr_365], 1)
xy_auto_corr, _, _ = normalize(xy_auto_corr, 0)

xy_lags, _, _ = normalize(F.make_lags(xy, [7*24, 14*24, 365*24]), axis=2)
xy_valid = np.expand_dims(xy_valid.astype("float32"), 1)
xy_lag_valid = np.concatenate([xy_lags,  xy_valid], axis=1).astype("float32").transpose([0, 2, 1])

weights = xy_valid.transpose([0, 2, 1])
# weights[:, :, np.where((power.columns >= "2020-02-01") & (power.columns < "2020-03-01"), 1, 0)] = 0.01
# weights = weights * xy_mean / xy_mean.mean()
# weights = weights.transpose([0, 2, 1])
xy_cat = np.expand_dims(np.arange(len(weights)), 1)

def get_holiday_features(dts):
    select_holidays = ["Spring Festival", "National Day", "Labour Day", "New Year's Day", "Mid-autumn Festival", "Tomb-sweeping Day"]

    def _get_holidays(x):
        is_holiday, holiday_name = calendar.get_holiday_detail(x)
        if holiday_name in select_holidays and is_holiday:
            return holiday_name

    holidays = pd.get_dummies(pd.Series(dts).apply(lambda x: _get_holidays(x)))
    holidays['sick'] = np.where((power.columns >= "2020-02-01") & (power.columns < "2020-03-01"), 1, 0)
    holidays.index = dts
    return holidays

def holiday_apply(x, holidays, func):
    result = pd.DataFrame()
    for h in holidays.columns:
        result[h] = x.loc[:, holidays[h].values.astype(bool)].agg(func, axis=1).values
    return result

holidays = get_holiday_features(power.columns)
xy_holiday_mean = holiday_apply(power, holidays, np.mean).values
xy_holiday_mean = normalize(xy_holiday_mean, 0)[0]

xy_weekday = pd.get_dummies(power.columns.weekday).values
xy_hour = pd.get_dummies(power.columns.hour).values
xy_month = pd.get_dummies(power.columns.month).values
xy_date = np.concatenate([xy_weekday, xy_hour, xy_month, holidays], 1)
xy_date = np.repeat(np.expand_dims(xy_date, 0), xy.shape[0], axis=0)


class ForwardSpliter:

    def split(self, time_idx, enc_len, valid_size):
        if valid_size < 1:
            valid_size = int(np.floor(len(time_idx) * valid_size))
        valid_idx = time_idx[-(valid_size + enc_len):]
        train_idx = time_idx[:-valid_size]
        return train_idx, valid_idx


spliter = ForwardSpliter()
train_idx, valid_idx = spliter.split(np.arange(xy.shape[1])[drop_before:], ENC_LEN, N_TEST + N_VALID)
valid_idx, test_idx = spliter.split(valid_idx, ENC_LEN, N_TEST)

xy = np.expand_dims(xy, 2)
train_xy = TimeSeries(xy[:, train_idx])
valid_xy = TimeSeries(xy[:, valid_idx])

trn_weight = TimeSeries(weights[:, train_idx])
val_weight = TimeSeries(weights[:, valid_idx])

trn_enc_cat = [Property(xy_cat)]
val_enc_cat = [Property(xy_cat)]

trn_dec_cat = [Property(xy_cat)]
val_dec_cat = [Property(xy_cat)]

trn_enc_num = [TimeSeries(xy_date[:, train_idx]), Property(xy_holiday_mean),
               TimeSeries(xy_lag_valid[:, train_idx]), Property(xy_auto_corr)]
val_enc_num = [TimeSeries(xy_date[:, valid_idx]), Property(xy_holiday_mean),
               TimeSeries(xy_lag_valid[:, valid_idx]), Property(xy_auto_corr)]

trn_dec_num = [TimeSeries(xy_date[:, train_idx]), Property(xy_holiday_mean),
               TimeSeries(xy_lag_valid[:, train_idx]), Property(xy_auto_corr)]
val_dec_num = [TimeSeries(xy_date[:, valid_idx]), Property(xy_holiday_mean),
               TimeSeries(xy_lag_valid[:, valid_idx]), Property(xy_auto_corr)]


train_frame = Seq2SeqDataLoader(train_xy, batch_size=8, enc_lens=ENC_LEN, dec_lens=DEC_LEN, use_cuda=True,
                                mode='train', time_free_space=time_free_space, enc_num_feats=trn_enc_num,
                                enc_cat_feats=trn_enc_cat, dec_num_feats=trn_dec_num,
                                dec_cat_feats=trn_dec_cat,
                                weights=trn_weight, seq_last=False)
valid_frame = Seq2SeqDataLoader(valid_xy, batch_size=64, enc_lens=ENC_LEN, dec_lens=DEC_LEN, use_cuda=True,
                                mode='train', time_free_space=0,
                                time_interval=48,
                                enc_num_feats=val_enc_num,
                                enc_cat_feats=val_enc_cat,
                                dec_num_feats=val_dec_num,
                                dec_cat_feats=val_dec_cat,
                                seq_last=False)


model = RNN2RNN(1, hidden_size=512, compress_size=128, enc_num_size=64,
                enc_cat_size=[(62, 2)], dec_num_size=64, dec_cat_size=[(62, 2)], residual=True,
                beta1=0., beta2=0., attn_heads=1, attn_size=128, num_layers=1, dropout=0.0, rnn_type='LSTM')
opt = Adam(model.parameters(), 0.001)
loss_fn = MSELoss()
model.cuda()
lr_scheduler = ReduceCosineAnnealingLR(opt, 64, eta_min=5e-5)
learner = Learner(model, opt, './power_preds', verbose=20, lr_scheduler=None)
learner.fit(500, train_frame, valid_frame, patient=128, start_save=1, early_stopping=True)
learner.load(460)
learner.model.eval()

preds = []
trues = []
for batch in valid_frame:
    batch[0].pop('dec_x')
    preds.append(learner.model(**batch[0])[0])
    trues.append(batch[1])

trues = torch.cat(trues, 2).squeeze().cpu().numpy() * xy_std + xy_mean
preds = torch.cat(preds, 2).squeeze().detach().cpu().numpy() * xy_std + xy_mean

k = 0

plt.plot(trues[k])
plt.plot(preds[k], label='preds')
plt.legend()


test_xy = torch.as_tensor(xy[:, test_idx]).float().cuda()
test_xy_num_feats = torch.as_tensor(
    np.concatenate([xy_date[:, test_idx], np.repeat(np.expand_dims(xy_holiday_mean, 1), len(test_idx), 1),
                    xy_lag_valid[:, test_idx], np.repeat(np.expand_dims(xy_auto_corr, 1), len(test_idx), 1)],
                   axis=2)).float().cuda()
test_xy_cat_feats = torch.as_tensor(np.repeat(np.expand_dims(xy_cat, 1), test_xy.shape[1], 1)).long().cuda()


def plot(x_true, y_true, y_pred):
    enc_ticks = np.arange(x_true.shape[1])
    dec_ticks = np.arange(y_pred.shape[1]) + x_true.shape[1]
    for idx, name in enumerate(power.index):
        plt.figure(figsize=(12, 3))
        plt.plot(enc_ticks, x_true[idx])
        plt.plot(dec_ticks, y_pred[idx], label='pred')
        plt.plot(dec_ticks, y_true[idx], label='true')
        plt.title(idx)
        plt.legend()


def wmape(y_hat, y):
    scores = []
    for day in range(int(y.shape[0] / 24)):
        scores.append(np.abs(y[day * 24: (day + 1) * 24] - y_hat[day * 24: (day + 1) * 24]).sum() / np.sum(
            y[day * 24: (day + 1) * 24]))
    return scores


def metric(y_true, y_pred):
    scores = {}
    for idx, name in enumerate(power.index):
        scores[name] = wmape(y_pred[idx], y_true[idx])
    return pd.DataFrame(scores)


def predict(learner, xy, x_num, x_cat, y_num, y_cat):
    preds = []
    days = int(xy.shape[1] / 24 - ENC_LEN / 24 - DEC_LEN / 24 + 1)
    for day in range(days):
        step = day * 24
        step_pred = learner.model(
            xy[:, step: step + ENC_LEN],
            enc_num=x_num[:, step: step + ENC_LEN],
            dec_num=y_num[:, step + ENC_LEN: step + ENC_LEN + DEC_LEN],
            enc_cat=x_cat[:, step: step + ENC_LEN],
            dec_cat=y_cat[:, step + ENC_LEN: step + ENC_LEN + DEC_LEN],
            dec_len=DEC_LEN
        )[0].cpu().detach().numpy()
        preds.append(step_pred[:, -24:])

    preds = np.concatenate(preds, axis=1)
    preds = preds.squeeze() * xy_std + xy_mean

    x_true = xy[:, :ENC_LEN + 24].cpu().numpy().squeeze() * xy_std + xy_mean
    y_true = xy[:, ENC_LEN + 24:].cpu().numpy().squeeze() * xy_std + xy_mean

    return x_true, y_true, preds


norm_data = pd.read_csv("./data/20200315_20200415.csv").drop(['Unnamed: 0', 'model_name'], axis=1)
norm_data = norm_data[norm_data.contributor_id.isin(power.index)].reset_index(drop=True)
norm_data = norm_data.set_index("contributor_id").loc[power.index].reset_index()
norm_data['data_time'] = pd.to_datetime(norm_data.data_time)
norm_data = norm_data.set_index("data_time").groupby("contributor_id").resample('1H')[['forecast_pwr', 'value']].sum().reset_index()
norm_true = norm_data.pivot(index='contributor_id', columns='data_time', values='value').iloc[:, 48:]
norm_pred = norm_data.pivot(index='contributor_id', columns='data_time', values='forecast_pwr').iloc[:, 48:]


x_true, y_true, y_pred  = predict(learner, test_xy, test_xy_num_feats, test_xy_cat_feats, test_xy_num_feats, test_xy_cat_feats)
scores = pd.DataFrame([metric(y_true, y_pred).mean().rename("wave"),
                       metric(norm_true.values, norm_pred.values).mean().rename("v1")]).T.dropna()
scores.describe()
plot(x_true, y_true, y_pred)
