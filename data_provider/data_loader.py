import os
import sys
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
from utils.augmentation import run_augmentation_single
import random

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0) 

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)
            
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_M4(Dataset):
    def __init__(self, args, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly'):
        # size [seq_len, label_len, pred_len]
        # init
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask


class PSMSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(os.path.join(root_path, 'train.csv'))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(os.path.join(root_path, 'test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = pd.read_csv(os.path.join(root_path, 'test_label.csv')).values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "MSL_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "MSL_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMAP_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=100, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMD_test_label.npy"))

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SWATSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
        test_data = pd.read_csv(os.path.join(root_path, 'swat2.csv'))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class UEAloader(Dataset):
    """
    Dataset class for datasets included in:
        Time Series Classification Archive (www.timeseriesclassification.com)
    Argument:
        limit_size: float in (0, 1) for debug
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """ 

    def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
        self.args = args
        self.root_path = root_path
        self.flag = flag
        self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=flag)
        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)
        

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        # pre_process
        normalizer = Normalizer()
        self.feature_df = normalizer.normalize(self.feature_df)
        print(len(self.all_IDs))

    def load_all(self, root_path, file_list=None, flag=None):
        """
        Loads datasets from csv files contained in `root_path` into a dataframe, optionally choosing from `pattern`
        Args:
            root_path: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_path` to consider.
                Otherwise, entire `root_path` contents will be used.
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """
        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_path, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_path, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_path, '*')))
        if flag is not None:
            data_paths = list(filter(lambda x: re.search(flag, x), data_paths))
        input_paths = [p for p in data_paths if os.path.isfile(p) and p.endswith('.ts')]
        if len(input_paths) == 0:
            pattern='*.ts'
            raise Exception("No .ts files found using pattern: '{}'".format(pattern))

        all_df, labels_df = self.load_single(input_paths[0])  # a single file contains dataset

        return all_df, labels_df

    def load_single(self, filepath):
        df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                             replace_missing_vals_with='NaN')
        labels = pd.Series(labels, dtype="category")
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(labels.cat.codes,
                                 dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

        lengths = df.applymap(
            lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series

        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            df = df.applymap(subsample)

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)

        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df

    def instance_norm(self, case):
        if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
            case /= stdev
            return case
        else:
            return case


    def __getitem__(self, ind):
        batch_x = self.feature_df.loc[self.all_IDs[ind]].values
        labels = self.labels_df.loc[self.all_IDs[ind]].values
        if self.flag == "TRAIN" and self.args.augmentation_ratio > 0:
            num_samples = len(self.all_IDs)
            num_columns = self.feature_df.shape[1]
            seq_len = int(self.feature_df.shape[0] / num_samples)
            batch_x = batch_x.reshape((1, seq_len, num_columns))
            batch_x, labels, augmentation_tags = run_augmentation_single(batch_x, labels, self.args)

            batch_x = batch_x.reshape((1 * seq_len, num_columns))

        return self.instance_norm(torch.from_numpy(batch_x)), \
               torch.from_numpy(labels)

    def __len__(self):
        return len(self.all_IDs)


class batteryloader(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='MS', data_path='final_data-CH41.csv',
                 target='Temperature', scale=True, timeenc=1, 
                 freq='5S', seasonal_patterns=None):
        '''self.timeenc == 1 自定义的时间特征编码函数'''
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                        self.data_path))

        '''
        df_raw.columns: ['time', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('time')
        df_raw = df_raw[['time'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['time']][border1:border2]
        df_stamp['time'] = pd.to_datetime(df_stamp.time)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.time.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.time.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.time.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.time.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['time'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['time'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

# 希望能够在这个内容中完成所有的train vail test的设置
class cameraloader_all(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='MS', data_path='data_20221110.csv/data_20220722.csv/data_20220729.csv/data_20221110.csv/data_20230315.csv',
                 target='label', scale=True, timeenc=1, 
                 freq='S', seasonal_patterns=None):
        '''self.timeenc == 1 自定义的时间特征编码函数'''
        self.args = args
        self.scale = self.args.scaling
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.source_path = self.args.source_path
        #source_path = 'data_20221110.csv/data_20220722.csv/data_20220729.csv/data_20220801.csv/data_20230315.csv'
        # total_data_20221110.csv/total_data_20220722.csv/total_data_20220729.csv/total_data_20220801.csv/total_data_20230315.csv
        # source_path = 'image_features_video20220722_with_label.csv/image_features_video20220729_with_label.csv/image_features_video20220801_with_label.csv/image_features_video20221110_with_label.csv/image_features_video20230315_with_label.csv'
        self.file_list = self.source_path.split('/')

        # 逐个读取每个CSV文件
        dataframes = []
        all_data_stamp = []
        if self.args.camera:
            index_range = []
            for file in self.file_list:
                self.data_path = file
                df_p = pd.read_csv(os.path.join(self.root_path,
                                            self.data_path))
                index_range.append(len(df_p))
                dataframes.append(df_p)
                if self.args.camera:
                    # 时间特征
                    df_stamp = df_p[['time']]
                    df_stamp['time'] = pd.to_datetime(df_stamp.time)
                    if self.timeenc == 0:
                        df_stamp['month'] = df_stamp.time.apply(lambda row: row.month, 1)
                        df_stamp['day'] = df_stamp.time.apply(lambda row: row.day, 1)
                        df_stamp['weekday'] = df_stamp.time.apply(lambda row: row.weekday(), 1)
                        df_stamp['hour'] = df_stamp.time.apply(lambda row: row.hour, 1)
                        data_stamp = df_stamp.drop(['time'], 1).values
                    elif self.timeenc == 1:
                        data_stamp = time_features(pd.to_datetime(df_stamp['time'].values), freq=self.freq)
                        data_stamp = data_stamp.transpose(1, 0)
                        # print(f"data_stamp 类型: {type(data_stamp)}, 形状: {data_stamp.shape}")
                        all_data_stamp.append(data_stamp)
            merged_data_stamp = np.concatenate(all_data_stamp, axis=0)
            all_data_stamp = torch.tensor(merged_data_stamp)
            # print(f"all_data_stamp 类型: {type(all_data_stamp)}, 形状: {all_data_stamp.shape}")
            self.data_stamp = all_data_stamp
            self.index_range = index_range
            # 如果你想合并所有的DataFrame，例如按行合并：
            self.combined_df = pd.concat(dataframes, ignore_index=True)
            self.__read_data__(self.combined_df)
        else:
            self.__read_data__()

    def __read_data__(self,df_raw = None):
        self.scaler = StandardScaler()
        if self.args.camera:
            df_raw = df_raw
        else:
            df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['time', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        # print(self.target)
        cols.remove(self.target)
        
        cols.remove('time')
        df_raw = df_raw[['time'] + cols + [self.target]]
        #
        length = len(df_raw) -(self.seq_len - self.pred_len + 1)*len(self.index_range)

        begin = self.index_range.copy()  # 使用 .copy() 来避免修改原始列表
        begin.insert(0, 0)  # 直接插入 0 在第一个位置
        list1 = [list(range(sum(begin[:_]),sum(self.index_range[:_]) - 1 - ((self.seq_len + self.pred_len - 1)),1)) for _ in range(1,6)]
        flattened_list = [item for sublist in list1 for item in sublist]
        index_list = list(range(len(flattened_list)))

        x = self.file_list.index(self.args.data_path)
        if self.args.testype == 'surgery-out':
            TandV_list = list1[x]
            Test_list = TandV_list[:int(len(TandV_list) * 0.7)]
            Vail_list = TandV_list[int(len(TandV_list) * 0.7):]
            Train_list = [item for sublist in list1[:x] + list1[x+1:] for item in sublist]
    
        elif self.args.testype == 'sequence-out':
            TandV_list = list1[x]
            Test_list = TandV_list[:int(len(TandV_list) * 0.4)]
            Vail_list = TandV_list[int(len(TandV_list) * 0.4):int(len(TandV_list) * 0.5)]
            Train_list = [item for sublist in list1[:x] + list1[x+1:] for item in sublist]
            Train_list.extend(TandV_list[int(len(TandV_list) * 0.5):])
        #
        random.shuffle(Test_list)
        random.shuffle(Vail_list)
        random.shuffle(Train_list)
        Test_list = Test_list[:int(len(Test_list))]
        Vail_list = Vail_list[:int(len(Vail_list))]
        Train_list = Train_list[:int(len(Train_list))]

        #
        index_list = list(range(len(Test_list)))
        self.Test_dict = dict(zip(index_list, flattened_list)) 
        index_list = list(range(len(Vail_list)))
        self.Vail_dict = dict(zip(index_list, flattened_list)) 
        index_list = list(range(len(Train_list)))
        self.Train_dict = dict(zip(index_list, flattened_list))  

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            # print(cols_data)
            # exit()
            df_data = df_raw[cols_data]
        if self.scale:
            features = df_data.drop(columns=[self.target])
            target = df_data[self.target]
            train_data = features 
            self.scaler.fit(train_data.values)
            scaled_features = self.scaler.transform(features.values)
            data = pd.DataFrame(scaled_features, columns=features.columns)
            data[self.target] = target.values
        else:
            data = df_data.values

        self.data_x = torch.tensor(data.values)
        self.data_y = torch.tensor(data.values)

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

    def __getitem__(self, index):
        if self.set_type == 0 :
            s_begin = self.Train_dict[index]
        elif self.set_type == 1:
            s_begin = self.Vail_dict[index]
        elif self.set_type == 2:
            s_begin = self.Test_dict[index]
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.set_type == 0:
            return len(self.Train_dict)
        elif self.set_type == 1:
            return len(self.Vail_dict)
        elif self.set_type == 2:
            return len(self.Test_dict)


    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class cameraloader_all_valeqtest(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='MS',
                 data_path='data_20221110.csv/data_20220722.csv/data_20220729.csv/data_20221110.csv/data_20230315.csv',
                 target='label', scale=True, timeenc=1, freq='S',
                 seasonal_patterns=None,  # ← 新增，保持兼容
                 **kwargs):                # ← 新增，吞掉其它未知参数
        # 其余代码保持不变...
        self.args = args
        self.root_path = root_path
        self.target = target
        self.features = features
        self.scale = bool(scale)
        self.timeenc = timeenc
        self.freq = freq

        # 窗口长度
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = size

        # split 标记
        assert flag in ['train', 'val', 'test']
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]

        # 文件列表
        self.source_path = getattr(self.args, 'source_path', data_path)
        self.file_list = self.source_path.split('/')

        # 随机种子
        seed = getattr(self.args, 'seed', None)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        # 读入并拼接各文件 + 生成时间编码
        dfs, stamps, index_range = [], [], []
        for fname in self.file_list:
            df = pd.read_csv(os.path.join(self.root_path, fname))
            index_range.append(len(df))
            dfs.append(df)

            t = df[['time']].copy()
            t['time'] = pd.to_datetime(t['time'])
            if self.timeenc == 0:
                t['month'] = t['time'].apply(lambda r: r.month)
                t['day'] = t['time'].apply(lambda r: r.day)
                t['weekday'] = t['time'].apply(lambda r: r.weekday())
                t['hour'] = t['time'].apply(lambda r: r.hour)
                s = t.drop(columns=['time']).values
            else:
                s = time_features(pd.to_datetime(t['time'].values), freq=self.freq).transpose(1, 0)
            stamps.append(s)

        self.index_range = index_range
        self.data_stamp = torch.tensor(np.concatenate(stamps, axis=0))  # [T, d_t]
        self.df_all = pd.concat(dfs, ignore_index=True)

        # 构建数据与切分
        self.__read_data__(self.df_all)

    def __read_data__(self, df_raw: pd.DataFrame) -> None:
        """
        依赖的成员变量：
        - self.args: 需包含 data_path（作为 test 基准文件名），可选 testype ∈ {'surgery-out','sequence-out'}
        - self.file_list: 源文件名列表（含 self.args.data_path）
        - self.index_range: 各文件的行数列表（与 file_list 一一对应）
        - self.seq_len, self.pred_len, self.features, self.target, self.scale
        - self.set_type: {'train':0,'val':1,'test':2} 由外层构造时设置
        该函数会设置：
        - self.Train_dict / self.Val_dict / self.Test_dict  (起点索引字典)
        - self.feature_idx / self.target_idx
        - self.data_x / self.data_y (torch.tensor)
        - self.scaler (StandardScaler, 仅在 train encoder 段上 fit)
        """
        import numpy as np
        import pandas as pd
        import torch
        from sklearn.preprocessing import StandardScaler

        # ---------- 列重排：['time'] + features + [target] ----------
        cols = list(df_raw.columns)
        if 'time' not in cols or self.target not in cols:
            raise ValueError("df_raw 需包含 'time' 与 target 列。")
        cols.remove(self.target); cols.remove('time')
        df_raw = df_raw[['time'] + cols + [self.target]]

        # ---------- 计算每个文件的不跨界合法起点 ----------
        offsets = [0]
        for i in range(1, len(self.index_range)):
            offsets.append(offsets[i - 1] + int(self.index_range[i - 1]))
        need_len = int(self.seq_len + self.pred_len)
        per_file_starts = []
        for i, flen in enumerate(self.index_range):
            start0 = offsets[i]
            last_start = start0 + int(flen) - need_len
            per_file_starts.append(list(range(start0, last_start + 1)) if last_start >= start0 else [])

        # ---------- 基准文件（test 来自该文件） ----------
        if getattr(self.args, 'data_path', None) not in self.file_list:
            raise ValueError("args.data_path 必须存在于源文件列表中。")
        cur_idx = self.file_list.index(self.args.data_path)
        base_list = per_file_starts[cur_idx]

        # ---------- 划分 ----------
        testype = getattr(self.args, 'testype', 'surgery-out')

        if testype == 'surgery-out':
            if len(per_file_starts) < 2:
                raise ValueError("surgery-out 至少需要 2 个源文件。")
            # 选一份不同文件做验证
            val_idx = (cur_idx - 1) % len(per_file_starts)
            if val_idx == cur_idx and len(per_file_starts) >= 3:
                val_idx = (cur_idx + 1) % len(per_file_starts)

            test_list = list(base_list)                   # 测试 = 基准文件全部窗口
            val_list  = list(per_file_starts[val_idx])    # 验证 = 另一文件全部窗口
            train_list = [s for fi, li in enumerate(per_file_starts)
                        if fi not in {cur_idx, val_idx} for s in li]

        elif testype == 'sequence-out':
            # 同文件时间切分：末尾 test，中间 gap，再取 val，再前面 train；其它文件也入 train
            starts = list(base_list)
            n = len(starts)
            if n < 8:
                raise ValueError("sequence-out 需要足够多的窗口（>=8）。")
            test_ratio = 0.2
            val_ratio  = 0.2
            gap = max(int(need_len), 1)

            n_test = max(1, int(round(n * test_ratio)))
            n_val  = max(1, int(round(n * val_ratio)))

            test_starts = starts[-n_test:]
            val_end     = n - n_test - gap
            val_start   = max(0, val_end - n_val)
            val_starts  = starts[val_start:val_end] if val_end > val_start else []

            train_end   = val_start - gap
            train_starts_same_file = starts[:max(0, train_end)]

            train_others = [s for fi, li in enumerate(per_file_starts) if fi != cur_idx for s in li]

            train_list = train_others + train_starts_same_file
            val_list   = val_starts
            test_list  = test_starts
        else:
            raise ValueError(f"Unknown testype: {testype}")

        # ---------- 合法性检查 ----------
        if not len(train_list): raise ValueError("Train 为空，请检查切分。")
        if not len(val_list):   raise ValueError("Val 为空，请检查切分。")
        if not len(test_list):  raise ValueError("Test 为空，请检查切分。")

        # ---------- 保存起点字典（互斥，不混入） ----------
        self.Train_dict = {i: s for i, s in enumerate(train_list)}
        self.Val_dict   = {i: s for i, s in enumerate(val_list)}
        self.Test_dict  = {i: s for i, s in enumerate(test_list)}

        # ---------- 特征表 ----------
        if self.features in ['M', 'MS']:
            df_data = df_raw[df_raw.columns[1:]]   # 去掉 time
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        else:
            raise ValueError(f"Unknown features mode: {self.features}")

        cols2 = list(df_data.columns)
        if self.target not in cols2:
            raise ValueError("features 模式导致 target 不在 df_data 中，请检查。")
        self.target_idx  = cols2.index(self.target)
        self.feature_idx = [i for i, c in enumerate(cols2) if c != self.target]

        # ---------- 仅用训练集 encoder 段拟合 scaler（避免泄露） ----------
        self.scaler = StandardScaler()
        if self.scale and len(self.feature_idx) > 0:
            feats = df_data.iloc[:, self.feature_idx]
            tgt   = df_data.iloc[:, self.target_idx]

            train_rows = set()
            for s in train_list:
                for r in range(int(s), int(s) + int(self.seq_len)):
                    train_rows.add(r)
            if not train_rows:
                train_rows = set(range(len(feats)))
            train_idx = np.array(sorted(train_rows), dtype=int)

            self.scaler.fit(feats.values[train_idx])
            scaled = self.scaler.transform(feats.values)

            data_mat = np.concatenate([scaled, tgt.values.reshape(-1, 1)], axis=1)
            new_cols = [cols2[i] for i in self.feature_idx] + [self.target]
            data_df = pd.DataFrame(data_mat, columns=new_cols)

            # 刷新索引（顺序未变，安全起见）
            cols3 = list(data_df.columns)
            self.feature_idx = [i for i, c in enumerate(cols3) if c != self.target]
            self.target_idx  = cols3.index(self.target)
        else:
            data_df = df_data.copy()

        # ---------- 存为 tensor ----------
        self.data_x = torch.tensor(data_df.values, dtype=torch.float32)
        self.data_y = torch.tensor(data_df.values, dtype=torch.float32)

        # ---------- 仅训练集做数据增强（如有该函数） ----------
        if self.set_type == 0 and getattr(self.args, 'augmentation_ratio', 0) > 0:
            try:
                self.data_x, self.data_y, _ = run_augmentation_single(self.data_x, self.data_y, self.args)  # noqa
            except Exception:
                pass


    # ----------------- Dataset 接口 ----------------- #
    def __getitem__(self, index: int):
        if self.set_type == 0:
            s_begin = self.Train_dict[index]
        elif self.set_type == 1:
            s_begin = self.Val_dict[index]
        else:
            s_begin = self.Test_dict[index]

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.set_type == 0:
            return len(self.Train_dict)
        elif self.set_type == 1:
            return len(self.Val_dict)
        else:
            return len(self.Test_dict)

    def inverse_transform(self, data: np.ndarray):
        return self.scaler.inverse_transform(data)