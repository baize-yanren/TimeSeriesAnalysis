import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import os

def load_and_preprocess_data(TRAIN_FILE, TEST_FILE, LABELS_FILE, SAMPLE_RATIO=0.8):
    print('加载PSM数据...')
    
    df_train = pd.read_csv(TRAIN_FILE)
    df_test = pd.read_csv(TEST_FILE)
    df_labels = pd.read_csv(LABELS_FILE)
    
    feature_cols = [col for col in df_train.columns if col.startswith('feature_')]
    
    train_data = df_train[feature_cols].values
    test_data = df_test[feature_cols].values
    test_labels = df_labels['label'].values
    
    print(f'训练数据NaN数量: {np.isnan(train_data).sum()}')
    print(f'测试数据NaN数量: {np.isnan(test_data).sum()}')
    
    col_means = np.nanmean(train_data, axis=0)
    nan_mask = np.isnan(train_data)
    train_data[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    
    test_data = np.nan_to_num(test_data, nan=0.0)
    
    train_size = int(len(train_data) * SAMPLE_RATIO)
    train_data = train_data[:train_size]
    test_size = int(len(test_data) * SAMPLE_RATIO)
    test_data = test_data[:test_size]
    test_labels = test_labels[:test_size]
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_data_scaled = scaler.fit_transform(train_data)
    test_data_scaled = scaler.transform(test_data)
    
    print(f'训练数据形状: {train_data_scaled.shape}')
    print(f'测试数据形状: {test_data_scaled.shape}')
    print(f'特征数量: {len(feature_cols)}')
    print(f'异常样本数量: {int(np.sum(test_labels))}')
    
    return train_data_scaled, test_data_scaled, test_labels, scaler

class PSMDataset(Dataset):
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size
    
    def __len__(self):
        return len(self.data) - self.window_size
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx:idx+self.window_size])