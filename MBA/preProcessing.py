import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import os

def load_and_preprocess_data(TRAIN_FILE, TEST_FILE, LABELS_FILE, SAMPLE_RATIO=0.8):
    print('加载MBA数据...')
    
    df_train = pd.read_excel(TRAIN_FILE)
    df_test = pd.read_excel(TEST_FILE)
    df_labels = pd.read_excel(LABELS_FILE)
    
    df_train = df_train.iloc[1:]
    df_test = df_test.iloc[1:]
    
    df_train = df_train.astype({'sample': int, 'ECG1': float, 'ECG2': float})
    df_test = df_test.astype({'sample': int, 'ECG1': float, 'ECG2': float})
    
    feature_cols = ['ECG1', 'ECG2']
    
    train_data = df_train[feature_cols].values
    test_data = df_test[feature_cols].values
    
    normal_mask = df_labels['#'] == 'N'
    anomaly_mask = df_labels['#'] != 'N'
    
    normal_sample_indices = df_labels[normal_mask]['Sample'].values
    anomaly_sample_indices = df_labels[anomaly_mask]['Sample'].values
    
    train_size = int(len(train_data) * SAMPLE_RATIO)
    train_data = train_data[:train_size]
    test_size = int(len(test_data) * SAMPLE_RATIO)
    test_data = test_data[:test_size]
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_data_scaled = scaler.fit_transform(train_data)
    test_data_scaled = scaler.transform(test_data)
    
    test_labels = np.zeros(len(test_data_scaled))
    for idx in anomaly_sample_indices:
        if idx < len(test_labels):
            test_labels[idx] = 1
    
    print(f'训练数据形状: {train_data_scaled.shape}')
    print(f'测试数据形状: {test_data_scaled.shape}')
    print(f'特征数量: {len(feature_cols)}')
    print(f'异常样本数量: {int(np.sum(test_labels))}')
    
    return train_data_scaled, test_data_scaled, test_labels, scaler

class MBADataset(Dataset):
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size
    
    def __len__(self):
        return len(self.data) - self.window_size
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx:idx+self.window_size])