import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import os

# 数据路径
# DATA_PATH = 'data/SWaT/'
# NORMAL_FILE = os.path.join(DATA_PATH, 'normal.csv')
# ATTACK_FILE = os.path.join(DATA_PATH, 'attack.csv')
# SAMPLE_RATIO = 0.1  # 采样比例（增加到30%以提升模型性能）

# 加载和预处理数据
def load_and_preprocess_data(NORMAL_FILE, ATTACK_FILE,SAMPLE_RATIO = 0.1):
    print('加载数据...')
    
    # 加载正常数据 - 只采样一部分
    df_normal = pd.read_csv(NORMAL_FILE)
    # 加载攻击数据
    df_attack = pd.read_csv(ATTACK_FILE)
    
    # 采样数据以减小计算量
    sample_size_normal = int(len(df_normal) * SAMPLE_RATIO)
    sample_size_attack = int(len(df_attack) * SAMPLE_RATIO)
    
    df_normal = df_normal.iloc[:sample_size_normal]
    df_attack = df_attack.iloc[:sample_size_attack]
    
    # 清理列名（去除前导空格）
    df_normal.columns = df_normal.columns.str.strip()
    df_attack.columns = df_attack.columns.str.strip()
    
    # 移除非数值列
    feature_cols = [col for col in df_normal.columns if col not in ['Timestamp', 'Normal/Attack']]
    
    # 提取特征
    normal_data = df_normal[feature_cols].values
    attack_data = df_attack[feature_cols].values
    
    # 获取标签
    attack_labels = (df_attack['Normal/Attack'] != 'Normal').values
    
    # 数据标准化
    scaler = MinMaxScaler(feature_range=(-1, 1))
    normal_data_scaled = scaler.fit_transform(normal_data)
    attack_data_scaled = scaler.transform(attack_data)
    
    print(f'正常数据形状: {normal_data_scaled.shape}')
    print(f'攻击数据形状: {attack_data_scaled.shape}')
    print(f'特征数量: {len(feature_cols)}')
    
    return normal_data_scaled, attack_data_scaled, attack_labels, scaler