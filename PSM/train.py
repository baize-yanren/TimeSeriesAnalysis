import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import sys

font_path = 'C:/Windows/Fonts/simhei.ttf'
font_prop = fm.FontProperties(fname=font_path)
matplotlib.rcParams['font.sans-serif'] = [font_prop.get_name()]
matplotlib.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import preProcessing as pp
import reference.LSTM as LSTM
import reference.RNN as RNN

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'PSM')
TRAIN_FILE = os.path.join(DATA_PATH, 'train.csv')
TEST_FILE = os.path.join(DATA_PATH, 'test.csv')
LABELS_FILE = os.path.join(DATA_PATH, 'test_label.csv')

WINDOW_SIZE = 50 # 窗口大小
BATCH_SIZE = 64 # 批次大小
EPOCHS = 50 # 训练轮数
LEARNING_RATE = 1e-4 # 学习率
HIDDEN_DIM = 64 # 隐藏层维度
NUM_LAYERS = 2 # 层数量
DROPOUT_RATE = 0.2 # Dropout率
SAMPLE_RATIO = 0.5 # 样本比例

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    model.train()
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            
            predictions = model(batch)
            loss = criterion(predictions, batch)
            
            if torch.isnan(loss):
                print('检测到NaN损失，跳过此批次')
                continue
            
            optimizer.zero_grad()
            loss.backward()
            
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            if torch.isnan(grad_norm):
                print('检测到NaN梯度，跳过此批次')
                optimizer.zero_grad()
                continue
            
            optimizer.step()
            
            epoch_loss += loss.item() * batch.size(0)
            num_batches += batch.size(0)
        
        avg_train_loss = epoch_loss / num_batches if num_batches > 0 else float('nan')
        train_losses.append(avg_train_loss)
        
        model.eval()
        val_loss = 0.0
        val_num_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                predictions = model(batch)
                loss = criterion(predictions, batch)
                if torch.isnan(loss):
                    continue
                val_loss += loss.item() * batch.size(0)
                val_num_batches += batch.size(0)
        
        avg_val_loss = val_loss / val_num_batches if val_num_batches > 0 else float('nan')
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
        
        if np.isnan(avg_train_loss) or np.isnan(avg_val_loss):
            print('检测到NaN损失，停止训练')
            break
    
    return train_losses, val_losses

def compute_anomaly_scores(model, data_loader, device):
    model.eval()
    scores = []
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            predictions = model(batch)
            mse = torch.mean((batch - predictions) ** 2, dim=(1, 2))
            scores.extend(mse.cpu().numpy())
    
    return np.array(scores)

def main():
    train_data, test_data, test_labels, scaler = pp.load_and_preprocess_data(
        TRAIN_FILE, TEST_FILE, LABELS_FILE, SAMPLE_RATIO
    )
    input_dim = train_data.shape[1]
    
    print(f'\n数据统计:')
    print(f'  训练数据样本数: {len(train_data)}')
    print(f'  测试数据样本数: {len(test_data)}')
    print(f'  特征维度: {input_dim}')
    print(f'  异常样本数量: {int(np.sum(test_labels))}')
    
    train_dataset = pp.PSMDataset(train_data, WINDOW_SIZE)
    test_dataset = pp.PSMDataset(test_data, WINDOW_SIZE)
    
    total_train = len(train_dataset)
    train_size = int(0.8 * total_train)
    val_size = total_train - train_size
    
    train_data_subset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    test_normal_mask = test_labels[:len(test_dataset)] == 0
    test_anomaly_mask = test_labels[:len(test_dataset)] == 1
    
    train_loader = DataLoader(train_data_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    os.makedirs('record', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    print('\n' + '='*50)
    print('训练LSTM模型')
    print('='*50)
    
    lstm_model = LSTM.LSTMAnomalyDetector(input_dim, HIDDEN_DIM, NUM_LAYERS, dropout_rate=DROPOUT_RATE).to(device)
    lstm_criterion = nn.MSELoss()
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE)
    
    lstm_train_losses, lstm_val_losses = train_model(lstm_model, train_loader, val_loader, lstm_criterion, lstm_optimizer, EPOCHS, device)
    
    print('\n计算LSTM异常分数...')
    lstm_val_scores = compute_anomaly_scores(lstm_model, val_loader, device)
    lstm_test_scores = compute_anomaly_scores(lstm_model, test_loader, device)
    
    lstm_threshold = np.percentile(lstm_val_scores, 85)
    lstm_y_pred = (lstm_test_scores > lstm_threshold).astype(int)
    lstm_y_true = test_labels[:len(lstm_test_scores)]
    
    lstm_acc = accuracy_score(lstm_y_true, lstm_y_pred)
    lstm_precision = precision_score(lstm_y_true, lstm_y_pred)
    lstm_recall = recall_score(lstm_y_true, lstm_y_pred)
    lstm_f1 = f1_score(lstm_y_true, lstm_y_pred)
    lstm_auc = roc_auc_score(lstm_y_true, lstm_test_scores)
    
    print(f'LSTM - AUC: {lstm_auc:.4f}, Acc: {lstm_acc:.4f}, Prec: {lstm_precision:.4f}, Rec: {lstm_recall:.4f}, F1: {lstm_f1:.4f}')
    
    torch.save(lstm_model.state_dict(), 'record/lstm_model.pth')
    
    print('\n' + '='*50)
    print('训练RNN模型')
    print('='*50)
    
    rnn_model = RNN.RNNAnomalyDetector(input_dim, HIDDEN_DIM, NUM_LAYERS, dropout_rate=DROPOUT_RATE).to(device)
    rnn_criterion = nn.MSELoss()
    rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=LEARNING_RATE)
    
    rnn_train_losses, rnn_val_losses = train_model(rnn_model, train_loader, val_loader, rnn_criterion, rnn_optimizer, EPOCHS, device)
    
    print('\n计算RNN异常分数...')
    rnn_val_scores = compute_anomaly_scores(rnn_model, val_loader, device)
    rnn_test_scores = compute_anomaly_scores(rnn_model, test_loader, device)
    
    rnn_threshold = np.percentile(rnn_val_scores, 85)
    rnn_y_pred = (rnn_test_scores > rnn_threshold).astype(int)
    rnn_y_true = test_labels[:len(rnn_test_scores)]
    
    rnn_acc = accuracy_score(rnn_y_true, rnn_y_pred)
    rnn_precision = precision_score(rnn_y_true, rnn_y_pred)
    rnn_recall = recall_score(rnn_y_true, rnn_y_pred)
    rnn_f1 = f1_score(rnn_y_true, rnn_y_pred)
    rnn_auc = roc_auc_score(rnn_y_true, rnn_test_scores)
    
    print(f'RNN - AUC: {rnn_auc:.4f}, Acc: {rnn_acc:.4f}, Prec: {rnn_precision:.4f}, Rec: {rnn_recall:.4f}, F1: {rnn_f1:.4f}')
    
    torch.save(rnn_model.state_dict(), 'record/rnn_model.pth')
    
    print('\n' + '='*50)
    print('结果对比')
    print('='*50)
    print(f'LSTM - AUC: {lstm_auc:.4f}, Acc: {lstm_acc:.4f}, Prec: {lstm_precision:.4f}, Rec: {lstm_recall:.4f}, F1: {lstm_f1:.4f}')
    print(f'RNN - AUC: {rnn_auc:.4f}, Acc: {rnn_acc:.4f}, Prec: {rnn_precision:.4f}, Rec: {rnn_recall:.4f}, F1: {rnn_f1:.4f}')
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(lstm_train_losses, label='LSTM')
    plt.plot(rnn_train_losses, label='RNN')
    plt.title('训练损失对比')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    lstm_fpr, lstm_tpr, _ = roc_curve(lstm_y_true, lstm_test_scores)
    plt.plot(lstm_fpr, lstm_tpr, label=f'LSTM (AUC = {lstm_auc:.4f})')
    
    rnn_fpr, rnn_tpr, _ = roc_curve(rnn_y_true, rnn_test_scores)
    plt.plot(rnn_fpr, rnn_tpr, label=f'RNN (AUC = {rnn_auc:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC曲线对比')
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/lstm_rnn_comparison.png')
    print('\n结果图已保存为 results/lstm_rnn_comparison.png')
    
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(lstm_test_scores[test_normal_mask[:len(lstm_test_scores)]], bins=50, alpha=0.5, label='正常数据', density=True)
    plt.hist(lstm_test_scores[test_anomaly_mask[:len(lstm_test_scores)]], bins=50, alpha=0.5, label='异常数据', density=True)
    plt.title(f'LSTM异常分数分布 (AUC={lstm_auc:.4f})')
    plt.xlabel('重构误差')
    plt.ylabel('密度')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(rnn_test_scores[test_normal_mask[:len(rnn_test_scores)]], bins=50, alpha=0.5, label='正常数据', density=True)
    plt.hist(rnn_test_scores[test_anomaly_mask[:len(rnn_test_scores)]], bins=50, alpha=0.5, label='异常数据', density=True)
    plt.title(f'RNN异常分数分布 (AUC={rnn_auc:.4f})')
    plt.xlabel('重构误差')
    plt.ylabel('密度')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/anomaly_scores_distribution.png')
    print('异常分数分布图已保存为 results/anomaly_scores_distribution.png')
    
    print('\n模型已保存: record/lstm_model.pth, record/rnn_model.pth')

if __name__ == '__main__':
    main()