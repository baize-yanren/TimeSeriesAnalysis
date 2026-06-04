import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
import matplotlib
import matplotlib.pyplot as plt
import os
import preProcessing as pp
import LSTM as LSTM
import RNN as RNN

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

DATA_PATH = 'SWaT/data/SWaT/'
NORMAL_FILE = os.path.join(DATA_PATH, 'normal.csv')
ATTACK_FILE = os.path.join(DATA_PATH, 'attack.csv')

WINDOW_SIZE = 50
BATCH_SIZE = 64
EPOCHS = 20  # 增加训练轮数
MADGAN_EPOCHS = 500
LEARNING_RATE = 1e-3
HIDDEN_DIM = 128  # 增加隐藏层维度
NUM_LAYERS = 3  # 增加层数
LATENT_DIM = 64
SAMPLE_RATIO = 0.3
CONTINUE_TRAINING = True
PATIENCE = 8  # 增加早停耐心值
DROPOUT_RATE = 0.0  # 暂时关闭Dropout


class SWaTDataset(Dataset):
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size
    
    def __len__(self):
        return len(self.data) - self.window_size
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx:idx+self.window_size])

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device, patience=10, model_name='model'):
    model.train()
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            
            predictions = model(batch)
            loss = criterion(predictions, batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch.size(0)
            num_batches += batch.size(0)
        
        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)
        
        model.eval()
        val_loss = 0.0
        val_num_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                predictions = model(batch)
                loss = criterion(predictions, batch)
                val_loss += loss.item() * batch.size(0)
                val_num_batches += batch.size(0)
        
        avg_val_loss = val_loss / val_num_batches
        val_losses.append(avg_val_loss)
        
        scheduler.step()
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'record/{model_name}_best.pth')
        else:
            patience_counter += 1
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}')
        
        if patience_counter >= patience:
            print(f'验证损失不再下降，早停在第 {epoch+1} 轮')
            break
    
    model.load_state_dict(torch.load(f'record/{model_name}_best.pth'))
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

def train_madgan(generator, discriminator, train_loader, device, epochs=5, lr=1e-4):
    criterion = nn.BCELoss()
    gen_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=lr * 0.5, betas=(0.5, 0.999))
    
    gen_losses = []
    disc_losses = []
    
    smooth_real_label = 0.9
    smooth_fake_label = 0.1
    
    for epoch in range(epochs):
        epoch_gen_loss = 0.0
        epoch_disc_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            batch = batch.to(device).float()
            batch_size = batch.size(0)
            
            real_labels = torch.full((batch_size, 1), smooth_real_label, dtype=torch.float32, device=device)
            fake_labels = torch.full((batch_size, 1), smooth_fake_label, dtype=torch.float32, device=device)
            
            z = torch.randn(batch_size, WINDOW_SIZE, LATENT_DIM, device=device)
            fake = generator(z)
            
            disc_optimizer.zero_grad()
            real_output = discriminator(batch)
            fake_output = discriminator(fake.detach())
            
            real_output = torch.clamp(real_output, 1e-7, 1 - 1e-7)
            fake_output = torch.clamp(fake_output, 1e-7, 1 - 1e-7)
            
            disc_loss_real = criterion(real_output, real_labels)
            disc_loss_fake = criterion(fake_output, fake_labels)
            disc_loss = disc_loss_real + disc_loss_fake
            disc_loss.backward()
            disc_optimizer.step()
            
            gen_optimizer.zero_grad()
            fake_output = discriminator(fake)
            fake_output = torch.clamp(fake_output, 1e-7, 1 - 1e-7)
            gen_loss = criterion(fake_output, real_labels)
            gen_loss.backward()
            gen_optimizer.step()
            
            epoch_disc_loss += disc_loss.item()
            epoch_gen_loss += gen_loss.item()
            num_batches += 1
        
        avg_gen_loss = epoch_gen_loss / num_batches
        avg_disc_loss = epoch_disc_loss / num_batches
        gen_losses.append(avg_gen_loss)
        disc_losses.append(avg_disc_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Gen Loss: {avg_gen_loss:.6f}, Disc Loss: {avg_disc_loss:.6f}')
    
    return gen_losses, disc_losses

def compute_madgan_scores(discriminator, data_loader, device):
    discriminator.eval()
    scores = []
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device).float()
            disc_scores = discriminator(batch)
            scores.extend(disc_scores.cpu().numpy())
    
    return np.array(scores)

def main():
    normal_data, attack_data, attack_labels, scaler = pp.load_and_preprocess_data(NORMAL_FILE, ATTACK_FILE)
    input_dim = normal_data.shape[1]
    
    normal_dataset = SWaTDataset(normal_data, WINDOW_SIZE)
    attack_dataset_full = SWaTDataset(attack_data, WINDOW_SIZE)
    
    total_normal = len(normal_dataset)
    train_size = int(0.7 * total_normal)
    val_size = int(0.15 * total_normal)
    test_size = total_normal - train_size - val_size
    
    print(f'\n数据划分:')
    print(f'  训练集（正常数据）: {train_size} 样本')
    print(f'  验证集（正常数据）: {val_size} 样本')
    print(f'  测试集（正常数据）: {test_size} 样本')
    print(f'  攻击数据样本数: {len(attack_dataset_full)} 样本')
    
    train_dataset, val_dataset, test_normal_dataset = torch.utils.data.random_split(
        normal_dataset, [train_size, val_size, test_size]
    )
    
    total_attack = len(attack_dataset_full)
    attack_val_size = int(0.3 * total_attack)
    attack_test_size = total_attack - attack_val_size
    
    attack_val_dataset, attack_test_dataset = torch.utils.data.random_split(
        attack_dataset_full, [attack_val_size, attack_test_size]
    )
    
    print(f'  验证集（攻击数据）: {attack_val_size} 样本')
    print(f'  测试集（攻击数据）: {attack_test_size} 样本')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    attack_val_loader = DataLoader(attack_val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    test_normal_loader = DataLoader(test_normal_dataset, batch_size=BATCH_SIZE, shuffle=False)
    attack_test_loader = DataLoader(attack_test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    adjusted_labels = attack_labels[WINDOW_SIZE:]
    
    val_data_for_threshold = val_dataset
    test_data_for_eval = (test_normal_dataset, attack_test_dataset)
    
    # ============ 训练LSTM模型 ============
    print('\n' + '='*50)
    print('训练LSTM模型')
    print('='*50)
    
    lstm_model = LSTM.LSTMAnomalyDetector(input_dim, HIDDEN_DIM, NUM_LAYERS, dropout_rate=DROPOUT_RATE).to(device)
    lstm_criterion = nn.MSELoss()
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE)
    
    if CONTINUE_TRAINING:
        lstm_model_path = 'SWaT/record/lstm_model.pth'
        if os.path.exists(lstm_model_path):
            lstm_model.load_state_dict(torch.load(lstm_model_path, map_location=device))
            print(f'已加载已有模型参数: {lstm_model_path}')
        else:
            print('未找到LSTM模型参数文件')
    
    lstm_train_losses, lstm_val_losses = train_model(lstm_model, train_loader, val_loader, lstm_criterion, lstm_optimizer, EPOCHS, device, PATIENCE, 'lstm')
    
    print('\n计算LSTM异常分数...')
    lstm_val_normal_scores = compute_anomaly_scores(lstm_model, val_loader, device)
    lstm_test_normal_scores = compute_anomaly_scores(lstm_model, test_normal_loader, device)
    lstm_test_attack_scores = compute_anomaly_scores(lstm_model, attack_test_loader, device)
    
    lstm_threshold = np.percentile(lstm_val_normal_scores, 80)  # 进一步降低阈值提高召回率
    lstm_y_true = np.concatenate([np.zeros(len(lstm_test_normal_scores)), np.ones(len(lstm_test_attack_scores))])
    lstm_y_scores = np.concatenate([lstm_test_normal_scores, lstm_test_attack_scores])
    lstm_auc = roc_auc_score(lstm_y_true, lstm_y_scores)
    
    lstm_y_pred = (lstm_y_scores > lstm_threshold).astype(int)
    lstm_acc = accuracy_score(lstm_y_true, lstm_y_pred)
    lstm_precision = precision_score(lstm_y_true, lstm_y_pred)
    lstm_recall = recall_score(lstm_y_true, lstm_y_pred)
    lstm_f1 = f1_score(lstm_y_true, lstm_y_pred)
    
    print(f'LSTM AUC-ROC: {lstm_auc:.4f}, Acc: {lstm_acc:.4f}, Prec: {lstm_precision:.4f}, Rec: {lstm_recall:.4f}, F1: {lstm_f1:.4f}')
    
    # ============ 训练RNN模型 ============
    print('\n' + '='*50)
    print('训练RNN模型')
    print('='*50)
    
    rnn_model = RNN.RNNAnomalyDetector(input_dim, HIDDEN_DIM, NUM_LAYERS, dropout_rate=DROPOUT_RATE).to(device)
    rnn_criterion = nn.MSELoss()
    rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=LEARNING_RATE)
    
    if CONTINUE_TRAINING:
        rnn_model_path = 'SWaT/record/rnn_model.pth'
        if os.path.exists(rnn_model_path):
            rnn_model.load_state_dict(torch.load(rnn_model_path, map_location=device))
            print(f'已加载已有模型参数: {rnn_model_path}')
        else:
            print('未找到RNN模型参数文件')
    
    rnn_train_losses, rnn_val_losses = train_model(rnn_model, train_loader, val_loader, rnn_criterion, rnn_optimizer, EPOCHS, device, PATIENCE, 'rnn')
    
    print('\n计算RNN异常分数...')
    rnn_val_normal_scores = compute_anomaly_scores(rnn_model, val_loader, device)
    rnn_test_normal_scores = compute_anomaly_scores(rnn_model, test_normal_loader, device)
    rnn_test_attack_scores = compute_anomaly_scores(rnn_model, attack_test_loader, device)
    
    rnn_threshold = np.percentile(rnn_val_normal_scores, 80)  # 进一步降低阈值提高召回率
    rnn_y_true = np.concatenate([np.zeros(len(rnn_test_normal_scores)), np.ones(len(rnn_test_attack_scores))])
    rnn_y_scores = np.concatenate([rnn_test_normal_scores, rnn_test_attack_scores])
    rnn_auc = roc_auc_score(rnn_y_true, rnn_y_scores)
    
    rnn_y_pred = (rnn_y_scores > rnn_threshold).astype(int)
    rnn_acc = accuracy_score(rnn_y_true, rnn_y_pred)
    rnn_precision = precision_score(rnn_y_true, rnn_y_pred)
    rnn_recall = recall_score(rnn_y_true, rnn_y_pred)
    rnn_f1 = f1_score(rnn_y_true, rnn_y_pred)
    
    print(f'RNN AUC-ROC: {rnn_auc:.4f}, Acc: {rnn_acc:.4f}, Prec: {rnn_precision:.4f}, Rec: {rnn_recall:.4f}, F1: {rnn_f1:.4f}')
    
    # # ============ 训练MAD-GAN模型 ============
    # print('\n' + '='*50)
    # print('训练MAD-GAN模型')
    # print('='*50)
    
    # madgan_model = madgan.MADGANAnomalyDetector(input_dim, HIDDEN_DIM, LATENT_DIM, NUM_LAYERS).to(device)
    # generator = madgan_model.generator
    # discriminator = madgan_model.discriminator
    
    # if CONTINUE_TRAINING:
    #     generator_path = 'record/generator_madgan.pth'
    #     discriminator_path = 'record/discriminator_madgan.pth'
    
    #     if os.path.exists(generator_path):
    #         generator.load_state_dict(torch.load(generator_path, map_location=device))
    #         print(f'已加载已有Generator参数: {generator_path}')
        
    #     if os.path.exists(discriminator_path):
    #         discriminator.load_state_dict(torch.load(discriminator_path, map_location=device))
    #         print(f'已加载已有Discriminator参数: {discriminator_path}')
    
    # gen_losses, disc_losses = train_madgan(generator, discriminator, train_loader, device, MADGAN_EPOCHS, LEARNING_RATE)
    
    # print('\n计算MAD-GAN异常分数...')
    # madgan_normal_scores = compute_madgan_scores(discriminator, val_loader, device)
    # madgan_attack_scores = compute_madgan_scores(discriminator, attack_loader, device)
    
    # madgan_y_true = np.concatenate([np.zeros(len(madgan_normal_scores)), np.ones(len(madgan_attack_scores))])
    # madgan_y_scores = np.concatenate([madgan_normal_scores, madgan_attack_scores])
    # madgan_auc = roc_auc_score(madgan_y_true, madgan_y_scores)
    # print(f'MAD-GAN AUC-ROC: {madgan_auc:.4f}')
    
    # ============ 可视化结果 ============
    print('\n' + '='*50)
    print('结果对比')
    print('='*50)
    print(f'LSTM AUC-ROC: {lstm_auc:.4f}')
    print(f'RNN AUC-ROC: {rnn_auc:.4f}')
    # print(f'MAD-GAN AUC-ROC: {madgan_auc:.4f}')
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(lstm_train_losses, label='LSTM')
    plt.plot(rnn_train_losses, label='RNN')
    # plt.plot(gen_losses, label='MAD-GAN Generator')
    plt.title('训练损失对比')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    lstm_fpr, lstm_tpr, _ = roc_curve(lstm_y_true, lstm_y_scores)
    plt.plot(lstm_fpr, lstm_tpr, label=f'LSTM (AUC = {lstm_auc:.4f})')
    
    rnn_fpr, rnn_tpr, _ = roc_curve(rnn_y_true, rnn_y_scores)
    plt.plot(rnn_fpr, rnn_tpr, label=f'RNN (AUC = {rnn_auc:.4f})')
    
    # madgan_fpr, madgan_tpr, _ = roc_curve(madgan_y_true, madgan_y_scores)
    # plt.plot(madgan_fpr, madgan_tpr, label=f'MAD-GAN (AUC = {madgan_auc:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC曲线对比')
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('SWaT/results/lstm_rnn_comparison.png')
    print('\n结果图已保存为 lstm_rnn_comparison.png')
    
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(lstm_test_normal_scores, bins=50, alpha=0.5, label='正常数据', density=True)
    plt.hist(lstm_test_attack_scores, bins=50, alpha=0.5, label='攻击数据', density=True)
    plt.title(f'LSTM异常分数分布 (AUC={lstm_auc:.4f})')
    plt.xlabel('重构误差')
    plt.ylabel('密度')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(rnn_test_normal_scores, bins=50, alpha=0.5, label='正常数据', density=True)
    plt.hist(rnn_test_attack_scores, bins=50, alpha=0.5, label='攻击数据', density=True)
    plt.title(f'RNN异常分数分布 (AUC={rnn_auc:.4f})')
    plt.xlabel('重构误差')
    plt.ylabel('密度')
    plt.legend()
    
    # plt.subplot(1, 3, 3)
    # plt.hist(madgan_normal_scores, bins=50, alpha=0.5, label='正常数据', density=True)
    # plt.hist(madgan_attack_scores, bins=50, alpha=0.5, label='攻击数据', density=True)
    # plt.title(f'MAD-GAN异常分数分布 (AUC={madgan_auc:.4f})')
    # plt.xlabel('判别器分数')
    # plt.ylabel('密度')
    # plt.legend()
    
    plt.tight_layout()
    plt.savefig('SWaT/results/anomaly_scores_distribution.png')
    print('异常分数分布图已保存为 anomaly_scores_distribution.png')
    
    torch.save(lstm_model.state_dict(), 'SWaT/record/lstm_model.pth')
    torch.save(rnn_model.state_dict(), 'SWaT/record/rnn_model.pth')
    # torch.save(generator.state_dict(), '../record/generator_madgan.pth')
    # torch.save(discriminator.state_dict(), '../record/discriminator_madgan.pth')
    print('\n模型已保存: record/lstm_model.pth, record/rnn_model.pth')

if __name__ == '__main__':
    main()
