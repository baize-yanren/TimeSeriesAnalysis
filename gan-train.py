import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

DATA_PATH = 'data/SWaT/'
NORMAL_FILE = os.path.join(DATA_PATH, 'normal.csv')
ATTACK_FILE = os.path.join(DATA_PATH, 'attack.csv')

WINDOW_SIZE = 50
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
HIDDEN_DIM = 128
NUM_LAYERS = 2
LATENT_DIM = 64

class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, num_layers=2):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        dropout = 0.2 if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=latent_dim, 
            hidden_size=hidden_dim,
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout,
            bidirectional=False
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        output = self.tanh(output)
        return output

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        dropout = 0.2 if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim,
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout,
            bidirectional=False
        )
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        output = self.fc(last_out)
        return output

class SWaTDataset(Dataset):
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size
    
    def __len__(self):
        return len(self.data) - self.window_size
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx:idx+self.window_size])

def load_data(normal_file, attack_file):
    df_normal = pd.read_csv(normal_file)
    df_attack = pd.read_csv(attack_file)
    
    df_normal.columns = df_normal.columns.str.strip()
    df_attack.columns = df_attack.columns.str.strip()
    
    df_normal = df_normal.drop(['Timestamp', 'Normal/Attack'], axis=1)
    df_attack = df_attack.drop(['Timestamp', 'Normal/Attack'], axis=1)
    
    normal_data = df_normal.values
    attack_data = df_attack.values
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    normal_data = scaler.fit_transform(normal_data)
    attack_data = scaler.transform(attack_data)
    
    attack_labels = np.ones(len(attack_data))
    
    return normal_data, attack_data, attack_labels, scaler

def compute_anomaly_scores(discriminator, data_loader, device):
    discriminator.eval()
    scores = []
    sigmoid = nn.Sigmoid()
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device).float()
            disc_logits = discriminator(batch)
            disc_scores = sigmoid(disc_logits)
            scores.extend(disc_scores.cpu().numpy().squeeze())
    
    return np.array(scores)

def train_gan(generator, discriminator, train_loader, device, epochs=100, lr=2e-5):
    criterion = nn.BCEWithLogitsLoss()
    
    gen_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=lr * 0.3, betas=(0.5, 0.999))
    
    gen_scheduler = optim.lr_scheduler.StepLR(gen_optimizer, step_size=30, gamma=0.5)
    disc_scheduler = optim.lr_scheduler.StepLR(disc_optimizer, step_size=30, gamma=0.5)
    
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
            
            real_logits = discriminator(batch)
            fake_logits = discriminator(fake.detach())
            
            disc_loss_real = criterion(real_logits, real_labels)
            disc_loss_fake = criterion(fake_logits, fake_labels)
            disc_loss = disc_loss_real + disc_loss_fake
            
            if not torch.isnan(disc_loss):
                disc_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 0.5)
                disc_optimizer.step()
            
            gen_optimizer.zero_grad()
            fake_logits = discriminator(fake)
            gen_loss = criterion(fake_logits, real_labels)
            
            if not torch.isnan(gen_loss):
                gen_loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), 0.5)
                gen_optimizer.step()
            
            epoch_disc_loss += disc_loss.item() if not torch.isnan(disc_loss) else 0.0
            epoch_gen_loss += gen_loss.item() if not torch.isnan(gen_loss) else 0.0
            num_batches += 1
        
        avg_gen_loss = epoch_gen_loss / num_batches
        avg_disc_loss = epoch_disc_loss / num_batches
        
        gen_scheduler.step()
        disc_scheduler.step()
        
        if np.isnan(avg_gen_loss) or np.isnan(avg_disc_loss):
            print(f'NaN detected at Epoch {epoch+1}, stopping training')
            break
        
        gen_losses.append(avg_gen_loss)
        disc_losses.append(avg_disc_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Gen Loss: {avg_gen_loss:.6f}, Disc Loss: {avg_disc_loss:.6f}')
    
    return gen_losses, disc_losses

def main():
    print('加载数据...')
    normal_data, attack_data, attack_labels, scaler = load_data(NORMAL_FILE, ATTACK_FILE)
    input_dim = normal_data.shape[1]
    
    print(f'正常数据形状: {normal_data.shape}')
    print(f'攻击数据形状: {attack_data.shape}')
    print(f'特征数量: {input_dim}')
    
    print('创建数据集...')
    normal_dataset = SWaTDataset(normal_data, WINDOW_SIZE)
    print(f'正常数据集大小: {len(normal_dataset)}')
    
    train_size = int(0.8 * len(normal_dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(
        normal_dataset, [train_size, len(normal_dataset) - train_size]
    )
    print(f'训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}')
    
    print('创建DataLoader...')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    attack_dataset = SWaTDataset(attack_data, WINDOW_SIZE)
    attack_loader = DataLoader(attack_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f'攻击数据集大小: {len(attack_dataset)}')
    
    print('\n初始化MAD-GAN模型...')
    generator = Generator(LATENT_DIM, HIDDEN_DIM, input_dim, NUM_LAYERS).to(device)
    discriminator = Discriminator(input_dim, HIDDEN_DIM, NUM_LAYERS).to(device)
    
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        elif classname.find('LSTM') != -1:
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0)
    
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    print('开始训练MAD-GAN...')
    print(f'Epochs: {EPOCHS}, Learning Rate: {LEARNING_RATE}')
    gen_losses, disc_losses = train_gan(generator, discriminator, train_loader, device, EPOCHS, LEARNING_RATE)
    
    print('\n计算MAD-GAN异常分数...')
    madgan_normal_scores = compute_anomaly_scores(discriminator, val_loader, device)
    madgan_attack_scores = compute_anomaly_scores(discriminator, attack_loader, device)
    
    madgan_y_true = np.concatenate([np.zeros(len(madgan_normal_scores)), np.ones(len(madgan_attack_scores))])
    madgan_y_scores = np.concatenate([madgan_normal_scores, madgan_attack_scores])
    
    valid_mask = ~np.isnan(madgan_y_scores)
    madgan_y_true = madgan_y_true[valid_mask]
    madgan_y_scores = madgan_y_scores[valid_mask]
    
    madgan_auc = roc_auc_score(madgan_y_true, madgan_y_scores)
    print(f'MAD-GAN AUC-ROC: {madgan_auc:.4f}')
    
    if len(gen_losses) > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(gen_losses, label='Generator Loss')
        plt.plot(disc_losses, label='Discriminator Loss')
        plt.title('MAD-GAN Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('gan_training_loss.png')
        print('训练损失图已保存为 gan_training_loss.png')
    
    if len(madgan_y_scores) > 0:
        plt.figure(figsize=(10, 5))
        madgan_fpr, madgan_tpr, _ = roc_curve(madgan_y_true, madgan_y_scores)
        plt.plot(madgan_fpr, madgan_tpr, label=f'MAD-GAN (AUC = {madgan_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title('MAD-GAN ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.savefig('gan_roc_curve.png')
        print('ROC曲线已保存为 gan_roc_curve.png')
        
        plt.figure(figsize=(10, 5))
        madgan_normal_scores = madgan_normal_scores[~np.isnan(madgan_normal_scores)]
        madgan_attack_scores = madgan_attack_scores[~np.isnan(madgan_attack_scores)]
        plt.hist(madgan_normal_scores, bins=50, alpha=0.5, label='Normal', density=True)
        plt.hist(madgan_attack_scores, bins=50, alpha=0.5, label='Attack', density=True)
        plt.title(f'MAD-GAN Anomaly Score Distribution (AUC={madgan_auc:.4f})')
        plt.xlabel('Discriminator Score')
        plt.ylabel('Density')
        plt.legend()
        plt.savefig('gan_anomaly_dist.png')
        print('异常分数分布图已保存为 gan_anomaly_dist.png')
    
    os.makedirs('record', exist_ok=True)
    torch.save(generator.state_dict(), 'record/gan_generator.pth')
    torch.save(discriminator.state_dict(), 'record/gan_discriminator.pth')
    print('\n模型已保存: record/gan_generator.pth, record/gan_discriminator.pth')

if __name__ == '__main__':
    main()