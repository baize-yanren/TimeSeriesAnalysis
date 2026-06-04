import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import os
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

from preProcessing import load_and_preprocess_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# ==================== 1. 环境模块（滑动窗口构造） ====================
class TimeSeriesEnvironment:
    """将原始时间序列转换为强化学习环境"""
    def __init__(self, data, window_len, step_size=1):
        """
        data: (T, D) 多维时间序列 (训练时仅使用正常数据)
        window_len: 滑动窗口长度
        step_size: 滑动步长
        """
        self.data = data
        self.window_len = window_len
        self.step_size = step_size
        self.num_samples = (len(data) - window_len) // step_size + 1
        self.current_idx = 0

    def reset(self):
        self.current_idx = 0
        return self._get_state(self.current_idx)

    def step(self, action):
        next_idx = min(self.current_idx + 1, self.num_samples - 1)
        next_state = self._get_state(next_idx)
        done = (next_idx == self.num_samples - 1)
        self.current_idx = next_idx
        return next_state, 0.0, done, {}

    def _get_state(self, idx):
        start = idx * self.step_size
        end = start + self.window_len
        return self.data[start:end]

# ==================== 2. 核心网络模块（Transformer + Q网络） ====================
class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]

class TransformerAutoencoder(nn.Module):
    """Transformer编码器-解码器，用于重构"""
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                    dim_feedforward=dim_feedforward,
                                                    batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                                    dim_feedforward=dim_feedforward,
                                                    batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, input_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x_emb = self.input_proj(x)
        x_emb = self.pos_enc(x_emb)
        memory = self.encoder(x_emb)
        tgt = x_emb
        out = self.decoder(tgt, memory)
        recon = self.output_proj(out)
        return recon, memory

class QNetwork(nn.Module):
    """主网络：从潜在表示预测Q值"""
    def __init__(self, d_model, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, z):
        # z: (batch, d_model) 或 (batch, seq_len, d_model) -> 取seq维度平均
        if z.dim() == 3:
            z = z.mean(dim=1)  # (batch, d_model)
        h = F.relu(self.fc1(z))
        return self.fc2(h)

# ==================== 3. 经验回放模块 ====================
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards).astype(np.float32),
                np.array(next_states))

    def __len__(self):
        return len(self.buffer)

# ==================== 4. 奖励自适应模块 ====================
class RewardSelfAdjustment:
    """基于对数增长调节奖励"""
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
        self.t = 0

    def compute(self, recon_error):
        self.t += 1
        if self.t == 1:
            f_t = 0.0
        else:
            f_t = np.log(self.t - 1 + self.epsilon) #ft=log(t-1+ε)
        reward = (1.0 / (recon_error + 1.0)) * (1.0 / (f_t + 1.0)) #Rt=1/(1+recon_error)*1/(ft+1)
        return reward

# ==================== 5. 目标网络模块 ====================
class TargetNetwork:
    """目标网络，定期从主网络复制参数"""
    def __init__(self, network):
        self.network = network
        self.target_network = self._clone(network)

    def _clone(self, net):
        import copy
        return copy.deepcopy(net)

    def update(self, network):
        self.target_network.load_state_dict(network.state_dict())

    def __call__(self, z):
        return self.target_network(z)

# ==================== 6. 完整TRLAD模型 ====================
class TRLAD(nn.Module):
    def __init__(self, input_dim, window_len=10, d_model=64, nhead=4, num_layers=2,
                 lr=1e-4, gamma=0.99, batch_size=64):
        super().__init__()
        self.input_dim = input_dim
        self.window_len = window_len
        self.gamma = gamma
        self.batch_size = batch_size

        self.autoencoder = TransformerAutoencoder(input_dim, d_model, nhead, num_layers)
        self.q_network = QNetwork(d_model)
        self.target_q = QNetwork(d_model)
        self.target_q.load_state_dict(self.q_network.state_dict())
        
        self.to(device)

        params = list(self.autoencoder.parameters()) + list(self.q_network.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)

        self.replay_buffer = ReplayBuffer(capacity=20000)

        self.reward_adj = RewardSelfAdjustment()

        self.train_step = 0

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        recon, z = self.autoencoder(x)
        q_values = self.q_network(z)
        return recon, q_values, z

    def select_action(self, state, epsilon=0.1):
        """ε-贪婪策略"""
        if random.random() < epsilon:
            return random.randint(0, 1)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            _, q, _ = self.forward(state_t)
            return q.argmax().item()

    def train_step_func(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)

        recon, q, z = self.forward(states) #当前Q值
        q_sa = q.gather(1, actions) #当前状态-动作对的Q值

        # target = r + γ × Q_target(s', argmax Q(s'))
        with torch.no_grad():
            _, next_q, next_z = self.forward(next_states)
            next_actions = next_q.argmax(1, keepdim=True)
            target_q = self.target_q(next_z)
            target_q_sa = target_q.gather(1, next_actions)
            target = rewards + self.gamma * target_q_sa

        L_rec = F.mse_loss(recon, states) #重构误差损失
        L_Q = F.mse_loss(q_sa, target) #Q值损失
        total_loss = L_rec + L_Q #总损失

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.train_step += 1
        if self.train_step % 10 == 0:
            self.target_q.load_state_dict(self.q_network.state_dict())
        
        return total_loss.item()
    
    def save_model(self, filepath):
        """保存模型参数"""
        torch.save({
            'autoencoder_state_dict': self.autoencoder.state_dict(),
            'q_network_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_step': self.train_step
        }, filepath)
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath):
        """加载模型参数"""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath)
            self.autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.train_step = checkpoint['train_step']
            print(f"模型已从 {filepath} 加载")
            return True
        return False

# ==================== 7. 训练算法 (Algorithm 1) ====================
def train_TRLAD(model, env, test_data=None, labels=None, window_len=10, 
                episodes=1000, epsilon_start=0.9, epsilon_end=0.05, 
                epsilon_decay=200, save_interval=50, eval_interval=30,
                eval_batch_size=1024):
    epsilon = epsilon_start
    episode_rewards = []
    episode_losses = []
    eval_metrics = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        episode_loss = 0
        loss_count = 0
        done = False
        
        while not done:
            action = model.select_action(state, epsilon)
            next_state, _, done, _ = env.step(action)
            
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                recon, _, _ = model.forward(state_t)
                err = F.mse_loss(recon, state_t).item()
            reward = model.reward_adj.compute(err)
            
            model.replay_buffer.push(state, action, reward, next_state)
            
            loss = model.train_step_func()
            if loss is not None:
                episode_loss += loss
                loss_count += 1
            
            total_reward += reward
            state = next_state
        
        avg_loss = episode_loss / loss_count if loss_count > 0 else 0
        episode_rewards.append(total_reward)
        episode_losses.append(avg_loss)
        
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-episode / epsilon_decay)
        
        if episode % eval_interval == 0:
             metrics_str = f"Episode {episode}/{episodes}, "
             metrics_str += f"loss: {avg_loss:.4f}, "
             metrics_str += f"reward: {total_reward:.4f}, "
             metrics_str += f"epsilon: {epsilon:.3f}"
             
             if test_data is not None and labels is not None:
                 scores = detect_anomalies(model, test_data, window_len, batch_size=eval_batch_size)
                 threshold = np.percentile(scores[:len(test_data)//2], 80)
                 metrics = evaluate_anomaly_detection(scores, labels, threshold, window_len)
                 metrics_str += f", Acc: {metrics['accuracy']:.4f}, "
                 metrics_str += f"Prec: {metrics['precision']:.4f}, "
                 metrics_str += f"Rec: {metrics['recall']:.4f}, "
                 metrics_str += f"F1: {metrics['f1']:.4f}"
                 eval_metrics.append({
                     'episode': episode,
                     'accuracy': metrics['accuracy'],
                     'precision': metrics['precision'],
                     'recall': metrics['recall'],
                     'f1': metrics['f1']
                 })
             
             print(metrics_str, flush=True)
        
        if episode > 0 and episode % save_interval == 0:
            save_path = f"record/trlad_model_ep{episode}.pth"
            model.save_model(save_path)
    
    return model, {
        'rewards': episode_rewards,
        'losses': episode_losses,
        'metrics': eval_metrics
    }

# ==================== 8. 异常检测函数 ====================
def detect_anomalies(model, test_data, window_len, batch_size=256, threshold=None):
    """
    使用训练好的模型对测试数据进行异常检测
    返回每个时间点的异常分数（重构误差）
    使用批量处理提高性能
    """
    model.eval()
    errors = []
    
    with torch.no_grad():
        for i in range(0, len(test_data) - window_len + 1, batch_size):
            batch_end = min(i + batch_size, len(test_data) - window_len + 1)
            batch_windows = []
            for j in range(i, batch_end):
                window = test_data[j:j+window_len]
                batch_windows.append(window)
            
            batch_tensor = torch.FloatTensor(np.array(batch_windows)).to(device)
            recon, _, _ = model.forward(batch_tensor)
            batch_errors = F.mse_loss(recon, batch_tensor, reduction='none').mean(dim=(1, 2)).cpu().numpy()
            errors.extend(batch_errors.tolist())
    
    anomaly_scores = np.zeros(len(test_data))
    for i, e in enumerate(errors):
        pos = i + window_len - 1
        anomaly_scores[pos] = e
    
    return anomaly_scores

def evaluate_anomaly_detection(scores, labels, threshold, window_len):
    """
    评估异常检测性能
    scores: 异常分数
    labels: 真实标签 (0=正常, 1=异常)
    threshold: 阈值
    window_len: 滑动窗口长度
    """
    predictions = (scores > threshold).astype(int)
    
    valid_len = len(scores) - window_len + 1
    predictions = predictions[window_len-1:]
    valid_labels = labels[:valid_len]
    
    accuracy = accuracy_score(valid_labels, predictions)
    precision = precision_score(valid_labels, predictions, zero_division=0)
    recall = recall_score(valid_labels, predictions, zero_division=0)
    f1 = f1_score(valid_labels, predictions, zero_division=0)
    cm = confusion_matrix(valid_labels, predictions)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': predictions,
        'valid_labels': valid_labels
    }

# ==================== 示例用法 ====================
if __name__ == "__main__":
    os.makedirs("SWaT/record", exist_ok=True)
    DATA_PATH = 'data/SWaT/'
    NORMAL_FILE = os.path.join(DATA_PATH, 'normal.csv')
    ATTACK_FILE = os.path.join(DATA_PATH, 'attack.csv')
    
    normal_data, attack_data, attack_labels, scaler = load_and_preprocess_data(NORMAL_FILE, ATTACK_FILE, SAMPLE_RATIO=0.01)
    
    D = normal_data.shape[1]
    window_len = 10
    
    print(f"特征维度: {D}")
    print(f"正常数据样本数: {len(normal_data)}")
    print(f"攻击数据样本数: {len(attack_data)}")
    
    train_data = normal_data
    
    env = TimeSeriesEnvironment(train_data, window_len, step_size=2)  # 步长2，加快训练
    model = TRLAD(input_dim=D, window_len=window_len, d_model=48, nhead=2, num_layers=1, lr=5e-5)
    
    model_path = "SWaT/record/trlad_model_latest.pth"
    if model.load_model(model_path):
        print("成功加载已有模型，继续训练...")
    else:
        print("未找到已有模型，从头开始训练...")
    
    test_data = np.concatenate([normal_data[-500:], attack_data[:200]], axis=0)
    true_labels = np.concatenate([np.zeros(500), attack_labels[:200]], axis=0)
    
    print("\n开始训练 TRLAD...")
    # print("提示：训练过程中不进行评估，训练完成后统一评估以提高速度")
    trained_model, train_history = train_TRLAD(
        model, env,
        test_data=None,  # 训练时不评估
        labels=None,
        window_len=window_len,
        episodes=10,
        epsilon_start=0.9,
        epsilon_end=0.1,
        epsilon_decay=100,
        save_interval=50,
        eval_interval=1024  # 几乎不评估
    )
    
    trained_model.save_model("SWaT/record/trlad_model_latest.pth")
    
    print("\n" + "="*50)
    print("训练完成，进行最终评估...")
    print("="*50)
    
    eval_data = np.concatenate([normal_data[-500:], attack_data[:200]], axis=0)
    eval_labels = np.concatenate([np.zeros(500), attack_labels[:200]], axis=0)
    
    scores = detect_anomalies(trained_model, eval_data, window_len)
    threshold = np.percentile(scores[:500], 80)
    final_metrics = evaluate_anomaly_detection(scores, eval_labels, threshold, window_len)
    
    print(f"\n最终评估结果:")
    print(f"准确率 (Accuracy):  {final_metrics['accuracy']:.4f}")
    print(f"精确率 (Precision): {final_metrics['precision']:.4f}")
    print(f"召回率 (Recall):    {final_metrics['recall']:.4f}")
    print(f"F1分数 (F1):        {final_metrics['f1']:.4f}")
    print(f"\n混淆矩阵:")
    print(final_metrics['confusion_matrix'])
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(train_history['losses'])
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Training Loss over Episodes')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(train_history['rewards'])
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Episode Rewards over Episodes')
    plt.grid(True)
    
    if train_history['metrics']:
        episodes = [m['episode'] for m in train_history['metrics']]
        plt.subplot(2, 2, 3)
        plt.plot(episodes, [m['accuracy'] for m in train_history['metrics']], label='Accuracy')
        plt.plot(episodes, [m['precision'] for m in train_history['metrics']], label='Precision')
        plt.plot(episodes, [m['recall'] for m in train_history['metrics']], label='Recall')
        plt.plot(episodes, [m['f1'] for m in train_history['metrics']], label='F1')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.title('Evaluation Metrics over Episodes')
        plt.legend()
        plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(scores, label='Anomaly Score', alpha=0.7)
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.4f})')
    plt.xlabel('Time')
    plt.ylabel('Score')
    plt.title('Anomaly Scores and Threshold')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('SWaT/record/training_evaluation.png', dpi=150)
    print(f"\n训练和评估图表已保存到: record/training_evaluation.png")
    plt.close()
    
    print(f"\n检测完成")
    print(f"预测异常点数: {final_metrics['predictions'].sum()}")
    print(f"真实异常点数: {final_metrics['valid_labels'].sum()}")
