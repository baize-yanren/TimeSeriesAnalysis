import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import os
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path = 'C:/Windows/Fonts/simhei.ttf'
font_prop = fm.FontProperties(fname=font_path)
matplotlib.rcParams['font.sans-serif'] = [font_prop.get_name()]
matplotlib.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preProcessing as pp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

class TimeSeriesEnvironment:
    def __init__(self, data, window_len, step_size=1):
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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerAutoencoder(nn.Module):
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
        x_emb = self.input_proj(x)
        x_emb = self.pos_enc(x_emb)
        memory = self.encoder(x_emb)
        tgt = x_emb
        out = self.decoder(tgt, memory)
        recon = self.output_proj(out)
        return recon, memory

class QNetwork(nn.Module):
    def __init__(self, d_model, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, z):
        if z.dim() == 3:
            z = z.mean(dim=1)
        h = F.relu(self.fc1(z))
        return self.fc2(h)

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

class RewardSelfAdjustment:
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
        self.t = 0

    def compute(self, recon_error):
        self.t += 1
        if self.t == 1:
            f_t = 0.0
        else:
            f_t = np.log(self.t - 1 + self.epsilon)
        reward = (1.0 / (recon_error + 1.0)) * (1.0 / (f_t + 1.0))
        return reward

class TargetNetwork:
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
        recon, z = self.autoencoder(x)
        q_values = self.q_network(z)
        return recon, q_values, z

    def select_action(self, state, epsilon=0.1):
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

        recon, q, z = self.forward(states)
        q_sa = q.gather(1, actions)

        with torch.no_grad():
            _, next_q, next_z = self.forward(next_states)
            next_actions = next_q.argmax(1, keepdim=True)
            target_q = self.target_q(next_z)
            target_q_sa = target_q.gather(1, next_actions)
            target = rewards + self.gamma * target_q_sa

        L_rec = F.mse_loss(recon, states)
        L_Q = F.mse_loss(q_sa, target)
        total_loss = L_rec + L_Q

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.train_step += 1
        if self.train_step % 10 == 0:
            self.target_q.load_state_dict(self.q_network.state_dict())

        return total_loss.item()
    
    def save_model(self, filepath):
        torch.save({
            'autoencoder_state_dict': self.autoencoder.state_dict(),
            'q_network_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_step': self.train_step
        }, filepath)
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath):
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath)
            self.autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.train_step = checkpoint['train_step']
            print(f"模型已从 {filepath} 加载")
            return True
        return False

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

def detect_anomalies(model, test_data, window_len, batch_size=256, threshold=None):
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

if __name__ == "__main__":
    os.makedirs("record", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'MBA')
    TRAIN_FILE = os.path.join(DATA_PATH, 'train.xlsx')
    TEST_FILE = os.path.join(DATA_PATH, 'test.xlsx')
    LABELS_FILE = os.path.join(DATA_PATH, 'labels.xlsx')
    
    train_data, test_data, test_labels, scaler = pp.load_and_preprocess_data(
        TRAIN_FILE, TEST_FILE, LABELS_FILE, SAMPLE_RATIO=0.5)
    
    D = train_data.shape[1]
    window_len = 10
    
    print(f"特征维度: {D}")
    print(f"训练数据样本数: {len(train_data)}")
    print(f"测试数据样本数: {len(test_data)}")
    print(f"异常样本数量: {int(np.sum(test_labels))}")
    
    env = TimeSeriesEnvironment(train_data, window_len, step_size=2)
    model = TRLAD(input_dim=D, window_len=window_len, d_model=48, nhead=2, num_layers=1, lr=5e-5)
    
    model_path = "record/trlad_model_latest.pth"
    if model.load_model(model_path):
        print("成功加载已有模型，继续训练...")
    else:
        print("未找到已有模型，从头开始训练...")
    
    print("\n开始训练 TRLAD...")
    trained_model, train_history = train_TRLAD(
        model, env,
        test_data=None,
        labels=None,
        window_len=window_len,
        episodes=30,
        epsilon_start=0.9,
        epsilon_end=0.1,
        epsilon_decay=100,
        save_interval=15,
        eval_interval=3
    )
    
    trained_model.save_model("record/trlad_model_latest.pth")
    
    print("\n" + "="*50)
    print("训练完成，进行最终评估...")
    print("="*50)
    
    scores = detect_anomalies(trained_model, test_data, window_len)
    threshold = np.percentile(scores[:len(test_data)//2], 80)
    final_metrics = evaluate_anomaly_detection(scores, test_labels, threshold, window_len)
    
    print("\n最终评估结果:")
    print(f"准确率 (Accuracy):  {final_metrics['accuracy']:.4f}")
    print(f"精确率 (Precision): {final_metrics['precision']:.4f}")
    print(f"召回率 (Recall):    {final_metrics['recall']:.4f}")
    print(f"F1分数 (F1):        {final_metrics['f1']:.4f}")
    print(f"\n混淆矩阵:\n{final_metrics['confusion_matrix']}")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].plot(train_history['losses'])
    axes[0, 0].set_title('训练损失')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Loss')
    
    axes[0, 1].plot(train_history['rewards'])
    axes[0, 1].set_title('奖励曲线')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Reward')
    
    if len(train_history['metrics']) > 0:
        episodes = [m['episode'] for m in train_history['metrics']]
        axes[1, 0].plot(episodes, [m['accuracy'] for m in train_history['metrics']], label='Accuracy')
        axes[1, 0].plot(episodes, [m['precision'] for m in train_history['metrics']], label='Precision')
        axes[1, 0].plot(episodes, [m['recall'] for m in train_history['metrics']], label='Recall')
        axes[1, 0].plot(episodes, [m['f1'] for m in train_history['metrics']], label='F1')
        axes[1, 0].set_title('评估指标')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].legend()
    
    normal_scores = scores[test_labels == 0]
    anomaly_scores = scores[test_labels == 1]
    axes[1, 1].hist(normal_scores, bins=50, alpha=0.5, label='正常数据', density=True)
    axes[1, 1].hist(anomaly_scores, bins=50, alpha=0.5, label='异常数据', density=True)
    axes[1, 1].axvline(threshold, color='r', linestyle='--', label=f'阈值 ({threshold:.4f})')
    axes[1, 1].set_title('异常分数分布')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('results/trlad_training_evaluation.png')
    print(f"\n训练和评估图表已保存到: results/trlad_training_evaluation.png")
