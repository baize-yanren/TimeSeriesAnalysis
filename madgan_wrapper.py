import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
import os
import madgan.constants as constants
from madgan.data import WindowDataset, LatentSpaceIterator, prepare_dataloader
from madgan.models import Generator, Discriminator
from madgan.engine import set_seed, train_one_epoch, evaluate


class MADGANAnomalyDetector:
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        latent_space_dim: int = constants.LATENT_SPACE_DIM,
        window_size: int = constants.WINDOW_SIZE,
        device: torch.device = None
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_space_dim = latent_space_dim
        self.window_size = window_size
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.generator = Generator(
            latent_space_dim=latent_space_dim,
            hidden_units=hidden_dim,
            output_dim=input_dim
        ).to(self.device)
        
        self.discriminator = Discriminator(
            input_dim=input_dim,
            hidden_units=hidden_dim,
            add_batch_mean=True
        ).to(self.device)
        
        self.train_losses = {'generator': [], 'discriminator': []}
        self.test_losses = {'generator': [], 'discriminator': []}
    
    def train(
        self,
        train_data: np.ndarray,
        test_data: np.ndarray = None,
        batch_size: int = 32,
        epochs: int = 8,
        lr: float = 1e-4,
        window_stride: int = constants.WINDOW_STRIDE,
        random_seed: int = 42
    ):
        set_seed(random_seed)
        
        train_df = pd.DataFrame(train_data)
        train_dataset = WindowDataset(
            train_df,
            window_size=self.window_size,
            window_slide=window_stride
        )
        train_dl = prepare_dataloader(train_dataset, batch_size=batch_size)
        
        if test_data is not None:
            test_df = pd.DataFrame(test_data)
            test_dataset = WindowDataset(
                test_df,
                window_size=self.window_size,
                window_slide=window_stride
            )
            test_dl = prepare_dataloader(test_dataset, batch_size=batch_size)
        else:
            test_dl = None
        
        latent_space = LatentSpaceIterator(
            noise_shape=[batch_size, self.window_size, self.latent_space_dim],
            device=self.device
        )
        
        discriminator_optim = optim.Adam(self.discriminator.parameters(), lr=lr)
        generator_optim = optim.Adam(self.generator.parameters(), lr=lr)
        
        criterion_fn = nn.BCELoss()
        
        print(f'\n开始训练 MAD-GAN (设备: {self.device})')
        print(f'训练数据集大小: {len(train_dataset)}')
        print(f'窗口大小: {self.window_size}, 批大小: {batch_size}, 轮数: {epochs}')
        
        for epoch in range(epochs):
            print(f'\n=== Epoch {epoch + 1}/{epochs} ===')
            
            gen_loss, disc_loss = train_one_epoch(
                generator=self.generator,
                discriminator=self.discriminator,
                loss_fn=criterion_fn,
                real_dataloader=train_dl,
                latent_dataloader=latent_space,
                discriminator_optimizer=discriminator_optim,
                generator_optimizer=generator_optim,
                normal_label=constants.REAL_LABEL,
                anomaly_label=constants.FAKE_LABEL,
                epoch=epoch
            )
            
            self.train_losses['generator'].append(gen_loss)
            self.train_losses['discriminator'].append(disc_loss)
            
            if test_dl is not None:
                gen_loss_test, disc_loss_test = evaluate(
                    generator=self.generator,
                    discriminator=self.discriminator,
                    real_dataloader=test_dl,
                    latent_dataloader=latent_space,
                    loss_fn=criterion_fn,
                    normal_label=constants.REAL_LABEL,
                    anomaly_label=constants.FAKE_LABEL
                )
                
                self.test_losses['generator'].append(gen_loss_test)
                self.test_losses['discriminator'].append(disc_loss_test)
                
                print(f'训练损失 - 生成器: {gen_loss:.6f}, 判别器: {disc_loss:.6f}')
                print(f'测试损失 - 生成器: {gen_loss_test:.6f}, 判别器: {disc_loss_test:.6f}')
            else:
                print(f'训练损失 - 生成器: {gen_loss:.6f}, 判别器: {disc_loss:.6f}')
        
        print('\nMAD-GAN 训练完成!')
    
    def compute_anomaly_scores(self, data: np.ndarray, batch_size: int = 32) -> np.ndarray:
        self.generator.eval()
        self.discriminator.eval()
        
        data_df = pd.DataFrame(data)
        dataset = WindowDataset(
            data_df,
            window_size=self.window_size,
            window_slide=self.window_size
        )
        dataloader = prepare_dataloader(dataset, batch_size=batch_size, shuffle=False)
        
        scores = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device).float()
                
                disc_scores = self.discriminator(batch)
                disc_scores = disc_scores.mean(dim=1).squeeze()
                
                scores.extend(disc_scores.cpu().numpy())
        
        return np.array(scores)
    
    def save(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        self.generator.save(os.path.join(model_dir, 'generator.pt'))
        self.discriminator.save(os.path.join(model_dir, 'discriminator.pt'))
        print(f'模型已保存到 {model_dir}')
    
    def load(self, model_dir: str):
        self.generator = Generator.from_pretrained(
            os.path.join(model_dir, 'generator.pt'),
            map_location=self.device
        )
        self.discriminator = Discriminator.from_pretrained(
            os.path.join(model_dir, 'discriminator.pt'),
            map_location=self.device
        )
        print(f'模型已从 {model_dir} 加载')
