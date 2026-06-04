import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, num_layers=2):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        dropout = 0.1 if num_layers > 1 else 0.0
        
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        return output


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        dropout = 0.1 if num_layers > 1 else 0.0
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        output = self.sigmoid(output)
        output = output.mean(dim=1)
        return output


class MADGANAnomalyDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=2):
        super(MADGANAnomalyDetector, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        self.generator = Generator(latent_dim, hidden_dim, input_dim, num_layers)
        self.discriminator = Discriminator(input_dim, hidden_dim, num_layers)
    
    def forward(self, x, mode='detect'):
        if mode == 'generate':
            return self.generator(x)
        elif mode == 'discriminate':
            return self.discriminator(x)
        else:
            fake = self.generator(x)
            score = self.discriminator(fake)
            return score
