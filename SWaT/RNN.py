import torch
import torch.nn as nn

# RNN模型
class RNNAnomalyDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate=0.0):
        super(RNNAnomalyDetector, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity='tanh',
            dropout=dropout_rate if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        rnn_out = self.dropout(rnn_out)
        predictions = self.fc(rnn_out)
        return predictions