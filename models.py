import torch
import torch.nn as nn

class LSTM_RUL(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=32, n_layers=5,
                 dropout=0.5, bid=True, device=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bid        = bid
        self.device     = device
        self.encoder = nn.LSTM(input_dim, hidden_dim, n_layers,
                               dropout=dropout, batch_first=True,
                               bidirectional=bid)
        out_dim = hidden_dim + hidden_dim * int(bid)
        self.regressor = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1)
        )
        self.feature_dim = out_dim

    def forward(self, src):
        enc_out, _ = self.encoder(src)     
        features   = enc_out[:, -1, :]    
        predictions = self.regressor(features)
        return predictions.squeeze(-1), features

class Generator(nn.Module):
    def __init__(self, input_dim=14):
        super().__init__()
        self.input_dim = input_dim
        self.encoder_1 = nn.Sequential(
            nn.Conv1d(input_dim, input_dim, 3, stride=2, padding=1, dilation=1),
            nn.LeakyReLU(),
            nn.Conv1d(input_dim, input_dim, 3, stride=2, padding=1, dilation=4),
            nn.LeakyReLU(),
            nn.MaxPool1d(5)
        )
        self.encoder_2 = nn.Sequential(
            nn.Conv1d(input_dim, input_dim, 3, stride=2, padding=1, dilation=1),
            nn.LeakyReLU(),
            nn.Conv1d(input_dim, input_dim, 5, stride=2, padding=1, dilation=2),
            nn.LeakyReLU(),
            nn.MaxPool1d(5)
        )
        self.encoder_3 = nn.Sequential(
            nn.Conv1d(input_dim, input_dim, 3, stride=2, padding=1, dilation=1),
            nn.LeakyReLU(),
            nn.Conv1d(input_dim, input_dim, 7, stride=2, padding=1, dilation=1),
            nn.LeakyReLU(),
            nn.MaxPool1d(6)
        )
        self.flat    = nn.Flatten()
        self.adapter = nn.Linear(42, 64)  

    def forward(self, src):
        src = src.permute(0, 2, 1)
        f1  = self.encoder_1(src)
        f2  = self.encoder_2(src)
        f3  = self.encoder_3(src)
        features = self.flat(torch.cat([f1, f2, f3], dim=2))
        hint = self.adapter(features)
        return features, hint

class CNN_Student(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=64, dropout=0.5):
        super().__init__()
        self.generator  = Generator(input_dim)
        self.regressor  = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        self.feature_dim = 64

    def forward(self, src):
        features, hint = self.generator(src)
        pred = self.regressor(hint)
        return pred.squeeze(-1), features, hint
