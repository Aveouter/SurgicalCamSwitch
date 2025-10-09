# ---------- LSTM 模块 ----------
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim = 512 ):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):  # x: [B, T, D]
        lstm_out, _ = self.lstm(x)
        return self.mlp(lstm_out)  # [B, T, 1]

