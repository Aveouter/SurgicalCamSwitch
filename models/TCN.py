# tcn_classification_model.py

import torch
import torch.nn as nn

# ---------- Temporal Convolution Block ----------
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        res = self.downsample(x)
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        if out.shape[-1] != res.shape[-1]:
            min_len = min(out.shape[-1], res.shape[-1])
            out = out[..., :min_len]
            res = res[..., :min_len]
        return self.relu(out + res)

# ---------- TCN Model for Multi-Step Classification ----------
class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        input_dim = args.enc_in         # 输入特征数，比如3901
        hidden_dim = getattr(args, 'hidden_dim', 512)
        num_layers = getattr(args, 'num_layers', 3)
        kernel_size = getattr(args, 'kernel_size', 3)
        self.pred_len = getattr(args, 'pred_len', 6)  # ✅ 预测长度（固定为6）
        output_dim = 6  # ✅ 分类类别数（固定为6）

        # TCN 层堆叠
        layers = []
        for i in range(num_layers):
            in_c = input_dim if i == 0 else hidden_dim
            layers.append(TemporalBlock(in_c, hidden_dim, kernel_size, dilation=2 ** i))
        self.network = nn.Sequential(*layers)

        # MLP 分类头
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, output_dim)  # ✅ 输出为6类
        )

    def forward(self, x_enc, *args, **kwargs):
        """
        x_enc: [B, T, C]  输入序列（例如历史24步，每步3901维）
        return: [B, T, 6] 每个时间步的分类 logits
        """
        # print(f"Input shape: {x_enc.shape}")
        x_enc = x_enc.transpose(1, 2)       # -> [B, C, T]
        out = self.network(x_enc)           # -> [B, H, T]
        out = out.transpose(1, 2)           # -> [B, T, H]
        out = self.mlp(out)                 # -> [B, T, 6]

        return out
