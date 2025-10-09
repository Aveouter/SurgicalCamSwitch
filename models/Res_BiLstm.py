import torch
import torch.nn as nn
import torch.optim as optim
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

# 定义空间特征聚合模块
class SpatialFeatureAggregation(nn.Module):
    def __init__(self):
        super(SpatialFeatureAggregation, self).__init__()

    def forward(self, features):
        B, T, C = features.shape
        global_feature = torch.max(features, dim=2)[0]
        # # 将全局特征与每个局部特征连接起来
        # print(global_feature.shape)
        features = features.view(B, T, 6, 128)
        context_features = torch.cat([features, global_feature.unsqueeze(-1).unsqueeze(-1).expand_as(features)], dim=-1)
        # print(context_features.shape)
        return context_features


class SequentialSelectionModule(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SequentialSelectionModule, self).__init__()
        # 双向LSTM
        self.bilstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        # MLP用于计算选择概率
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),  # LSTM是双向的，因此隐藏维度需要乘以2
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # LSTM进行时序特征提取
        lstm_output, _ = self.bilstm(x)
        # MLP计算选择概率
        selection_probabilities = self.mlp(lstm_output)
        return selection_probabilities
    
class LinearFeatureExtractor(nn.Module):
    def __init__(self, input_dim, N, output_dim=128):
        super(LinearFeatureExtractor, self).__init__()
        self.N = N  # 分割份数
        self.output_dim = output_dim  # 每份特征缩小后的维度
        self.split_dim = (input_dim - 1) // N  
        
        # 定义多个线性层，每个线性层用于处理一份数据
        self.linears = nn.ModuleList([nn.Linear(self.split_dim, self.output_dim) for _ in range(N)])
    
    def forward(self, x):
        B, T, C = x.shape
        assert (C - 1) % self.N == 0, "去除第一列和最后一列后，输入的最后一个维度必须能被 N 整除"
        
        last_col = x[:, :, -1].unsqueeze(-1)  # [B, T, 1]
        x_middle = x[:, :, :-1]
        # print(B, T, C)
        # print(self.output_dim)
        # exit()
        x_split = x_middle.view(B, T, self.N, self.split_dim)
        x_transformed = []
        for i in range(self.N):
            x_transformed.append(self.linears[i](x_split[:, :, i]))  # 对第 i 份应用线性层
        x_out = torch.cat(x_transformed, dim=-1)
        x_final = x_out
        return x_final
    
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.final = SequentialSelectionModule(256, 64)
        self.spatial_aggregation = SpatialFeatureAggregation()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len        
        self.camera = configs.camera
        self.N = 6
        self.LinearFeatureExtractor = LinearFeatureExtractor(configs.enc_in,self.N)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # standard
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        B, T, C = x_enc.shape
        # embedding
        enc_out = self.LinearFeatureExtractor(x_enc)
        context_features = self.spatial_aggregation(enc_out)
        probabilities = []
        for i in range(self.N):
            probability = self.final(context_features[:, :, i, :])
            # print(probability.shape)
            probabilities.append(probability)
            # selection_probabilities = selection_probabilities.view(B, T, N)
        probabilities = torch.cat(probabilities, dim=2)
        # print(probabilities.shape)
        return probabilities[:,-self.pred_len:,:]



