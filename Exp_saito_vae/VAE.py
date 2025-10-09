import os
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import pandas as pd

# ========== Residual Blocks ==========
class ResidualDecodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False, stride=1):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) if upsample else nn.Identity()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.upsample(x)
        identity = self.shortcut(out)
        out = self.relu(self.bn1(self.conv1(out)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        return self.relu(out + identity)

# ========== VAE Model ==========
class VAE_ResNet18_Encoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        self.layer1 = self._make_layer(64, 64, 2, 1)
        self.layer2 = self._make_layer(64, 128, 2, 2)
        self.layer3 = self._make_layer(128, 256, 2, 2)
        self.layer4 = self._make_layer(256, 512, 2, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(512, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        for i in range(blocks):
            s = stride if i == 0 else 1
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, s),
                nn.BatchNorm2d(out_channels)
            ) if i == 0 and (s != 1 or in_channels != out_channels) else None
            layers.append(BasicBlock(in_channels if i == 0 else out_channels, out_channels, s, downsample))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x).view(x.size(0), -1)
        x = self.fc(x)
        return self.fc_mu(x), self.fc_logvar(x)

class VAE_ResNet18_Decoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(latent_dim, 512), nn.BatchNorm1d(512), nn.ReLU())
        self.init_upsample = nn.Upsample(scale_factor=7, mode='bilinear', align_corners=True)
        self.blocks = nn.Sequential(
            ResidualDecodeBlock(512, 512, upsample=True),
            ResidualDecodeBlock(512, 512),
            ResidualDecodeBlock(512, 256, upsample=True),
            ResidualDecodeBlock(256, 256),
            ResidualDecodeBlock(256, 128, upsample=True),
            ResidualDecodeBlock(128, 128),
            ResidualDecodeBlock(128, 64, upsample=True),
            ResidualDecodeBlock(64, 64)
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z).unsqueeze(-1).unsqueeze(-1)
        x = self.init_upsample(x)
        x = self.blocks(x)
        return self.deconv(x)

class VAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.encoder = VAE_ResNet18_Encoder(latent_dim)
        self.decoder = VAE_ResNet18_Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar