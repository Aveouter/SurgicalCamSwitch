import glob
import os
import random
from PIL import Image
from math import ceil  # 引入向上取整函数
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import InterpolationMode


class EgoImageDataset(Dataset):
    def __init__(self, image_dir, mode, surgery, stage = "train",augment=True):
        file_list = ['video20220722','video20220729', 'video20220801', 'video20221110', 'video20230315']
        if mode == 'Surgery-out' and stage == "train":
            file_list.remove(surgery)
        if mode == 'Surgery-out' and stage == "val":
            file_list = [surgery] # 只使用指定的手术视频

        sub_folders = ['1', '2', '3', '4', '5', '6']
        image_list =  []

        if stage == "train":
            for video in file_list:
                folder = os.path.join(
                    "/baksv/CIGIT/GXN_Liuxy/LenSe",
                    video,
                    "screenshots",
                    sub_folders[random.randint(0, 5)] # 随机选择一个子文件夹
                )
                all_images = sorted(glob.glob(os.path.join(folder, '**', '*.*'), recursive=True))
                all_images = [f for f in all_images if f.lower().endswith(('.jpg', '.png', '.jpeg')) and os.path.isfile(f)]

                if mode == 'Senquence-out':
                    num_use = ceil(len(all_images) * 0.7)
                    image_list.extend(all_images[:num_use])  # 前70%
                else:
                    image_list.extend(all_images)  # 全部加入
        else:  # stage == "val"
            pass

        self.image_files = image_list  # 已经提前处理好了
        self.augment = augment
        self.base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.augment_transform = transforms.Compose([
            transforms.RandomApply([
                transforms.Lambda(lambda img: transforms.Resize((224, 224))(
                    transforms.CenterCrop(size=random.randint(180, 200))(img)))
            ], p=0.5),
            
            transforms.RandomApply([
                transforms.RandomPerspective(distortion_scale=0.5, p=1.0, interpolation=InterpolationMode.BICUBIC)
            ], p=0.5),

            transforms.RandomApply([
                transforms.RandomRotation(degrees=(-180, 180))
            ], p=0.5),

            transforms.RandomApply([
                transforms.ColorJitter(brightness=(0.9, 1.65),
                                    contrast=(1.0, 1.35),
                                    saturation=(1.0, 1.4))
            ], p=0.5),

            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.5))
            ], p=0.5),

            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert("RGB")

        # 先做基础 resize / ToTensor 等
        anchor = self.base_transform(img)          # 原始 / 轻量处理

        if self.augment:
            positive = self.augment_transform(img) # 随机增强版
        else:
            positive = anchor                      # 不增强时两者相同

        return {"anchor": anchor, "positive": positive}

# ========== Loss ==========
class VAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + 1e-3 * kl_loss